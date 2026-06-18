from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vla.bev_qa.dataloader import BEVQADataset, bev_qa_collate_fn
from vla.bev_qa.model import BEVGemmaQA


@dataclass(frozen=True)
class TrainBEVQAConfig:
    qa_dir: str = "/zfsauton/scratch/mineuih/waymax_rs/qa_dataset/"
    img_root: str = "/zfsauton/scratch/eshau/imgs_past/"
    output_dir: str = "/home/scratch/sbellad/waymax_rs/bev_qa_output_gemma"
    gemma_name: str = "google/gemma-4-E2B-it"

    file_indices: list[int] | None = None
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 1
    max_steps: int | None = None
    max_samples: int | None = None
    validation_fraction: float = 0.04
    seed: int = 0

    freeze_gemma: bool = True
    freeze_vision: bool = True

    max_prompt_length: int = 128
    max_answer_length: int = 8
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    dtype: str = "bf16"

    log_every: int = 10
    save_every: int = 1000
    eval_every: int = 500
    eval_num_samples: int = 64

    num_workers: int = 0
    pin_memory: bool = True

    wandb_project: str | None = "waymax_rs_bev_qa"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str = "online"


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def _split_indices(n: int, validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = int(round(n * validation_fraction))
    if validation_fraction > 0 and n > 1:
        n_val = max(1, n_val)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    return train_indices, val_indices


def _build_loaders(cfg: TrainBEVQAConfig) -> tuple[DataLoader, DataLoader | None]:
    dataset = BEVQADataset(
        qa_dir=cfg.qa_dir,
        img_root=cfg.img_root,
        file_indices=cfg.file_indices,
        max_samples=cfg.max_samples,
        one_qa_per_scenario=True,
        seed=cfg.seed,
    )

    print(f"[dataset] samples={len(dataset)}")
    print(f"[dataset] missing_images={dataset.missing_images}")
    print(f"[dataset] skipped_no_qas={dataset.skipped_no_qas}")

    train_indices, val_indices = _split_indices(
        len(dataset),
        cfg.validation_fraction,
        cfg.seed,
    )

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices) if val_indices else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=bev_qa_collate_fn,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            collate_fn=bev_qa_collate_fn,
        )

    return train_loader, val_loader


@torch.no_grad()
def evaluate(
    model: BEVGemmaQA,
    loader: DataLoader,
    max_samples: int,
    max_prompt_length: int,
    max_answer_length: int,
) -> dict[str, float]:
    model.eval()

    total = 0
    correct = 0

    for batch in loader:
        remaining = max_samples - total
        if remaining <= 0:
            break

        images = batch["images"][:remaining]
        questions = batch["questions"][:remaining]
        answers = batch["answers"][:remaining]

        predictions = model.generate_answers(
            images=images,
            questions=questions,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_answer_length,
        )

        for pred, target in zip(predictions, answers):
            if _normalize_text(pred) == _normalize_text(target):
                correct += 1
            total += 1

        if total >= max_samples:
            break

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "num_samples": float(total),
    }


def _trainable_state_dict(model: BEVGemmaQA) -> dict[str, torch.Tensor]:
    trainable_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    state = model.state_dict()
    return {
        name: tensor.detach().cpu()
        for name, tensor in state.items()
        if name in trainable_names
    }


def save_checkpoint(
    output_dir: str,
    step: int,
    model: BEVGemmaQA,
    optimizer: torch.optim.Optimizer,
    cfg: TrainBEVQAConfig,
) -> None:
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"step_{step:08d}.pt"

    torch.save(
        {
            "step": step,
            "trainable_model_state_dict": _trainable_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dataclasses.asdict(cfg),
        },
        ckpt_path,
    )

    print(f"[checkpoint] saved {ckpt_path}")


def run_training(cfg: TrainBEVQAConfig) -> None:
    device = _resolve_device()
    dtype = _resolve_dtype(cfg.dtype)

    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    train_loader, val_loader = _build_loaders(cfg)

    model = BEVGemmaQA(
        gemma_name=cfg.gemma_name,
        freeze_gemma=cfg.freeze_gemma,
        freeze_vision=cfg.freeze_vision,
    )

    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    model = model.to(device=device, dtype=dtype)
    model.train()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    num_trainable = sum(param.numel() for param in trainable_params)
    num_total = sum(param.numel() for param in model.parameters())

    print(f"[params] trainable={num_trainable:,}")
    print(f"[params] total={num_total:,}")

    if not trainable_params:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler() if cfg.dtype == "fp16" else None

    wandb_run = None
    if cfg.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name,
                mode=cfg.wandb_mode,
                config=dataclasses.asdict(cfg),
            )
        except ImportError:
            print("[wandb] not installed; continuing without wandb")
            wandb_run = None

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.num_epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.num_epochs}", unit="batch")

        for batch in pbar:
            amp_enabled = device.type == "cuda"

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
                outputs = model(
                    images=batch["images"],
                    questions=batch["questions"],
                    answers=batch["answers"],
                    max_prompt_length=cfg.max_prompt_length,
                    max_answer_length=cfg.max_answer_length,
                )

                loss = outputs["loss"] / cfg.grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % cfg.grad_accum_steps == 0:
                if cfg.max_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    clip_grad_norm_(trainable_params, cfg.max_grad_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            if global_step % cfg.log_every == 0:
                train_loss = float(outputs["loss"].detach().cpu())
                pbar.set_postfix(loss=f"{train_loss:.4f}")

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/step": global_step,
                            "train/loss": train_loss,
                            "train/batch_size": len(batch["questions"]),
                        },
                        step=global_step,
                    )

            if (
                val_loader is not None
                and cfg.eval_every > 0
                and global_step > 0
                and global_step % cfg.eval_every == 0
            ):
                metrics = evaluate(
                    model=model,
                    loader=val_loader,
                    max_samples=cfg.eval_num_samples,
                    max_prompt_length=cfg.max_prompt_length,
                    max_answer_length=cfg.max_answer_length,
                )

                print(f"[eval] step={global_step} {metrics}")

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/step": global_step,
                            "eval/accuracy": metrics["accuracy"],
                            "eval/num_samples": metrics["num_samples"],
                        },
                        step=global_step,
                    )

                model.train()

            if cfg.save_every > 0 and global_step > 0 and global_step % cfg.save_every == 0:
                save_checkpoint(cfg.output_dir, global_step, model, optimizer, cfg)

            global_step += 1

            if cfg.max_steps is not None and global_step >= cfg.max_steps:
                break

        if cfg.max_steps is not None and global_step >= cfg.max_steps:
            break

    save_checkpoint(cfg.output_dir, global_step, model, optimizer, cfg)

    if wandb_run is not None:
        wandb_run.finish()


def _parse_file_indices(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None

    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))

    return values or None


def parse_args() -> TrainBEVQAConfig:
    parser = argparse.ArgumentParser(description="Train BEV-Gemma QA adapter.")

    parser.add_argument("--qa_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/qa_dataset/")
    parser.add_argument("--img_root", type=str, default="/zfsauton/scratch/eshau/imgs_past/")
    parser.add_argument("--output_dir", type=str, default="/home/scratch/sbellad/waymax_rs/bev_qa_output_gemma")
    parser.add_argument("--gemma_name", type=str, default="google/gemma-4-E2B-it")

    parser.add_argument("--file_indices", type=str, default=None, help="Comma-separated tfrecord indices, e.g. 0,1,2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--validation_fraction", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--unfreeze_gemma", action="store_true")
    parser.add_argument("--unfreeze_vision", action="store_true")

    parser.add_argument("--max_prompt_length", type=int, default=128)
    parser.add_argument("--max_answer_length", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=("bf16", "fp16"))

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_num_samples", type=int, default=64)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_pin_memory", action="store_true")

    parser.add_argument("--wandb_project", type=str, default="waymax_rs_bev_qa")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=("online", "offline", "disabled"))

    args = parser.parse_args()

    return TrainBEVQAConfig(
        qa_dir=args.qa_dir,
        img_root=args.img_root,
        output_dir=args.output_dir,
        gemma_name=args.gemma_name,
        file_indices=_parse_file_indices(args.file_indices),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        max_samples=args.max_samples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        freeze_gemma=not args.unfreeze_gemma,
        freeze_vision=not args.unfreeze_vision,
        max_prompt_length=args.max_prompt_length,
        max_answer_length=args.max_answer_length,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        dtype=args.dtype,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_num_samples=args.eval_num_samples,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )


def main() -> None:
    cfg = parse_args()
    run_training(cfg)


if __name__ == "__main__":
    main()
