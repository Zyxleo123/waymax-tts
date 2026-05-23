from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class CacheFileInfo:
	path: Path
	tfrecord_index: int
	anchor_step: int


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Embed per-scenario instructions and append them to cache NPZ files."
	)
	parser.add_argument(
		"--cache_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/cache",
		help="Directory containing *_t{anchor}.npz cache files.",
	)
	parser.add_argument(
		"--instruction_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/instructions/training",
		help="Directory containing tfrecord_{i}.jsonl and tfrecord_{i}_index.json.",
	)
	parser.add_argument(
		"--anchor_step",
		type=int,
		default=10,
		help="Only process cache files ending with _t{anchor_step}.npz.",
	)
	parser.add_argument(
		"--model_name",
		type=str,
		default="google/embeddinggemma-300m",
		help="Hugging Face model name for text embeddings.",
	)
	parser.add_argument(
		"--num_instructions",
		type=int,
		default=5,
		help="Number of instructions per (scenario, timestep) to read from index.",
	)
	parser.add_argument(
		"--embed_dim",
		type=int,
		default=256,
		help="Target embedding dimension written to cache.",
	)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--max_length", type=int, default=128)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Embedding device.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing instruction feature keys if present.",
	)
	parser.add_argument(
		"--dry_run",
		action="store_true",
		help="Run parsing/embedding checks without writing NPZ files.",
	)
	return parser.parse_args()


def _discover_cache_files(cache_dir: Path, anchor_step: int) -> list[CacheFileInfo]:
	pattern = re.compile(r"training_tfexample\.tfrecord-(\d{5})-of-\d{5}_t(\d+)\.npz$")
	infos: list[CacheFileInfo] = []
	for path in sorted(cache_dir.glob(f"*_t{anchor_step}.npz")):
		match = pattern.search(path.name)
		if match is None:
			continue
		infos.append(
			CacheFileInfo(
				path=path,
				tfrecord_index=int(match.group(1)),
				anchor_step=int(match.group(2)),
			)
		)
	return infos


def _load_instruction_index(index_path: Path) -> dict[str, list[int]]:
	if not index_path.exists():
		raise FileNotFoundError(f"Instruction index not found: {index_path}")
	with index_path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	return {str(k): [int(v) for v in values] for k, values in data.items()}


def _read_instruction_at_offset(data_file: Path, offset: int) -> str:
	with data_file.open("rb") as f:
		f.seek(int(offset))
		raw = f.readline()
	if not raw:
		return ""
	try:
		obj = json.loads(raw.decode("utf-8"))
	except json.JSONDecodeError:
		return ""
	text = obj.get("instruction", "")
	return str(text).strip()


def _is_valid_instruction(text: str) -> bool:
	stripped = text.strip()
	if not stripped:
		return False
	return "unclear" not in stripped.lower()


def _load_instruction_texts(
	*,
	tfrecord_index: int,
	scenario_indices: np.ndarray,
	timesteps: np.ndarray,
	instruction_dir: Path,
	num_instructions: int,
) -> tuple[list[list[str]], np.ndarray]:
	data_path = instruction_dir / f"tfrecord_{tfrecord_index}.jsonl"
	index_path = instruction_dir / f"tfrecord_{tfrecord_index}_index.json"

	if not data_path.exists():
		raise FileNotFoundError(f"Instruction JSONL not found: {data_path}")
	index = _load_instruction_index(index_path)

	all_texts: list[list[str]] = []
	all_valid = np.zeros((scenario_indices.shape[0], num_instructions), dtype=np.bool_)

	for i, (scenario_idx, timestep) in enumerate(zip(scenario_indices.tolist(), timesteps.tolist())):
		key = f"{int(scenario_idx)}:{int(timestep)}"
		offsets = index.get(key, [])
		texts: list[str] = []
		for j in range(num_instructions):
			if j < len(offsets):
				text = _read_instruction_at_offset(data_path, int(offsets[j]))
			else:
				text = ""
			is_valid = _is_valid_instruction(text)
			all_valid[i, j] = is_valid
			texts.append(text if is_valid else "")
		all_texts.append(texts)

	return all_texts, all_valid


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
	mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
	summed = (last_hidden_state * mask).sum(dim=1)
	denom = mask.sum(dim=1).clamp(min=1e-6)
	return summed / denom


def _to_target_dim(emb: np.ndarray, target_dim: int) -> np.ndarray:
	if emb.shape[-1] == target_dim:
		return emb.astype(np.float32, copy=False)
	if emb.shape[-1] > target_dim:
		return emb[:, :target_dim].astype(np.float32, copy=False)
	out = np.zeros((emb.shape[0], target_dim), dtype=np.float32)
	out[:, : emb.shape[-1]] = emb.astype(np.float32)
	return out


class EmbeddingGemmaEncoder:
	def __init__(self, model_name: str, device: str, max_length: int, target_dim: int):
		self.device = torch.device(device)
		self.max_length = int(max_length)
		self.target_dim = int(target_dim)
		try:
			self.tokenizer = AutoTokenizer.from_pretrained(model_name)
			self.model = AutoModel.from_pretrained(model_name)
		except Exception as exc:
			raise RuntimeError(
				"Failed to load embedding model. If this is a gated repo, "
				"authenticate first (e.g., huggingface-cli login)."
			) from exc
		self.model.to(self.device)
		self.model.eval()

	@torch.inference_mode()
	def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
		all_embs: list[np.ndarray] = []
		for i in range(0, len(texts), batch_size):
			batch = texts[i : i + batch_size]
			tokenized = self.tokenizer(
				batch,
				padding=True,
				truncation=True,
				max_length=self.max_length,
				return_tensors="pt",
			)
			tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
			outputs = self.model(**tokenized)
			pooled = _mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])  # [B, H]
			pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
			all_embs.append(pooled.detach().cpu().numpy())

		embs = np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, self.target_dim), dtype=np.float32)
		return embs


def _build_payload_with_instruction_features(
	npz_path: Path,
	inst_features: np.ndarray,
	inst_valid: np.ndarray,
	inst_features_multi: np.ndarray,
	inst_valid_multi: np.ndarray,
	*,
	overwrite: bool,
) -> dict[str, np.ndarray]:
	with np.load(npz_path, allow_pickle=False) as npz_data:
		payload = {k: npz_data[k] for k in npz_data.files}

	target_keys = {
		"features/inst_features",
		"features/inst_valid",
		"features/inst_features_multi",
		"features/inst_valid_multi",
	}
	exists = target_keys.intersection(payload.keys())
	if exists and not overwrite:
		raise ValueError(
			f"Instruction feature keys already exist in {npz_path}. "
			f"Use --overwrite to replace them. Existing keys: {sorted(exists)}"
		)

	payload["features/inst_features"] = inst_features.astype(np.float32)
	payload["features/inst_valid"] = inst_valid.astype(np.bool_)
	payload["features/inst_features_multi"] = inst_features_multi.astype(np.float32)
	payload["features/inst_valid_multi"] = inst_valid_multi.astype(np.bool_)
	return payload


def _write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	with tmp_path.open("wb") as f:
		np.savez(f, **payload)
	tmp_path.replace(path)


def main() -> None:
	args = parse_args()
	cache_dir = Path(args.cache_dir)
	instruction_dir = Path(args.instruction_dir)

	cache_files = _discover_cache_files(cache_dir, args.anchor_step)
	if not cache_files:
		raise FileNotFoundError(
			f"No cache files found in {cache_dir} matching *_t{args.anchor_step}.npz"
		)

	encoder = EmbeddingGemmaEncoder(
		model_name=args.model_name,
		device=args.device,
		max_length=args.max_length,
		target_dim=args.embed_dim,
	)

	for info in tqdm(cache_files, desc="Embedding instructions into cache"):
		with np.load(info.path, allow_pickle=False) as npz_data:
			if "scenario_index" not in npz_data.files:
				raise KeyError(f"scenario_index is missing in {info.path}")
			if "timestep" not in npz_data.files:
				raise KeyError(f"timestep is missing in {info.path}")
			scenario_indices = np.asarray(npz_data["scenario_index"]).astype(np.int32)
			timesteps = np.asarray(npz_data["timestep"]).astype(np.int32)

		instruction_texts, valid_multi = _load_instruction_texts(
			tfrecord_index=info.tfrecord_index,
			scenario_indices=scenario_indices,
			timesteps=timesteps,
			instruction_dir=instruction_dir,
			num_instructions=args.num_instructions,
		)

		flat_texts = [txt for texts in instruction_texts for txt in texts]
		flat_embs = encoder.encode(flat_texts, batch_size=args.batch_size)
		num_rows = scenario_indices.shape[0]

		inst_multi = flat_embs.reshape(num_rows, args.num_instructions, -1)
		valid_counts = valid_multi.sum(axis=1, keepdims=True).clip(min=1)
		masked = inst_multi * valid_multi[..., None].astype(np.float32)
		pooled = masked.sum(axis=1) / valid_counts.astype(np.float32)
		pooled_valid = valid_multi.any(axis=1)

		payload = _build_payload_with_instruction_features(
			info.path,
			pooled,
			pooled_valid,
			inst_multi,
			valid_multi,
			overwrite=args.overwrite,
		)

		if not args.dry_run:
			_write_npz_atomic(info.path, payload)

		print(
			f"updated {info.path.name}: rows={num_rows}, "
			f"valid_rows={int(pooled_valid.sum())}/{num_rows}"
		)


if __name__ == "__main__":
	main()
