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
		description="Embed manual instructions into cache NPZ files and write new cache files."
	)
	parser.add_argument(
		"--cache_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/cache",
		help="Directory containing training_tfexample cache NPZ files.",
	)
	parser.add_argument(
		"--instruction_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/manual_instruction",
		help="Directory containing tfrecord_{i}.jsonl and tfrecord_{i}.idx.json.",
	)
	parser.add_argument(
		"--out_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/cache_inst",
		help="Directory to write updated cache NPZ files.",
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
		help="Hugging Face model name used to embed instructions.",
	)
	parser.add_argument(
		"--embed_dim",
		type=int,
		default=768,
		help="Target embedding dimension written to the cache.",
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
		"--dry_run",
		action="store_true",
		help="Run parsing and embedding without writing NPZ files.",
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


def _load_instruction_offsets(index_path: Path) -> list[int]:
	if not index_path.exists():
		raise FileNotFoundError(f"Instruction index not found: {index_path}")
	with index_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	if isinstance(data, list):
		return [int(v) for v in data]
	if isinstance(data, dict):
		if "byte_offsets" in data and isinstance(data["byte_offsets"], list):
			return [int(v) for v in data["byte_offsets"]]
		if len(data) == 1:
			value = next(iter(data.values()))
			if isinstance(value, list):
				return [int(v) for v in value]
	raise ValueError(f"Unsupported instruction index format: {index_path}")


def _read_instruction_at_offset(data_file: Path, offset: int) -> str:
	with data_file.open("rb") as f:
		f.seek(int(offset))
		raw = f.readline()
	if not raw:
		raise ValueError(f"Missing instruction at offset {offset} in {data_file}")
	try:
		obj = json.loads(raw.decode("utf-8"))
	except json.JSONDecodeError as exc:
		raise ValueError(f"Failed to decode instruction at offset {offset} in {data_file}") from exc
	text = str(obj.get("instruction", "")).strip()
	if not text:
		raise ValueError(f"Empty instruction at offset {offset} in {data_file}")
	return text


def _load_instruction_texts(
	*,
	tfrecord_index: int,
	anchor_step: int,
	scenario_indices: np.ndarray,
	instruction_dir: Path,
) -> list[str]:
	data_path = instruction_dir / f"training_tfexample.tfrecord-{tfrecord_index:05d}-of-01000_t{anchor_step}.jsonl"
	index_path = instruction_dir / f"training_tfexample.tfrecord-{tfrecord_index:05d}-of-01000_t{anchor_step}.idx.json"

	if not data_path.exists():
		raise FileNotFoundError(f"Instruction JSONL not found: {data_path}")
	offsets = _load_instruction_offsets(index_path)

	texts: list[str] = []
	for scenario_index in scenario_indices.tolist():
		line_index = int(scenario_index)
		if line_index < 0 or line_index >= len(offsets):
			raise IndexError(
				f"Scenario index {line_index} is out of range for {data_path} (have {len(offsets)} lines)"
			)
		texts.append(_read_instruction_at_offset(data_path, offsets[line_index]))
	return texts


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
				"Failed to load embedding model. If this is a gated repo, authenticate first."
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
			pooled = _mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])
			pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
			all_embs.append(pooled.detach().cpu().numpy())

		embs = np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, self.target_dim), dtype=np.float32)
		return _to_target_dim(embs, self.target_dim)


def _strip_existing_instruction_keys(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
	keys_to_drop = {
		"features/inst_features",
		"features/inst_valid",
		"features/inst_features_multi",
		"features/inst_valid_multi",
		"features/inst_features_mult",
	}
	return {key: value for key, value in payload.items() if key not in keys_to_drop}


def _build_payload_with_instruction_features(
	npz_path: Path,
	inst_features: np.ndarray,
	inst_valid: np.ndarray,
	*,
	overwrite: bool,
) -> dict[str, np.ndarray]:
	with np.load(npz_path, allow_pickle=False) as npz_data:
		payload = {k: npz_data[k] for k in npz_data.files}

	if not overwrite:
		payload = _strip_existing_instruction_keys(payload)
	else:
		payload = _strip_existing_instruction_keys(payload)

	payload["features/inst_features"] = inst_features.astype(np.float32)
	payload["features/inst_valid"] = inst_valid.astype(np.bool_)
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
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

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

	for info in tqdm(cache_files, desc="Embedding manual instructions into cache"):
		with np.load(info.path, allow_pickle=False) as npz_data:
			if "scenario_index" not in npz_data.files:
				raise KeyError(f"scenario_index is missing in {info.path}")
			scenario_indices = np.asarray(npz_data["scenario_index"]).astype(np.int32)

		instruction_texts = _load_instruction_texts(
				tfrecord_index=info.tfrecord_index,
				scenario_indices=scenario_indices,
				anchor_step=info.anchor_step,
				instruction_dir=instruction_dir,
		)

		flat_embs = encoder.encode(instruction_texts, batch_size=args.batch_size)
		inst_features = flat_embs.reshape(scenario_indices.shape[0], -1)
		inst_valid = np.ones((scenario_indices.shape[0],), dtype=np.bool_)

		payload = _build_payload_with_instruction_features(
			info.path,
			inst_features,
			inst_valid,
			overwrite=True,
		)

		out_path = out_dir / info.path.name
		if not args.dry_run:
			_write_npz_atomic(out_path, payload)

		print(f"updated {out_path.name}: rows={scenario_indices.shape[0]}")


if __name__ == "__main__":
	main()
