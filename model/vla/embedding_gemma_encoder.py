from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
	mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
	summed = (last_hidden_state * mask).sum(dim=1)
	denom = mask.sum(dim=1).clamp(min=1e-6)
	return summed / denom


class EmbeddingGemmaEncoder:
	def __init__(self, model_name: str, device: str, max_length: int, target_dim: int = 768) -> None:
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