import os
import json
import numpy as np
from pathlib import Path
from typing import Any
import math
import jax
import jax.numpy as jnp
from tqdm import tqdm


def _parse_int_csv(raw: str) -> list[int]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return [int(v) for v in values]


def _summarize_metric(value_bkt: np.ndarray, valid_bkt: np.ndarray) -> dict[str, Any]:
    batch_size, num_samples, _ = value_bkt.shape
    per_entry: list[list[dict[str, Any]]] = []
    for b in range(batch_size):
        scenario_entries: list[dict[str, Any]] = []
        for k in range(num_samples):
            valid = valid_bkt[b, k].astype(bool)
            if np.any(valid):
                vals = value_bkt[b, k][valid]
                final = float(vals[-1])
                mean = float(vals.mean())
                max_value = float(vals.max())
                valid_steps = int(valid.sum())
            else:
                final = float("nan")
                mean = float("nan")
                max_value = float("nan")
                valid_steps = 0
            scenario_entries.append(
                {
                    "final": final,
                    "mean": mean,
                    "max": max_value,
                    "valid_steps": valid_steps,
                }
            )
        per_entry.append(scenario_entries)
    return {"per_scenario_sample": per_entry}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
