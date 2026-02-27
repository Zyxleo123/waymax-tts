from __future__ import annotations

import json
import os
from typing import Any

import jax
import orbax.checkpoint as ocp


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return str(x)


def save_checkpoint(
    save_dir: str,
    train_state: dict[str, Any],
    config_dict: dict[str, Any],
    tag: str,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, tag)
    if os.path.exists(ckpt_path):
        if os.path.isdir(ckpt_path):
            for root, dirs, files in os.walk(ckpt_path, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                for d in dirs:
                    os.rmdir(os.path.join(root, d))
            os.rmdir(ckpt_path)
        else:
            os.remove(ckpt_path)

    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(ckpt_path, train_state)

    meta_path = os.path.join(ckpt_path, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(config_dict), f, indent=2, sort_keys=True)

    return ckpt_path


def restore_checkpoint(path: str) -> dict[str, Any]:
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(path)
    return restored


def tree_global_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jax.numpy.array(0.0, dtype=jax.numpy.float32)
    sq = [jax.numpy.sum(jax.numpy.square(x)) for x in leaves if x is not None]
    if not sq:
        return jax.numpy.array(0.0, dtype=jax.numpy.float32)
    return jax.numpy.sqrt(jax.numpy.sum(jax.numpy.stack(sq)))
