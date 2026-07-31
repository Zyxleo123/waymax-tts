from __future__ import annotations

import json
import os
import warnings
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


def _save_checkpoint_tag(
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


def save_checkpoint(
    save_dir: str,
    config_dict: dict[str, Any],
    *,
    epoch: int,
    global_step,
    params_state,
    nonparam_state,
    opt_state,
    ema_params,
    rng_key,
    save_every: int | None = None,
) -> None:
    train_state = {
        "epoch": jax.numpy.array(epoch, dtype=jax.numpy.int32),
        "global_step": global_step,
        "params_state": params_state,
        "nonparam_state": nonparam_state,
        "opt_state": opt_state,
        "ema_params": ema_params,
        "rng_key": rng_key,
    }
    _save_checkpoint_tag(save_dir, train_state, config_dict, tag="latest")
    if save_every is not None and epoch % save_every == 0:
        _save_checkpoint_tag(save_dir, train_state, config_dict, tag=f"epoch_{epoch:04d}")


def restore_checkpoint(
    path: str,
    *,
    params_state: Any | None = None,
    nonparam_state: Any | None = None,
    tx: Any | None = None,
    coerce_tree_like_fn=None,
) -> Any:
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(path)

    if params_state is None or tx is None:
        return restored

    loaded_params_state = restored["params_state"]
    loaded_nonparam_state = restored.get("nonparam_state", nonparam_state)

    if coerce_tree_like_fn is not None:
        loaded_params_state = coerce_tree_like_fn(params_state, loaded_params_state)
        if nonparam_state is not None and loaded_nonparam_state is not None:
            loaded_nonparam_state = coerce_tree_like_fn(nonparam_state, loaded_nonparam_state)

    opt_state = restored["opt_state"]
    
    opt_template = tx.init(loaded_params_state)
    if coerce_tree_like_fn is not None:
        try:
            loaded_opt_state = coerce_tree_like_fn(opt_template, opt_state)
        except Exception as exc:
            warnings.warn(
                "Failed to coerce checkpoint opt_state to current optimizer structure; "
                "falling back to freshly initialized optimizer state. "
                f"This can happen after optimizer implementation changes. Details: {exc}",
                RuntimeWarning,
            )
            loaded_opt_state = opt_template
    else:
        loaded_opt_state = restored["opt_state"]
    loaded_global_step_host = int(restored["global_step"])
    loaded_ema_params = restored["ema_params"]
    if coerce_tree_like_fn is not None:
        loaded_ema_params = coerce_tree_like_fn(loaded_params_state, loaded_ema_params)

    return (
        loaded_params_state,
        loaded_nonparam_state,
        loaded_opt_state,
        loaded_ema_params,
        restored["rng_key"],
        int(restored["epoch"]) + 1,
        jax.numpy.array(loaded_global_step_host, dtype=jax.numpy.int32),
        loaded_global_step_host,
    )


def tree_global_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jax.numpy.array(0.0, dtype=jax.numpy.float32)
    sq = [jax.numpy.sum(jax.numpy.square(x)) for x in leaves if x is not None]
    if not sq:
        return jax.numpy.array(0.0, dtype=jax.numpy.float32)
    return jax.numpy.sqrt(jax.numpy.sum(jax.numpy.stack(sq)))
