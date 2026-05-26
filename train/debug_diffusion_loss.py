from __future__ import annotations
import argparse
import sys

import jax
import jax.numpy as jnp
from flax import nnx

from data.cache_dataloader_jax import build_cache_dataloader_jax
from train.train_diffusion_policy import (
    _build_model_from_checkpoint,
    _load_pretrained_state,
    _state_to_pure_dict,
    _pure_dict_to_state,
)


def tree_l2_norm(tree) -> float:
    leaves = [x for x in jax.tree_util.tree_leaves(tree) if x is not None]
    if len(leaves) == 0:
        return 0.0
    sq = sum([jnp.sum(jnp.asarray(x) ** 2) for x in leaves])
    return float(jnp.sqrt(sq))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--anchor_step", default="all")
    args = p.parse_args(argv)

    model_meta = _build_model_from_checkpoint(args.checkpoint, seed=0)
    graphdef, params_state, nonparam_state = _load_pretrained_state(args.checkpoint, model_meta, use_ema=False)
    params_state = _state_to_pure_dict(params_state)
    nonparam_state = _state_to_pure_dict(nonparam_state)

    loader = build_cache_dataloader_jax(
        args.cache_dir,
        anchor_step=args.anchor_step,
        batch_size=args.batch_size,
        shuffle_seed=0,
        instruction_seed=0,
        num_workers=0,
        pin_memory=False,
    )

    it = iter(loader)
    batch = next(it)
    features = batch.features

    def print_stats(name, arr):
        a = jnp.asarray(arr)
        print(f"{name}: shape={a.shape}, mean={float(a.mean()):.6g}, std={float(a.std()):.6g}, min={float(a.min()):.6g}, max={float(a.max()):.6g}")

    print("-- Batch feature stats --")
    if "ego_trajectory" in features:
        print_stats("ego_trajectory", features["ego_trajectory"])
    if "ego_state" in features:
        print_stats("ego_state", features["ego_state"])
    if "inst_valid" in features:
        print_stats("inst_valid", features["inst_valid"])

    merged = nnx.merge(graphdef, _pure_dict_to_state(params_state), _pure_dict_to_state(nonparam_state))

    rng = jax.random.PRNGKey(0)
    rng, sub = jax.random.split(rng)
    loss = merged.loss(features, rng=sub)
    print(f"loss: {float(loss):.6g}")

    masked_features = dict(features)
    masked_features["inst_valid"] = jnp.zeros_like(masked_features["inst_valid"])
    rng = jax.random.PRNGKey(0)
    rng, sub = jax.random.split(rng)
    masked_loss = merged.loss(masked_features, rng=sub)
    print(f"loss_with_inst_valid_zero: {float(masked_loss):.6g}")

    def loss_fn(p):
        m = nnx.merge(graphdef, _pure_dict_to_state(p), _pure_dict_to_state(nonparam_state))
        rng = jax.random.PRNGKey(0)
        return m.loss(features, rng=rng)

    grads = jax.grad(loss_fn)(params_state)
    print(f"grad L2 norm: {tree_l2_norm(grads):.6g}")

    # Check for NaN or inf in grads
    def any_nan_inf(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        for l in leaves:
            if l is None:
                continue
            a = jnp.asarray(l)
            if jnp.isnan(a).any() or jnp.isinf(a).any():
                return True
        return False

    print(f"grads contain NaN/inf: {any_nan_inf(grads)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
