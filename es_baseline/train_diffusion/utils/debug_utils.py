from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import make_array_from_single_device_arrays, tree_util
from jax.sharding import PmapSharding


def _normalize_index_parts(index, ndim: int):
    if not isinstance(index, tuple):
        return (index,)
    if len(index) < ndim:
        index = index + (slice(None),) * (ndim - len(index))
    return index[:ndim]


def _is_full_slice(part, size: int) -> bool:
    return (
        isinstance(part, slice)
        and part.start in (None, 0)
        and part.stop in (None, size)
        and part.step in (None, 1)
    )


def _infer_split_axes(x):
    if not hasattr(x, "addressable_shards") or not x.addressable_shards:
        return []
    axes = set()
    for shard in x.addressable_shards:
        idx = _normalize_index_parts(shard.index, x.ndim)
        for axis, part in enumerate(idx):
            if not _is_full_slice(part, x.shape[axis]):
                axes.add(axis)
    return sorted(axes)


def _projection_info(x):
    split_axes = _infer_split_axes(x)
    if x.ndim <= 2 or not split_axes:
        raise ValueError("no projection needed or split axes not found")

    keep = split_axes[:2]
    if len(keep) == 1:
        a = keep[0]
        second = int(jnp.prod(jnp.array([s for i, s in enumerate(x.shape) if i != a])).item())
        projection_shape = (x.shape[a], max(1, second))
    else:
        a, b = keep
        projection_shape = (x.shape[a], x.shape[b])

    try:
        if len(keep) == 1:
            projection_sharding = PmapSharding.default(
                shape=projection_shape, sharded_dim=0, devices=x.sharding.devices
            )
        else:
            projection_sharding = x.sharding.reshape(projection_shape)
    except Exception:
        projection_sharding = PmapSharding.default(
            shape=projection_shape, sharded_dim=0, devices=x.sharding.devices
        )

    return projection_shape, projection_sharding


def _visualize_with_layout(x):
    if x.ndim <= 2:
        jax.debug.visualize_array_sharding(x)
        return

    projection_shape, projection_sharding = _projection_info(x)
    shard_shape = projection_sharding.shard_shape(projection_shape)
    per_device = [
        jax.device_put(jnp.zeros(shard_shape, dtype=x.dtype), device)
        for device in projection_sharding.devices.flat
    ]
    y = make_array_from_single_device_arrays(projection_shape, projection_sharding, per_device)
    jax.debug.visualize_array_sharding(y)


def visualize_pytree_sharding_like_jax(pytree):
    for path, x in tree_util.tree_flatten_with_path(pytree)[0]:
        if not isinstance(x, jax.Array):
            continue
        name = "/".join(str(getattr(k, "key", k)) for k in path) or "<root>"
        print(f"{name}:")
        try:
            _visualize_with_layout(x)
        except Exception as exc:
            print(f"  <error rendering sharding: {exc}>")
        print()
