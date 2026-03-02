from __future__ import annotations

import jax
import jax.numpy as jnp

from train.preprocess import _rotate_xy, _wrap_to_pi, resample_trajectory_timebase_jax
from train.types import PreprocessConfig


def ego_normalized_to_world(
    pred_btd: jax.Array,
    origin_xy: jax.Array,
    anchor_yaw: jax.Array,
    max_distance: float,
    max_velocity: float,
) -> jax.Array:
    # pred_btd uses [x, y, vx, vy, yaw] in ego-normalized coordinates.
    pos = pred_btd[:, :, :2] * max_distance
    vel = pred_btd[:, :, 2:4] * max_velocity
    yaw = pred_btd[:, :, 4] * jnp.pi

    pos_world = _rotate_xy(pos, -anchor_yaw[:, None]) + origin_xy[:, None, :]
    vel_world = _rotate_xy(vel, -anchor_yaw[:, None])
    yaw_world = _wrap_to_pi(yaw + anchor_yaw[:, None])

    # World trajectory consumers expect [x, y, yaw, vel_x, vel_y].
    return jnp.concatenate([pos_world, yaw_world[:, :, None], vel_world], axis=-1).astype(jnp.float32)


def resample_to_world_dt(
    pred_world_btd: jax.Array,
    model_t_s: jax.Array,
    world_t_s: jax.Array,
) -> jax.Array:
    return resample_trajectory_timebase_jax(
        traj_btd=pred_world_btd,
        src_t_s=model_t_s,
        dst_t_s=world_t_s,
        yaw_index=2,
    )


def postprocess_predictions(
    pred_norm_btd: jax.Array,
    aux: dict[str, jax.Array],
    cfg: PreprocessConfig,
) -> dict[str, jax.Array]:
    pred_world_model_dt = ego_normalized_to_world(
        pred_btd=pred_norm_btd,
        origin_xy=aux["origin_xy"],
        anchor_yaw=aux["anchor_yaw"],
        max_distance=cfg.ego_range,
        max_velocity=cfg.max_velocity,
    )

    # anchor_world_state is [x, y, vx, vy, yaw] from preprocessing; reorder to world layout.
    anchor_world = aux["anchor_world_state"][:, [0, 1, 4, 2, 3]][:, None, :]
    pred_world_with_anchor = jnp.concatenate([anchor_world, pred_world_model_dt], axis=1)

    model_t_s = aux["model_t_seconds"]
    world_t_s = aux["world_t_seconds"]

    pred_world_world_dt = resample_to_world_dt(
        pred_world_btd=pred_world_with_anchor,
        model_t_s=model_t_s,
        world_t_s=world_t_s,
    )

    return {
        "trajectory_world_model_dt": pred_world_with_anchor,
        "trajectory_world_world_dt": pred_world_world_dt,
        "world_t_seconds": world_t_s,
        "world_t_valid": aux["world_t_valid"],
    }
