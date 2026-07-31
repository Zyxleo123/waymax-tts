"""Dense-return oracle scorer for the Diffusion-ES reward-resolution experiment.

Stage 1 of the plan: replace the binary ES selection score
(``collision * offroad in {0,1}``, goal-agnostic) with a dense trajectory return

    G(traj) = sum_t gamma^t * ( sum_i w_i r_{i,t} ) / normalizer            (in [0,1])

built from the *same* per-step simulator quantities the evaluator cares about:
collision, off-road, goal progress / closeness, agent separation, speed
compliance, and comfort (accel / yaw-rate). Because the other agents are
non-reactive log-replay in this simulator, scoring the planned candidate
open-loop against the log is equivalent to the closed-loop rollout, so this is a
faithful *oracle* dense fitness, not a learned model.

Design notes
------------
* This lives outside the vendored baseline and never mutates it. It reuses the
  vendored ``scores.polygon`` primitives and mirrors ``Scorer.compute_score``'s
  ego/object construction and ``compute_offroad_score_v2``'s road-edge geometry,
  but keeps the **time axis** so we get per-step rewards instead of one bit.
* Every per-step component is bounded to [0, 1]. ``G`` is a weight-normalized,
  discounted mean, so no single large-magnitude term dominates (plan requirement).
* ``compute_components`` returns the per-candidate component breakdown for
  Stage-0 instrumentation and rank-correlation analysis.
* ``binary_baseline_score`` reproduces the exact reference selection score so the
  reference arm and the oracle arm share one code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from waymax import datatypes

from scores.polygon import pairwise_intersections, polygon_corners


@dataclass
class DenseWeights:
    """Per-component weights for the dense return. Set a weight to 0 to disable."""

    collision: float = 1.0
    offroad: float = 1.0
    progress: float = 1.0
    closeness: float = 0.5
    separation: float = 0.25
    speed: float = 0.0        # only meaningful if target_speed is provided
    comfort: float = 0.25

    def as_dict(self) -> Dict[str, float]:
        return {
            "collision": self.collision,
            "offroad": self.offroad,
            "progress": self.progress,
            "closeness": self.closeness,
            "separation": self.separation,
            "speed": self.speed,
            "comfort": self.comfort,
        }


@dataclass
class DenseConfig:
    weights: DenseWeights = field(default_factory=DenseWeights)
    gamma: float = 1.0                 # finite horizon; 1.0 = undiscounted mean
    goal_ref_m: float = 30.0           # normalizer for goal closeness
    step_ref_m: float = 2.0            # normalizer for per-step progress (~v_max*dt)
    sep_ref_m: float = 5.0             # separation margin saturates here
    speed_scale_mps: float = 10.0      # speed-error scale
    accel_max_mps2: float = 6.0        # comfort accel normalizer
    yawrate_max_rps: float = 1.0       # comfort yaw-rate normalizer
    target_speed_mps: Optional[float] = None
    dt: float = 0.2                    # model_dt of the diffusion checkpoint


class DenseReturnScorer:
    """Computes the dense oracle return for a batch of candidate trajectories.

    Candidate ``trajectories`` are world-frame ``[K, T, 5]`` = (x, y, yaw, vx, vy),
    the same tensor the vendored ``Scorer.compute_score`` consumes.
    """

    def __init__(self, cfg: Optional[DenseConfig] = None):
        self.cfg = cfg or DenseConfig()

    # ---- ego / object construction (mirrors Scorer.compute_score) -----------
    def _build_ego_object(self, trajectories, sim_state, timestep, world_idx):
        trajectories = jnp.asarray(trajectories)
        if trajectories.ndim != 3 or trajectories.shape[-1] != 5:
            raise ValueError(
                f"trajectories must be [K, T, 5], got {trajectories.shape}"
            )
        ego_mask = jnp.asarray(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
        if int(jnp.sum(ego_mask)) != 1:
            raise ValueError(f"Expected exactly one SDC, got {int(jnp.sum(ego_mask))}.")

        object_xy = sim_state.log_trajectory.xy[world_idx]
        object_yaw = sim_state.log_trajectory.yaw[world_idx]
        object_length = sim_state.log_trajectory.length[world_idx]
        object_width = sim_state.log_trajectory.width[world_idx]
        object_valid = sim_state.log_trajectory.valid[world_idx]

        traj_length = int(trajectories.shape[1])
        max_available = int(object_xy.shape[1]) - int(timestep)
        traj_length = min(traj_length, max_available)

        sl = slice(int(timestep), int(timestep) + traj_length)
        object_xy = object_xy[:, sl, :]
        object_yaw = object_yaw[:, sl]
        object_length = object_length[:, sl]
        object_width = object_width[:, sl]
        object_masks = (object_valid[:, sl] > 0) & (~ego_mask[:, None])

        object_trajectories = jnp.stack(
            [object_xy[..., 0], object_xy[..., 1], object_yaw, object_length, object_width],
            axis=-1,
        )
        ego_idx = int(jnp.argmax(ego_mask.astype(jnp.int32)))
        ego_length_t = object_length[ego_idx]
        ego_width_t = object_width[ego_idx]
        num_k = int(trajectories.shape[0])
        ego_length = jnp.broadcast_to(ego_length_t[None, :], (num_k, traj_length))
        ego_width = jnp.broadcast_to(ego_width_t[None, :], (num_k, traj_length))
        ego_trajectories = jnp.stack(
            [
                trajectories[:, :traj_length, 0],
                trajectories[:, :traj_length, 1],
                trajectories[:, :traj_length, 2],
                ego_length,
                ego_width,
            ],
            axis=-1,
        )
        ego_vel = trajectories[:, :traj_length, 3:5]
        return ego_trajectories, object_trajectories, object_masks, ego_vel

    # ---- per-time safety ----------------------------------------------------
    def _per_time_collision(self, ego_trajectories, object_trajectories, object_masks):
        """Returns (collision_flags [K, T] bool, min_separation [K, T] float meters)."""
        ego_polygons = polygon_corners(ego_trajectories)        # [K, T, 4, 2]
        object_polygons = polygon_corners(object_trajectories)  # [O, T, 4, 2]

        def _time_pairwise(ego_t, object_t):
            return pairwise_intersections(ego_t, object_t)      # [K, O] bool

        inter_t_k_o = jax.vmap(_time_pairwise, in_axes=(1, 1), out_axes=0)(
            ego_polygons, object_polygons
        )                                                       # [T, K, O]
        valid_t_o = jnp.swapaxes(object_masks > 0, 0, 1)        # [T, O]
        inter_t_k_o = inter_t_k_o & valid_t_o[:, None, :]
        collision_flags = jnp.swapaxes(jnp.any(inter_t_k_o, axis=2), 0, 1)  # [K, T]

        # continuous separation margin: ego-center to nearest valid object-center
        ego_c = ego_trajectories[:, :, :2]                      # [K, T, 2]
        obj_c = object_trajectories[:, :, :2]                   # [O, T, 2]
        d = jnp.linalg.norm(
            ego_c[:, None, :, :] - obj_c[None, :, :, :], axis=-1
        )                                                       # [K, O, T]
        big = jnp.asarray(1e6, dtype=d.dtype)
        valid_o_t = (object_masks > 0)                          # [O, T]
        d = jnp.where(valid_o_t[None, :, :], d, big)
        min_sep = jnp.min(d, axis=1)                            # [K, T]
        return collision_flags, min_sep

    def _per_time_offroad(self, ego_trajectories, sim_state, world_idx,
                          k_chunk: int = 4):
        """Per-time off-road flags [K, T] bool. Mirrors compute_offroad_score_v2
        but reduces only over bbox corners, keeping time. Returns all-False if the
        road graph is unavailable (same fallback semantics as the baseline).

        Chunks over K so we never materialize the full [K,T,4,P,3] broadcast
        (that OOMs on preempt/debug cards when K=64, T~52, P~2k — ~350MiB+ for
        a single reduce_sum autotune buffer).
        """
        K, T = ego_trajectories.shape[0], ego_trajectories.shape[1]
        rg = getattr(sim_state, "roadgraph_points", None)
        if rg is None or not hasattr(rg, "xyz"):
            return jnp.zeros((K, T), dtype=bool)

        xyz = jnp.asarray(rg.xyz, dtype=jnp.float32)
        dir_xyz = jnp.asarray(rg.dir_xyz, dtype=jnp.float32)
        types = jnp.asarray(rg.types)
        valid = jnp.asarray(rg.valid).astype(bool)
        ids = jnp.asarray(rg.ids)
        if xyz.ndim >= 3:
            xyz, dir_xyz, types, valid, ids = (
                xyz[world_idx], dir_xyz[world_idx], types[world_idx],
                valid[world_idx], ids[world_idx],
            )
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            return jnp.zeros((K, T), dtype=bool)

        xyz = xyz[valid]; dir_xyz = dir_xyz[valid]; types = types[valid]; ids = ids[valid]
        road_edge = datatypes.is_road_edge(types)
        if not bool(jnp.any(road_edge)):
            return jnp.zeros((K, T), dtype=bool)
        xyz = xyz[road_edge]; dir_xyz = dir_xyz[road_edge]; ids = ids[road_edge]
        if xyz.shape[0] == 0:
            return jnp.zeros((K, T), dtype=bool)

        corners_xy = polygon_corners(ego_trajectories)          # [K, T, 4, 2]
        chunk = max(1, int(k_chunk))
        parts = []
        for i0 in range(0, K, chunk):
            parts.append(self._offroad_flags_chunk(
                corners_xy[i0:i0 + chunk], xyz, dir_xyz, ids))
        return jnp.concatenate(parts, axis=0)

    @staticmethod
    def _offroad_flags_chunk(corners_xy, xyz, dir_xyz, ids):
        """Off-road flags for a K-chunk of corners. corners_xy: [Ck, T, 4, 2]."""
        z = jnp.zeros_like(corners_xy[..., :1])
        corners = jnp.concatenate([corners_xy, z], axis=-1)     # [Ck, T, 4, 3]
        diffs = xyz - jnp.expand_dims(corners, axis=-2)         # [Ck, T, 4, P, 3]
        zstretch = diffs * jnp.asarray([1.0, 1.0, 2.0], dtype=jnp.float32)
        sq = jnp.sum(zstretch**2, axis=-1)                      # [Ck, T, 4, P]
        nearest = jnp.argmin(sq, axis=-1)                       # [Ck, T, 4]
        prior = jnp.maximum(jnp.zeros_like(nearest), nearest - 1)

        nearest_xy = xyz[nearest, :2]
        nearest_vec = dir_xyz[nearest, :2]
        prior_vec = dir_xyz[prior, :2]
        p2e = corners[..., :2] - nearest_xy
        cross = p2e[..., 0] * nearest_vec[..., 1] - p2e[..., 1] * nearest_vec[..., 0]
        cross_prior = p2e[..., 0] * prior_vec[..., 1] - p2e[..., 1] * prior_vec[..., 0]
        same_curve = ids[nearest] == ids[prior]
        sign = jnp.sign(
            jnp.where(jnp.logical_and(same_curve, cross_prior < cross), cross_prior, cross)
        )
        dist = jnp.linalg.norm(nearest_xy - corners[..., :2], axis=-1) * sign  # [Ck, T, 4]
        return jnp.any(dist > 0.0, axis=-1)                     # [Ck, T]

    # ---- dense return -------------------------------------------------------
    def compute_components(self, trajectories, sim_state, timestep, world_idx=0,
                           goal_xy=None):
        """Returns a dict of per-candidate component scores, each shape [K] in [0,1],
        plus per-step arrays used for the discounted sum. All jax arrays."""
        cfg = self.cfg
        ego, obj, obj_masks, ego_vel = self._build_ego_object(
            trajectories, sim_state, timestep, world_idx
        )
        K, T = ego.shape[0], ego.shape[1]
        collision_flags, min_sep = self._per_time_collision(ego, obj, obj_masks)
        offroad_flags = self._per_time_offroad(ego, sim_state, world_idx)

        r_collision = (~collision_flags).astype(jnp.float32)     # [K, T]
        r_offroad = (~offroad_flags).astype(jnp.float32)         # [K, T]
        r_sep = jnp.clip(min_sep / cfg.sep_ref_m, 0.0, 1.0)      # [K, T]

        # goal terms
        ego_xy = ego[:, :, :2]                                   # [K, T, 2]
        if goal_xy is not None:
            goal = jnp.asarray(goal_xy, dtype=jnp.float32).reshape(2)
            d_goal = jnp.linalg.norm(ego_xy - goal[None, None, :], axis=-1)  # [K, T]
            r_close = jnp.clip(1.0 - d_goal / cfg.goal_ref_m, 0.0, 1.0)      # [K, T]
            dd = d_goal[:, :-1] - d_goal[:, 1:]                  # approach>0  [K, T-1]
            r_prog_step = jnp.clip(0.5 + dd / (2.0 * cfg.step_ref_m), 0.0, 1.0)
            r_prog = jnp.concatenate([r_prog_step[:, :1], r_prog_step], axis=1)  # [K, T]
        else:
            r_close = jnp.ones((K, T), dtype=jnp.float32)
            r_prog = jnp.ones((K, T), dtype=jnp.float32)

        # speed compliance
        speed = jnp.linalg.norm(ego_vel, axis=-1)               # [K, T]
        if cfg.target_speed_mps is not None:
            err = jnp.abs(speed - cfg.target_speed_mps)
            r_speed = 1.0 - jnp.minimum(err / cfg.speed_scale_mps, 1.0)
        else:
            r_speed = jnp.ones((K, T), dtype=jnp.float32)

        # comfort: accel + yaw-rate from finite differences of the trajectory
        accel = jnp.abs(jnp.diff(speed, axis=1)) / cfg.dt        # [K, T-1]
        accel = jnp.concatenate([accel[:, :1], accel], axis=1)
        yaw = ego[:, :, 2]
        dyaw = jnp.arctan2(jnp.sin(jnp.diff(yaw, axis=1)), jnp.cos(jnp.diff(yaw, axis=1)))
        yawrate = jnp.abs(dyaw) / cfg.dt                         # [K, T-1]
        yawrate = jnp.concatenate([yawrate[:, :1], yawrate], axis=1)
        r_comfort = 1.0 - jnp.clip(
            0.5 * (accel / cfg.accel_max_mps2 + yawrate / cfg.yawrate_max_rps), 0.0, 1.0
        )

        per_step = {
            "collision": r_collision,
            "offroad": r_offroad,
            "progress": r_prog,
            "closeness": r_close,
            "separation": r_sep,
            "speed": r_speed,
            "comfort": r_comfort,
        }
        # discounted mean over time -> per-candidate component score in [0,1]
        gamma_t = (cfg.gamma ** jnp.arange(T, dtype=jnp.float32))
        wsum = jnp.sum(gamma_t)
        comp = {k: jnp.sum(v * gamma_t[None, :], axis=1) / wsum for k, v in per_step.items()}
        return comp, per_step

    def compute_dense_return(self, trajectories, sim_state, timestep, world_idx=0,
                             goal_xy=None):
        """Weight-normalized dense return G [K] in [0, 1]."""
        comp, _ = self.compute_components(
            trajectories, sim_state, timestep, world_idx, goal_xy
        )
        weights = self.cfg.weights.as_dict()
        total_w = sum(w for w in weights.values() if w > 0)
        if total_w <= 0:
            raise ValueError("At least one dense weight must be > 0.")
        G = jnp.zeros((trajectories.shape[0],), dtype=jnp.float32)
        for name, w in weights.items():
            if w > 0:
                G = G + w * comp[name]
        return G / total_w

    @staticmethod
    def binary_baseline_score(scorer, trajectories, sim_state, timestep, world_idx=0):
        """Exact reference selection score (collision*offroad in {0,1}) via the
        vendored Scorer, so the reference arm shares one code path with the oracle."""
        return scorer.compute_score(
            trajectories=trajectories, sim_state=sim_state,
            timestep=timestep, world_idx=world_idx,
        )
