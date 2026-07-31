"""Instrumented Diffusion-ES planner for the Stage-0 / Stage-1 / Stage-2 experiments.

Subclasses the vendored ``DiffusionESPlanner`` and reuses every search
primitive (sampling, truncated-reverse mutation, top-k elitism, elite carry-over)
unchanged. The ONLY things it changes are:

  * ``selection_mode``: which score drives top-k selection
        "binary" -> vendored collision*offroad in {0,1}   (reference arm, Stage 0)
        "dense"  -> DenseReturnScorer oracle return in [0,1] (Stage 1 arm B)
  * ``init_select`` / ``init_bank_multiplier``: Stage-2 diversified initialization.
        Sample an oversized diffusion bank, then down-select to ``population_size``
        via farthest-point sampling on behavior descriptors (optionally safe-aware).
  * ``init_select`` ``sac`` / ``sac_safe``: Stage-3 offline SAC traj bank as init
        (mutation still diffusion). Requires ``sac_bank`` + batch scenario indices.
  * per-(world, replan-step) instrumentation logged to ``self.diagnostics``.

Keeping selection/mutation identical across modes is exactly the paired-seed
control the plan requires: population size, ES iterations, mutation schedule,
diffusion model, and rollout budget are all inherited from the parent, so any
difference between arms is attributable to the selection score / init policy alone.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from planner.abstract_planner import PlannerResult
from planner.diffusion_es_planner import DiffusionESPlanner
from data.preprocess import preprocess_simulator_state

from experiments.dense_scorer import DenseReturnScorer, DenseConfig
from experiments.descriptors import (
    compute_descriptors, descriptor_matrix, select_diverse, count_modes,
)
from experiments.sac_bank import (
    SacInitBank, sac_slice_to_ego_norm, select_sac_population,
)


# ``diverse_sac*`` mixes both banks (SAC augments diffusion); ``sac*`` replaces
# the diffusion init bank outright.
MIXED_INIT_CHOICES = ("diverse_sac", "diverse_sac_safe")
SAC_INIT_CHOICES = ("sac", "sac_safe") + MIXED_INIT_CHOICES
INIT_SELECT_CHOICES = ("none", "diverse", "diverse_safe") + SAC_INIT_CHOICES


class InstrumentedESPlanner(DiffusionESPlanner):
    def __init__(self, *args, selection_mode: str = "binary",
                 dense_cfg: Optional[DenseConfig] = None,
                 init_bank_multiplier: int = 1,
                 init_select: str = "none",
                 sac_bank: Optional[SacInitBank] = None,
                 tf_shard: Optional[str] = None,
                 sac_frac: float = 0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if selection_mode not in ("binary", "dense"):
            raise ValueError(f"selection_mode must be binary|dense, got {selection_mode}")
        if init_select not in INIT_SELECT_CHOICES:
            raise ValueError(
                f"init_select must be one of {'|'.join(INIT_SELECT_CHOICES)}, "
                f"got {init_select}")
        if not 0.0 <= float(sac_frac) <= 1.0:
            raise ValueError(f"sac_frac must be in [0, 1], got {sac_frac}")
        self.selection_mode = selection_mode
        self.dense_scorer = DenseReturnScorer(dense_cfg or DenseConfig())
        self.init_bank_multiplier = max(1, int(init_bank_multiplier))
        self.init_select = init_select
        self.sac_bank = sac_bank
        self.sac_frac = float(sac_frac)
        self.tf_shard = tf_shard  # e.g. "00000"; set per-tfrecord by runner/driver
        # Filled by simulation.runner before each batch:
        self.batch_scenario_indices: Optional[List[int]] = None
        if (self.init_select in ("diverse", "diverse_safe") + MIXED_INIT_CHOICES
                and self.init_bank_multiplier < 2):
            # diversity over a size-K bank is a no-op; force at least 2x
            self.init_bank_multiplier = 2
        if self.init_select in SAC_INIT_CHOICES and self.sac_bank is None:
            raise ValueError(f"init_select {self.init_select} requires sac_bank=")
        self.diagnostics: List[dict] = []

    # ---- scoring helpers ----------------------------------------------------
    def _binary_scores_bk(self, world_bkt5, sim_state, timestep):
        out = []
        for w in range(world_bkt5.shape[0]):
            out.append(self.scorers[w].compute_score(
                trajectories=world_bkt5[w], sim_state=sim_state,
                timestep=timestep, world_idx=w))
        return jnp.stack(out)

    def _dense_scores_bk(self, world_bkt5, sim_state, timestep, goal_b2):
        out = []
        for w in range(world_bkt5.shape[0]):
            gxy = None if goal_b2 is None else np.asarray(goal_b2)[w]
            out.append(self.dense_scorer.compute_dense_return(
                trajectories=world_bkt5[w], sim_state=sim_state,
                timestep=timestep, world_idx=w, goal_xy=gxy))
        return jnp.stack(out)

    def _selection_scores_bk(self, world_bkt5, sim_state, timestep, goal_b2):
        if self.selection_mode == "binary":
            return self._binary_scores_bk(world_bkt5, sim_state, timestep)
        return self._dense_scores_bk(world_bkt5, sim_state, timestep, goal_b2)

    # ---- init-bank helpers --------------------------------------------------
    @staticmethod
    def _pick_safe_then_diverse(mat, binary, n, *, seed, safe_first: bool) -> np.ndarray:
        """Pick ``n`` rows of ``mat``: binary-safe first, remainder by FPS.

        With ``safe_first=False`` this is plain FPS over the whole bank. Safe-first
        exists because Stage-0 showed the rare safe modes are exactly what diversity
        selection would otherwise throw away.
        """
        n = int(min(int(n), mat.shape[0]))
        if n <= 0:
            return np.zeros((0,), dtype=np.int64)
        safe_idx = (np.flatnonzero(binary >= 1.0 - 1e-6) if safe_first
                    else np.zeros((0,), dtype=np.int64))
        if len(safe_idx) >= n:
            chosen = select_diverse(mat[safe_idx], n, seed=seed)
            return np.asarray(safe_idx[np.asarray(chosen, dtype=np.int64)], dtype=np.int64)
        rest = np.setdiff1d(np.arange(mat.shape[0], dtype=np.int64), safe_idx)
        n_fill = n - len(safe_idx)
        fill = (select_diverse(mat[rest], n_fill, seed=seed)
                if n_fill > 0 and len(rest) > 0 else np.zeros((0,), dtype=np.int64))
        return np.concatenate(
            [safe_idx, rest[np.asarray(fill, dtype=np.int64)]]).astype(np.int64)

    def _descriptor_matrix_for(self, world_kt5, *, goal_xy):
        """Descriptor matrix for one world's candidate set, anchored on member 0."""
        ref_xy, ref_yaw = world_kt5[0, 0, :2], float(world_kt5[0, 0, 2])
        desc = compute_descriptors(
            world_kt5, dt=float(self.preprocess_cfg.model_dt),
            goal_xy=goal_xy, ref_xy=ref_xy, ref_yaw=ref_yaw)
        return descriptor_matrix(desc)

    def _build_diffusion_bank(
        self, *, cond_bf, inst_cond_bf, instruction_mask, rng,
        pre_batch, sim_state, timestep,
    ):
        """Sample ``multiplier * K`` diffusion candidates in K-sized chunks.

        Chunked and accumulated on host to avoid a single [B, K*M, ...] device alloc.
        """
        norm_chunks: List[np.ndarray] = []
        world_chunks: List[np.ndarray] = []
        binary_chunks: List[np.ndarray] = []
        world_t_seconds_bt = world_t_valid_bt = None
        for _ in range(self.init_bank_multiplier):
            rng, key = jax.random.split(rng)
            chunk_norm = self._sample_population_jit(
                cond_bf=cond_bf, inst_cond_bf=inst_cond_bf,
                inst_cond_mask_bf=instruction_mask, rng=key)
            chunk_world, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(
                chunk_norm, pre_batch)
            chunk_binary = np.asarray(self._binary_scores_bk(chunk_world, sim_state, timestep))
            norm_chunks.append(np.asarray(chunk_norm))
            world_chunks.append(np.asarray(chunk_world))
            binary_chunks.append(chunk_binary)

        return (
            np.concatenate(norm_chunks, axis=1),      # [B, M, T, D]
            np.concatenate(world_chunks, axis=1),     # [B, M, T, 5]
            np.concatenate(binary_chunks, axis=1),    # [B, M]
            world_t_seconds_bt, world_t_valid_bt, rng,
        )

    # ---- Stage-2 diversified initialization ---------------------------------
    def _sample_and_downselect_diverse(
        self, *, cond_bf, inst_cond_bf, instruction_mask, rng,
        pre_batch, sim_state, timestep, goal_b2,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, List[dict], Any]:
        """Sample an oversized diffusion bank, then FPS down-select to K."""
        B = int(self.num_worlds)
        K = int(self.population_size)
        seed = int(timestep) * 1009 + K

        (bank_norm, bank_world, bank_binary,
         world_t_seconds_bt, world_t_valid_bt, rng) = self._build_diffusion_bank(
            cond_bf=cond_bf, inst_cond_bf=inst_cond_bf,
            instruction_mask=instruction_mask, rng=rng,
            pre_batch=pre_batch, sim_state=sim_state, timestep=timestep)
        M = bank_norm.shape[1]

        sel_norm, sel_world, meta = [], [], []
        for w in range(B):
            world = bank_world[w]
            binary = bank_binary[w]
            gxy = None if goal_b2 is None else np.asarray(goal_b2)[w]
            mat = self._descriptor_matrix_for(world, goal_xy=gxy)
            n_modes_bank = count_modes(mat)
            n_safe_bank = int((binary >= 1.0 - 1e-6).sum())

            idx = self._pick_safe_then_diverse(
                mat, binary, K, seed=seed + w,
                safe_first=(self.init_select == "diverse_safe"))

            sel_norm.append(bank_norm[w, idx])
            sel_world.append(bank_world[w, idx])
            meta.append({
                "init_bank_size": int(M),
                "init_bank_num_safe": n_safe_bank,
                "init_bank_num_modes": int(n_modes_bank),
                "init_selected_num_modes": int(count_modes(mat[idx])),
                "init_selected_num_safe": int((binary[idx] >= 1.0 - 1e-6).sum()),
            })
        return (jnp.asarray(np.stack(sel_norm)),
                jnp.asarray(np.stack(sel_world)),
                world_t_seconds_bt, world_t_valid_bt, meta, rng)

    # ---- Stage-3 SAC offline-bank initialization ----------------------------
    def _require_sac_bank(self) -> None:
        if self.sac_bank is None:
            raise RuntimeError("sac_bank not set")
        if not self.batch_scenario_indices:
            raise RuntimeError(
                "batch_scenario_indices not set — runner must assign them before plan")
        if self.tf_shard is None:
            raise RuntimeError("tf_shard not set on planner (e.g. '00000')")
        if len(self.batch_scenario_indices) < int(self.num_worlds):
            raise RuntimeError(
                f"batch_scenario_indices has {len(self.batch_scenario_indices)} entries "
                f"< num_worlds={self.num_worlds}")

    def _sample_and_downselect_sac(
        self, *, pre_batch, sim_state, timestep, goal_b2, rng,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, List[dict], Any]:
        """Load offline SAC trajs for this batch, convert to ego-norm, optionally safe-filter."""
        self._require_sac_bank()

        B = int(self.num_worlds)
        K = int(self.population_size)
        cfg = self.preprocess_cfg
        H = int(cfg.predict_horizon)
        origin_xy = np.asarray(pre_batch.aux["origin_xy"])
        anchor_yaw = np.asarray(pre_batch.aux["anchor_yaw"])

        sel_norm = []
        meta = []
        for w in range(B):
            if w >= len(self.batch_scenario_indices):
                raise RuntimeError(
                    f"world {w} >= len(batch_scenario_indices)="
                    f"{len(self.batch_scenario_indices)}")
            scen_idx = int(self.batch_scenario_indices[w])
            traj_kt5, safe_k, len_k = self.sac_bank.get(self.tf_shard, scen_idx)
            traj_sel, _safe_sel, len_sel, m = select_sac_population(
                traj_kt5, safe_k, len_k,
                population_size=K,
                sac_safe=(self.init_select == "sac_safe"),
            )
            norm_ktd = sac_slice_to_ego_norm(
                traj_sel, len_sel,
                timestep=int(timestep),
                origin_xy=origin_xy[w],
                anchor_yaw=float(anchor_yaw[w]),
                predict_horizon=H,
                model_dt=float(cfg.model_dt),
                ego_range=float(cfg.ego_range),
                max_velocity=float(cfg.max_velocity),
                bank_dt=float(self.sac_bank.dt),
                bank_start_timestep=int(self.sac_bank.start_timestep),
            )
            sel_norm.append(norm_ktd)
            m.update(self._sac_scene_stats(traj_kt5, len_k, origin_xy[w], timestep,
                                  int(self.sac_bank.start_timestep)))
            meta.append(m)

        current_norm_bktd = jnp.asarray(np.stack(sel_norm, axis=0))  # [B,K,H,5]
        current_world_bkt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(
            current_norm_bktd, pre_batch)
        return (current_norm_bktd, current_world_bkt5,
                world_t_seconds_bt, world_t_valid_bt, meta, rng)

    @staticmethod
    def _sac_scene_stats(traj_kt5, lengths_k, origin_xy, timestep, bank_start_timestep=10) -> dict:
        """Bank sanity for one scene: is it actually diverse, is it in frame?

        ``unique_members`` < K means the bank stores duplicate rollouts, so the K
        dimension buys no coverage. ``anchor_gap_m`` is the distance from the ego's
        current world pose to the bank pose at the same timestep — metres if the
        bank belongs to this scene, hundreds/thousands if the row is mismatched or
        in another frame, in which case selecting it teleports the ego.
        """
        K = int(traj_kt5.shape[0])
        L = int(np.max(lengths_k)) if K else 0
        t = int(min(max(int(timestep) - int(bank_start_timestep), 0), max(L - 1, 0)))
        gap = np.linalg.norm(
            traj_kt5[:, t, :2] - np.asarray(origin_xy, dtype=np.float64).reshape(1, 2), axis=1)
        flat = np.round(traj_kt5[:, :max(L, 1), :2].reshape(K, -1), 3)
        return {
            "init_sac_bank_unique_members": int(len(np.unique(flat, axis=0))),
            "init_sac_anchor_gap_m": float(np.median(gap)),
            "init_sac_anchor_gap_min_m": float(np.min(gap)),
        }

    def _build_sac_bank(self, *, pre_batch, sim_state, timestep):
        """Convert the *whole* per-scene SAC bank to ego-norm + world + binary score.

        Unlike ``_sample_and_downselect_sac`` this keeps every bank member as a
        candidate; the caller decides how many survive into the population.
        """
        self._require_sac_bank()
        B = int(self.num_worlds)
        cfg = self.preprocess_cfg
        origin_xy = np.asarray(pre_batch.aux["origin_xy"])
        anchor_yaw = np.asarray(pre_batch.aux["anchor_yaw"])

        norms, bank_safe, stats = [], [], []
        for w in range(B):
            scen_idx = int(self.batch_scenario_indices[w])
            traj_kt5, safe_k, len_k = self.sac_bank.get(self.tf_shard, scen_idx)
            stats.append(self._sac_scene_stats(traj_kt5, len_k, origin_xy[w], timestep,
                                  int(self.sac_bank.start_timestep)))
            norms.append(sac_slice_to_ego_norm(
                traj_kt5, len_k,
                timestep=int(timestep),
                origin_xy=origin_xy[w],
                anchor_yaw=float(anchor_yaw[w]),
                predict_horizon=int(cfg.predict_horizon),
                model_dt=float(cfg.model_dt),
                ego_range=float(cfg.ego_range),
                max_velocity=float(cfg.max_velocity),
                bank_dt=float(self.sac_bank.dt),
                bank_start_timestep=int(self.sac_bank.start_timestep),
            ))
            bank_safe.append(np.asarray(safe_k).astype(bool))

        sac_norm_bktd = jnp.asarray(np.stack(norms, axis=0))     # [B, Ks, H, D]
        sac_world_bkt5, _, _ = self._postprocess_population(sac_norm_bktd, pre_batch)
        sac_binary = np.asarray(self._binary_scores_bk(sac_world_bkt5, sim_state, timestep))
        return (np.asarray(sac_norm_bktd), np.asarray(sac_world_bkt5),
                sac_binary, np.stack(bank_safe, axis=0), stats)

    # ---- Stage-4 mixed init: SAC trajectories diversify the diffusion bank ---
    def _sample_and_downselect_mixed(
        self, *, cond_bf, inst_cond_bf, instruction_mask, rng,
        pre_batch, sim_state, timestep, goal_b2,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, List[dict], Any]:
        """Augment the diffusion init bank with SAC trajectories.

        The population is split by ``sac_frac``: a reserved quota of K comes from
        the SAC bank and the rest from an oversized diffusion bank, each side
        chosen safe-first then FPS-diversified. The quota is what makes this an
        *augmentation* — SAC modes cannot be diversity-selected away, which is what
        pooling both banks into one FPS pass would allow.
        """
        B = int(self.num_worlds)
        K = int(self.population_size)
        seed = int(timestep) * 1009 + K
        safe_first = self.init_select == "diverse_sac_safe"

        (dif_norm, dif_world, dif_binary,
         world_t_seconds_bt, world_t_valid_bt, rng) = self._build_diffusion_bank(
            cond_bf=cond_bf, inst_cond_bf=inst_cond_bf,
            instruction_mask=instruction_mask, rng=rng,
            pre_batch=pre_batch, sim_state=sim_state, timestep=timestep)
        sac_norm, sac_world, sac_binary, sac_bank_safe, sac_stats = self._build_sac_bank(
            pre_batch=pre_batch, sim_state=sim_state, timestep=timestep)

        if dif_norm.shape[2:] != sac_norm.shape[2:]:
            raise RuntimeError(
                f"diffusion/SAC init banks disagree on trajectory layout: "
                f"{dif_norm.shape[2:]} vs {sac_norm.shape[2:]}")

        M = int(dif_norm.shape[1])
        Ks = int(sac_norm.shape[1])
        n_sac = int(min(Ks, round(self.sac_frac * K)))
        n_dif = K - n_sac
        if n_dif > M:
            raise RuntimeError(
                f"diffusion bank too small: need {n_dif} of M={M}; "
                f"raise --init_bank_multiplier or --sac_frac")

        sel_norm, sel_world, meta = [], [], []
        for w in range(B):
            gxy = None if goal_b2 is None else np.asarray(goal_b2)[w]
            # One descriptor space over both banks so the two FPS passes are
            # measured on the same scale, and mode counts stay comparable.
            world_all = np.concatenate([sac_world[w], dif_world[w]], axis=0)
            binary_all = np.concatenate([sac_binary[w], dif_binary[w]], axis=0)
            mat_all = self._descriptor_matrix_for(world_all, goal_xy=gxy)
            mat_sac, mat_dif = mat_all[:Ks], mat_all[Ks:]

            idx_sac = self._pick_safe_then_diverse(
                mat_sac, sac_binary[w], n_sac, seed=seed + w, safe_first=safe_first)
            idx_dif = self._pick_safe_then_diverse(
                mat_dif, dif_binary[w], n_dif, seed=seed + w, safe_first=safe_first)

            sel_norm.append(np.concatenate(
                [sac_norm[w, idx_sac], dif_norm[w, idx_dif]], axis=0))
            sel_world.append(np.concatenate(
                [sac_world[w, idx_sac], dif_world[w, idx_dif]], axis=0))
            idx_all = np.concatenate([idx_sac, Ks + idx_dif])
            meta.append({
                "init_bank_size": int(Ks + M),
                "init_bank_num_safe": int((binary_all >= 1.0 - 1e-6).sum()),
                "init_bank_num_modes": int(count_modes(mat_all)),
                "init_selected_num_modes": int(count_modes(mat_all[idx_all])),
                "init_selected_num_safe": int((binary_all[idx_all] >= 1.0 - 1e-6).sum()),
                "init_sac_bank_size": Ks,
                "init_sac_bank_num_safe_es": int((sac_binary[w] >= 1.0 - 1e-6).sum()),
                "init_sac_bank_num_safe_rollout": int(sac_bank_safe[w].sum()),
                "init_selected_num_sac": int(len(idx_sac)),
                "init_selected_num_sac_safe": int(
                    (sac_binary[w][idx_sac] >= 1.0 - 1e-6).sum()),
                "init_selected_num_diffusion": int(len(idx_dif)),
                **sac_stats[w],
            })

        return (jnp.asarray(np.stack(sel_norm)),
                jnp.asarray(np.stack(sel_world)),
                world_t_seconds_bt, world_t_valid_bt, meta, rng)

    # ---- main loop (mirrors parent, adds logging + pluggable score) ---------
    def plan_trajectory(self, sim_state, goal, *, rng, instruction=None,
                        instruction_mask=None, timestep=0, mask_goal=False,
                        **kwargs) -> PlannerResult:
        rng, key_pre, key_sample = jax.random.split(rng, 3)
        pre_batch, _ = preprocess_simulator_state(
            sim_state, key_pre, self.preprocess_cfg,
            anchor_step_override=timestep, goal_step_override=90, goal_xy_override=goal)
        features = pre_batch.features
        instruction = jnp.zeros((self.num_worlds, self.preprocess_cfg.inst_dim), dtype=jnp.float32)
        instruction_mask = jnp.zeros((self.num_worlds,), dtype=jnp.float32)
        features["inst_features"] = instruction
        features["inst_valid"] = instruction_mask
        if mask_goal:
            features["goal_xy"] = jnp.zeros_like(features["goal_xy"])
            features["remaining_timesteps"] = jnp.zeros_like(features["remaining_timesteps"])
        features["subgoal_xy"] = jnp.zeros((self.num_worlds, 2), dtype=jnp.float32)
        features["subgoal_valid"] = jnp.zeros((self.num_worlds,), dtype=jnp.float32)
        cond_bf, inst_cond_bf = self._compute_condition_jit(pre_batch.features)

        init_meta = [{"init_bank_size": self.population_size,
                      "init_bank_num_safe": None,
                      "init_bank_num_modes": None,
                      "init_selected_num_modes": None,
                      "init_selected_num_safe": None}
                     for _ in range(self.num_worlds)]

        if self.init_select in ("sac", "sac_safe"):
            (current_norm_bktd, current_world_bkt5,
             world_t_seconds_bt, world_t_valid_bt, init_meta, rng) = (
                self._sample_and_downselect_sac(
                    pre_batch=pre_batch, sim_state=sim_state,
                    timestep=timestep, goal_b2=goal, rng=rng))
        elif self.init_select in MIXED_INIT_CHOICES:
            (current_norm_bktd, current_world_bkt5,
             world_t_seconds_bt, world_t_valid_bt, init_meta, rng) = (
                self._sample_and_downselect_mixed(
                    cond_bf=cond_bf, inst_cond_bf=inst_cond_bf,
                    instruction_mask=instruction_mask, rng=key_sample,
                    pre_batch=pre_batch, sim_state=sim_state,
                    timestep=timestep, goal_b2=goal))
        elif self.init_select == "none" or self.init_bank_multiplier <= 1:
            current_norm_bktd = self._sample_population_jit(
                cond_bf=cond_bf, inst_cond_bf=inst_cond_bf,
                inst_cond_mask_bf=instruction_mask, rng=key_sample)
            current_world_bkt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(
                current_norm_bktd, pre_batch)
        else:
            (current_norm_bktd, current_world_bkt5,
             world_t_seconds_bt, world_t_valid_bt, init_meta, rng) = (
                self._sample_and_downselect_diverse(
                    cond_bf=cond_bf, inst_cond_bf=inst_cond_bf,
                    instruction_mask=instruction_mask, rng=key_sample,
                    pre_batch=pre_batch, sim_state=sim_state,
                    timestep=timestep, goal_b2=goal))

        current_scores_bk = self._selection_scores_bk(current_world_bkt5, sim_state, timestep, goal)

        # instrumentation: initial population, both scores for rank-correlation
        init_binary = np.asarray(self._binary_scores_bk(current_world_bkt5, sim_state, timestep))
        init_dense = np.asarray(self._dense_scores_bk(current_world_bkt5, sim_state, timestep, goal))
        # fill bank-safe from selected if we skipped the bank path
        for w, m in enumerate(init_meta):
            if m["init_bank_num_safe"] is None:
                m["init_bank_num_safe"] = int((init_binary[w] >= 1.0 - 1e-6).sum())
                m["init_selected_num_safe"] = m["init_bank_num_safe"]
        per_iter = []

        for it in range(self.num_iterations + 1):
            elite_idx_bk, elite_scores_bk = self._select_topk_per_scenario(
                current_scores_bk, top_k=self.elite_size)
            sc = np.asarray(current_scores_bk)
            ei = np.asarray(elite_idx_bk)
            per_iter.append({
                "iteration": it,
                "best": sc.max(axis=1).tolist(),
                "mean": sc.mean(axis=1).tolist(),
                "std": sc.std(axis=1).tolist(),
                "num_max": (sc >= (sc.max(axis=1, keepdims=True) - 1e-6)).sum(axis=1).tolist(),
                "num_unique_parents": [int(len(np.unique(ei[w]))) for w in range(ei.shape[0])],
                "pop_size": int(sc.shape[1]),
            })
            if it == self.num_iterations:
                break
            elite_norm_betd = self._gather_population(current_norm_bktd, elite_idx_bk)
            elite_world_bet5, _, _ = self._postprocess_population(elite_norm_betd, pre_batch)
            replicated_norm_bktd = self._replicate_elites(elite_norm_betd, population_size=self.population_size)
            rng, key_resample = jax.random.split(rng)
            current_norm_bktd = self._resample_population_jit(
                cond_bf=cond_bf, inst_cond_bf=inst_cond_bf,
                inst_cond_mask_bf=instruction_mask,
                proposals_bktd=replicated_norm_bktd, rng=key_resample)
            current_world_bkt5, _, _ = self._postprocess_population(current_norm_bktd, pre_batch)
            current_scores_bk = self._selection_scores_bk(current_world_bkt5, sim_state, timestep, goal)
            current_norm_bktd = jnp.concat([elite_norm_betd, current_norm_bktd], axis=1)
            current_scores_bk = jnp.concat([elite_scores_bk, current_scores_bk], axis=1)
            current_world_bkt5 = jnp.concat([elite_world_bet5, current_world_bkt5], axis=1)

        best_idx_b1, best_scores_b1 = self._select_topk_per_scenario(current_scores_bk, top_k=1)
        best_norm_bt1d = self._gather_population(current_norm_bktd, best_idx_b1)
        best_world_bt15 = self._gather_population(current_world_bkt5, best_idx_b1)

        # final-population both-score logging
        final_binary = np.asarray(self._binary_scores_bk(current_world_bkt5, sim_state, timestep))
        final_dense = np.asarray(self._dense_scores_bk(current_world_bkt5, sim_state, timestep, goal))
        for w in range(self.num_worlds):
            rec = {
                "timestep": int(timestep),
                "world_idx": int(w),
                "selection_mode": self.selection_mode,
                "init_select": self.init_select,
                "init_bank_multiplier": self.init_bank_multiplier,
                "sac_frac": self.sac_frac if self.init_select in SAC_INIT_CHOICES else None,
                "init_has_safe": bool((init_binary[w] >= 1.0 - 1e-6).any()),
                "init_num_safe": int((init_binary[w] >= 1.0 - 1e-6).sum()),
                "init_best_binary": float(init_binary[w].max()),
                "init_best_dense": float(init_dense[w].max()),
                "final_num_safe": int((final_binary[w] >= 1.0 - 1e-6).sum()),
                "final_best_binary": float(final_binary[w].max()),
                "final_best_dense": float(final_dense[w].max()),
                "selected_binary": float(final_binary[w][int(np.asarray(best_idx_b1)[w, 0])]),
                "selected_dense": float(final_dense[w][int(np.asarray(best_idx_b1)[w, 0])]),
                "per_iter": per_iter and [{k: (v[w] if isinstance(v, list) else v)
                                           for k, v in pi.items()} for pi in per_iter],
            }
            rec.update(init_meta[w])
            self.diagnostics.append(rec)

        return PlannerResult(
            start_t_b=pre_batch.aux["anchor_step"].astype(jnp.int32),
            trajectory_norm_btd=jnp.squeeze(best_norm_bt1d, axis=1),
            trajectory_world_bt5=jnp.squeeze(best_world_bt15, axis=1),
            world_t_seconds_bt=world_t_seconds_bt,
            world_t_valid_bt=world_t_valid_bt,
            aux=pre_batch.aux,
        )
