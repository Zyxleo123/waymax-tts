"""Weights & Biases logging for the ES experiment arms.

Logs *metrics*, not trajectories: per-(scene, replan-step) init-bank coverage and
safety, SAC provenance, bank-sanity, and the per-scene binary outcome. Everything
logged here is also on disk (diagnostics_<arm>.json + <scene>.json), so plots stay
redrawable without rerunning the sim.

Import is lazy and every entry point is a no-op when --wandb is off or wandb is
missing, so the sim never dies because a logger did.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

# Per-(scene, step) diagnostics fields worth a time series.
STEP_METRICS = (
    "init_num_safe", "final_num_safe",
    "init_best_binary", "final_best_binary", "selected_binary",
    "init_best_dense", "final_best_dense", "selected_dense",
    "init_bank_num_safe", "init_bank_num_modes", "init_selected_num_modes",
    "init_selected_num_safe",
    # SAC-init arms only
    "init_sac_bank_num_safe_es", "init_sac_bank_num_safe_rollout",
    "init_selected_num_sac", "init_selected_num_sac_safe",
    "init_selected_num_diffusion",
    "init_sac_bank_unique_members", "init_sac_anchor_gap_m", "init_sac_anchor_gap_min_m",
)

# A SAC bank row whose pose sits further than this from the ego at the same
# timestep does not belong to this scene (or is in another frame); selecting it
# teleports the ego. Loud on stdout because it invalidates the arm.
ANCHOR_GAP_WARN_M = 25.0


class WandbLogger:
    """Thin wrapper; `enabled=False` makes every method a no-op."""

    def __init__(self, args, enabled: bool = False):
        self.enabled = False
        self.run = None
        if not enabled:
            return
        try:
            import wandb  # noqa: F401
        except Exception as e:
            print(f"[wandb] disabled — import failed: {e}")
            return
        self._wandb = wandb
        try:
            self.run = wandb.init(
                project=getattr(args, "wandb_project", None) or "es-reward-resolution",
                entity=getattr(args, "wandb_entity", None) or None,
                name=getattr(args, "wandb_run_name", None) or os.path.basename(args.output_dir),
                group=getattr(args, "arm", None),
                job_type="es-eval",
                config={k: v for k, v in vars(args).items() if _jsonable(v)},
                dir=getattr(args, "wandb_dir", None) or None,
                mode=getattr(args, "wandb_mode", None) or "online",
            )
            self.enabled = True
            print(f"[wandb] logging to {self.run.url}")
        except Exception as e:
            print(f"[wandb] disabled — init failed: {e}")

    # ---- per-batch logging -------------------------------------------------
    def log_diagnostics(self, records: List[dict], scenario_indices: List[int],
                        shard: Optional[str]) -> None:
        """One time series per (scene, metric), x-axis = replan timestep."""
        if not self.enabled or not records:
            return
        for r in records:
            w = int(r.get("world_idx", 0))
            if w >= len(scenario_indices):
                continue
            scene = f"{shard or 'tf'}_{int(scenario_indices[w]):03d}"
            payload = {f"scene/{scene}/{k}": _num(r[k])
                       for k in STEP_METRICS if _num(r.get(k)) is not None}
            if payload:
                self._safe_log(payload, step=int(r.get("timestep", 0)))

    def warn_on_bank_mismatch(self, records: List[dict], scenario_indices: List[int]) -> List[str]:
        """Print (and return) scenes whose SAC bank row is not in the ego's frame.

        Runs whether or not wandb is on — this is a correctness signal, not telemetry.
        """
        bad = []
        for r in records:
            gap = _num(r.get("init_sac_anchor_gap_min_m"))
            w = int(r.get("world_idx", 0))
            if gap is None or w >= len(scenario_indices):
                continue
            if gap > ANCHOR_GAP_WARN_M:
                scene = int(scenario_indices[w])
                msg = (f"[bank] scene {scene} t={r.get('timestep')}: nearest SAC bank pose is "
                       f"{gap:.1f} m from the ego — bank row does not match this scene")
                if msg not in bad:
                    bad.append(msg)
        for m in bad[:10]:
            print(m)
        if len(bad) > 10:
            print(f"[bank] ... {len(bad) - 10} more mismatched (scene, step) pairs")
        return bad

    # ---- end-of-run summary ------------------------------------------------
    def log_run_summary(self, output_dir: str, records: List[dict], arm: str) -> Dict[str, Any]:
        """Join per-scene outcomes with their diagnostics; log scalars + a table."""
        scenes = _load_scene_results(output_dir)
        by_world = defaultdict(list)
        for r in records:
            by_world[int(r.get("world_idx", 0))].append(r)

        summary: Dict[str, Any] = {"arm": arm, "num_scenarios": len(scenes)}
        if scenes:
            summary["success_rate"] = float(np.mean([s["success"] for s in scenes.values()]))
            summary["goal_reach_rate"] = float(np.mean([s["goal_reached"] for s in scenes.values()]))
            summary["offroad_rate"] = float(np.mean([s["offroad"] for s in scenes.values()]))
            summary["overlap_rate"] = float(np.mean([s["overlap"] for s in scenes.values()]))
        for k in STEP_METRICS:
            vals = [_num(r.get(k)) for r in records if _num(r.get(k)) is not None]
            if vals:
                summary[f"mean/{k}"] = float(np.mean(vals))

        # The two failure modes worth a headline number.
        gaps = [_num(r.get("init_sac_anchor_gap_min_m")) for r in records]
        gaps = [g for g in gaps if g is not None]
        if gaps:
            summary["bank/frac_steps_mismatched"] = float(np.mean([g > ANCHOR_GAP_WARN_M for g in gaps]))
            summary["bank/max_anchor_gap_m"] = float(np.max(gaps))
        uniq = [_num(r.get("init_sac_bank_unique_members")) for r in records]
        uniq = [u for u in uniq if u is not None]
        if uniq:
            summary["bank/min_unique_members"] = float(np.min(uniq))

        print("[summary] " + json.dumps(summary, indent=2, sort_keys=True))
        if not self.enabled:
            return summary

        self._safe_log({f"summary/{k}": v for k, v in summary.items() if _num(v) is not None})
        try:
            cols = ["scene", "success", "goal_reached", "offroad", "overlap",
                    "min_goal_distance_m", "final_goal_distance_m"] + list(STEP_METRICS)
            table = self._wandb.Table(columns=cols)
            idx_order = sorted(scenes)
            for w, scene_idx in enumerate(idx_order):
                s = scenes[scene_idx]
                rs = by_world.get(w, [])
                row = [scene_idx, s["success"], s["goal_reached"], s["offroad"], s["overlap"],
                       s["min_goal_distance_m"], s["final_goal_distance_m"]]
                for k in STEP_METRICS:
                    vals = [_num(r.get(k)) for r in rs if _num(r.get(k)) is not None]
                    row.append(float(np.mean(vals)) if vals else None)
                table.add_data(*row)
            self._safe_log({"scenes": table})
        except Exception as e:
            print(f"[wandb] table logging failed: {e}")
        return summary

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass

    def _safe_log(self, payload: dict, step: Optional[int] = None) -> None:
        try:
            self.run.log(payload, step=step) if step is not None else self.run.log(payload)
        except Exception as e:
            print(f"[wandb] log failed ({e}); continuing")


def _num(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v)):
        return float(v)
    return None


def _jsonable(v) -> bool:
    return isinstance(v, (str, int, float, bool, type(None), list, tuple, dict))


def _load_scene_results(output_dir: str) -> Dict[int, dict]:
    out = {}
    for f in glob.glob(os.path.join(output_dir, "**", "*.scenario_*.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "scenario_idx" in d and "success" in d:
            out[int(d["scenario_idx"])] = d
    return out
