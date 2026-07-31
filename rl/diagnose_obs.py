"""Report per-block occupancy of the observation on *real* scenarios.

Both bugs that have cost a training run so far were of the same shape: a block
silently came back empty (route not loaded; road edges filtered out) while the
observation width stayed correct, so nothing raised. The unit tests only prove
the layout is self-consistent on synthetic data -- this checks the blocks are
actually populated on real WOMD scenes.

For each block it reports the fraction of valid tokens and the magnitude of the
features, aggregated over scenarios and over a short rollout. A block sitting at
0% valid (or all-zero features) means the policy is blind to it.

CPU-only; no GPU needed::

    JAX_PLATFORMS=cpu python -m rl.diagnose_obs --limit 16
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from rl.obs_layout import DEFAULT_OBS_BLOCKS, block_offsets
from rl.scenario_source import ScenarioSource
from rl.waymax_env import RewardConfig, WaymaxGymEnv


def block_stats(obs: np.ndarray) -> dict[str, tuple[float, float]]:
    """Per-block (valid-token fraction, mean |feature|) for one observation."""
    out: dict[str, tuple[float, float]] = {}
    for offset, block in zip(block_offsets(DEFAULT_OBS_BLOCKS), DEFAULT_OBS_BLOCKS):
        rows = obs[offset : offset + block.size].reshape(block.num_tokens, block.stride)
        if block.has_valid:
            valid = rows[:, -1] > 0.5
            feats = rows[:, :-1]
        else:
            # path_target has no valid bit; treat a non-zero row as populated.
            valid = np.abs(rows).sum(axis=-1) > 0
            feats = rows
        frac = float(valid.mean())
        mag = float(np.abs(feats[valid]).mean()) if valid.any() else 0.0
        out[block.name] = (frac, mag)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Observation block occupancy on real scenarios.")
    p.add_argument("--failure-dir", type=str,
                   default="/zfsauton/scratch/mineuih/waymax_rs/failure_samples")
    p.add_argument("--limit", type=int, default=16, help="Scenarios to inspect.")
    p.add_argument("--steps", type=int, default=5, help="Zero-action steps per scenario.")
    args = p.parse_args()

    source = ScenarioSource.from_failure_dir(args.failure_dir, limit=args.limit)
    env = WaymaxGymEnv(source, reward_config=RewardConfig(), sequential=True)

    samples: list[dict[str, tuple[float, float]]] = []
    for _ in range(len(source)):
        obs, _ = env.reset()
        samples.append(block_stats(np.asarray(obs)))
        for _ in range(args.steps):
            obs, _, term, trunc, _ = env.step(np.zeros(env.action_space.shape, np.float32))
            samples.append(block_stats(np.asarray(obs)))
            if term or trunc:
                break

    print(f"\n{len(source)} scenarios, {len(samples)} observations\n")
    print(f"{'block':<16}{'tokens':>8}{'valid %':>10}{'mean|feat|':>12}   status")
    print("-" * 62)
    for block in DEFAULT_OBS_BLOCKS:
        fracs = np.array([s[block.name][0] for s in samples])
        mags = np.array([s[block.name][1] for s in samples])
        frac, mag = float(fracs.mean()), float(mags.mean())
        if frac == 0.0:
            status = "EMPTY - policy is blind to this block"
        elif mag == 0.0:
            status = "valid but all-zero features"
        elif frac < 0.05:
            status = "nearly empty"
        else:
            status = "ok"
        print(f"{block.name:<16}{block.num_tokens:>8}{100 * frac:>9.1f}%{mag:>12.4f}   {status}")

    empty = [b.name for b in DEFAULT_OBS_BLOCKS
             if np.mean([s[b.name][0] for s in samples]) == 0.0]
    print("\n" + ("FAIL: empty blocks: " + ", ".join(empty) if empty
                  else "All blocks populated."))


if __name__ == "__main__":
    main()
