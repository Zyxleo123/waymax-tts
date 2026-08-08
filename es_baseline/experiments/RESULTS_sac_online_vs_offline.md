# WOMD-checkpoint online SAC init vs. offline bank (ES failure set)

**Date:** 2026-08-08 · **Branch:** `tts` · **Fix commit:** `a945b60`

Does rolling the new WOMD-supported V-Max SAC checkpoint *online* as the ES
initialization help on the ES failure set, compared to the old offline init bank?

## TL;DR

Yes, modestly, and **with no regressions**. On the scenes both arms share:

| metric                        | offline (old bank) | online (new WOMD ckpt) |
|-------------------------------|:------------------:|:----------------------:|
| failure scenes solved (n=18)  | 0                  | **2**                  |
| control scenes kept (n=10)    | 3                  | **4**                  |
| conversions (loss→win)        | —                  | **3**                  |
| regressions (win→loss)        | —                  | **0**                  |

- Failures newly solved: **tf00001 #360, tf00002 #20**
- Control newly solved: **tf00001 #0**

## The catch: the raw comparison was wrong, not the runs

A naive first pass reported "online solves 5 failures but regresses 2 controls."
That was **entirely an output-labeling bug**, not real behavior.

`es_baseline/data/scenario_loader.py::load_scenario_state_batch_fast` returns each
batch in **ascending record order** (plus a `scenario_to_batch_idx` remap dict).
`runner.py` discarded the remap and wrote each per-scene result file using the
**requested** order instead. The online arm requests failure-set-first then
controls (unsorted), so every `.json` / `.trajectory.npz` filename was shifted
relative to the data inside it. The SDC/ego and goal were always **correct** —
only the filename/index was scrambled. The offline arm requests already-sorted
indices, so it was never affected.

Verification: re-keying each online result file to its true scene (sort each
`num_worlds` batch, same position) makes the online `(ego_idx, goal_xy)` match the
offline true-scene values **28/28** across the three shards. See
`corrected_compare.py` (and `remap_verify.py` in the scratch analysis).

## Fix (commit `a945b60`)

- `runner.py`: relabel `scenario_indices` to the loader's actual order right after
  load — `scenario_indices = sorted(scenario_to_batch_idx, key=scenario_to_batch_idx.get)`.
  No-op for a sorted request, so offline behavior is unchanged. This also corrects
  the lane-graph lookup and `planner.batch_scenario_indices`, which shared the bug.
- `analyze_arms.py`: map `world_idx`→scene via the same per-batch sort.

## Caveats / status

1. **This table is a post-hoc remap of the OLD mislabeled runs**, verified byte-for-byte
   against the offline ground truth — not yet a re-run. The online arm is being
   re-run with the fix (`es_baseline/slurm_es_online.sbatch`, repointed to gpu26) so
   the on-disk files are correctly labeled. Re-confirm with `corrected_compare.py`
   after it lands.
2. **Not a clean online-vs-offline ablation.** "online" is confounded with "new WOMD
   checkpoint": the offline bank predates the checkpoint (built 2026-07-28), while the
   online arm rolls the 2026-08-04 `womd_raw_parity_sac_lq` checkpoint live. To isolate
   the "roll live" mechanism, rebuild `sac_init_bank.npz` from the same checkpoint and
   rerun the offline arm.
3. Only the vanilla `binary`/`dense` arms flow through `analyze_arms.py`; the SAC-init
   arms are compared with `corrected_compare.py`.

## Redraw

```bash
python3 es_baseline/experiments/corrected_compare.py
```

Reads the per-scene result JSONs and one `diagnostics_*.json` (for the requested
order + `num_worlds`) from each run under
`/zfsauton/scratch/yixiz/waymax_rs/es_baseline/experiments/out`. Edit the `ONLINE`
/ `OFFLINE` run-dir maps at the top to point at the re-run outputs.
