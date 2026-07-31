"""Paired success + failure-taxonomy report for dense vs dense+diverse arms.

Reads ``experiments/out/`` result JSONs + ``diagnostics_*.json`` and prints:

  * Stage-1 success (failures / controls) per arm
  * paired failure conversions (diverse over vanilla dense)
  * Stage-0/2 coverage (init_has_safe, bank safe/modes when logged)
  * failure taxonomy table (same buckets used in the post-hoc dense audit)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

ORIG_FAIL = {
    "tfrecord-00000-of-01000": {71, 92, 113, 134, 187, 252, 315, 377},
    "tfrecord-00001-of-01000": {45, 62, 147, 158, 200, 243, 289, 360, 463},
    "tfrecord-00002-of-01000": {0, 13, 20, 24, 70, 94, 97},
}
ORIG_CONTROL = {
    "tfrecord-00000-of-01000": {0, 1, 2, 3},
    "tfrecord-00001-of-01000": {0, 1, 2, 3},
    "tfrecord-00002-of-01000": {1, 2, 3, 4},
}


def _canon_tfrecord(raw: str) -> str:
    m = re.search(r"(tfrecord-\d{5}-of-\d+)", os.path.basename(raw or ""))
    return m.group(1) if m else (os.path.basename(raw or "") or "?")


def _arm_of_path(path: str) -> str | None:
    # Parse the run-dir name first: es_<arm>_<tag>_<timestamp>. The prefix list
    # below cannot be trusted for arms whose names extend another arm's name —
    # e.g. dense_diverse_sac_safe_m4_f0.5 starts with dense_diverse.
    m = re.search(r"(?:^|[/\\])es_([^/\\]+?)_tf\d{5}_\d{8}_\d{6}", path)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|[/\\])es_([^/\\]+?)_\d{8}_\d{6}", path)
    if m:
        return m.group(1)
    # longer / more-specific prefixes first
    for arm in (
        "dense_diverse_sac_safe", "dense_diverse_sac",
        "binary_diverse_sac_safe", "binary_diverse_sac",
        "dense_sac_safe", "dense_sac", "binary_sac_safe", "binary_sac",
        "dense_diverse_safe_m4", "dense_diverse_safe_m8", "dense_diverse_m4",
        "dense_diverse_safe", "dense_diverse", "binary_diverse_safe", "binary",
        "dense",
    ):
        token = f"es_{arm}"
        if f"/{token}" in path or path.startswith(token) or f"{os.sep}{token}" in path:
            return arm
    return None


def _arm_of_diag(blob: dict, path: str) -> str:
    a = blob.get("args") or {}
    if a.get("arm"):
        return str(a["arm"])
    sel = a.get("selection", "binary")
    init = a.get("init_select", "none")
    mult = a.get("init_bank_multiplier", 1)
    if init and init != "none":
        if init in ("sac", "sac_safe"):
            return f"{sel}_{init}"
        if init in ("diverse_sac", "diverse_sac_safe"):
            return f"{sel}_{init}_m{mult}_f{float(a.get('sac_frac', 0.5)):g}"
        return f"{sel}_{init}_m{mult}"
    return _arm_of_path(path) or sel


def _load(out: str):
    results = {}  # (arm, rec, idx) -> result dict
    for f in glob.glob(os.path.join(out, "**", "*.scenario_*.json"), recursive=True):
        if "142833" in f:  # aborted early binary run
            continue
        arm = _arm_of_path(f)
        if arm is None:
            continue
        d = json.load(open(f))
        results[(arm, _canon_tfrecord(d.get("tfrecord", "")), int(d["scenario_idx"]))] = d

    diags = defaultdict(list)  # (arm, rec, idx) -> list[records]
    for f in glob.glob(os.path.join(out, "**", "diagnostics_*.json"), recursive=True):
        if "142833" in f:
            continue
        blob = json.load(open(f))
        arm = _arm_of_diag(blob, f)
        a = blob["args"]
        m = re.search(r"tf(\d{5})", a.get("tfrecord_dir", ""))
        canon = f"tfrecord-{m.group(1)}-of-01000" if m else "?"
        idx_list = [int(x) for x in str(a.get("scenario_indices", "")).split(",") if x != ""]
        for r in blob["records"]:
            w = r["world_idx"]
            if w < len(idx_list):
                diags[(arm, canon, idx_list[w])].append(r)
    return results, diags


def _taxonomy(d: dict, rs: list) -> str:
    any_safe = any(r.get("init_has_safe") or r.get("final_num_safe", 0) > 0 for r in rs) if rs else False
    always_safe_sel = bool(rs) and all(
        r.get("final_num_safe", 0) > 0 or r.get("selected_binary", 0) >= 0.5 for r in rs)
    prefer_unsafe = any(
        r.get("final_num_safe", 0) > 0 and r.get("selected_binary", 0) < 0.5 for r in rs)
    bank_has_safe_sel_empty = any(
        (r.get("init_bank_num_safe") or 0) > 0 and not r.get("init_has_safe") for r in rs)

    if d.get("offroad") and not d.get("overlap"):
        mode = "offroad"
    elif d.get("overlap") and not d.get("offroad"):
        mode = "collision"
    elif d.get("overlap") and d.get("offroad"):
        mode = "both"
    elif not d.get("goal_reached"):
        mode = "no_goal"
    else:
        mode = "other"

    if prefer_unsafe:
        return f"{mode}+prefer_unsafe_over_safe"
    if bank_has_safe_sel_empty:
        return f"{mode}+bank_had_safe_FPS_dropped"  # should not happen with diverse_safe
    if not any_safe:
        return f"{mode}+no_safe_in_bank"
    if always_safe_sel and not d.get("goal_reached"):
        return f"{mode}+safe_but_no_goal"
    if any_safe:
        return f"{mode}+intermittent_safe"
    return mode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline_arm", default="dense",
                    help="vanilla arm to compare against (default: dense)")
    ap.add_argument("--diverse_arm", default=None,
                    help="comparison arm name; auto-detect dense_*diverse* / dense_sac* if omitted")
    ap.add_argument("--overlap_only", action="store_true",
                    help="only compare scenes present in BOTH arms (for 29-hit SAC runs)")
    args = ap.parse_args()

    results, diags = _load(args.out)
    arms = sorted({a for a, _, _ in results.keys()})
    print(f"arms found: {arms}")

    base = args.baseline_arm
    div = args.diverse_arm
    if div is None:
        cands = [a for a in arms if ("diverse" in a or "sac" in a) and a != base]
        if not cands:
            print("ERROR: no diverse/sac arm found in out/. Did the run finish?")
            return
        # prefer sac_safe over diverse when both exist and user didn't specify
        sac = [a for a in cands if "sac" in a]
        div = sorted(sac)[-1] if sac else sorted(cands)[-1]
        print(f"auto-selected comparison arm: {div}")

    if args.overlap_only:
        base_keys = {(rec, idx) for (a, rec, idx) in results if a == base}
        div_keys = {(rec, idx) for (a, rec, idx) in results if a == div}
        overlap_keys = base_keys & div_keys
        print(f"overlap_only: comparing {len(overlap_keys)} scenes present in both "
              f"{base} and {div} (denom != full 36)")
        results = {k: v for k, v in results.items()
                   if (k[1], k[2]) in overlap_keys and k[0] in (base, div)}
        diags = {k: v for k, v in diags.items()
                 if (k[1], k[2]) in overlap_keys and k[0] in (base, div)}

    def is_fail(rec, idx): return idx in ORIG_FAIL.get(rec, set())
    def is_ctrl(rec, idx): return idx in ORIG_CONTROL.get(rec, set())

    print("=" * 78)
    print("STAGE 1 — final binary success by arm")
    for arm in (base, div):
        f_succ = f_tot = c_succ = c_tot = 0
        for (a, rec, idx), d in results.items():
            if a != arm:
                continue
            ok = bool(d.get("success"))
            if is_fail(rec, idx):
                f_tot += 1; f_succ += int(ok)
            elif is_ctrl(rec, idx):
                c_tot += 1; c_succ += int(ok)
        n_ok = sum(1 for (a, _, _), d in results.items() if a == arm and d.get("success"))
        n_tot = sum(1 for (a, _, _) in results if a == arm)
        print(f"  {arm:>28}:  failures {f_succ}/{f_tot}   controls {c_succ}/{c_tot}   "
              f"overall {n_ok}/{n_tot}")

    print("=" * 78)
    print(f"PAIRED failures   (B={base} success, D={div} success)"
          + ("  [overlap_only]" if args.overlap_only else ""))
    print(f"  {'tfrecord':<22} {'idx':>4}  {'B':>1} {'D':>1}   delta")
    conv = regress = skipped = 0
    for rec in sorted(ORIG_FAIL):
        for idx in sorted(ORIG_FAIL[rec]):
            b = results.get((base, rec, idx))
            d = results.get((div, rec, idx))
            if not b or not d:
                skipped += 1
                continue
            bs, ds = int(bool(b.get("success"))), int(bool(d.get("success")))
            tag = ""
            if ds and not bs:
                tag = "  <-- CONVERTED"; conv += 1
            elif bs and not ds:
                tag = "  <-- regressed"; regress += 1
            print(f"  {rec:<22} {idx:>4}  {bs:>1} {ds:>1}   {ds - bs:+d}{tag}")
    print(f"  conversions ({div} over {base}): {conv}   regressions: {regress}"
          + (f"   skipped(missing either arm): {skipped}" if skipped else ""))

    print("=" * 78)
    print("STAGE 0/2 — init coverage (averaged over replan steps)")
    for arm in (base, div):
        recs = [r for (a, _, _), rs in diags.items() if a == arm for r in rs]
        if not recs:
            print(f"  {arm:>28}: (no diagnostics)"); continue
        n = len(recs)
        init_safe = sum(int(r.get("init_has_safe", False)) for r in recs) / n
        avg_safe = sum(r.get("init_num_safe", 0) for r in recs) / n
        bank_safe = [r["init_bank_num_safe"] for r in recs if r.get("init_bank_num_safe") is not None]
        bank_modes = [r["init_bank_num_modes"] for r in recs if r.get("init_bank_num_modes") is not None]
        sel_modes = [r["init_selected_num_modes"] for r in recs if r.get("init_selected_num_modes") is not None]
        prefer = sum(1 for r in recs
                     if r.get("final_num_safe", 0) > 0 and r.get("selected_binary", 0) < 0.5)
        line = (f"  {arm:>28}: init_has_safe={init_safe:.2f}  avg_init_#safe={avg_safe:.1f}  "
                f"prefer_unsafe={prefer}/{n}")
        if bank_safe:
            line += f"  avg_bank_#safe={sum(bank_safe)/len(bank_safe):.1f}"
        if bank_modes:
            line += f"  avg_bank_modes={sum(bank_modes)/len(bank_modes):.1f}"
        if sel_modes:
            line += f"  avg_sel_modes={sum(sel_modes)/len(sel_modes):.1f}"
        print(line)

    print("=" * 78)
    print("FAILURE TAXONOMY")
    print(f"  {'scene':<32} {'arm':<28} taxonomy")
    tax = defaultdict(lambda: defaultdict(int))
    for arm in (base, div):
        for (a, rec, idx), d in sorted(results.items()):
            if a != arm or d.get("success"):
                continue
            # skip pure controls from taxonomy headline? include all fails
            tag = _taxonomy(d, diags.get((arm, rec, idx), []))
            tax[arm][tag] += 1
            print(f"  {rec} {idx:<4d}  {arm:<28} {tag}  "
                  f"goal={d.get('goal_reached')} offroad={d.get('offroad')} "
                  f"overlap={d.get('overlap')}")

    print("-" * 78)
    print(f"  {'taxonomy':<40} {base:>8} {div:>28}")
    all_tags = sorted(set(tax[base]) | set(tax[div]), key=lambda t: -(tax[base][t] + tax[div][t]))
    for t in all_tags:
        print(f"  {t:<40} {tax[base][t]:>8} {tax[div][t]:>28}")
    print("=" * 78)
    print("Interpretation:")
    print("  no_safe_in_bank drop under diverse  -> oversized bank recovered a safe mode (GO diversity)")
    print("  safe_but_no_goal persists           -> need stronger goal terms / different modes, not just coverage")
    print("  prefer_unsafe_over_safe             -> reward soft-safety issue (not fixed by diversity alone)")
    print("  sac_safe + overlap_only             -> denom is SMX-hit intersection, not full 36")


if __name__ == "__main__":
    main()
