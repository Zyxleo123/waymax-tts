"""Paired Stage-0 / Stage-1 analysis of the binary vs dense-oracle ES arms.

Reads the per-scene result JSONs and per-run ``diagnostics_*.json`` written by
``run_experiment.py`` and reports the measures the plan asks for:

  * final binary success rate per arm (failures + controls separately)
  * failure->success conversion on the 24 known failure scenes
  * scene-by-scene paired comparison (binary vs dense)
  * Stage-0 instrumentation summary (init has-safe rate, unique parents, ...)
  * rank correlation between the dense return and final binary success
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


def _arm_of(path: str) -> str | None:
    # Match longer arm names first so es_dense_diverse_* is not classified as dense.
    for arm in ("dense_diverse_safe_m4", "dense_diverse_safe_m8", "dense_diverse_m4",
                "dense_diverse_safe", "dense_diverse", "binary_diverse_safe",
                "dense", "binary"):
        token = f"{os.sep}es_{arm}"
        if token in path or f"/es_{arm}" in path:
            return arm
    return None


def _spearman(xs, ys) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # results: (arm, tfrecord, idx) -> result dict
    results = {}
    for f in glob.glob(os.path.join(args.out, "**", "*.scenario_*.json"), recursive=True):
        if os.path.basename(f) == "summary.json":
            continue
        arm = _arm_of(f)
        if arm is None:
            continue
        d = json.load(open(f))
        # basename is e.g. training_tfexample.tfrecord-00000-of-01000;
        # ORIG_FAIL/CONTROL keys are tfrecord-XXXXX-of-01000.
        rec_raw = os.path.basename(d.get("tfrecord", "")) or "?"
        m = re.search(r"(tfrecord-\d{5}-of-\d+)", rec_raw)
        rec = m.group(1) if m else rec_raw
        # keep only vanilla binary/dense in this Stage-0/1 report
        if arm not in ("binary", "dense"):
            continue
        results[(arm, rec, int(d["scenario_idx"]))] = d

    # diagnostics: (arm, tfrecord, idx) -> list of per-replan records
    diags = defaultdict(list)
    for f in glob.glob(os.path.join(args.out, "**", "diagnostics_*.json"), recursive=True):
        blob = json.load(open(f))
        a = blob["args"]
        # Stage-0/1 report only uses vanilla arms (init_select=none / missing).
        init = a.get("init_select", "none")
        if init not in (None, "none"):
            continue
        arm = a.get("arm") or a["selection"]
        if arm not in ("binary", "dense"):
            continue
        rec = os.path.basename(a["tfrecord_dir"].rstrip("/"))
        # map tfrecord_dir (es_baseline/repro/tf0000X/training) back to canonical name
        m = re.search(r"tf(\d{5})", a["tfrecord_dir"])
        canon = f"tfrecord-{m.group(1)}-of-01000" if m else rec
        idx_list = [int(x) for x in str(a["scenario_indices"]).split(",")] if a.get("scenario_indices") else []
        for r in blob["records"]:
            w = r["world_idx"]
            if w < len(idx_list):
                diags[(arm, canon, idx_list[w])].append(r)

    def is_fail(rec, idx):
        return idx in ORIG_FAIL.get(rec, set())
    def is_ctrl(rec, idx):
        return idx in ORIG_CONTROL.get(rec, set())

    print("=" * 78)
    print("STAGE 1 — final binary success by arm")
    for arm in ("binary", "dense"):
        f_succ = f_tot = c_succ = c_tot = 0
        for (a, rec, idx), d in results.items():
            if a != arm:
                continue
            ok = bool(d.get("success"))
            if is_fail(rec, idx):
                f_tot += 1; f_succ += int(ok)
            elif is_ctrl(rec, idx):
                c_tot += 1; c_succ += int(ok)
        print(f"  {arm:>6}:  failures solved {f_succ}/{f_tot}   controls kept {c_succ}/{c_tot}")

    print("=" * 78)
    print("PAIRED per failure scene   (B=binary success, D=dense success)")
    print(f"  {'tfrecord':<22} {'idx':>4}  {'B':>1} {'D':>1}   delta")
    conv = regress = 0
    for rec in sorted(ORIG_FAIL):
        for idx in sorted(ORIG_FAIL[rec]):
            b = results.get(("binary", rec, idx))
            d = results.get(("dense", rec, idx))
            if not b or not d:
                continue
            bs, ds = int(bool(b.get("success"))), int(bool(d.get("success")))
            tag = ""
            if ds and not bs:
                tag = "  <-- CONVERTED"; conv += 1
            elif bs and not ds:
                tag = "  <-- regressed"; regress += 1
            print(f"  {rec:<22} {idx:>4}  {bs:>1} {ds:>1}   {ds-bs:+d}{tag}")
    print(f"  failure->success conversions (dense over binary): {conv}   regressions: {regress}")

    print("=" * 78)
    print("STAGE 0 — instrumentation (binary arm, averaged over replan steps/scenes)")
    for arm in ("binary", "dense"):
        recs = [r for k, rs in diags.items() if k[0] == arm for r in rs]
        if not recs:
            continue
        n = len(recs)
        init_safe = sum(int(r["init_has_safe"]) for r in recs) / n
        avg_safe = sum(r["init_num_safe"] for r in recs) / n
        # unique parents across iterations (first iteration)
        up = []
        for r in recs:
            pit = r.get("per_iter") or []
            if pit:
                up.append(pit[0]["num_unique_parents"])
        avg_up = sum(up) / len(up) if up else float("nan")
        print(f"  {arm:>6}: init_has_safe={init_safe:.2f}  avg_init_#safe={avg_safe:.1f}  "
              f"avg_unique_parents(it0)={avg_up:.2f}  (n={n} replan-steps)")

    print("=" * 78)
    print("RANK CORRELATION — selected dense return vs final binary success (per scene)")
    for arm in ("binary", "dense"):
        xs, ys = [], []
        for (a, rec, idx), rs in diags.items():
            if a != arm:
                continue
            res = results.get((arm, rec, idx))
            if not res:
                continue
            xs.append(sum(r["selected_dense"] for r in rs) / len(rs))
            ys.append(int(bool(res.get("success"))))
        rho = _spearman(xs, ys)
        print(f"  {arm:>6}: spearman(mean selected_dense, success) = {rho:.3f}  (n={len(xs)})")

    print("=" * 78)
    print("Interpretation guide:")
    print("  conversions>0 & controls kept  -> dense reward resolution has oracle leverage (GO reward-model)")
    print("  init_has_safe high but failures persist in binary -> selection/scoring mismatch, not coverage")
    print("  dense improves & rho high -> dense return predicts success; value model justified")


if __name__ == "__main__":
    main()
