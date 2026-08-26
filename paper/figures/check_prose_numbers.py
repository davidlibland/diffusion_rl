#!/usr/bin/env python3
"""Verify the numbers Section 5's prose quotes against the ablation data.

A reviewer caught S5 asserting a value-RMSE advantage of '0.37 to 0.75 nats'
when the paired data gave 0.36 to 0.61 -- stale text that survived the table's
regeneration.  The table is script-emitted; the prose is not.  This script
closes that gap by recomputing every paired comparison S5 quotes and printing
it next to the source, so drift is caught mechanically rather than by a reader.

Run after any change to the ablation data or to Section 5.
"""
import glob, json, math, os, re, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "experiments", "loss_ablation", "results")
SEC = os.path.join(os.path.dirname(HERE), "sections", "05_offpolicy_exp.tex")

C = {}
for f in glob.glob(f"{RES}/grid_*.json"):
    r = json.load(open(f))
    C[(r["loss"], r["dim"])] = {
        e["seed"]: e for e in r["seeds"]
        if e.get("frac_closed") is not None and math.isfinite(e["frac_closed"])}

DIMS = sorted({d for _, d in C})


def paired(a_, b_, d, m):
    a, b = C.get((a_, d), {}), C.get((b_, d), {})
    ks = sorted(set(a) & set(b))
    if len(ks) < 3:
        return None
    x = np.array([a[s][m] for s in ks], float)
    y = np.array([b[s][m] for s in ks], float)
    if m == "frac_closed":
        x, y = x * 100, y * 100
    dd = x - y
    _, p = stats.ttest_rel(x, y)
    return dd.mean(), p, len(ks)


COMPARISONS = [
    ("Spence vs exp-MSE",   "quad", "mse",    "v_rmse"),
    ("Spence vs exp-MSE",   "quad", "mse",    "frac_closed"),
    ("Spence vs gen-KL",    "quad", "gkl",    "v_rmse"),
    ("Spence vs gen-KL",    "quad", "gkl",    "frac_closed"),
    ("Spence vs Itakura",   "quad", "is",     "v_rmse"),
    ("Spence vs Itakura",   "quad", "is",     "frac_closed"),
    ("log-MSE vs Spence",   "logmse", "quad", "frac_closed"),
    ("log-MSE vs Spence",   "logmse", "quad", "v_rmse"),
]

print("Recomputed paired comparisons (the ground truth for Section 5's prose)\n")
for label, a, b, m in COMPARISONS:
    cells = []
    for d in DIMS:
        r = paired(a, b, d, m)
        cells.append(f"d{d}: {r[0]:+6.2f}{'*' if r[1] < 0.05 else ' '}(n={r[2]})"
                     if r else f"d{d}: --")
    print(f"  {label:20s} {m:12s} " + "  ".join(cells))

print("\nNumbers appearing in Section 5's prose:")
txt = open(SEC).read()
txt = re.sub(r"%.*", "", txt)
nums = sorted({m.group(0) for m in re.finditer(r"[+-]?\d+\.\d+", txt)},
              key=lambda s: abs(float(s)))
print("  " + "  ".join(nums))
print("\nCheck each against the table above by hand; any prose number that is not"
      "\na recomputed value or a table entry is drift.")
