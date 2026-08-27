#!/usr/bin/env python3
"""Does gradient clipping close the gap that the Spence weight opens?

Section 5 argues that Itakura-Saito and generalised KL buy their accuracy with
a gradient that explodes on one side, so they depend on clipping that a
two-sided weight makes unnecessary -- and then admits no arm was clipped, so
the argument is untested.  This runs the same grid with gradient_clip_val=1.0
and asks two questions:

  1. Does clipping HELP the one-sided arms more than it helps ours?
     (If it helps everyone equally, tail control buys nothing.)
  2. Does the RANKING change?
     (If Itakura-Saito still wins on value, our residual claim is gone.)
"""
import glob, json, math, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
NICE = {"quad": "Spence", "mse": "exp-MSE", "is": "Itakura-Saito",
        "gkl": "gen. KL"}


def load(tag):
    out = {}
    for f in glob.glob(f"{HERE}/results/{tag}_*_d*.json"):
        r = json.load(open(f))
        out[(r["loss"], r["dim"])] = {
            e["seed"]: e for e in r["seeds"]
            if e.get("frac_closed") is not None and math.isfinite(e["frac_closed"])}
    return out


U, K = load("grid"), load("clip1")
dims = sorted({d for _, d in U})
arms = [a for a in ("quad", "mse", "is", "gkl") if any((a, d) in K for d in dims)]
if not K:
    print("no clipped runs yet"); raise SystemExit

def paired(a, b, m):
    ks = sorted(set(a) & set(b))
    if len(ks) < 3: return None
    x = np.array([a[s][m] for s in ks], float); y = np.array([b[s][m] for s in ks], float)
    if m == "frac_closed": x, y = x*100, y*100
    d = x - y; _, p = stats.ttest_rel(x, y)
    return d.mean(), p, len(ks)

print("Q1  effect of clipping on each arm (clipped minus unclipped, same seeds)\n")
for m, lab, better in (("frac_closed", "headroom closed", "higher"),
                       ("v_rmse", "value RMSE", "lower")):
    print(f"  {lab} ({better} is better)")
    for a in arms:
        cells = []
        for d in dims:
            r = paired(K.get((a, d), {}), U.get((a, d), {}), m)
            cells.append(f"d{d}: {r[0]:+6.2f}{'*' if r[1]<0.05 else ' '}" if r else f"d{d}:   --  ")
        print(f"    {NICE[a]:14s} " + "  ".join(cells))
    print()

print("Q2  ranking WITH clipping: Spence minus each competitor\n")
for m, lab in (("v_rmse", "value RMSE (negative = Spence better)"),
               ("frac_closed", "headroom closed (positive = Spence better)")):
    print(f"  {lab}")
    for a in arms:
        if a == "quad": continue
        cells = []
        for d in dims:
            r = paired(K.get(("quad", d), {}), K.get((a, d), {}), m)
            cells.append(f"d{d}: {r[0]:+6.2f}{'*' if r[1]<0.05 else ' '}(n={r[2]})" if r else f"d{d}: --")
        print(f"    vs {NICE[a]:12s} " + "  ".join(cells))
    print()
