#!/usr/bin/env python3
"""Turn the loss-ablation seed grid into the numbers Section 5 quotes.

Writes ablation.json (consumed by the paper figure/table) and prints a summary.
Comparisons are PAIRED: seed s is the same problem instance for every arm.
"""
import glob, json, math, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["quad", "mse", "is", "gkl", "logmse"]
NICE = {"quad": "Spence", "mse": "exp-MSE", "is": "Itakura-Saito",
        "gkl": "gen. KL (beta=1)", "logmse": "log-MSE (biased)"}
METRICS = ["frac_closed", "v_rmse", "v_bias", "skips"]


def load():
    d = {}
    for f in glob.glob(f"{HERE}/results/grid_*_d*.json"):
        r = json.load(open(f))
        per = {}
        for e in r["seeds"]:
            fc = e.get("frac_closed")
            if fc is None or not math.isfinite(fc):
                continue
            per[e["seed"]] = {"frac_closed": 100*fc, "v_rmse": e.get("v_rmse"),
                              "v_bias": e.get("v_bias"), "skips": e.get("skips", 0)}
        if per:
            d[(r["loss"], r["dim"])] = {"lr": r["lr"], "seeds": per}
    return d


def agg(per, m):
    v = np.array([s[m] for s in per.values() if s[m] is not None
                  and math.isfinite(s[m])], dtype=float)
    if not len(v):
        return {"n": 0, "mean": float("nan"), "se": float("nan")}
    return {"n": len(v), "mean": float(v.mean()),
            "se": float(v.std(ddof=1)/math.sqrt(len(v))) if len(v) > 1 else float("nan")}


def paired(a, b, m):
    ks = sorted(set(a) & set(b))
    xs = [(a[k][m], b[k][m]) for k in ks
          if a[k][m] is not None and b[k][m] is not None
          and math.isfinite(a[k][m]) and math.isfinite(b[k][m])]
    if len(xs) < 3:
        return None
    x = np.array([p[0] for p in xs]); y = np.array([p[1] for p in xs])
    t, p = stats.ttest_rel(x, y)
    dd = x - y
    return {"n": len(xs), "delta": float(dd.mean()),
            "se": float(dd.std(ddof=1)/math.sqrt(len(dd))), "p": float(p),
            "wins": int((dd > 0).sum())}


def main():
    data = load()
    if not data:
        print("no grid results yet"); return
    dims = sorted({d for _, d in data})
    out = {"dims": dims, "arms": ARMS, "cells": {}, "paired_vs_mse": {}}
    for (loss, dim), rec in data.items():
        out["cells"][f"{loss}_d{dim}"] = {
            "lr": rec["lr"], **{m: agg(rec["seeds"], m) for m in METRICS}}
    for loss in ARMS:
        if loss == "mse":
            continue
        row = {}
        for dim in dims:
            a = data.get((loss, dim)); b = data.get(("mse", dim))
            if not a or not b:
                continue
            r = {m: paired(a["seeds"], b["seeds"], m) for m in
                 ("frac_closed", "v_rmse")}
            if r["frac_closed"]:
                row[str(dim)] = r
        out["paired_vs_mse"][loss] = row
    json.dump(out, open(f"{HERE}/ablation.json", "w"), indent=1)

    print("\n=== headroom closed (%), mean +- s.e. ===")
    hdr = "arm            " + "".join(f"{'d='+str(d):>16s}" for d in dims)
    print(hdr)
    for loss in ARMS:
        cells = []
        for d in dims:
            c = out["cells"].get(f"{loss}_d{d}")
            cells.append(f"{c['frac_closed']['mean']:7.1f}+-{c['frac_closed']['se']:<5.1f}"
                         if c else " " * 14)
        print(f"{NICE[loss]:15s}" + "".join(f"{c:>16s}" for c in cells))

    print("\n=== value RMSE vs analytic oracle (nats) ===")
    print(hdr)
    for loss in ARMS:
        cells = []
        for d in dims:
            c = out["cells"].get(f"{loss}_d{d}")
            cells.append(f"{c['v_rmse']['mean']:7.2f}+-{c['v_rmse']['se']:<5.2f}"
                         if c else " " * 14)
        print(f"{NICE[loss]:15s}" + "".join(f"{c:>16s}" for c in cells))

    print("\n=== signed value bias (nats) -- log-MSE should show the Jensen deficit ===")
    print(hdr)
    for loss in ARMS:
        cells = []
        for d in dims:
            c = out["cells"].get(f"{loss}_d{d}")
            cells.append(f"{c['v_bias']['mean']:+7.2f}" if c else " " * 8)
        print(f"{NICE[loss]:15s}" + "".join(f"{c:>16s}" for c in cells))

    print("\n=== paired delta vs exp-MSE, headroom closed (same instances) ===")
    for loss, row in out["paired_vs_mse"].items():
        s = " ".join(
            f"d{d}:{row[d]['frac_closed']['delta']:+6.1f}"
            f"{'*' if row[d]['frac_closed']['p'] < 0.05 else ' '}"
            f"(n={row[d]['frac_closed']['n']})" for d in sorted(row, key=int))
        print(f"  {NICE[loss]:16s} {s}")
    print(f"\nwrote {HERE}/ablation.json")


if __name__ == "__main__":
    main()
