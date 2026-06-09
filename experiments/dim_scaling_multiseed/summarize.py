#!/usr/bin/env python3
"""Aggregate the multi-seed cells into dimension-scaling trends (mean ± SEM)."""

import glob
import json
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

ORDER = ["off_policy", "single_seed_mc", "single_seed_td_lambda", "ancestral_mc_td_lambda"]
COLORS = {"off_policy": "#2ca02c", "single_seed_mc": "#ff7f0e",
          "single_seed_td_lambda": "#1f77b4", "ancestral_mc_td_lambda": "#9467bd"}
SHORT = {"off_policy": "off-policy", "single_seed_mc": "ssmc",
         "single_seed_td_lambda": "ssmc-td(λ)", "ancestral_mc_td_lambda": "anc-mc-td(λ)"}


def main():
    cells = {}
    dims, methods = set(), set()
    for p in glob.glob(f"{RESULTS}/*_d*.json"):
        if os.path.basename(p) == "summary.json":
            continue
        d = json.load(open(p))
        if not d.get("seeds"):
            continue
        cells[(d["method"], d["dim"])] = d
        dims.add(d["dim"]); methods.add(d["method"])
    if not cells:
        print("no cells yet"); return
    dims = sorted(dims)
    methods = [m for m in ORDER if m in methods] + [m for m in sorted(methods) if m not in ORDER]

    def arr(m, dim, key):
        c = cells.get((m, dim))
        if not c:
            return np.array([])
        return np.array([r[key] for r in c["seeds"]
                         if key in r and isinstance(r[key], (int, float))
                         and math.isfinite(r[key])])

    def stat(m, dim, key):
        a = arr(m, dim, key)
        if len(a) == 0:
            return float("nan"), float("nan"), 0
        sem = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0
        return float(a.mean()), sem, len(a)

    hdr = f"{'dim':>5} | " + " | ".join(f"{SHORT[m]:>20}" for m in methods)
    print("\n" + "=" * len(hdr))
    print("MULTI-SEED REGRET vs DIMENSION  (plateau − optimal; mean ± SEM over seeds; "
          "0 = optimal)")
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for dim in dims:
        cellstr = []
        for m in methods:
            mu, sem, n = stat(m, dim, "regret")
            cellstr.append(f"{mu:>8.3f} ± {sem:.3f} (n{n})".rjust(20)
                           if n else f"{'—':>20}")
        print(f"{dim:>5} | " + " | ".join(cellstr))

    # ── plots ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Multi-seed performance by dimension @ BS=4 (quad, fixed hparams, "
                 "30 paired seeds)", fontsize=13, fontweight="bold")

    ax = axes[0]
    for m in methods:
        mus, sems = [], []
        for dim in dims:
            mu, sem, _ = stat(m, dim, "regret")
            mus.append(mu); sems.append(sem)
        mus, sems = np.array(mus), np.array(sems)
        ax.plot(dims, mus, "o-", color=COLORS.get(m), lw=2, ms=6, label=SHORT[m])
        ax.fill_between(dims, mus - sems, mus + sems, color=COLORS.get(m), alpha=0.2)
    ax.axhline(0, color="gray", ls=":", alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("symlog")
    ax.set_xlabel("dimension d"); ax.set_ylabel("regret = plateau − optimal (symlog)")
    ax.set_title("Regret vs dimension (mean ± SEM, 30 seeds)")
    ax.set_xticks(dims); ax.set_xticklabels(dims); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    for m in methods:
        mus, sems = [], []
        for dim in dims:
            mu, sem, _ = stat(m, dim, "plateau")
            mus.append(mu); sems.append(sem)
        mus, sems = np.array(mus), np.array(sems)
        ax.plot(dims, mus, "o-", color=COLORS.get(m), lw=2, ms=6, label=SHORT[m])
        ax.fill_between(dims, mus - sems, mus + sems, color=COLORS.get(m), alpha=0.2)
    optmu = [stat(methods[0], dim, "opt_reward")[0] for dim in dims]
    # optimal is method-independent; average across methods for robustness
    optmu = []
    for dim in dims:
        vals = [stat(m, dim, "opt_reward")[0] for m in methods
                if not math.isnan(stat(m, dim, "opt_reward")[0])]
        optmu.append(np.mean(vals) if vals else float("nan"))
    ax.plot(dims, optmu, "k--", lw=2, label="optimal $E_{p^*}[r]$")
    ax.set_xscale("log"); ax.set_xlabel("dimension d"); ax.set_ylabel("plateau reward")
    ax.set_title("Plateau reward vs dimension (mean ± SEM)")
    ax.set_xticks(dims); ax.set_xticklabels(dims); ax.grid(True, alpha=0.3); ax.legend()

    plt.tight_layout()
    plt.savefig(f"{RESULTS}/summary.png", dpi=140, bbox_inches="tight")
    print(f"\nSaved {RESULTS}/summary.png")

    out = {"dims": dims, "methods": methods,
           "regret": {m: {d: dict(zip(("mean", "sem", "n"), stat(m, d, "regret")))
                          for d in dims} for m in methods},
           "plateau": {m: {d: dict(zip(("mean", "sem", "n"), stat(m, d, "plateau")))
                           for d in dims} for m in methods},
           "optimal": {d: optmu[i] for i, d in enumerate(dims)}}
    json.dump(out, open(f"{RESULTS}/summary.json", "w"), indent=2)
    print(f"Saved {RESULTS}/summary.json")


if __name__ == "__main__":
    main()
