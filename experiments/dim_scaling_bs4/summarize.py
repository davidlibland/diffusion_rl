#!/usr/bin/env python3
"""Aggregate the per-cell results into a dimension-scaling table + plot.

The per-dimension baseline is the analytical optimal expected reward
E_opt_reward = E_{p*}[r] (from optimal_baseline.py / results/optimal_baseline.json),
NOT V(0,0): V(0,0) is a log-partition value that the achievable reward can
exceed, whereas E_opt_reward is the best reward any policy targeting the
exp(r)-tilt can attain.  Regret = plateau - E_opt_reward (<= 0; 0 = optimal).
"""

import glob
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

METHOD_ORDER = ["off_policy", "single_seed_mc", "single_seed_td_lambda",
                "ancestral_mc_td_lambda"]
COLORS = {"off_policy": "#2ca02c", "single_seed_mc": "#ff7f0e",
          "single_seed_td_lambda": "#1f77b4", "ancestral_mc_td_lambda": "#9467bd"}
SHORT = {"off_policy": "off-policy", "single_seed_mc": "ssmc",
         "single_seed_td_lambda": "ssmc-td(λ)", "ancestral_mc_td_lambda": "anc-mc-td(λ)"}


def main():
    cells = {}
    dims, methods = set(), set()
    for path in glob.glob(f"{RESULTS}/*_d*.json"):
        if os.path.basename(path) in ("summary.json", "optimal_baseline.json"):
            continue
        d = json.load(open(path))
        if "convergence" not in d:
            continue
        cells[(d["method"], d["dim"])] = d
        dims.add(d["dim"]); methods.add(d["method"])
    dims = sorted(dims)
    methods = [m for m in METHOD_ORDER if m in methods] + \
              [m for m in sorted(methods) if m not in METHOD_ORDER]
    if not cells:
        print("No result cells found yet.")
        return

    # optimal-reward baseline per dimension (fallback to V00 if not computed yet)
    opt = {}
    bpath = f"{RESULTS}/optimal_baseline.json"
    if os.path.exists(bpath):
        ob = json.load(open(bpath))
        opt = {int(k): v["E_opt_reward"] for k, v in ob.items()}

    def baseline(dim):
        if dim in opt:
            return opt[dim]
        c = next((cells[(m, dim)] for m in methods if (m, dim) in cells), None)
        return c["problem"]["V00_analytical"] if c else float("nan")

    def plateau(m, dim):
        c = cells.get((m, dim))
        return c["convergence"]["plateau_reward"] if c else float("nan")

    def final_lcb(m, dim):
        c = cells.get((m, dim))
        return c["convergence"]["final_lcb"] if c else float("nan")

    hdr = f"{'dim':>5} | " + " | ".join(f"{SHORT[m]:>13}" for m in methods) + " | optimal"
    print("\n" + "=" * len(hdr))
    print("DIMENSION SCALING @ BS=4 (quad loss) — converged plateau reward")
    print("=" * len(hdr))
    print(hdr); print("-" * len(hdr))
    for dim in dims:
        row = " | ".join(f"{plateau(m, dim):>13.3f}" for m in methods)
        print(f"{dim:>5} | {row} | {baseline(dim):>7.3f}")

    print("\nregret = plateau - optimal  (0 = optimal; closer to 0 is better):")
    print(hdr); print("-" * len(hdr))
    for dim in dims:
        b = baseline(dim)
        row = " | ".join(f"{plateau(m, dim) - b:>13.3f}" for m in methods)
        print(f"{dim:>5} | {row} | {0.0:>7.3f}")

    print("\nfinal-LCB (lower confidence bound on tail reward):")
    print(hdr); print("-" * len(hdr))
    for dim in dims:
        row = " | ".join(f"{final_lcb(m, dim):>13.3f}" for m in methods)
        print(f"{dim:>5} | {row} | {baseline(dim):>7.3f}")

    # ── plots ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Dimension scaling @ BS=4 (quad loss) — bs4-style Optuna per cell",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    for m in methods:
        ys = [plateau(m, d) for d in dims]
        ax.plot(dims, ys, "o-", color=COLORS.get(m), lw=2, ms=7, label=SHORT[m])
    ax.plot(dims, [baseline(d) for d in dims], "k--", lw=2, label="optimal $E_{p^*}[r]$")
    ax.set_xscale("log"); ax.set_xlabel("dimension d"); ax.set_ylabel("plateau reward")
    ax.set_title("Converged reward vs dimension"); ax.grid(True, alpha=0.3)
    ax.set_xticks(dims); ax.set_xticklabels(dims); ax.legend(fontsize=9)

    ax = axes[1]
    for m in methods:
        reg = [plateau(m, d) - baseline(d) for d in dims]
        ax.plot(dims, reg, "o-", color=COLORS.get(m), lw=2, ms=7, label=SHORT[m])
    ax.axhline(0, color="gray", ls=":", alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("symlog")
    ax.set_xlabel("dimension d"); ax.set_ylabel("regret = plateau - optimal (symlog)")
    ax.set_title("Sub-optimality vs dimension"); ax.grid(True, alpha=0.3)
    ax.set_xticks(dims); ax.set_xticklabels(dims); ax.legend(fontsize=9)

    ax = axes[2]
    for m in methods:
        for d in dims:
            c = cells.get((m, d))
            if not c:
                continue
            cc = c["convergence_curve"]
            st, cv = np.array(cc["steps"]), np.array(cc["val_reward"])
            if len(cv):
                sm = pd.Series(cv).rolling(8, min_periods=1).mean()
                ax.plot(st, sm, color=COLORS.get(m),
                        alpha=0.30 + 0.45 * (d == dims[-1]), lw=1.5)
    for m in methods:
        ax.plot([], [], color=COLORS.get(m), lw=2, label=SHORT[m])
    ax.set_xlabel("training step"); ax.set_ylabel("val reward (8-step mean)")
    ax.set_title("Convergence curves (all dims; bold = largest d)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    plt.tight_layout()
    fig_path = f"{RESULTS}/summary.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"\nSaved {fig_path}")

    summary = {
        "dims": dims, "methods": methods,
        "optimal_reward": {d: baseline(d) for d in dims},
        "plateau": {m: {d: plateau(m, d) for d in dims} for m in methods},
        "regret": {m: {d: plateau(m, d) - baseline(d) for d in dims} for m in methods},
        "final_lcb": {m: {d: final_lcb(m, d) for d in dims} for m in methods},
        "winner_params": {f"{m}_d{d}": cells[(m, d)]["winner"]["params"]
                          for m in methods for d in dims if (m, d) in cells},
    }
    json.dump(summary, open(f"{RESULTS}/summary.json", "w"), indent=2)
    print(f"Saved {RESULTS}/summary.json")


if __name__ == "__main__":
    main()
