#!/usr/bin/env python3
"""Plot law-v2 grid vs off-policy and the previous studies' curves.

Controls read from ../dim_scaling_consth/results (off-policy + old law
configs) and ../dim_scaling_recipe/results (fixed recipe) -- identical
protocol, same nested instances/seeds.  Writes lawv2_vs_offpolicy.png and
summary.md.
"""

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CONSTH = os.path.join(os.path.dirname(HERE), "dim_scaling_consth", "results")
RECIPE = os.path.join(os.path.dirname(HERE), "dim_scaling_recipe", "results")
DIMS = [2, 8, 16, 32, 64, 128, 256, 512]

SERIES = [
    ("off-policy (control)", CONSTH, "grid_off_policy_d{d}.json",
     dict(color="k", ls="--", marker="s")),
    ("ssmc law-v1", CONSTH, "grid_single_seed_mc_d{d}.json",
     dict(color="tab:gray", ls=":", marker="o")),
    ("ssmc fixed recipe", RECIPE, "recipe_expand_ns60_sub_single_seed_mc_d{d}.json",
     dict(color="tab:cyan", ls=":", marker="o", alpha=0.7)),
    ("ssmc law-v2", RES, "gridv2_single_seed_mc_d{d}.json",
     dict(color="tab:blue", marker="o", lw=2)),
    ("ssmc-td law-v1", CONSTH, "grid_single_seed_td_lambda_d{d}.json",
     dict(color="tab:gray", ls=":", marker="^", alpha=0.6)),
    ("ssmc-td law-v2", RES, "gridv2_single_seed_td_lambda_d{d}.json",
     dict(color="tab:red", marker="^", lw=2)),
]


def cell(dirname, pattern, d):
    p = os.path.join(dirname, pattern.format(d=d))
    if not os.path.exists(p):
        return None
    fr = [e["frac_closed"] for e in json.load(open(p))["seeds"]
          if isinstance(e.get("frac_closed"), float)
          and math.isfinite(e["frac_closed"])]
    if not fr:
        return None
    fr = np.array(fr)
    se = fr.std(ddof=1) / math.sqrt(len(fr)) if len(fr) > 1 else 0.0
    return 100 * fr.mean(), 100 * se, len(fr)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.2))
    lines = ["# Law-v2 vs off-policy across dimension (frac_closed %)\n",
             "| series | " + " | ".join(f"d={d}" for d in DIMS) + " |",
             "|---|" + "---|" * len(DIMS)]
    for label, dirname, pattern, style in SERIES:
        xs, ys, es, cells = [], [], [], []
        for d in DIMS:
            c = cell(dirname, pattern, d)
            cells.append(c)
            if c:
                xs.append(d); ys.append(c[0]); es.append(c[1])
        if xs:
            ax.errorbar(xs, ys, yerr=es, label=label, capsize=3, **style)
        lines.append(f"| {label} | " + " | ".join(
            f"{c[0]:.1f}±{c[1]:.1f} (n={c[2]})" if c else "—"
            for c in cells) + " |")
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIMS); ax.set_xticklabels(DIMS)
    ax.set_xlabel("dimension d")
    ax.set_ylabel("headroom captured (%)")
    ax.set_title("Law-v2 (recipe knobs in the hyperparameter laws) vs off-policy\n"
                 "constant 6-nat headroom, 30 nested paired seeds, 15k steps")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "lawv2_vs_offpolicy.png")
    fig.savefig(out, dpi=150)
    md = "\n".join(lines) + "\n"
    open(os.path.join(HERE, "summary.md"), "w").write(md)
    print(md)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
