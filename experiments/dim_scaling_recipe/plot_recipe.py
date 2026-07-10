#!/usr/bin/env python3
"""Plot the recipe-vs-off-policy dimension-scaling comparison.

Recipe cells from results/; law controls (off_policy, ssmc, ssmc-td) read
straight from ../dim_scaling_consth/results (identical protocol, same nested
instances and seeds).  Writes recipe_vs_offpolicy.png and prints the summary
table (also saved as summary.md).
"""

import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CTRL = os.path.join(os.path.dirname(HERE), "dim_scaling_consth", "results")
DIMS = [2, 8, 16, 32, 64, 128, 256, 512]

SERIES = [
    # (label, results dir, file pattern, plot style)
    ("off-policy (control)", CTRL, "grid_off_policy_d{d}.json",
     dict(color="k", ls="--", marker="s")),
    ("ssmc law (control)", CTRL, "grid_single_seed_mc_d{d}.json",
     dict(color="tab:gray", ls=":", marker="o")),
    ("ssmc-td law (control)", CTRL, "grid_single_seed_td_lambda_d{d}.json",
     dict(color="tab:gray", ls=":", marker="^", alpha=0.6)),
    ("ssmc + expand_ns60", RES, "recipe_expand_ns60_single_seed_mc_d{d}.json",
     dict(color="tab:blue", marker="o")),
    ("ssmc + expand_ns60_sub", RES,
     "recipe_expand_ns60_sub_single_seed_mc_d{d}.json",
     dict(color="tab:cyan", marker="o")),
    ("ssmc-td + expand_ns60", RES,
     "recipe_expand_ns60_single_seed_td_lambda_d{d}.json",
     dict(color="tab:red", marker="^")),
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
    lines = ["# Recipe vs off-policy across dimension (frac_closed %)\n",
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
    ax.set_title("On-policy recipe (expansion + n_steps=60) vs off-policy\n"
                 "constant 6-nat headroom, 30 nested paired seeds, 15k steps")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "recipe_vs_offpolicy.png")
    fig.savefig(out, dpi=150)
    md = "\n".join(lines) + "\n"
    open(os.path.join(HERE, "summary.md"), "w").write(md)
    print(md)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
