#!/usr/bin/env python3
"""Dimensional-scaling figure: on- vs off-policy across d = 2..512.

Reads figures/results.json (produced by collect_results.py from per-seed
records), so the figure and the numbers quoted in the text share one source.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))

STYLE = [
    ("off_policy",  "off-policy (control)", "#444444", "o", "--"),
    ("ssmc_lawv1",  "on-policy, laws w/o recipe knobs", "#7570b3", "s", ":"),
    ("ssmc_lawv2",  "on-policy, laws incl. expansion + steps", "#d95f02", "D", "-"),
]

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))

for key, lab, c, m, ls in STYLE:
    s = R["series"][key]
    ds = sorted((int(k) for k in s), key=int)
    mu = np.array([s[str(d)]["mean"] for d in ds])
    se = np.array([s[str(d)]["se"] for d in ds])
    ax[0].errorbar(ds, mu, yerr=se, color=c, marker=m, ls=ls, lw=1.7, ms=4.5,
                   capsize=2, label=lab)
ax[0].set_xscale("log", base=2)
ax[0].set_xlabel("dimension $d$"); ax[0].set_ylabel("headroom closed (\\%)")
ax[0].set_title("Absolute performance", fontsize=11)
ax[0].grid(alpha=0.25); ax[0].legend(fontsize=7.5, loc="upper right")

for key, lab, c, m, ls in STYLE[1:]:
    p = R["paired_vs_off_policy"][key]
    ds = sorted((int(k) for k in p), key=int)
    dl = np.array([p[str(d)]["delta"] for d in ds])
    se = np.array([p[str(d)]["se"] for d in ds])
    sig = np.array([p[str(d)]["p"] < 0.05 for d in ds])
    ax[1].errorbar(ds, dl, yerr=se, color=c, marker=m, ls=ls, lw=1.7, ms=4.5,
                   capsize=2, label=lab)
    ax[1].plot(np.array(ds)[sig], dl[sig], marker="*", ls="none", ms=11,
               color=c, markeredgecolor="k", markeredgewidth=0.4)
ax[1].axhline(0, color="k", lw=1.0)
ax[1].set_xscale("log", base=2)
ax[1].set_xlabel("dimension $d$")
ax[1].set_ylabel("paired $\\Delta$ vs off-policy (pts)")
ax[1].set_title("Paired difference (same instances; $\\star$: $p<0.05$)", fontsize=11)
ax[1].grid(alpha=0.25); ax[1].legend(fontsize=7.5, loc="lower left")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"dimscaling.{ext}"), dpi=150, bbox_inches="tight")
print("wrote dimscaling.pdf / .png  (n=%d seeds per point)"
      % R["series"]["off_policy"]["512"]["n"])
