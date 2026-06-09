#!/usr/bin/env python3
"""Extra plots for the multi-seed run EXCLUDING the out-of-regime d=2 point.

Non-destructive: reads results/summary.json (produced by summarize.py) and
writes results/summary_no_d2.png — it does NOT touch summary.png/json.
"""

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
S = json.load(open(f"{RESULTS}/summary.json"))

COLORS = {"off_policy": "#2ca02c", "single_seed_mc": "#ff7f0e",
          "single_seed_td_lambda": "#1f77b4", "ancestral_mc_td_lambda": "#9467bd"}
SHORT = {"off_policy": "off-policy", "single_seed_mc": "ssmc",
         "single_seed_td_lambda": "ssmc-td(λ)", "ancestral_mc_td_lambda": "anc-mc-td(λ)"}

methods = S["methods"]
dims = [d for d in S["dims"] if d > 2]            # drop out-of-regime d=2


def get(block, m, d):
    return S[block][m][str(d)]


fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Multi-seed performance by dimension @ BS=4 (quad, fixed hparams, "
             "30 seeds) — d≥8 (d=2 excluded: out of selector regime)",
             fontsize=13, fontweight="bold")

# Panel 1: regret vs d, linear y
ax = axes[0]
for m in methods:
    mu = np.array([get("regret", m, d)["mean"] for d in dims])
    se = np.array([get("regret", m, d)["sem"] for d in dims])
    ax.plot(dims, mu, "o-", color=COLORS[m], lw=2, ms=6, label=SHORT[m])
    ax.fill_between(dims, mu - se, mu + se, color=COLORS[m], alpha=0.2)
ax.axhline(0, color="gray", ls=":", alpha=0.6)
ax.set_xscale("log"); ax.set_xticks(dims); ax.set_xticklabels(dims)
ax.set_xlabel("dimension d"); ax.set_ylabel("regret = plateau − optimal")
ax.set_title("Regret vs dimension (mean ± SEM)"); ax.grid(True, alpha=0.3); ax.legend()

# Panel 2: |regret| log-log with power-law slope
ax = axes[1]
logd = np.log(dims)
for m in methods:
    absr = np.array([abs(get("regret", m, d)["mean"]) for d in dims])
    ax.plot(dims, absr, "o-", color=COLORS[m], lw=2, ms=6, label=SHORT[m])
    sl, ic = np.polyfit(logd, np.log(absr), 1)
    ax.plot([], [], " ", label=f"   {SHORT[m]}: |regret|∝d^{sl:.2f}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(dims); ax.set_xticklabels(dims)
ax.set_xlabel("dimension d"); ax.set_ylabel("|regret| (log)")
ax.set_title("Regret decay (log-log; power-law exponent)")
ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8)

# Panel 3: plateau vs d with optimal
ax = axes[2]
for m in methods:
    mu = np.array([get("plateau", m, d)["mean"] for d in dims])
    se = np.array([get("plateau", m, d)["sem"] for d in dims])
    ax.plot(dims, mu, "o-", color=COLORS[m], lw=2, ms=6, label=SHORT[m])
    ax.fill_between(dims, mu - se, mu + se, color=COLORS[m], alpha=0.2)
ax.plot(dims, [S["optimal"][str(d)] for d in dims], "k--", lw=2,
        label="optimal $E_{p^*}[r]$")
ax.set_xscale("log"); ax.set_xticks(dims); ax.set_xticklabels(dims)
ax.set_xlabel("dimension d"); ax.set_ylabel("plateau reward")
ax.set_title("Plateau reward vs dimension"); ax.grid(True, alpha=0.3); ax.legend()

plt.tight_layout()
plt.savefig(f"{RESULTS}/summary_no_d2.png", dpi=140, bbox_inches="tight")
print(f"Saved {RESULTS}/summary_no_d2.png")

# also print the power-law exponents
print("\npower-law fit |regret| ∝ d^a  (d≥8):")
for m in methods:
    absr = np.array([abs(get("regret", m, d)["mean"]) for d in dims])
    sl, ic = np.polyfit(np.log(dims), np.log(absr), 1)
    print(f"  {SHORT[m]:>14}: a = {sl:+.3f}")
