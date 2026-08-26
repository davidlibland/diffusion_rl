#!/usr/bin/env python3
"""Loss-ablation figure: the four losses across dimension, off-policy only.

Reads experiments/loss_ablation/ablation.json (written by analyse_ablation.py).
Left  : control quality  (fraction of the 6-nat headroom closed)
Right : value quality    (RMSE of V_theta against the analytic value)
Inset : signed value bias, where the log-space arm shows the Jensen deficit.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                   "diffusion_rl", "experiments", "loss_ablation", "ablation.json")
if not os.path.exists(SRC):
    SRC = os.path.join(os.path.dirname(HERE), "..", "experiments",
                       "loss_ablation", "ablation.json")
R = json.load(open(os.path.abspath(SRC)))

STYLE = [("quad",   "Spence (ours)",     "#d95f02", "D", "-"),
         ("mse",    "exp-space MSE",     "#444444", "o", "--"),
         ("is",     "Itakura--Saito",    "#1b9e77", "s", "-."),
         ("logmse", "log-space MSE",     "#7570b3", "^", ":")]
dims = R["dims"]

fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
for key, lab, c, m, ls in STYLE:
    mu, se, ds = [], [], []
    for d in dims:
        cell = R["cells"].get(f"{key}_d{d}")
        if cell and np.isfinite(cell["frac_closed"]["mean"]):
            ds.append(d); mu.append(cell["frac_closed"]["mean"]); se.append(cell["frac_closed"]["se"])
    if ds:
        ax[0].errorbar(ds, mu, yerr=se, color=c, marker=m, ls=ls, lw=1.7, ms=4.5,
                       capsize=2, label=lab)
    mu, se, ds = [], [], []
    for d in dims:
        cell = R["cells"].get(f"{key}_d{d}")
        if cell and np.isfinite(cell["v_rmse"]["mean"]):
            ds.append(d); mu.append(cell["v_rmse"]["mean"]); se.append(cell["v_rmse"]["se"])
    if ds:
        ax[1].errorbar(ds, mu, yerr=se, color=c, marker=m, ls=ls, lw=1.7, ms=4.5,
                       capsize=2, label=lab)
    mu, ds = [], []
    for d in dims:
        cell = R["cells"].get(f"{key}_d{d}")
        if cell and np.isfinite(cell["v_bias"]["mean"]):
            ds.append(d); mu.append(cell["v_bias"]["mean"])
    if ds:
        ax[2].plot(ds, mu, color=c, marker=m, ls=ls, lw=1.7, ms=4.5, label=lab)

ax[0].set_ylabel("headroom closed (\\%)"); ax[0].set_title("Control quality", fontsize=11)
ax[1].set_ylabel("RMSE$(V_\\theta, V)$ (nats)"); ax[1].set_title("Value quality vs analytic $V$", fontsize=11)
ax[2].set_ylabel("signed bias (nats)"); ax[2].set_title("Value bias", fontsize=11)
ax[2].axhline(0, color="k", lw=0.8)
for a in ax:
    a.set_xscale("log", base=2); a.set_xlabel("dimension $d$"); a.grid(alpha=0.25)
ax[0].legend(fontsize=7.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"ablation.{ext}"), dpi=150, bbox_inches="tight")
print("wrote ablation.pdf / .png")
