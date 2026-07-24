#!/usr/bin/env python3
"""Shared loss-family figure: value and gradient of D_F(e^{v_pred}, e^{v_true})
as functions of v_pred, for a few v_true (reused across all three losses).

Spence uses the repo's actual value/gradient (losses/log_quadratic_bregman);
MSE and Itakura-Saito are the standard Bregman divergences with prediction as
the first argument (so E[D(pred, Target)] is minimized at pred = E[Target]).

Output: paper/figures/loss_curves.{pdf,png}
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from diffusion_rl.losses.log_quadratic_bregman import (  # noqa: E402
    _log_quadratic_bregman_value, log_quadratic_bregman_grad)

V_TRUE = [-2.0, 0.0, 2.0]
COLORS = ["#1b9e77", "#7570b3", "#d95f02"]
VP = np.linspace(-6.0, 8.0, 1400)


def mse_value(p, t):      # (e^p - e^t)^2
    return (np.exp(p) - np.exp(t)) ** 2

def mse_grad(p, t):       # 2 e^p (e^p - e^t)
    return 2 * np.exp(p) * (np.exp(p) - np.exp(t))

def is_value(p, t):       # P/T - log(P/T) - 1,  P=e^p, T=e^t
    lr = p - t
    return np.exp(lr) - lr - 1.0

def is_grad(p, t):        # e^{p-t} - 1
    return np.exp(p - t) - 1.0

def spence_value(p, t):
    P = torch.tensor(p, dtype=torch.float64)
    T = torch.full_like(P, t)
    return _log_quadratic_bregman_value(P, T).numpy()

def spence_grad(p, t):
    P = torch.tensor(p, dtype=torch.float64)
    T = torch.full_like(P, t)
    return log_quadratic_bregman_grad(P, T).numpy()


LOSSES = [
    ("Squared error", mse_value, mse_grad,
     r"$D(e^{p},e^{t})=(e^{p}-e^{t})^2$"),
    ("Itakura--Saito", is_value, is_grad,
     r"$D_{\mathrm{IS}}=e^{p-t}-(p-t)-1$"),
    ("Spence (ours)", spence_value, spence_grad,
     r"$\partial_p L = p\,(e^{p}-e^{t})/(e^{p}-1)$"),
]


def main():
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True)
    for j, (name, vfn, gfn, sub) in enumerate(LOSSES):
        for t, c in zip(V_TRUE, COLORS):
            ax[0, j].plot(VP, vfn(VP, t), color=c, lw=1.8,
                          label=fr"$v_{{\rm true}}={t:+.0f}$")
            ax[1, j].plot(VP, gfn(VP, t), color=c, lw=1.8)
            ax[0, j].plot([t], [0], "o", color=c, ms=4)
            ax[1, j].plot([t], [0], "o", color=c, ms=4)
        ax[0, j].set_title(name, fontsize=12)
        ax[0, j].set_yscale("symlog", linthresh=1.0)
        ax[1, j].set_yscale("symlog", linthresh=1.0)
        ax[1, j].axhline(0, color="k", lw=0.5, alpha=0.4)
        ax[1, j].set_xlabel(r"prediction $v_{\rm pred}$")
        ax[0, j].text(0.03, 0.95, sub, transform=ax[0, j].transAxes,
                      va="top", ha="left", fontsize=8.5)
        for a in (ax[0, j], ax[1, j]):
            a.grid(alpha=0.25); a.set_xlim(VP[0], VP[-1])
    ax[0, 0].set_ylabel("loss value")
    ax[1, 0].set_ylabel(r"gradient $\partial L/\partial v_{\rm pred}$")
    ax[0, 0].legend(fontsize=8, loc="upper center")
    # annotate the pathologies
    ax[1, 0].annotate("vanishes", xy=(-5, -1e-1), xytext=(-5.5, -30),
                      fontsize=8, color="0.3",
                      arrowprops=dict(arrowstyle="->", color="0.5"))
    ax[1, 0].annotate("explodes", xy=(6.5, 3e5), xytext=(1.5, 3e5),
                      fontsize=8, color="0.3",
                      arrowprops=dict(arrowstyle="->", color="0.5"))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(os.path.dirname(__file__), f"loss_curves.{ext}"),
                    dpi=150, bbox_inches="tight")
    print("wrote loss_curves.pdf / .png")


if __name__ == "__main__":
    main()
