#!/usr/bin/env python3
"""Shared loss-family figure: value and gradient of D_F(e^{v_true}, e^{v_pred})
as functions of v_pred, for a few v_true (reused across all three losses).

Spence and Itakura-Saito use the repo's shipped implementations, so the figure
cannot drift from the trained losses; squared error is inlined (it is symmetric,
so slot order is immaterial).  The prediction is the SECOND Bregman argument --
the slot whose population minimizer is the plain mean,
argmin_u E[D(T, u)] = E[T].  (With the slots swapped one gets a quasi-arithmetic
mean instead: the harmonic mean of T for Itakura-Saito.)

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
from diffusion_rl.losses.itakura_saito import (  # noqa: E402
    itakura_saito, itakura_saito_grad)

V_TRUE = [-2.0, 0.0, 2.0]
COLORS = ["#1b9e77", "#7570b3", "#d95f02"]
VP = np.linspace(-6.0, 8.0, 1400)


def mse_value(p, t):      # (e^t - e^p)^2   (symmetric; slot order immaterial)
    return (np.exp(t) - np.exp(p)) ** 2

def mse_grad(p, t):       # 2 e^p (e^p - e^t)
    return 2 * np.exp(p) * (np.exp(p) - np.exp(t))

def is_value(p, t):       # repo implementation: D_IS(e^t, e^p)
    P = torch.tensor(p, dtype=torch.float64)
    T = torch.full_like(P, t)
    return itakura_saito(P, T).numpy()

def is_grad(p, t):
    P = torch.tensor(p, dtype=torch.float64)
    T = torch.full_like(P, t)
    return itakura_saito_grad(P, T).numpy()

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
     "$w(y)=2y$\n" + r"$\partial_p L = 2e^{p}(e^{p}-e^{q})$"),
    ("Itakura--Saito", is_value, is_grad,
     "$w(y)=1/y$\n" + r"$\partial_p L = 1-e^{q-p}$"),
    ("Spence (ours)", spence_value, spence_grad,
     r"$w(y)=\ln y/(y-1)$" + "\n" + r"$\partial_p L = p\,(e^{p}-e^{q})/(e^{p}-1)$"),
]


def main():
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True)
    for j, (name, vfn, gfn, sub) in enumerate(LOSSES):
        for t, c in zip(V_TRUE, COLORS):
            ax[0, j].plot(VP, vfn(VP, t), color=c, lw=1.8,
                          label=fr"$q={t:+.0f}$")
            ax[1, j].plot(VP, gfn(VP, t), color=c, lw=1.8)
            ax[0, j].plot([t], [0], "o", color=c, ms=4)
            ax[1, j].plot([t], [0], "o", color=c, ms=4)
        ax[0, j].set_title(name, fontsize=12)
        ax[0, j].set_yscale("symlog", linthresh=1.0)
        ax[1, j].set_yscale("symlog", linthresh=1.0)
        ax[1, j].axhline(0, color="k", lw=0.5, alpha=0.4)
        ax[1, j].set_xlabel(r"prediction $p$")
        ax[0, j].text(0.03, 0.95, sub, transform=ax[0, j].transAxes,
                      va="top", ha="left", fontsize=8.5)
        for a in (ax[0, j], ax[1, j]):
            a.grid(alpha=0.25); a.set_xlim(VP[0], VP[-1])
    ax[0, 0].set_ylabel("loss value")
    ax[1, 0].set_ylabel(r"gradient $\partial L/\partial p$")
    ax[0, 0].legend(fontsize=8, loc="upper center")
    # annotate the pathologies
    ax[1, 0].annotate("vanishes", xy=(-5, -1e-1), xytext=(-5.5, -30),
                      fontsize=8, color="0.3",
                      arrowprops=dict(arrowstyle="->", color="0.5"))
    ax[1, 0].annotate("explodes", xy=(6.5, 3e5), xytext=(1.5, 3e5),
                      fontsize=8, color="0.3",
                      arrowprops=dict(arrowstyle="->", color="0.5"))
    ax[1, 1].annotate("explodes", xy=(-5, -2e2), xytext=(-3.5, -3e4),
                      fontsize=8, color="0.3",
                      arrowprops=dict(arrowstyle="->", color="0.5"))
    ax[1, 1].annotate("bounded", xy=(6.5, 1.0), xytext=(2.0, 30),
                      fontsize=8, color="0.3",
                      arrowprops=dict(arrowstyle="->", color="0.5"))
    ax[1, 2].annotate("linear both sides", xy=(-5.0, -30), xytext=(-5.5, -3e3),
                      fontsize=8, color="0.3",
                      arrowprops=dict(arrowstyle="->", color="0.5"))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(os.path.dirname(__file__), f"loss_curves.{ext}"),
                    dpi=150, bbox_inches="tight")
    print("wrote loss_curves.pdf / .png")


if __name__ == "__main__":
    main()
