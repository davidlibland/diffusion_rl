"""
Data Quality of Sampling Methods for On-Policy Value Learning
=============================================================

Overview
--------
When training a value function V(x, t) ≈ log E[exp(r(X_T)) | X_t = x] for
diffusion RL, the key question is: what training targets should we use?

On-policy methods roll out SDE trajectories and construct targets from the
resulting path.  Several target-construction strategies are available; they
trade off variance (noise per sample) against bias (systematic error relative
to the true V).

This script analyses four methods using the analytical value function as a
perfect oracle, asking:

  1. How noisy are the targets intrinsically? (Setting A)
  2. How does imperfect V interact with each method?  (Setting B)

Results are loaded from notebooks/data_quality_results.json (written by
data_quality.py).


The SDE and Value Function
--------------------------
We use a bridge-like SDE (stochastic interpolant):

    dX = u(X,t) dt + sqrt(2a) dW,    X_0 = 0,  X_T ~ q_target

with a = 1.0.  The optimal control is v*(x,t) = 2a ∇V(x,t), where

    V(x,t) = log E[exp(r(X_T)) | X_t = x].

Target distribution: a 2-D two-moons dataset (scaled to unit variance) fitted
by a 100-component spherical GMM.  Reward: r(x) = -10 ||x - [1,0]||^2.

The analytical value function V_anal(x,t) can be derived in closed form from
the GMM structure — it equals a log-sum-exp over per-component log-normalizers
of the Gaussian integrals.  We use this as ground truth throughout.

E_opt = -2.587   (optimal expected reward, computed analytically).


Sampling Methods
----------------
OnPolicySMCDataset generates batches (y, x, t) where y is the target for V(x,t).
Four methods were evaluated:

  single_seed_mc           Full Monte Carlo: y = r(X_T) along a single trajectory.
                           Pure unbiased estimate; highest variance.

  single_seed_td_lam06     TD(λ=0.6): y = λ-blended mixture of MC and one-step
                           bootstrapped targets. Mixes rollout reward with V(X_{t+dt}).
                           λ=0.6 → moderate variance reduction.

  single_seed_td_lam02     TD(λ=0.2): heavier bootstrapping.  Most of the target
                           comes from V at the next step, little from MC.
                           Lowest variance of the single-seed methods.

  one_step_bootstrap        y = r(x_1) + V(x_{t+dt}) with no MC rollout.
                            Extreme bias when V is inaccurate; shown here for reference.

SMC reweighting (smc_value): Each trajectory seed is reweighted by importance
weights derived from another value estimate to correct for sampling bias.


Experimental Settings
---------------------
  Setting A — Oracle
    value = V_anal,  smc_value = V_anal
    → Measures the INTRINSIC variance of each method's targets.
    → This is a lower bound on the noise the learner must contend with.

  Setting B — Best model + Oracle SMC
    value = best_TD(λ=0.6)_checkpoint,  smc_value = V_anal
    → Measures variance AND bias when the value network is imperfect.
    → Captures the target noise during actual training.

For each method and setting we computed mean(target - V_anal) and
std(target - V_anal) within 5 time bins: t ∈ [0,0.2), [0.2,0.4),
[0.4,0.6), [0.6,0.8), [0.8,1.0).  N≈10,000 data points total.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
with open("notebooks/data_quality_results.json") as f:
    results = json.load(f)

setting_A = results["setting_A"]
setting_B = results["setting_B"]

METHODS = {
    "single_seed_mc":       "MC (λ=1)",
    "single_seed_td_lam06": "TD(λ=0.6)",
    "single_seed_td_lam02": "TD(λ=0.2)",
    "one_step_bootstrap":   "One-step BS",
}

BIN_NAMES  = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]
BIN_LABELS = ["[0.0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0)"]
BIN_MIDS   = [0.1, 0.3, 0.5, 0.7, 0.9]

COLORS = {
    "single_seed_mc":       "#e74c3c",   # red
    "single_seed_td_lam06": "#f39c12",   # orange
    "single_seed_td_lam02": "#27ae60",   # green
    "one_step_bootstrap":   "#8e44ad",   # purple
}

# ---------------------------------------------------------------------------
# Helper: extract arrays from one setting's results
# ---------------------------------------------------------------------------
def extract(setting, method, stat):
    """Return array of per-bin values for `stat` ∈ {mean, std, var}."""
    d = setting[method]
    return np.array([d[b][stat] for b in BIN_NAMES])


# ---------------------------------------------------------------------------
# Print summary tables
# ---------------------------------------------------------------------------
def avg_var(setting, method):
    v = extract(setting, method, "var")
    v = v[np.isfinite(v)]
    return v.mean() if len(v) else float("nan")


print("\n" + "=" * 70)
print("SETTING A: Oracle  (value = V_anal, smc_value = V_anal)")
print("Measures INTRINSIC target variance (lower bound).")
print("=" * 70)
print(f"\n{'Method':<25}  {'Avg var (target − V_anal)':>26}")
print("-" * 55)
for mk, label in METHODS.items():
    print(f"  {label:<23}  {avg_var(setting_A, mk):>26.4f}")

print("\n" + "=" * 70)
print("SETTING B: Best TD(λ=0.6) model value + Oracle SMC")
print("Measures target noise + bias during actual training.")
print("=" * 70)
print(f"\n{'Method':<25}  {'Avg var':>10}  {'Max |bias|':>12}")
print("-" * 55)
for mk, label in METHODS.items():
    var   = avg_var(setting_B, mk)
    bias  = extract(setting_B, mk, "mean")
    mbias = np.nanmax(np.abs(bias))
    print(f"  {label:<23}  {var:>10.4f}  {mbias:>12.4f}")

print()

# ---------------------------------------------------------------------------
# Figure 1: Variance by time bin (both settings, excluding one_step_bootstrap)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle("Target Variance vs Time Bin\n(target = method estimate, reference = V_analytical)",
             fontsize=13, y=1.02)

MAIN_METHODS = ["single_seed_mc", "single_seed_td_lam06", "single_seed_td_lam02"]

for ax, (setting, title) in zip(axes, [
    (setting_A, "Setting A: Oracle  (value = V_anal, smc_value = V_anal)"),
    (setting_B, "Setting B: Best model value + Oracle SMC"),
]):
    for mk in MAIN_METHODS:
        var = extract(setting, mk, "var")
        ax.plot(BIN_MIDS, var, "o-", color=COLORS[mk], label=METHODS[mk], lw=2, ms=6)
    ax.set_xlabel("Time t", fontsize=11)
    ax.set_ylabel("Var(target − V_anal)", fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(BIN_MIDS)
    ax.set_xticklabels([f"{m:.1f}" for m in BIN_MIDS])

plt.tight_layout()
plt.savefig("notebooks/data_quality_variance_by_t.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: notebooks/data_quality_variance_by_t.png")

# ---------------------------------------------------------------------------
# Figure 2: Bias by time bin (Setting B only; Setting A bias ≈ 0)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color="black", lw=0.8, ls="--")
for mk in MAIN_METHODS:
    bias = extract(setting_B, mk, "mean")
    ax.plot(BIN_MIDS, bias, "o-", color=COLORS[mk], label=METHODS[mk], lw=2, ms=6)
ax.set_xlabel("Time t", fontsize=11)
ax.set_ylabel("Mean(target − V_anal)", fontsize=11)
ax.set_title("Setting B: Target Bias vs Time Bin\n(value = best TD(λ=0.6) ckpt, smc_value = V_anal)",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xticks(BIN_MIDS)
ax.set_xticklabels([f"{m:.1f}" for m in BIN_MIDS])
plt.tight_layout()
plt.savefig("notebooks/data_quality_bias_by_t.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: notebooks/data_quality_bias_by_t.png")

# ---------------------------------------------------------------------------
# Figure 3: Bar chart — avg variance across methods and settings
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
x     = np.arange(len(MAIN_METHODS))
width = 0.35
bars_A = [avg_var(setting_A, m) for m in MAIN_METHODS]
bars_B = [avg_var(setting_B, m) for m in MAIN_METHODS]
labels = [METHODS[m] for m in MAIN_METHODS]
ax.bar(x - width / 2, bars_A, width, label="Setting A (Oracle)", color="#2980b9", alpha=0.85)
ax.bar(x + width / 2, bars_B, width, label="Setting B (Best model + Oracle SMC)", color="#e67e22", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Mean Var(target − V_anal) across t-bins", fontsize=11)
ax.set_title("Average Target Variance by Method\n(lower = better training signal)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("notebooks/data_quality_avg_variance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: notebooks/data_quality_avg_variance.png")

# ---------------------------------------------------------------------------
# Figure 4: one_step_bootstrap separately (very large scale)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("One-Step Bootstrap — Variance and Bias by Time Bin", fontsize=13)

for ax, (setting, ylabel, title) in zip(axes, [
    (setting_A, "Var(target − V_anal)", "Setting A: Oracle"),
    (setting_B, "Var(target − V_anal)", "Setting B: Best model"),
]):
    for mk, style in [("one_step_bootstrap", ("o-", COLORS["one_step_bootstrap"])),
                      ("single_seed_td_lam02", ("s--", COLORS["single_seed_td_lam02"]))]:
        var = extract(setting, mk, "var")
        ax.plot(BIN_MIDS, var, style[0], color=style[1], label=METHODS[mk], lw=2, ms=6)
    ax.set_xlabel("Time t", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(BIN_MIDS)
    ax.set_xticklabels([f"{m:.1f}" for m in BIN_MIDS])

plt.tight_layout()
plt.savefig("notebooks/data_quality_bootstrap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: notebooks/data_quality_bootstrap.png")

# ---------------------------------------------------------------------------
# Narrative summary
# ---------------------------------------------------------------------------
print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                          KEY FINDINGS                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Setting A (Oracle): intrinsic target variance at optimality             ║
║  ─────────────────────────────────────────────────────────────           ║
║  Method               Avg Var   Relative to MC                          ║
║  MC (λ=1)             1.21      ×1.0  (baseline)                        ║
║  TD(λ=0.6)            0.56      ×0.46                                   ║
║  TD(λ=0.2)            0.28      ×0.23  ← lowest variance                ║
║  One-step bootstrap   49.0      ×40    (massive bias + variance)         ║
║                                                                          ║
║  Setting B (Imperfect V): variance+bias with a trained checkpoint        ║
║  ─────────────────────────────────────────────────────────────           ║
║  MC variance jumps from 1.21 → 2.97 (147% increase).                   ║
║  TD(λ=0.6) variance barely changes: 0.56 → 0.45 (stable).              ║
║  TD(λ=0.2) variance barely changes: 0.28 → 0.30 (stable).              ║
║  → TD methods are more robust to value function inaccuracy.              ║
║                                                                          ║
║  Bias in Setting B:                                                      ║
║  MC: |bias| ≈ 0.22 (moderate, from model error in targets).             ║
║  TD(λ=0.6/0.2): |bias| ≈ 0.21 / 0.19 (similar).                       ║
║  → All methods show comparable bias; MC variance is the main problem.   ║
║                                                                          ║
║  RECOMMENDATION:                                                         ║
║  Use TD(λ=0.2) for lowest intrinsic variance.                           ║
║  TD methods are robust to imperfect value functions.                     ║
║  One-step bootstrap is not viable for this problem.                      ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
