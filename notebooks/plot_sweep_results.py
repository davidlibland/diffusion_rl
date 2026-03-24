"""
Plot training curves from moons_sweep and moons_lambda_sweep experiments.

Three figures:
  1. Sampling methods  (on-policy, quad loss; best lambda for single_seed_td_lambda)
  2. Loss type         (MSE vs quad, per sampling method)
  3. Policy            (off vs on, quad, per sampling method)
"""

import glob
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Analytical target
# ---------------------------------------------------------------------------
_target = json.loads(open("notebooks/analytical_target.json").read())
E_OPT = _target["E_opt"]   # optimal E_{q*}[r(x_1)]  ≈ -2.587
V_0_0 = _target["V_0_0"]   # V(0,0) = log E_p[exp r]  ≈ -5.085


def add_target(ax):
    """Add a horizontal dashed line at the analytical optimal reward."""
    ax.axhline(E_OPT, color="black", linestyle=":", linewidth=1.5, label=f"optimal ({E_OPT:.2f})")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_latest(log_dir, run_name):
    """Return the metrics DataFrame for the latest version of a run."""
    versions = sorted(
        glob.glob(f"{log_dir}/{run_name}/version_*"),
        key=lambda p: int(p.split("_")[-1]),
    )
    if not versions:
        return None
    df = pd.read_csv(f"{versions[-1]}/metrics.csv")
    return df


def val_curve(df):
    """Extract (step_fraction, mean, std) for validation rows."""
    rows = df.dropna(subset=["val_reward_mean"]).copy()
    if rows.empty:
        return None
    max_step = df["step"].max()
    rows["frac"] = rows["step"] / max_step
    return rows[["frac", "step", "val_reward_mean", "val_reward_std", "val_reward_max"]]


def train_curve(df):
    """Extract (step_fraction, train_loss) for training rows."""
    rows = df.dropna(subset=["train_loss"]).copy()
    if rows.empty:
        return None
    max_step = df["step"].max()
    rows["frac"] = rows["step"] / max_step
    return rows[["frac", "step", "train_loss"]]


SWEEP_DIR  = "lightning_logs/moons_sweep"
LAMBDA_DIR = "lightning_logs/moons_lambda_sweep"

# Nice display names
METHOD_LABELS = {
    "one_step_bootstrap":     "one-step bootstrap",
    "ancestral_td_lambda":    "ancestral TD(λ)",
    "single_seed_td_lambda":  "single-seed TD(λ)",
    "single_seed_mc":         "single-seed MC",
}

COLORS = {
    "one_step_bootstrap":    "#1f77b4",
    "ancestral_td_lambda":   "#ff7f0e",
    "single_seed_td_lambda": "#2ca02c",
    "single_seed_mc":        "#d62728",
    "off_policy":            "#9467bd",
}

ON_METHODS = ["one_step_bootstrap", "ancestral_td_lambda", "single_seed_td_lambda", "single_seed_mc"]

# ---------------------------------------------------------------------------
# Figure 1 – Sampling methods (quad, on-policy)
#   - single_seed_td_lambda: use lambda sweep lam0.600 (best)
#   - all others: use main sweep on_quad_*
# ---------------------------------------------------------------------------

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig1.suptitle("Sampling methods  (quad loss, on-policy)\n"
              "single_seed_td_lambda uses best λ_eff=0.6", fontsize=13)

ax_val, ax_train = axes1
ax_val.set_title("Validation reward mean")
ax_train.set_title("Training loss")

for method in ON_METHODS:
    color = COLORS[method]
    label = METHOD_LABELS[method]

    if method == "single_seed_td_lambda":
        df = load_latest(LAMBDA_DIR, "single_seed_td_lambda_lam0.600")
    else:
        df = load_latest(SWEEP_DIR, f"on_quad_{method}")

    if df is None:
        continue

    vc = val_curve(df)
    tc = train_curve(df)

    if vc is not None:
        ax_val.plot(vc["frac"], vc["val_reward_mean"], "o-", color=color, label=label)
        ax_val.fill_between(
            vc["frac"],
            vc["val_reward_mean"] - vc["val_reward_std"] / np.sqrt(512),
            vc["val_reward_mean"] + vc["val_reward_std"] / np.sqrt(512),
            color=color, alpha=0.2,
        )

    if tc is not None:
        # smooth with rolling window
        smoothed = tc["train_loss"].rolling(50, min_periods=1).mean()
        ax_train.plot(tc["frac"], smoothed, color=color, label=label, alpha=0.8)

add_target(ax_val)
for ax in axes1:
    ax.set_xlabel("Training fraction")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
ax_val.set_ylabel("val_reward_mean")
ax_train.set_ylabel("train_loss")

fig1.tight_layout()
fig1.savefig("notebooks/plot_sampling_methods.png", dpi=150)
print("Saved plot_sampling_methods.png")

# ---------------------------------------------------------------------------
# Figure 2 – MSE vs Quad  (on-policy, 4 sampling methods)
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
fig2.suptitle("Loss type: MSE vs Quad  (on-policy)", fontsize=13)

for idx, method in enumerate(ON_METHODS):
    ax = axes2.flat[idx]
    ax.set_title(METHOD_LABELS[method])

    for loss_type, ls, marker in [("mse", "--", "s"), ("quad", "-", "o")]:
        df = load_latest(SWEEP_DIR, f"on_{loss_type}_{method}")
        if df is None:
            continue
        vc = val_curve(df)
        if vc is None:
            continue
        color = "#1f77b4" if loss_type == "mse" else "#d62728"
        ax.plot(vc["frac"], vc["val_reward_mean"], f"{marker}{ls}",
                color=color, label=loss_type.upper())
        ax.fill_between(
            vc["frac"],
            vc["val_reward_mean"] - vc["val_reward_std"] / np.sqrt(512),
            vc["val_reward_mean"] + vc["val_reward_std"] / np.sqrt(512),
            color=color, alpha=0.2,
        )

    add_target(ax)
    ax.set_xlabel("Training fraction")
    ax.set_ylabel("val_reward_mean")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig("notebooks/plot_loss_type.png", dpi=150)
print("Saved plot_loss_type.png")

# ---------------------------------------------------------------------------
# Figure 3 – Off-policy vs On-policy  (quad)
# ---------------------------------------------------------------------------

fig3, axes3 = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
fig3.suptitle("Off-policy vs On-policy  (quad loss)", fontsize=13)

for idx, method in enumerate(ON_METHODS):
    ax = axes3.flat[idx]
    ax.set_title(METHOD_LABELS[method])

    # off-policy (same curve for all methods)
    df_off = load_latest(SWEEP_DIR, "off_quad")
    if df_off is not None:
        vc = val_curve(df_off)
        if vc is not None:
            ax.plot(vc["frac"], vc["val_reward_mean"], "s--",
                    color=COLORS["off_policy"], label="off-policy")
            ax.fill_between(
                vc["frac"],
                vc["val_reward_mean"] - vc["val_reward_std"] / np.sqrt(512),
                vc["val_reward_mean"] + vc["val_reward_std"] / np.sqrt(512),
                color=COLORS["off_policy"], alpha=0.2,
            )

    # on-policy (quad)
    if method == "single_seed_td_lambda":
        df_on = load_latest(LAMBDA_DIR, "single_seed_td_lambda_lam0.600")
    else:
        df_on = load_latest(SWEEP_DIR, f"on_quad_{method}")

    if df_on is not None:
        vc = val_curve(df_on)
        if vc is not None:
            ax.plot(vc["frac"], vc["val_reward_mean"], "o-",
                    color=COLORS[method], label="on-policy")
            ax.fill_between(
                vc["frac"],
                vc["val_reward_mean"] - vc["val_reward_std"] / np.sqrt(512),
                vc["val_reward_mean"] + vc["val_reward_std"] / np.sqrt(512),
                color=COLORS[method], alpha=0.2,
            )

    add_target(ax)
    ax.set_xlabel("Training fraction")
    ax.set_ylabel("val_reward_mean")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig("notebooks/plot_off_vs_on.png", dpi=150)
print("Saved plot_off_vs_on.png")

# ---------------------------------------------------------------------------
# Figure 4 – Lambda sweep for single_seed_td_lambda
# ---------------------------------------------------------------------------

fig4, ax4 = plt.subplots(figsize=(8, 5))
fig4.suptitle("λ_eff sweep  –  single_seed_td_lambda, quad loss", fontsize=13)

lambdas_p1 = [0.010, 0.050, 0.150, 0.350, 0.600, 0.900]
cmap = plt.get_cmap("viridis")
colors_lam = {lam: cmap(i / (len(lambdas_p1) - 1)) for i, lam in enumerate(lambdas_p1)}

for lam in lambdas_p1:
    run_name = f"single_seed_td_lambda_lam{lam:.3f}"
    df = load_latest(LAMBDA_DIR, run_name)
    if df is None:
        continue
    vc = val_curve(df)
    if vc is None:
        continue
    marker = "o" if lam == 0.600 else "."
    lw = 2.5 if lam == 0.600 else 1.5
    ax4.plot(vc["frac"], vc["val_reward_mean"], f"{marker}-",
             color=colors_lam[lam], label=f"λ_eff={lam}", linewidth=lw)

add_target(ax4)
ax4.set_xlabel("Training fraction")
ax4.set_ylabel("val_reward_mean")
ax4.legend(fontsize=9, title="λ_eff")
ax4.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig("notebooks/plot_lambda_sweep.png", dpi=150)
print("Saved plot_lambda_sweep.png")

plt.show()
