"""Generate batch size sweep plots."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

BASE = "/Users/dlibland/dev/diffusion_rl/lightning_logs/batch_sweep"
OUT = "/Users/dlibland/dev/diffusion_rl/notebooks"
E_OPT = -2.5872

BATCH_SIZES = [1, 2, 4, 16, 256]

METHODS = [
    ("offpolicy", "Off-Policy", "black", "--"),
    ("ssmc_frac0", "SSMC (frac=0)", "tab:red", "-"),
    ("ssmc_frac50", "SSMC (frac=50)", "tab:red", "--"),
    ("fbrrt_frac0", "FBRRT (frac=0)", "teal", "-"),
    ("fbrrt_frac50", "FBRRT (frac=50)", "teal", "--"),
    ("amctd_frac0", "AMCTD (frac=0)", "navy", "-"),
    ("amctd_frac50", "AMCTD (frac=50)", "navy", "--"),
]


def load_metrics(name):
    path = os.path.join(BASE, name, "version_0", "metrics.csv")
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    df = pd.read_csv(path)
    return df


def dir_name(method_key, bs):
    if method_key == "offpolicy":
        return f"offpolicy_bs{bs}"
    parts = method_key.split("_frac")
    return f"{parts[0]}_bs{bs}_frac{parts[1]}"


for bs in BATCH_SIZES:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Batch Size = {bs}", fontsize=14, fontweight="bold")

    for method_key, label, color, ls in METHODS:
        dn = dir_name(method_key, bs)
        df = load_metrics(dn)
        if df is None:
            continue

        # Filter to rows with val_reward_mean
        reward_df = df.dropna(subset=["val_reward_mean"])
        mae_df = df.dropna(subset=["traj_avg_mae_guided"])

        if len(reward_df) > 0:
            axes[0].plot(
                reward_df["step"], reward_df["val_reward_mean"],
                label=label, color=color, linestyle=ls, linewidth=1.5, alpha=0.85,
            )
        if len(mae_df) > 0:
            axes[1].plot(
                mae_df["step"], mae_df["traj_avg_mae_guided"],
                label=label, color=color, linestyle=ls, linewidth=1.5, alpha=0.85,
            )

    # Optimal line on reward plot
    axes[0].axhline(E_OPT, color="green", linestyle=":", linewidth=1, alpha=0.7, label=f"E_OPT={E_OPT}")

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Terminal Reward (mean)")
    axes[0].set_title("Terminal Reward vs Steps")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Guided MAE")
    axes[1].set_title("Guided MAE vs Steps")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(OUT, f"batch_sweep_bs{bs}.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")

print("Done.")
