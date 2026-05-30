"""Generate clean bias/variance plots for the simplified data quality report."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

with open("experiments/data_quality/data_quality_v2_results.json") as f:
    data = json.load(f)

BIN_NAMES = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]
BIN_MIDS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Methods to include
METHOD_FAMILIES = {
    "Ancestral TD(λ)": {"prefix": "Ancestral TD(λ)", "exclude": "MC", "color_base": "Reds", "ls": "-"},
    "Ancestral MC-TD(λ)": {"prefix": "Ancestral MC-TD(λ)", "exclude": None, "color_base": "Blues", "ls": "--"},
    "FBRRT-TD(λ)": {"prefix": "FBRRT-TD(λ)", "exclude": None, "color_base": "Greens", "ls": "-"},
}

LAMBDA_ORDER = ["λ=0", "λ_s=0.1", "λ_s=0.5", "λ_eff=0.1", "λ_eff=0.5", "λ_eff=0.8", "λ=1"]

STAGES = [
    ("stage3", "Oracle Lower Bound\n(oracle V + oracle SMC)"),
    ("stage7a", "Self-Consistent Early\n(early ckpt V + early ckpt SMC)"),
    ("stage7b", "Self-Consistent Mid\n(mid ckpt V + mid ckpt SMC)"),
    ("stage6", "Self-Consistent Best\n(best model V + best model SMC)"),
]


def get_entries(stage_data, family_prefix, exclude=None):
    """Get entries matching a method family, ordered by lambda."""
    out = []
    for lam_label in LAMBDA_ORDER:
        for e in stage_data:
            label = e["label"]
            if family_prefix in label and f"({lam_label})" in label:
                if exclude and exclude in label:
                    continue
                out.append((lam_label, e))
                break
    return out


def get_offpolicy(stage_data):
    for e in stage_data:
        if e.get("is_offpolicy"):
            return e
    return None


for stage_key, stage_title in STAGES:
    entries = data[stage_key]
    off = get_offpolicy(entries)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey='row')
    fig.suptitle(f"Target Quality: {stage_title}", fontsize=15, fontweight="bold", y=1.02)

    for col, (family_name, cfg) in enumerate(METHOD_FAMILIES.items()):
        family_entries = get_entries(entries, cfg["prefix"], cfg.get("exclude"))
        cmap_fn = cm.get_cmap(cfg["color_base"], max(7, len(family_entries) + 2))

        ax_var = axes[0, col]
        ax_bias = axes[1, col]
        ax_var.set_title(family_name, fontsize=12, fontweight="bold")

        # Plot off-policy baseline
        if off:
            stats = off["stats"]
            var_vals = [max(stats[b]["var"], 1e-9) for b in BIN_NAMES]
            bias_vals = [max(abs(stats[b]["mean"]), 1e-9) for b in BIN_NAMES]
            ax_var.plot(BIN_MIDS, var_vals, "k--", lw=2.5, label="Off-Policy", zorder=10)
            ax_bias.plot(BIN_MIDS, bias_vals, "k--", lw=2.5, label="Off-Policy", zorder=10)

        # Plot each lambda
        for i, (lam_label, e) in enumerate(family_entries):
            stats = e["stats"]
            color = cmap_fn(i + 2)  # skip lightest colors
            var_vals = [max(stats[b]["var"], 1e-9) for b in BIN_NAMES]
            bias_vals = [max(abs(stats[b]["mean"]), 1e-9) for b in BIN_NAMES]

            ax_var.plot(BIN_MIDS, var_vals, marker="o", markersize=4,
                       color=color, ls=cfg["ls"], lw=1.5, label=lam_label)
            ax_bias.plot(BIN_MIDS, bias_vals, marker="o", markersize=4,
                        color=color, ls=cfg["ls"], lw=1.5, label=lam_label)

        ax_var.set_yscale("log")
        ax_bias.set_yscale("log")
        ax_var.grid(True, alpha=0.3, which="both")
        ax_bias.grid(True, alpha=0.3, which="both")
        ax_var.legend(fontsize=6, loc="upper left")
        ax_bias.legend(fontsize=6, loc="upper left")

        if col == 0:
            ax_var.set_ylabel("Variance", fontsize=11)
            ax_bias.set_ylabel("|Bias|", fontsize=11)
        ax_bias.set_xlabel("t", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"experiments/data_quality/dq2_clean_{stage_key}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: experiments/data_quality/dq2_clean_{stage_key}.png")
    plt.close()

print("Done.")
