"""
FBRRT data-quality re-run (2026-06-10): OLD (buggy) vs NEW (fixed) FBRRT.
=========================================================================

Measures the bias/variance of the FBRRT-SMC regression targets against the
analytical value function V_anal(x, t), using the same per-t-bin metric as the
archived data_quality_v2.py (../2026-05-28/).  For every FBRRT method and value
scenario we run BOTH the pre-fix estimators (frozen in old_fbrrt.py, from commit
master/8c5782e) and the post-fix estimators (live package), so the only
difference is the code.

Methods
  fbrrt            grad-control one-step      (fix B: entropy weight off target)
  fbrrt_td_lambda  GAE multi-step             (fix D: ancestor align + multinom)
  fbrrt_cv         residual control variate   (fix A,C: driver + Malliavin scale)
  fbrrt_mc_z       MC-Z estimator             (issue E: unstable, documented)

Value scenarios
  oracle   v_policy = v_target = analytical V        (clean correctness check)
  model    v_policy = v_target = trained model V      (self-consistent, eps=0)
  lagged   v_policy = lagged model, v_target = model   (eps != 0; exercises RCV)

Outputs (this folder):
  fbrrt_dq_results.json          raw per-bin stats
  fbrrt_dq_avg.png               avg |bias| and avg var, old vs new
  fbrrt_dq_bias_by_t.png         bias vs t per method/scenario, old vs new
"""

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dq_setup as S  # noqa: E402
import old_fbrrt  # noqa: E402  (frozen pre-fix estimators)
from diffusion_rl.models import on_policy as new  # noqa: E402

warnings.filterwarnings("ignore")
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Collection parameters (match the archived experiment's FBRRT regime)
# ---------------------------------------------------------------------------
N_STEPS = 100
N_PARTICLES = 16
BRANCH = 4
N_CALLS = 24
ENTROPY_LAMBDA = 1.0  # the default; the regime where the (B) bias appeared
ALPHA = 1.0
LAMBDA_EFF = 0.5  # for td_lambda

CKPT_MODEL = "checkpoints/dim_scaling_bs4/single_seed_td_lambda_d2/best.ckpt"
CKPT_LAG = "checkpoints/dim_scaling_bs4/single_seed_td_lambda_d2/last.ckpt"

print("Loading model value functions...")
model_fn = S.load_value_fn(CKPT_MODEL, "on_policy")
lag_fn = S.load_value_fn(CKPT_LAG, "on_policy")
oracle_fn = S.anal_fn


def _common(mod_fn, **extra):
    return dict(
        a=S.a, n_steps=N_STEPS, n_particles=N_PARTICLES, branch=BRANCH,
        f=S.base_drift, reward=S.reward, d=S.D, alpha=ALPHA,
        entropy_lambda=ENTROPY_LAMBDA, device=torch.device(S.DEVICE), **extra,
    )


def collect(fbrrt_mod, method, v_policy, v_target):
    """Run a method (from old_fbrrt or new) N_CALLS times; return (x,t,target)."""
    xs, ts, tg = [], [], []
    for _ in range(N_CALLS):
        if method == "fbrrt":
            out = fbrrt_mod.fbrrt_smc_grad_control(v_theta=v_target, **_common(v_target))
        elif method == "fbrrt_td_lambda":
            out = fbrrt_mod.fbrrt_smc_grad_control_td_lambda(
                v_theta=v_target, lambda_eff=LAMBDA_EFF, **_common(v_target))
        elif method == "fbrrt_cv":
            out = fbrrt_mod.fbrrt_smc_grad_control_variate(
                v_policy=v_policy, v_target=v_target, **_common(v_target))
        elif method == "fbrrt_mc_z":
            out = fbrrt_mod.fbrrt_smc_grad_mc_Z(
                v_policy=v_policy, v_target=v_target, **_common(v_target))
        else:
            raise ValueError(method)
        xs.append(out.x)
        ts.append(out.t)
        tg.append(out.v_hat)
    return torch.cat(xs), torch.cat(ts), torch.cat(tg)


# method -> which scenarios apply (single-value methods skip "lagged")
SINGLE = ["fbrrt", "fbrrt_td_lambda"]
DUAL = ["fbrrt_cv", "fbrrt_mc_z"]
METHODS = SINGLE + DUAL

SCENARIOS = {
    "oracle": dict(v_policy=oracle_fn, v_target=oracle_fn),
    "model": dict(v_policy=model_fn, v_target=model_fn),
    # oracle_lag isolates the control-variate / driver: the TARGET value is exact
    # (v_target = oracle), so the only source of target bias is the driver built
    # from a *wrong* policy gradient (v_policy = lagged model).  This is the
    # regime where the (A,C) fix matters; with a model v_target the ~2-nat model
    # error swamps the driver term.
    "oracle_lag": dict(v_policy=lag_fn, v_target=oracle_fn),
}


def scenario_keys(method):
    if method in SINGLE:
        return ["oracle", "model"]  # single V: oracle_lag == oracle
    return ["oracle", "model", "oracle_lag"]


CODES = {"old": old_fbrrt, "new": new}

RESULTS_PATH = os.path.join(HERE, "fbrrt_dq_results.json")
REPLOT = "--replot" in sys.argv

results = {}  # (method, scenario, code) -> {stats, avg_var, avg_bias, frac_nonfinite}
if REPLOT:
    print(f"--replot: loading {RESULTS_PATH}")
    with open(RESULTS_PATH) as f:
        results = json.load(f)
print("\nRunning OLD vs NEW FBRRT data-quality sweep "
      f"(n_steps={N_STEPS}, M={N_PARTICLES}, B={BRANCH}, calls={N_CALLS}, "
      f"entropy_lambda={ENTROPY_LAMBDA})\n")
print(f"{'method':18s} {'scenario':8s} {'code':4s} "
      f"{'avg|bias|':>10s} {'avg var':>10s} {'%nonfinite':>11s}")
print("-" * 70)

for method in (METHODS if not REPLOT else []):
    for scen in scenario_keys(method):
        for code, mod in CODES.items():
            torch.manual_seed(0)  # same particles old vs new -> controlled diff
            try:
                x, t, tgt = collect(mod, method, **SCENARIOS[scen])
                frac_nf = float((~torch.isfinite(tgt)).float().mean().item())
                stats = S.binned_stats(x, t, tgt)
                avg_var, avg_bias = S.avg_stats(stats)
            except Exception as e:  # numerical blow-ups etc.
                stats, avg_var, avg_bias, frac_nf = {}, float("nan"), float("nan"), 1.0
                print(f"  [error] {method}/{scen}/{code}: {type(e).__name__}: {e}")
            results[f"{method}|{scen}|{code}"] = {
                "method": method, "scenario": scen, "code": code,
                "stats": stats, "avg_var": avg_var, "avg_bias": avg_bias,
                "frac_nonfinite": frac_nf,
            }
            print(f"{method:18s} {scen:8s} {code:4s} "
                  f"{avg_bias:10.4f} {avg_var:10.4f} {100*frac_nf:10.2f}%")

if not REPLOT:
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_PATH}")

# ---------------------------------------------------------------------------
# Plot 1: avg |bias| and avg var, old vs new, per (method, scenario).
# fbrrt_mc_z is excluded from the bars (it diverges, ~1e16..1e32, off-scale);
# it is annotated instead -- see the table / report for its numbers.
# ---------------------------------------------------------------------------
PLOT_METHODS = [m for m in METHODS if m != "fbrrt_mc_z"]
labels, old_bias, new_bias, old_var, new_var = [], [], [], [], []
for method in PLOT_METHODS:
    for scen in scenario_keys(method):
        labels.append(f"{method}\n{scen}")
        old_bias.append(results[f"{method}|{scen}|old"]["avg_bias"])
        new_bias.append(results[f"{method}|{scen}|new"]["avg_bias"])
        old_var.append(results[f"{method}|{scen}|old"]["avg_var"])
        new_var.append(results[f"{method}|{scen}|new"]["avg_var"])

xpos = np.arange(len(labels))
w = 0.38


def _clip(vals):
    return [v if np.isfinite(v) else np.nan for v in vals]


fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels) * 1.1), 9))
for ax, old_v, new_v, title in [
    (axes[0], old_bias, new_bias, "avg |bias|  (mean |target - V_anal| over t-bins)"),
    (axes[1], old_var, new_var, "avg variance  (var(target - V_anal) over t-bins)"),
]:
    ax.bar(xpos - w / 2, _clip(old_v), w, label="OLD (buggy)", color="#c0392b", alpha=0.85)
    ax.bar(xpos + w / 2, _clip(new_v), w, label="NEW (fixed)", color="#1abc9c", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", axis="y", alpha=0.3)
mcz = results.get("fbrrt_mc_z|oracle|new", {})
axes[0].text(
    0.5, 0.95,
    "fbrrt_mc_z omitted: diverges (|target| up to ~1e16, "
    f"{100 * results['fbrrt_mc_z|oracle|old']['frac_nonfinite']:.0f}% non-finite) "
    "for OLD and NEW alike (issue E, not fixed)",
    transform=axes[0].transAxes, ha="center", va="top", fontsize=8,
    bbox=dict(boxstyle="round", fc="#fdecea", ec="#c0392b"),
)
fig.suptitle("FBRRT data quality: OLD vs NEW (lower is better; fbrrt_mc_z off-scale)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fbrrt_dq_avg.png"), dpi=140, bbox_inches="tight")
print(f"Saved {os.path.join(HERE, 'fbrrt_dq_avg.png')}")

# ---------------------------------------------------------------------------
# Plot 2: bias vs t per (method, scenario), old vs new
# ---------------------------------------------------------------------------
panels = [(m, s) for m in METHODS for s in scenario_keys(m)]
ncol = 3
nrow = int(np.ceil(len(panels) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow), squeeze=False)
for idx, (method, scen) in enumerate(panels):
    ax = axes[idx // ncol][idx % ncol]
    for code, color in [("old", "#c0392b"), ("new", "#1abc9c")]:
        st = results[f"{method}|{scen}|{code}"]["stats"]
        if not st:
            continue
        ys = [st[b]["mean"] if (b in st and np.isfinite(st[b]["mean"])) else np.nan
              for b in S.BIN_NAMES]
        ax.plot(S.BIN_MIDS, ys, "o-", color=color, label=f"{code}")
    ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_title(f"{method} / {scen}", fontsize=9)
    ax.set_xlabel("t")
    ax.set_ylabel("bias = E[target - V_anal]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
for j in range(len(panels), nrow * ncol):
    axes[j // ncol][j % ncol].axis("off")
fig.suptitle("FBRRT target bias vs t: OLD vs NEW", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fbrrt_dq_bias_by_t.png"), dpi=140, bbox_inches="tight")
print(f"Saved {os.path.join(HERE, 'fbrrt_dq_bias_by_t.png')}")
print("\nDone.")
