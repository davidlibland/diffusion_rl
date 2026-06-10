#!/usr/bin/env python3
"""Combine the three per-method FBRRT re-tune results into the cross-method
comparison plot + table used by REPORT.md.

Reads optuna_fbrrt2_bs4_sweep_{method}_results.json (written by the per-method
final passes) and the archived 2026-05-28 winners; writes
fbrrt2_summary_convergence.png and prints the comparison table as markdown.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "..", "2026-05-28")
METHODS = ["fbrrt", "fbrrt_td_lambda", "fbrrt_cv"]
COLORS = {"fbrrt": "#16a085", "fbrrt_td_lambda": "#e67e22",
          "fbrrt_cv": "#8e44ad"}

# ── archived (pre-fix) winners ──────────────────────────────────────────────
prior = {}
for pth in [f"{ARCHIVE}/optuna_other_onpolicy_pipeline_results.json",
            f"{ARCHIVE}/optuna_confirm_converge_results.json"]:
    if os.path.exists(pth):
        for k, v in json.load(open(pth)).get("convergence", {}).items():
            prior[k] = v
opp = f"{ARCHIVE}/optuna_offpolicy_pipeline_results.json"
if os.path.exists(opp):
    op = json.load(open(opp)).get("comparison", {}).get("off_policy")
    if op:
        prior["offpolicy_t0_converge"] = op
amc = f"{ARCHIVE}/optuna_amctl_bs4_sweep_results.json"
if os.path.exists(amc):
    cv_ = json.load(open(amc)).get("convergence", {})
    if "plateau_reward" in cv_:
        prior["amctl_retuned_t22"] = cv_

# ── new per-method results ──────────────────────────────────────────────────
new = {}
for m in METHODS:
    pth = f"{HERE}/optuna_fbrrt2_bs4_sweep_{m}_results.json"
    if not os.path.exists(pth):
        print(f"[skip] {pth} missing")
        continue
    d = json.load(open(pth))
    conv = list(d["convergence"].values())[0]
    curves = list(d["convergence_curves"].values())[0]
    new[m] = {"conv": conv, "curves": curves, "winner": d["winner"],
              "confirm": d["confirm"]}

# ── plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_title("FBRRT re-tune (FIXED estimators, 2026-06-10) — 50k convergence "
             "vs archived BS=4 winners (pre-fix)", fontsize=11)
for m, d in new.items():
    st = np.array(d["curves"]["steps"]); cv = np.array(d["curves"]["val_reward"])
    ax.plot(st, cv, color=COLORS[m], alpha=0.22, lw=0.8)
    label = (f"{m} (fixed)  plateau={d['conv']['plateau_reward']:.2f}")
    if d["conv"].get("diverged_at"):
        label += f"  (diverged@{d['conv']['diverged_at']})"
    ax.plot(st, pd.Series(cv).rolling(8, min_periods=1).mean(),
            color=COLORS[m], lw=2.2, label=label)
ref_styles = {
    "single_seed_td_lambda_t80_converge": ("#1f77b4", "single_seed_td_lambda"),
    "amctl_retuned_t22": ("#2ca02c", "ancestral_mc_td_lambda (re-tuned)"),
    "single_seed_mc_t1_converge": ("#7f7f7f", "single_seed_mc"),
    "offpolicy_t0_converge": ("#d62728", "off-policy"),
    "famB_t2_fbrrt_td_lambda_converge": ("#8c564b", "fbrrt_td_lambda (OLD code)"),
}
for k, (col, lbl) in ref_styles.items():
    if k in prior:
        ax.axhline(prior[k]["plateau_reward"], color=col, ls="--", alpha=0.85,
                   label=f"{lbl}  plateau={prior[k]['plateau_reward']:.2f}")
ax.set_xlabel("training step"); ax.set_ylabel("val reward (512 rollouts)")
ax.set_ylim(bottom=-26)
ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
out = f"{HERE}/fbrrt2_summary_convergence.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved {out}\n")

# ── markdown table ──────────────────────────────────────────────────────────
print("| config | code | plateau | best smoothed | conv. step | final LCB |")
print("|---|---|---:|---:|---:|---:|")
rows = []
for k, lbl in [("single_seed_td_lambda_t80_converge", "`single_seed_td_lambda`"),
               ("amctl_retuned_t22", "`ancestral_mc_td_lambda` (re-tuned)"),
               ("single_seed_mc_t1_converge", "`single_seed_mc`"),
               ("offpolicy_t0_converge", "off-policy"),
               ("famB_t2_fbrrt_td_lambda_converge", "`fbrrt_td_lambda`")]:
    if k in prior:
        v = prior[k]
        code = "pre-fix" if "fbrrt" in k else "archived"
        rows.append((lbl, code, v["plateau_reward"], None,
                     v["convergence_step"], v["final_lcb"]))
for m, d in new.items():
    cv = np.array(d["curves"]["val_reward"])
    sm = pd.Series(cv).rolling(8, min_periods=1).mean()
    rows.append((f"`{m}`", "**fixed**", d["conv"]["plateau_reward"],
                 float(sm.max()), d["conv"]["convergence_step"],
                 d["conv"]["final_lcb"]))
rows.sort(key=lambda r: r[2], reverse=True)
for lbl, code, plat, best, cs, flcb in rows:
    b = f"{best:.2f}" if best is not None else "—"
    print(f"| {lbl} | {code} | {plat:.2f} | {b} | {cs} | {flcb:.2f} |")

print("\nWinner params per method:")
for m, d in new.items():
    print(f"  {m}: {d['winner'].get('params')}")
