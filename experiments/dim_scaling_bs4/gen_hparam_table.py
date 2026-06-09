"""Generate hparams_by_dimension.md: the confirmed-winner hyperparameters for
each algorithm across the moons BS=4 problem and the calibrated-GMM dimensions.

Sources (all confirmed winners, i.e. best by 5-seed LCB):
  moons d=2 : experiments/bs4_moons/*.json
  GMM d=2..128: experiments/dim_scaling_bs4/results/<method>_d<dim>.json
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MOONS = os.path.join(ROOT, "experiments", "bs4_moons")
RES = os.path.join(HERE, "results")
GMM_DIMS = [2, 8, 32, 128]

METHODS = ["off_policy", "single_seed_mc", "single_seed_td_lambda",
           "ancestral_mc_td_lambda"]
TITLE = {"off_policy": "off-policy", "single_seed_mc": "ssmc (single_seed_mc)",
         "single_seed_td_lambda": "ssmc-td(λ) (single_seed_td_lambda)",
         "ancestral_mc_td_lambda": "anc-mc-td(λ) (ancestral_mc_td_lambda)"}

# hparam display order + optuna sampling scale (for the table footnote + task 2)
HP_ORDER = {
    "off_policy": ["lr", "grad_decay"],
    "single_seed_mc": ["lr", "grad_decay", "n_steps", "mc_samples", "off_policy_frac",
                       "smc_type", "k", "l", "ema_decay", "random_t"],
    "single_seed_td_lambda": ["lr", "grad_decay", "n_steps", "mc_samples",
                              "off_policy_frac", "smc_type", "k", "l", "ema_decay",
                              "random_t", "lambda_eff"],
    "ancestral_mc_td_lambda": ["lr", "grad_decay", "n_steps", "mc_samples",
                               "off_policy_frac", "smc_type", "k", "l", "ema_decay",
                               "lambda_eff"],
}
SCALE = {
    "lr": "log", "grad_decay": "log (+ on/off toggle)", "k": "log", "l": "log",
    "mc_samples": "log-int", "n_steps": "int (linear)", "off_policy_frac": "linear [0,.5]",
    "ema_decay": "linear [.9,.999]", "lambda_eff": "linear [0,1]",
    "smc_type": "categorical", "random_t": "categorical (bool)",
}


def fmt(hp, params):
    if hp == "grad_decay":
        if not params.get("use_grad_decay"):
            return "off"
        return f"{params['grad_decay']:.2e}"
    if hp not in params:
        return "—"
    v = params[hp]
    if hp in ("lr", "k", "l"):
        return f"{v:.2e}"
    if hp in ("ema_decay",):
        return f"{v:.3f}"
    if hp in ("off_policy_frac", "lambda_eff"):
        return f"{v:.3f}"
    if hp in ("n_steps", "mc_samples"):
        return f"{int(v)}"
    if hp == "random_t":
        return "yes" if v else "no"
    return str(v)


def moons_winner(method):
    if method == "off_policy":
        d = json.load(open(f"{MOONS}/optuna_offpolicy_pipeline_results.json"))
        return d["confirm"][str(d["winner_trial"])]["params"]
    if method in ("single_seed_mc", "single_seed_td_lambda"):
        d = json.load(open(f"{MOONS}/optuna_confirm_converge_results.json"))
        return d["winners"][method]["params"]
    if method == "ancestral_mc_td_lambda":
        d = json.load(open(f"{MOONS}/optuna_other_onpolicy_pipeline_results.json"))
        return d["winners"]["A"]["params"]  # t56, discrete smc_type (same space)
    raise ValueError(method)


def gmm_winner(method, dim):
    p = f"{RES}/{method}_d{dim}.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))["winner"]["params"]


def cell_perf(method, dim):
    """(plateau, regret) for the GMM cells; None for moons."""
    p = f"{RES}/{method}_d{dim}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    opt = json.load(open(f"{RES}/optimal_baseline.json")).get(str(dim), {})
    pl = d["convergence"]["plateau_reward"]
    er = opt.get("E_opt_reward")
    return pl, (pl - er if er is not None else None)


COLS = ["moons d=2"] + [f"GMM d={d}" for d in GMM_DIMS]


def col_params(method):
    out = [("moons d=2", moons_winner(method))]
    for d in GMM_DIMS:
        out.append((f"GMM d={d}", gmm_winner(method, d)))
    return out


lines = ["# BS=4 winning hyperparameters by dimension", "",
         "Confirmed-winner hyperparameters (best by 5-seed LCB) for each algorithm, "
         "across the original **moons** BS=4 problem and the calibrated random-GMM "
         "problem at **d = 2, 8, 32, 128**.  Quad loss throughout.", "",
         "- *moons d=2* and *GMM d=2* are the **same dimension but different problems** "
         "(moons dataset with fixed reward scale s=10 vs. random GMM with the "
         "per-dimension gap-calibrated reward).",
         "- `grad_decay = off` means the `use_grad_decay` toggle was False.",
         "- `—` means the hyperparameter is inactive for that config (e.g. `l` only "
         "exists when `smc_type = kV_plus_ltr`; `ema_decay` only when `smc_type = k_Vema`).",
         ""]

for method in METHODS:
    lines.append(f"## {TITLE[method]}")
    lines.append("")
    cps = col_params(method)
    # performance context row
    perf = []
    for col, _ in cps:
        if col == "moons d=2":
            perf.append("— (diff. problem)")
        else:
            dim = int(col.split("=")[1])
            pr = cell_perf(method, dim)
            perf.append(f"{pr[0]:.2f} (reg {pr[1]:+.2f})" if pr else "—")
    # build table
    header = "| hyperparameter | scale | " + " | ".join(c for c, _ in cps) + " |"
    sep = "|" + "---|" * (len(cps) + 2)
    lines.append(header); lines.append(sep)
    for hp in HP_ORDER[method]:
        row = [hp, SCALE.get(hp, "")]
        for _, params in cps:
            row.append(fmt(hp, params) if params else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| *plateau reward (regret)* | — | " + " | ".join(perf) + " |")
    lines.append("")
    if method == "ancestral_mc_td_lambda":
        lines.append("> Note: the GMM cells and the moons row both use the **discrete "
                     "`smc_type`** twist space, on the **fixed** estimator. The separate "
                     "re-tuned moons sweep (`optuna_amctl`) used a more general "
                     "linear-combination twist (`cr·r + cV·V`), not directly comparable "
                     "in parameter space, so it is omitted here.")
        lines.append("")

out_path = f"{HERE}/hparams_by_dimension.md"
open(out_path, "w").write("\n".join(lines))
print(f"Wrote {out_path}\n")
print("\n".join(lines))
