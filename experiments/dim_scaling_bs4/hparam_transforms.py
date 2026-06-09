"""Per-hyperparameter transforms (matching Optuna's sampling scale) and the
per-method hyperparameter layout, shared by fit_hparams.py and selector.py.

Each continuous hparam is fit on its `fwd`-transformed scale (so a straight line
in the transformed space corresponds to the scale Optuna sampled on), then
`inv`-transformed, clipped to the sampling bounds, and rounded if integer.
"""

import numpy as np


def _logit(p):
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# kind: 'cont' (continuous, LOO const-vs-slope), 'cat' (categorical mode),
#       'bool' (majority).  conditional => only active when smc_type == that value.
SPEC = {
    "lr":         dict(kind="cont", fwd=np.log, inv=np.exp, lo=1e-4, hi=3e-3),
    "k":          dict(kind="cont", fwd=np.log, inv=np.exp, lo=1e-3, hi=1.0),
    "mc_samples": dict(kind="cont", fwd=np.log, inv=np.exp, lo=1, hi=24, integer=True),
    "n_steps":    dict(kind="cont", fwd=lambda v: float(v), inv=lambda z: z,
                       lo=10, hi=60, integer=True),
    "off_policy_frac": dict(kind="cont",
                            fwd=lambda v: _logit(np.clip(v, 1e-3, 0.5 - 1e-3) / 0.5),
                            inv=lambda z: 0.5 * _sigmoid(z), lo=0.0, hi=0.5),
    "lambda_eff": dict(kind="cont",
                       fwd=lambda v: _logit(np.clip(v, 1e-3, 1 - 1e-3)),
                       inv=_sigmoid, lo=0.0, hi=1.0),
    "grad_decay": dict(kind="cont", fwd=np.log, inv=np.exp, lo=1e-5, hi=1e-1),
    "l":          dict(kind="cont", fwd=np.log, inv=np.exp, lo=1e-3, hi=1.0,
                       conditional="kV_plus_ltr"),
    "ema_decay":  dict(kind="cont",
                       fwd=lambda v: np.log(1.0 - np.clip(v, 0.9, 0.999)),
                       inv=lambda z: 1.0 - np.exp(z), lo=0.9, hi=0.999,
                       conditional="k_Vema"),
    "smc_type":   dict(kind="cat"),
    "random_t":   dict(kind="bool"),
    "use_grad_decay": dict(kind="bool"),
}


def apply_inv(hp, z):
    s = SPEC[hp]
    v = float(s["inv"](z))
    v = float(np.clip(v, s["lo"], s["hi"]))
    if s.get("integer"):
        v = int(round(v))
        v = int(np.clip(v, s["lo"], s["hi"]))
    return v


# Per-method layout.
#   cont   : continuous hparams that get LOO const-vs-slope model selection
#   cat    : categorical (mode)
#   bool   : majority-vote booleans
#   cond   : constants used only when smc_type selects them (+ grad_decay value)
LAYOUT = {
    "off_policy": dict(cont=["lr"], cat=[], bool=["use_grad_decay"], cond=["grad_decay"]),
    "single_seed_mc": dict(
        cont=["lr", "n_steps", "mc_samples", "off_policy_frac", "k"],
        cat=["smc_type"], bool=["use_grad_decay", "random_t"],
        cond=["grad_decay", "l", "ema_decay"]),
    "single_seed_td_lambda": dict(
        cont=["lr", "n_steps", "mc_samples", "off_policy_frac", "k", "lambda_eff"],
        cat=["smc_type"], bool=["use_grad_decay", "random_t"],
        cond=["grad_decay", "l", "ema_decay"]),
    "ancestral_mc_td_lambda": dict(
        cont=["lr", "n_steps", "mc_samples", "off_policy_frac", "k", "lambda_eff"],
        cat=["smc_type"], bool=["use_grad_decay"],
        cond=["grad_decay", "l", "ema_decay"]),
}
