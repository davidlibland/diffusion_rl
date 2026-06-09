"""Dimension -> hyperparameters selector, fit by fit_hparams.py.

Loads fitted_models.json and returns a params dict ready for
``sweep.build(method, params, ...)`` — the bridge for the next experiment
(fixed hparams-by-dimension, multi-seed problem instances) so we no longer
sweep Optuna per dimension.

    from selector import hparams_for_dim
    params = hparams_for_dim("single_seed_td_lambda", 16)
"""

import json
import os

import numpy as np

from hparam_transforms import LAYOUT, apply_inv

HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = json.load(open(f"{HERE}/fitted_models.json"))


def _cont(models, hp, z):
    m = models[hp]
    if m["model"] == "slope":
        a, b = m["slope"]
        zval = float(a) + float(b) * z
    else:
        zval = float(m["const"])
    return apply_inv(hp, zval)


def hparams_for_dim(method, d):
    """Predicted hyperparameters for `method` at dimension `d` (quad loss)."""
    if method not in _MODELS:
        raise ValueError(f"unknown method {method}")
    models = _MODELS[method]
    z = float(np.log(d))
    p = {"off_policy_frac": 0.0, "lr": _cont(models, "lr", z)}

    gd = models.get("use_grad_decay")
    if gd and gd["value"] and "grad_decay" in models:
        p["use_grad_decay"] = True
        p["grad_decay"] = _cont(models, "grad_decay", z)
    else:
        p["use_grad_decay"] = False

    if method == "off_policy":
        return p

    for hp in ("n_steps", "mc_samples", "off_policy_frac", "k", "lambda_eff"):
        if hp in models:
            p[hp] = _cont(models, hp, z)
    smc = models["smc_type"]["mode"]
    p["smc_type"] = smc
    if smc == "kV_plus_ltr" and "l" in models:
        p["l"] = _cont(models, "l", z)
    if smc == "k_Vema" and "ema_decay" in models:
        p["ema_decay"] = _cont(models, "ema_decay", z)
    if "random_t" in models:
        p["random_t"] = bool(models["random_t"]["value"])
    return p


if __name__ == "__main__":
    for method in LAYOUT:
        print(f"\n=== {method} ===")
        for d in (2, 8, 16, 32, 64, 128):
            print(f"  d={d:>3}: {hparams_for_dim(method, d)}")
