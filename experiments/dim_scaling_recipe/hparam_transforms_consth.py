"""Hyperparameter transforms + per-method layout for the constant-headroom
dim-scaling study.

Extends ../dim_scaling_bs4/hparam_transforms.py with:
  - guided-proposal knobs for the SMC methods (use_guidance, guidance_scale,
    guidance_source) -- the new gradient-based driver;
  - the two FBRRT methods (fbrrt with an always-EMA generation network per the
    bs4_moons stabilization study, fbrrt_cv with its policy-EMA), with their
    alpha / branch / entropy_lambda knobs;
  - ema_decay is now a shared conditional (active when any of k_Vema twist,
    EMA guidance source, or an FBRRT method needs it).
"""

import numpy as np


def _logit(p):
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


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
    "l":          dict(kind="cont", fwd=np.log, inv=np.exp, lo=1e-3, hi=1.0),
    "ema_decay":  dict(kind="cont",
                       fwd=lambda v: np.log(1.0 - np.clip(v, 0.9, 0.999)),
                       inv=lambda z: 1.0 - np.exp(z), lo=0.9, hi=0.999),
    # guided-proposal knobs (SMC methods)
    "guidance_scale": dict(kind="cont", fwd=np.log, inv=np.exp, lo=0.05, hi=1.5),
    "guidance_source": dict(kind="cat"),
    "use_guidance": dict(kind="bool"),
    # FBRRT knobs
    "alpha":      dict(kind="cont", fwd=lambda v: float(v), inv=lambda z: z,
                       lo=0.0, hi=1.5),
    "branch":     dict(kind="cont", fwd=np.log, inv=np.exp, lo=2, hi=16,
                       integer=True),
    "entropy_lambda": dict(kind="cont", fwd=np.log, inv=np.exp, lo=0.05, hi=5.0),
    "ent_inf":    dict(kind="bool"),
    # categoricals/bools shared with the old study
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


_SMC_COMMON = dict(
    cont=["lr", "n_steps", "mc_samples", "off_policy_frac", "k",
          "guidance_scale"],
    cat=["smc_type", "guidance_source"],
    bool=["use_grad_decay", "use_guidance"],
    cond=["grad_decay", "l", "ema_decay"])

LAYOUT = {
    "off_policy": dict(cont=["lr"], cat=[], bool=["use_grad_decay"],
                       cond=["grad_decay"]),
    "single_seed_mc": dict(
        cont=_SMC_COMMON["cont"], cat=_SMC_COMMON["cat"],
        bool=_SMC_COMMON["bool"] + ["random_t"], cond=_SMC_COMMON["cond"]),
    "single_seed_td_lambda": dict(
        cont=_SMC_COMMON["cont"] + ["lambda_eff"], cat=_SMC_COMMON["cat"],
        bool=_SMC_COMMON["bool"] + ["random_t"], cond=_SMC_COMMON["cond"]),
    "ancestral_mc_td_lambda": dict(
        cont=_SMC_COMMON["cont"] + ["lambda_eff"], cat=_SMC_COMMON["cat"],
        bool=_SMC_COMMON["bool"], cond=_SMC_COMMON["cond"]),
    "fbrrt": dict(
        cont=["lr", "n_steps", "mc_samples", "branch", "alpha",
              "off_policy_frac", "entropy_lambda", "ema_decay"],
        cat=[], bool=["use_grad_decay", "ent_inf"], cond=["grad_decay"]),
    "fbrrt_cv": dict(
        cont=["lr", "n_steps", "mc_samples", "branch", "alpha",
              "off_policy_frac", "entropy_lambda", "ema_decay"],
        cat=[], bool=["use_grad_decay", "ent_inf"], cond=["grad_decay"]),
}
