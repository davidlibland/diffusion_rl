"""Hyperparameter transforms + layout for the law-v2 dim-scaling study.

Extends ../dim_scaling_consth/hparam_transforms_consth.py for the two SMC
methods that reached/beat off-policy at high d, adding the recipe knobs to
the swept space so the laws can move them with dimension:

  - n_steps     : now LOG scale, range 12..120 (the old 10..60 linear range
                  plus the consth anchors' d<=128 blind spot produced the
                  n_steps=19 lock-in; the recipe grid showed 60 wins at high d)
  - expand_frac : fraction of eligible rows given one backward-noising
                  expansion sample (0 = law config, 1 = probe-3 recipe)
  - epoch_rows  : cap on rows per generated epoch (uniform subsample after
                  expansion); controls the dataset regeneration cadence that
                  the probe-3 *_sub arms showed matters (freshness vs
                  implicit-target smoothing)
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
    "n_steps":    dict(kind="cont", fwd=np.log, inv=np.exp, lo=12, hi=120,
                       integer=True),
    "expand_frac": dict(kind="cont",
                        fwd=lambda v: _logit(np.clip(v, 1e-3, 1 - 1e-3)),
                        inv=_sigmoid, lo=0.0, hi=1.0),
    "epoch_rows": dict(kind="cont", fwd=np.log, inv=np.exp, lo=768, hi=16384,
                       integer=True),
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
    "guidance_scale": dict(kind="cont", fwd=np.log, inv=np.exp, lo=0.05, hi=1.5),
    "guidance_source": dict(kind="cat"),
    "use_guidance": dict(kind="bool"),
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
          "guidance_scale", "expand_frac", "epoch_rows"],
    cat=["smc_type", "guidance_source"],
    bool=["use_grad_decay", "use_guidance"],
    cond=["grad_decay", "l", "ema_decay"])

LAYOUT = {
    "single_seed_mc": dict(
        cont=_SMC_COMMON["cont"], cat=_SMC_COMMON["cat"],
        bool=_SMC_COMMON["bool"] + ["random_t"], cond=_SMC_COMMON["cond"]),
    "single_seed_td_lambda": dict(
        cont=_SMC_COMMON["cont"] + ["lambda_eff"], cat=_SMC_COMMON["cat"],
        bool=_SMC_COMMON["bool"] + ["random_t"], cond=_SMC_COMMON["cond"]),
}
