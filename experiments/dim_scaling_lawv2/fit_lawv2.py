#!/usr/bin/env python3
"""Fit law-v2 hparam(dimension) laws from the 4-anchor sweeps and serve them.

Same machinery as ../dim_scaling_consth/fit_consth.py (Theil-Sen on log(d) in
the Optuna sampling scale, LOO const-vs-slope with a 10% bias toward the
constant, rank-weighted top-3 records per anchor), with two changes:
  - anchors now d in {2, 16, 128, 512} (the d=512 anchor removes the high-d
    blind spot that produced the old n_steps=19 law);
  - the swept space includes the recipe knobs n_steps / expand_frac /
    epoch_rows (see hparam_transforms_lawv2.py).

Run `python fit_lawv2.py` to (re)fit -> fitted_models_lawv2.json +
hparam_fit_lawv2.md; import hparams_for_dim(method, d) to use the laws.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import theilslopes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hparam_transforms_lawv2 import LAYOUT, SPEC, apply_inv  # noqa: E402

RES = os.environ.get("DSC_RESULTS_DIR", os.path.join(HERE, "results"))
FITTED = os.path.join(HERE, "fitted_models_lawv2.json")
ANCHOR_DIMS = [2, 16, 128, 512]
METHODS = list(LAYOUT)
SLOPE_GAIN = 0.10


def records(method):
    recs = []
    for dim in ANCHOR_DIMS:
        p = f"{RES}/anchor_{method}_d{dim}.json"
        if not os.path.exists(p):
            continue
        chosen = json.load(open(p))["chosen"][:3]
        for i, e in enumerate(chosen):
            recs.append((dim, 3 - i, e["params"]))
    return recs


def _replicate(xs, ys, ws):
    X, Y = [], []
    for x, y, w in zip(xs, ys, ws):
        X += [x] * int(w); Y += [y] * int(w)
    return np.array(X, float), np.array(Y, float)


def _wmedian(ys, ws):
    Y = []
    for y, w in zip(ys, ws):
        Y += [y] * int(w)
    return float(np.median(Y))


def fit_cont(hp, recs, force_const=False):
    pts = [(np.log(dim), SPEC[hp]["fwd"](p[hp]), w)
           for (dim, w, p) in recs if hp in p and p.get(hp) is not None]
    if not pts:
        return None
    xs, ys, ws = map(list, zip(*pts))
    dims = sorted(set(xs))
    const_all = _wmedian(ys, ws)
    out = {"kind": "cont", "n_points": len(pts), "n_dims": len(dims),
           "const": const_all}
    if force_const or len(dims) < 3:
        out.update(model="const", noise=True, slope=None,
                   loo_const=None, loo_slope=None)
        return out
    ec = es = wt = 0.0
    for D in dims:
        tr = [(x, y, w) for x, y, w in zip(xs, ys, ws) if x != D]
        te = [(x, y, w) for x, y, w in zip(xs, ys, ws) if x == D]
        txs, tys, tws = map(list, zip(*tr))
        cpred = _wmedian(tys, tws)
        RX, RY = _replicate(txs, tys, tws)
        sl, ic, *_ = theilslopes(RY, RX)
        spred = ic + sl * D
        for x, y, w in te:
            ec += w * (y - cpred) ** 2
            es += w * (y - spred) ** 2
            wt += w
    loo_c, loo_s = ec / wt, es / wt
    RX, RY = _replicate(xs, ys, ws)
    sl, ic, *_ = theilslopes(RY, RX)
    use_slope = loo_s < (1 - SLOPE_GAIN) * loo_c
    out.update(model="slope" if use_slope else "const",
               slope=[float(ic), float(sl)], loo_const=float(loo_c),
               loo_slope=float(loo_s), noise=not use_slope)
    return out


def fit_cat(hp, recs):
    votes = {}
    for dim, w, p in recs:
        v = p.get(hp)
        if v is None:
            continue
        votes[str(v)] = votes.get(str(v), 0) + w
    if not votes:
        return None
    return {"kind": "cat", "mode": max(votes, key=votes.get), "votes": votes}


def fit_bool(hp, recs):
    tw = fw = 0.0
    for dim, w, p in recs:
        v = p.get(hp)
        if v is None:
            continue
        tw, fw = (tw + w, fw) if v else (tw, fw + w)
    if tw + fw == 0:
        return None
    return {"kind": "bool", "p_true": tw / (tw + fw), "value": bool(tw >= fw)}


def fit_method(method):
    recs = records(method)
    lay = LAYOUT[method]
    models = {}
    for hp in lay["cont"]:
        m = fit_cont(hp, recs)
        if m:
            models[hp] = m
    for hp in lay["cat"]:
        m = fit_cat(hp, recs)
        if m:
            models[hp] = m
    for hp in lay["bool"]:
        m = fit_bool(hp, recs)
        if m:
            models[hp] = m
    for hp in lay["cond"]:
        m = fit_cont(hp, recs, force_const=True)
        if m:
            models[hp] = m
    return models, recs


def _cont_value(models, hp, d):
    m = models[hp]
    z = (m["slope"][0] + m["slope"][1] * np.log(d)
         if m["model"] == "slope" and m["slope"] else m["const"])
    return apply_inv(hp, z)


def predict(method, models, d):
    """Params dict ready for sweep_lawv2.build(method, params, ...)."""
    p = {"lr": _cont_value(models, "lr", d)}
    p["use_grad_decay"] = models.get("use_grad_decay", {}).get("value", False)
    if p["use_grad_decay"] and "grad_decay" in models:
        p["grad_decay"] = _cont_value(models, "grad_decay", d)
    for hp in ("n_steps", "expand_frac", "epoch_rows", "mc_samples",
               "off_policy_frac"):
        p[hp] = _cont_value(models, hp, d)
    p["smc_type"] = models["smc_type"]["mode"]
    p["k"] = _cont_value(models, "k", d)
    if p["smc_type"] == "kV_plus_ltr" and "l" in models:
        p["l"] = _cont_value(models, "l", d)
    p["use_guidance"] = models.get("use_guidance", {}).get("value", False)
    if p["use_guidance"]:
        p["guidance_scale"] = _cont_value(models, "guidance_scale", d)
        p["guidance_source"] = models.get(
            "guidance_source", {}).get("mode", "ema")
    if (p["smc_type"] == "k_Vema"
            or (p["use_guidance"] and p.get("guidance_source") == "ema")):
        p["ema_decay"] = (_cont_value(models, "ema_decay", d)
                          if "ema_decay" in models else 0.99)
    p["random_t"] = models.get("random_t", {}).get("value", False)
    if method == "single_seed_td_lambda":
        p["lambda_eff"] = _cont_value(models, "lambda_eff", d)
    return p


_FITTED_CACHE = None


def hparams_for_dim(method, d):
    global _FITTED_CACHE
    if _FITTED_CACHE is None:
        _FITTED_CACHE = json.load(open(FITTED))
    return predict(method, _FITTED_CACHE[method], d)


def main():
    fitted = {}
    lines = ["# Law-v2 hyperparameter-vs-dimension fits (anchors 2/16/128/512)\n"]
    for method in METHODS:
        models, recs = fit_method(method)
        if not recs:
            print(f"{method}: NO RECORDS -- skipping")
            continue
        fitted[method] = models
        lines.append(f"\n## {method}  ({len(recs)} records)\n")
        lines.append("| hparam | model | LOO const→slope | d=2 | d=16 | d=128 | d=512 |")
        lines.append("|---|---|---|---|---|---|---|")
        for hp, m in models.items():
            if m["kind"] == "cont":
                loo = (f"{m['loo_const']:.2f}→{m['loo_slope']:.2f}"
                       if m.get("loo_const") is not None else "—")
                vals = " | ".join(
                    f"{_cont_value(models, hp, d):.4g}" for d in (2, 16, 128, 512))
                lines.append(f"| {hp} | {m['model']} | {loo} | {vals} |")
            elif m["kind"] == "cat":
                lines.append(f"| {hp} | cat | — | {m['mode']} ||||")
            else:
                lines.append(f"| {hp} | bool (p={m['p_true']:.2f}) | — | {m['value']} ||||")
        print(f"{method}: {len(recs)} records")
    json.dump(fitted, open(FITTED, "w"), indent=2)
    open(os.path.join(HERE, "hparam_fit_lawv2.md"), "w").write(
        "\n".join(lines) + "\n")
    print(f"saved {FITTED}")
    for method in fitted:
        for d in (2, 8, 16, 32, 64, 128, 256, 512):
            p = predict(method, fitted[method], d)
            assert "lr" in p and "expand_frac" in p, (method, d)
    print("predict() sanity OK for all methods/dims")


if __name__ == "__main__":
    main()
