"""Fit hparam(dimension) selectors per algorithm from the top-3 Optuna configs.

Default pipeline (approved):
  data      : top-3 per setting (moons d=2 + GMM d=2,8,32,128), rank-weighted 3/2/1
              within each setting (scale-free; LCB not comparable across problems).
  predictor : log(d).
  transforms: per hparam_transforms.SPEC (log / logit / linear), matching Optuna.
  per cont hparam: robust fit (Theil-Sen, weights via replication) with
              leave-one-dimension-out (LOO) model selection between a CONSTANT
              (weighted median) and a SLOPE; slope kept only if it beats the
              constant out-of-sample by >10%.  noise flag = constant chosen.
  categoricals/conditionals: rank-weighted mode / majority / median constant.

Outputs: fitted_models.json, hparam_fit.md, hparam_fit_<method>.png.
"""

import json
import os

import numpy as np
from scipy.stats import theilslopes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hparam_transforms import SPEC, LAYOUT, apply_inv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MOONS = os.path.join(ROOT, "experiments", "bs4_moons")
RES = os.path.join(HERE, "results")
GMM_DIMS = [2, 8, 32, 128]
METHODS = list(LAYOUT)
SLOPE_GAIN = 0.10  # slope must cut LOO error by >10% to be kept


# ── assemble rank-weighted top-3 records per method ────────────────────────
def moons_top(method):
    if method == "off_policy":
        d = json.load(open(f"{MOONS}/optuna_offpolicy_pipeline_results.json"))
        return [e["params"] for e in d["sweep_top3"]]
    if method in ("single_seed_mc", "single_seed_td_lambda"):
        d = json.load(open(f"{MOONS}/optuna_top_configs.json"))
        return [e["params"] for e in d[method]]
    if method == "ancestral_mc_td_lambda":
        d = json.load(open(f"{MOONS}/optuna_other_top.json"))
        return [e["params"] for e in d
                if e["params"].get("method") == method]
    return []


def records(method):
    """List of (dim, weight, params), weight = within-setting rank (3/2/1)."""
    recs = []
    for i, params in enumerate(moons_top(method)[:3]):
        recs.append((2, 3 - i, params))
    for dim in GMM_DIMS:
        p = f"{RES}/{method}_d{dim}.json"
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


def fit_cont(hp, recs):
    """Records active for hp -> chosen model dict."""
    pts = [(np.log(dim), SPEC[hp]["fwd"](p[hp]), w)
           for (dim, w, p) in recs if hp in p and p.get(hp) is not None]
    if not pts:
        return None
    xs, ys, ws = map(list, zip(*pts))
    dims = sorted(set(xs))
    const_all = _wmedian(ys, ws)
    out = {"kind": "cont", "n_points": len(pts), "n_dims": len(dims),
           "transform": "fwd", "const": const_all}

    if len(dims) < 3:                      # not enough distinct dims to trust a slope
        out.update(model="const", noise=True, slope=None,
                   loo_const=None, loo_slope=None)
        return out

    # LOO over distinct dims
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
        votes[v] = votes.get(v, 0) + w
    if not votes:
        return None
    mode = max(votes, key=votes.get)
    return {"kind": "cat", "mode": mode, "votes": votes}


def fit_bool(hp, recs):
    tw = fw = 0.0
    for dim, w, p in recs:
        v = p.get(hp)
        if v is None:
            continue
        if v:
            tw += w
        else:
            fw += w
    if tw + fw == 0:
        return None
    p_true = tw / (tw + fw)
    return {"kind": "bool", "p_true": p_true, "value": bool(p_true >= 0.5)}


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
    for hp in lay["cond"]:           # constants (grad_decay value, l, ema_decay)
        m = fit_cont(hp, recs)       # reuse cont fitter but force constant
        if m:
            m["model"] = "const"; m["slope"] = None
            models[hp] = m
    return models, recs


def predict(method, models, d):
    """Return a params dict ready for sweep.build(method, params, ...)."""
    z = np.log(d)
    p = {"off_policy_frac": 0.0}

    def cont_val(hp):
        m = models[hp]
        zval = m["slope"][0] + m["slope"][1] * z if m["model"] == "slope" else m["const"]
        return apply_inv(hp, zval)

    p["lr"] = cont_val("lr")
    # grad_decay toggle + value
    if "use_grad_decay" in models and models["use_grad_decay"]["value"] and "grad_decay" in models:
        p["use_grad_decay"] = True
        p["grad_decay"] = cont_val("grad_decay")
    else:
        p["use_grad_decay"] = False

    if method == "off_policy":
        return p

    for hp in ("n_steps", "mc_samples", "off_policy_frac", "k", "lambda_eff"):
        if hp in models:
            p[hp] = cont_val(hp)
    smc = models["smc_type"]["mode"]
    p["smc_type"] = smc
    if smc == "kV_plus_ltr" and "l" in models:
        p["l"] = cont_val("l")
    if smc == "k_Vema" and "ema_decay" in models:
        p["ema_decay"] = cont_val("ema_decay")
    if "random_t" in models:
        p["random_t"] = models["random_t"]["value"]
    return p


# ── run ────────────────────────────────────────────────────────────────────
all_models = {}
md = ["# Hyperparameter-vs-dimension fits (BS=4, quad loss)", "",
      "Per algorithm, each hyperparameter is fit as a function of **log(dimension)** "
      "from the **top-3** Optuna configs at moons d=2 + GMM d∈{2,8,32,128}, "
      "rank-weighted 3/2/1 within each setting.  Continuous hparams use the "
      "Optuna sampling-scale transform (log / logit / linear) and a robust "
      "Theil-Sen fit; a **slope** is kept over a **constant** only if it lowers "
      "leave-one-dimension-out error by >10% (else the hparam is flagged *noise* "
      "and set to a dimension-independent constant).  Categoricals/booleans use "
      "the rank-weighted mode/majority.", ""]
PRED_DIMS = [2, 8, 16, 32, 64, 128]

for method in METHODS:
    models, recs = fit_method(method)
    all_models[method] = models
    md.append(f"## {method}")
    md.append("")
    md.append("| hparam | scale | model | noise? | LOO const→slope | "
              + " | ".join(f"d={d}" for d in PRED_DIMS) + " |")
    md.append("|" + "---|" * (5 + len(PRED_DIMS)))
    # build predictions per dim once
    preds = {d: predict(method, models, d) for d in PRED_DIMS}
    order = (LAYOUT[method]["cont"] + LAYOUT[method]["cat"]
             + LAYOUT[method]["bool"] + LAYOUT[method]["cond"])
    for hp in order:
        if hp not in models:
            continue
        m = models[hp]
        scale = "log" if SPEC[hp].get("fwd") in (np.log,) else SPEC[hp]["kind"]
        if m["kind"] == "cont":
            model = m["model"]
            noise = "noise" if m.get("noise") else "trend"
            loo = (f"{m['loo_const']:.2f}→{m['loo_slope']:.2f}"
                   if m.get("loo_slope") is not None else "—(const)")
        else:
            model = m["kind"]; noise = "—"; loo = "—"
        # value strings per pred dim
        def vstr(d):
            v = preds[d].get(hp)
            if v is None:
                return "—"
            if hp in ("lr", "k", "l", "grad_decay"):
                return f"{v:.2e}"
            if hp in ("ema_decay", "off_policy_frac", "lambda_eff"):
                return f"{v:.3f}"
            if hp in ("n_steps", "mc_samples"):
                return str(int(v))
            if hp == "use_grad_decay":
                return "on" if preds[d].get("use_grad_decay") else "off"
            return str(v)
        # grad_decay row only meaningful if toggle on
        vals = " | ".join(vstr(d) for d in PRED_DIMS)
        md.append(f"| {hp} | {scale} | {model} | {noise} | {loo} | {vals} |")
    md.append("")

    # ── plot continuous hparams: transformed scatter + fit ──
    conts = [hp for hp in LAYOUT[method]["cont"] if hp in models]
    if conts:
        ncol = min(3, len(conts)); nrow = (len(conts) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow),
                                 squeeze=False)
        for ax in axes.flat[len(conts):]:
            ax.axis("off")
        for ax, hp in zip(axes.flat, conts):
            pts = [(np.log(dim), SPEC[hp]["fwd"](p[hp]), w)
                   for (dim, w, p) in recs if hp in p and p.get(hp) is not None]
            xs, ys, ws = map(np.array, zip(*pts))
            ax.scatter(xs, ys, s=20 + 25 * ws, alpha=0.6, color="#1f77b4")
            xx = np.log(np.array(PRED_DIMS, float))
            m = models[hp]
            if m["model"] == "slope":
                yy = m["slope"][0] + m["slope"][1] * xx
            else:
                yy = np.full_like(xx, m["const"])
            ax.plot(xx, yy, "r-", lw=2)
            ax.set_xticks(np.log(GMM_DIMS)); ax.set_xticklabels(GMM_DIMS)
            ax.set_title(f"{hp}  [{m['model']}{'*noise' if m.get('noise') else ''}]",
                         fontsize=10)
            ax.set_xlabel("dimension"); ax.set_ylabel(f"{hp} (fit scale)")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"{method}: hparam vs dimension (top-3, transformed scale)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{HERE}/hparam_fit_{method}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

json.dump(all_models, open(f"{HERE}/fitted_models.json", "w"), indent=2, default=str)
open(f"{HERE}/hparam_fit.md", "w").write("\n".join(md))
print("Wrote fitted_models.json, hparam_fit.md, hparam_fit_<method>.png\n")
print("\n".join(md))
