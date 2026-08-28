#!/usr/bin/env python3
"""Table: what gradient clipping does to each loss, at a threshold that binds
about a tenth of the time.

Answers three questions the text needs: how often the clip actually binds at
convergence, how much each loss gains from it, and what it costs in bias.
Emitted from the run records, like every other number in the paper.
"""
import glob, json, math, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                   "diffusion_rl", "experiments", "loss_ablation", "results")
if not os.path.isdir(RES):
    RES = os.path.join(os.path.dirname(HERE), "..", "experiments",
                       "loss_ablation", "results")
RES = os.path.abspath(RES)
OUT = os.path.join(os.path.dirname(HERE), "sections", "05_clip_table.tex")

NICE = {"quad": r"\textbf{Spence (ours)}", "mse": "exp-space squared error",
        "is": "Itakura--Saito", "gkl": r"generalised KL ($\beta=1$)",
        "logmse": "log-space squared error"}
ORDER = ["quad", "mse", "is", "gkl", "logmse"]
DIMS = [2, 8, 32, 128]


def load(tag):
    o = {}
    for f in glob.glob(f"{RES}/{tag}_*_d*.json"):
        r = json.load(open(f))
        o[(r["loss"], r["dim"])] = {
            e["seed"]: e for e in r["seeds"]
            if e.get("frac_closed") is not None and math.isfinite(e["frac_closed"])}
    return o


U, K = load("grid"), load("clipp90")


def delta(a, d, m):
    x_, y_ = K.get((a, d), {}), U.get((a, d), {})
    ks = sorted(set(x_) & set(y_))
    if len(ks) < 3:
        return None
    x = np.array([x_[s][m] for s in ks], float)
    y = np.array([y_[s][m] for s in ks], float)
    if m == "frac_closed":
        x, y = x * 100, y * 100
    dd = x - y
    _, p = stats.ttest_rel(x, y)
    return dd.mean(), p


def rate(a, d):
    v = [e.get("clip_rate_tail") for e in K.get((a, d), {}).values()
         if e.get("clip_rate_tail") is not None]
    return float(np.mean(v)) if v else None


def cell(a, d, m, fmt="{:+.1f}"):
    r = delta(a, d, m)
    if r is None:
        return "---"
    s = fmt.format(r[0])
    return r"\textbf{%s}" % s if r[1] < 0.05 else s


L = [r"\begin{table}[t]", r"\centering\small",
     r"\caption{Effect of gradient clipping, at a per-arm threshold set to each",
     r"loss's own $90$th-percentile gradient norm. \emph{Clip rate} is the",
     r"fraction of steps on which the clip engages over the final fifth of",
     r"training; the remaining rows are paired differences (clipped $-$",
     r"unclipped) over the same $n=12$ instances, \textbf{bold} for $p<0.05$.",
     r"The threshold is calibrated identically for every arm, so the spread in",
     r"clip rate is a property of the losses, not of the protocol.}",
     r"\label{tab:clip}",
     r"\begin{tabular}{llcccc}", r"\toprule",
     r"loss & quantity & $d=2$ & $d=8$ & $d=32$ & $d=128$\\", r"\midrule"]
for k, a in enumerate(ORDER):
    rs = [rate(a, d) for d in DIMS]
    L.append(" & ".join([r"\multirow{4}{*}{%s}" % NICE[a], "clip rate"]
             + [f"{r:.2f}" if r is not None else "---" for r in rs]) + r"\\")
    L.append(" & ".join(["", r"$\Delta$ headroom (pts)"]
             + [cell(a, d, "frac_closed") for d in DIMS]) + r"\\")
    L.append(" & ".join(["", r"$\Delta$ value RMSE"]
             + [cell(a, d, "v_rmse", "{:+.2f}") for d in DIMS]) + r"\\")
    L.append(" & ".join(["", r"$\Delta$ signed bias"]
             + [cell(a, d, "v_bias", "{:+.2f}") for d in DIMS]) + r"\\")
    L.append(r"\midrule" if k < len(ORDER) - 1 else "")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT}")
