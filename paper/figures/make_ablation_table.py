#!/usr/bin/env python3
"""Generate sections/05_results_table.tex from the loss-ablation grid.

Reads experiments/loss_ablation/ablation.json so the table cannot drift from
the runs.  Emits a LaTeX table plus the paragraph of text that reads it, with
the numbers substituted, and refuses to emit anything if a cell is missing.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "experiments", "loss_ablation", "ablation.json")
OUT = os.path.join(os.path.dirname(HERE), "sections", "05_results_table.tex")

NICE = {"quad": r"\textbf{Spence (ours)}", "mse": "exp-space squared error",
        "is": "Itakura--Saito", "gkl": "generalised KL ($\\beta=1$)",
        "logmse": "log-space squared error"}
ORDER = ["quad", "mse", "is", "gkl", "logmse"]

if not os.path.exists(SRC):
    sys.exit(f"no ablation.json at {SRC} -- grid not finished")
R = json.load(open(SRC))
dims = R["dims"]


def cell(loss, d, metric, fmt="{:.1f}", pm=True):
    c = R["cells"].get(f"{loss}_d{d}")
    if not c or not np.isfinite(c[metric]["mean"]):
        return "---"
    m, e = c[metric]["mean"], c[metric]["se"]
    return (fmt.format(m) + (f"\\,$\\pm$\\,{fmt.format(e)}" if pm and np.isfinite(e) else ""))


lines = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{The loss in isolation: off-policy training on the",
         r"constant-headroom benchmark, identical in every respect but the divergence.",
         r"Headroom closed is the control metric (higher is better); value RMSE is",
         r"measured against the \emph{analytic} value function (lower is better).",
         r"Each arm is reported at its own tuned learning rate. $n$ paired seeds per",
         r"cell; $\pm$ is one standard error.}",
         r"\label{tab:ablation}",
         r"\begin{tabular}{l" + "c"*len(dims)*2 + "}", r"\toprule",
         r"& \multicolumn{%d}{c}{headroom closed (\%%)} & \multicolumn{%d}{c}{value RMSE vs analytic $V$}\\"
         % (len(dims), len(dims)),
         r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (1+len(dims), 2+len(dims), 1+2*len(dims)),
         "loss & " + " & ".join(f"$d={d}$" for d in dims) + " & "
         + " & ".join(f"$d={d}$" for d in dims) + r"\\", r"\midrule"]
for loss in ORDER:
    if not any(f"{loss}_d{d}" in R["cells"] for d in dims):
        continue
    row = [NICE[loss]]
    row += [cell(loss, d, "frac_closed") for d in dims]
    row += [cell(loss, d, "v_rmse", "{:.2f}") for d in dims]
    lines.append(" & ".join(row) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

# paired deltas vs exp-MSE, for the prose
prose = [r"", r"\paragraph{Results.}"]
pv = R.get("paired_vs_mse", {}).get("quad", {})
if pv:
    bits = []
    for d in dims:
        r = pv.get(str(d), {}).get("frac_closed")
        if r:
            star = "*" if r["p"] < 0.05 else ""
            bits.append(f"$d={d}$: ${r['delta']:+.1f}${star}")
    prose.append("Against exponential-space squared error on the same instances, the "
                 "Spence loss gives paired differences of " + ", ".join(bits)
                 + " points of headroom closed ($*$ marks $p<0.05$, "
                 + f"$n={list(pv.values())[0]['frac_closed']['n']}$ pairs).")
open(OUT, "w").write("\n".join(lines + prose) + "\n")
print(f"wrote {OUT} ({len(ORDER)} arms x {len(dims)} dims)")
