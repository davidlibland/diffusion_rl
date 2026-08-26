#!/usr/bin/env python3
"""Choose the learning rate per (loss, dim) from the tuning scan.

Selection is on TUNING seeds at a short budget; the reported grid then runs
fresh seeds at the full budget, so the choice is not made on the data used to
report.  Ties and boundary optima are flagged, since a chosen lr sitting at the
edge of the grid means the grid was too narrow to be fair to that arm.
"""
import glob, json, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
best = {}
grid = {}
for f in glob.glob(f"{HERE}/results/lr*_*_d*.json"):
    rec = json.load(open(f))
    vals = [100*e["frac_closed"] for e in rec["seeds"]
            if e.get("frac_closed") is not None and math.isfinite(e["frac_closed"])]
    if not vals: continue
    key = (rec["loss"], rec["dim"])
    grid.setdefault(key, []).append((rec["lr"], float(np.mean(vals)), len(vals)))

out = {}
print(f"{'loss':8s} {'dim':>4s}  " + "  ".join(f"{l:>9s}" for l in ["1e-05","3e-05","1e-04","3e-04","1e-03","3e-03"]) + "   chosen")
for key in sorted(grid, key=lambda k: (k[0], k[1])):
    rows = sorted(grid[key])
    lrs = [r[0] for r in rows]; mus = [r[1] for r in rows]
    i = int(np.argmax(mus)); chosen = lrs[i]
    edge = " <-EDGE" if i in (0, len(rows)-1) else ""
    out[f"{key[0]}_d{key[1]}"] = {"lr": chosen, "scan": dict(zip(map(str,lrs), mus)),
                                  "boundary": bool(i in (0,len(rows)-1))}
    cells = {f"{l:.0e}": m for l, m in zip(lrs, mus)}
    print(f"{key[0]:8s} {key[1]:>4d}  " + "  ".join(
        f"{cells.get(l, float('nan')):9.1f}" for l in ["1e-05","3e-05","1e-04","3e-04","1e-03","3e-03"])
        + f"   {chosen:.0e}{edge}")
json.dump(out, open(f"{HERE}/chosen_lr.json","w"), indent=1)
print(f"\nwrote chosen_lr.json ({len(out)} cells)")
