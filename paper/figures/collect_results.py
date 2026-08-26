#!/usr/bin/env python3
"""Collect per-seed experiment records into ONE canonical results.json.

Every number the paper quotes for the dimensional-scaling study comes from
here, computed from per-seed records rather than transcribed from a report,
so a rewrite cannot drift the numbers.  Paired statistics use the nested-seed
construction: seed s of one arm and seed s of another are the SAME problem
instance, so differences are paired.
"""
import glob, json, math, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(ROOT, "experiments")

SERIES = {   # label -> (results dir, filename prefix)
    "off_policy":   (f"{EXP}/dim_scaling_consth/results", "grid_off_policy"),
    "ssmc_lawv1":   (f"{EXP}/dim_scaling_consth/results", "grid_single_seed_mc"),
    "ssmctd_lawv1": (f"{EXP}/dim_scaling_consth/results", "grid_single_seed_td_lambda"),
    "ssmc_lawv2":   (f"{EXP}/dim_scaling_lawv2/results",  "gridv2_single_seed_mc"),
    "ssmctd_lawv2": (f"{EXP}/dim_scaling_lawv2/results",  "gridv2_single_seed_td_lambda"),
    "ssmc_recipe":  (f"{EXP}/dim_scaling_recipe/results", "recipe_expand_ns60_sub_single_seed_mc"),
}


def load(d, prefix):
    out = {}
    for f in glob.glob(f"{d}/{prefix}_d*.json"):
        dim = int(f.rsplit("_d", 1)[1].split(".")[0])
        rec = json.load(open(f))
        by_seed = {}
        for e in rec["seeds"]:
            v = e.get("frac_closed", float("nan"))
            if v is not None and math.isfinite(v):
                by_seed[e["seed"]] = 100.0 * v
        if by_seed:
            out[dim] = by_seed
    return out


def summarise(by_seed):
    v = np.array(list(by_seed.values()))
    return {"n": len(v), "mean": float(v.mean()),
            "se": float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")}


def paired(a, b):
    """Paired delta a - b over the seeds both arms share."""
    ks = sorted(set(a) & set(b))
    if len(ks) < 3:
        return None
    d = np.array([a[k] - b[k] for k in ks])
    t, p = stats.ttest_rel([a[k] for k in ks], [b[k] for k in ks])
    return {"n_pairs": len(ks), "delta": float(d.mean()),
            "se": float(d.std(ddof=1) / math.sqrt(len(d))), "p": float(p),
            "wins": int((d > 0).sum())}


def main():
    data = {k: load(*v) for k, v in SERIES.items()}
    dims = sorted({d for s in data.values() for d in s})
    out = {"dims": dims, "metric": "frac_closed_pct", "series": {}, "paired_vs_off_policy": {}}
    for name, per_dim in data.items():
        out["series"][name] = {str(d): summarise(per_dim[d]) for d in sorted(per_dim)}
    ctrl = data["off_policy"]
    for name, per_dim in data.items():
        if name == "off_policy":
            continue
        row = {}
        for d in sorted(set(per_dim) & set(ctrl)):
            r = paired(per_dim[d], ctrl[d])
            if r:
                row[str(d)] = r
        out["paired_vs_off_policy"][name] = row
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}")
    for name in out["series"]:
        s = out["series"][name]
        print(f"  {name:14s} " + " ".join(
            f"d{d}:{s[d]['mean']:.1f}(n={s[d]['n']})" for d in sorted(s, key=int)))


if __name__ == "__main__":
    main()
