#!/usr/bin/env python3
"""Does the on-policy-vs-off-policy result depend on which loss is used?

Section 7.2 compares samplers while holding the loss fixed at Spence; Section 5
compares losses while holding the sampler fixed at off-policy.  Each is
internally valid, but the quantity one holds fixed is the quantity the other
varies, and Section 5 shows the fixed choice is ~10 points suboptimal on
Section 7.2's own metric.  This measures the interaction directly: the same
sampler comparison, run under two losses.

If the sampler delta survives the loss change, the two studies are separable
and Section 7.2's conclusion is safe.  If it does not, they are one experiment
and must be reported as such.
"""
import argparse, gc, json, math, os, sys, time
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass
import lightning as L
from lightning.pytorch.callbacks import Callback

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "dim_scaling_bs4"))
sys.path.insert(0, os.path.join(EXP, "dim_scaling_consth"))
sys.path.insert(0, os.path.join(EXP, "dim_scaling_lawv2"))
os.environ.setdefault("OPT_SKIP_NONFINITE", "1")

RESULTS = os.path.join(HERE, "results")


class VC(Callback):
    def __init__(self): super().__init__(); self.v = []
    def on_validation_end(self, tr, pl):
        m = tr.callback_metrics.get("val_reward_mean")
        if m is not None: self.v.append(float(m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--loss", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=8000)
    a = ap.parse_args()
    os.environ["DSC_LOSS"] = a.loss          # read by sweep_consth at import
    import sweep_consth as sw
    from fit_lawv2 import hparams_for_dim as hp2
    import sweep_lawv2 as sw2
    from problem_consth import make_problem
    from problem import optimal_terminal_and_reward
    assert sw.LOSS == a.loss, f"loss override failed: {sw.LOSS}"

    os.makedirs(RESULTS, exist_ok=True)
    path = f"{RESULTS}/int_{a.method}_{a.loss}_d{a.dim}.json"
    rec = (json.load(open(path)) if os.path.exists(path)
           else {"method": a.method, "loss": a.loss, "dim": a.dim, "seeds": []})
    done = {e["seed"] for e in rec["seeds"]}
    hidden = min(256, max(64, 32 * a.dim))
    for s in range(a.seeds):
        if s in done: continue
        t0 = time.time()
        prob = make_problem(a.dim, seed=s)
        _, e_opt, _ = optimal_terminal_and_reward(
            prob["means"], prob["sigma2"], prob["weights"], prob["c"],
            prob["reward_scale"], a.dim)
        if a.method == "off_policy":
            from fit_consth import hparams_for_dim as hp1
            params = hp1("off_policy", a.dim)
        else:
            params = hp2(a.method, a.dim)
        build = sw2.build if a.method != "off_policy" else sw.build
        model, vm, ds, loader = build(a.method, params, prob, a.dim, hidden, s)
        vc = VC()
        tr = L.Trainer(max_steps=a.steps, val_check_interval=500, callbacks=[vc],
                       logger=False, enable_checkpointing=False,
                       enable_progress_bar=False, num_sanity_val_steps=0)
        err = None
        try: tr.fit(model, loader, val_dataloaders=sw.val_loader)
        except (RuntimeError, ValueError) as e: err = f"{type(e).__name__}: {str(e)[:80]}"
        v = np.array(vc.v, float)
        plateau = (float(np.nanmean(pd.Series(v).rolling(8, min_periods=1).mean().to_numpy()[-8:]))
                   if len(v) else float("nan"))
        out = {"seed": s, "plateau": plateau, "E_base": prob["diag"]["E_base_r"],
               "frac_closed": ((plateau - prob["diag"]["E_base_r"]) / 6.0
                               if math.isfinite(plateau) else float("nan")),
               "error": err}
        rec["seeds"].append(out); json.dump(rec, open(path, "w"), indent=1)
        print(f"  {a.method}/{a.loss} d{a.dim} seed {s}: "
              f"closed={100*out['frac_closed']:.1f}% ({(time.time()-t0)/60:.1f}m)", flush=True)
        del model, vm, ds, loader, tr; gc.collect(); sw.empty_cache()


if __name__ == "__main__":
    main()
