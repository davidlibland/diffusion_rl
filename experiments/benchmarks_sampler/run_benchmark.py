#!/usr/bin/env python3
"""Train a value learner on a sampler benchmark and report literature metrics.

Reuses dim_scaling_consth.build (driftless base + our reward is a drop-in
problem dict from problem_sampler).  After training we roll out the CONTROLLED
process X_0=0, dX = 2a*grad V_theta dt + sqrt(2a) dW to get terminal samples,
then report: log-Z error (learned V(0,0) vs true log Z), mode coverage,
energy-W1 to ground-truth samples, and (GMM only) the learned-V error vs the
analytic oracle.

Usage: python run_benchmark.py --problem gmm40 --method off_policy --steps 8000
Env:   BM_RESULTS_DIR, BM_GEN=20000 (rollout samples), BM_NSTEP=100
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import lightning as L

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "dim_scaling_bs4"))
sys.path.insert(0, os.path.join(EXP, "dim_scaling_consth"))
sys.path.insert(0, HERE)

sys.path.insert(0, os.path.join(EXP, "dim_scaling_lawv2"))
import sweep_consth as sw  # noqa: E402
from fit_consth import hparams_for_dim  # noqa: E402
import problem_sampler as ps  # noqa: E402

RESULTS = os.environ.get("BM_RESULTS_DIR", os.path.join(HERE, "results"))
N_GEN = int(os.environ.get("BM_GEN", 20000))
N_STEP = int(os.environ.get("BM_NSTEP", 100))
DEVICE = sw.DEVICE
A = ps.A


def make_problem(name, seed):
    if name == "gmm40":
        return ps.make_gmm40()
    if name == "manywell":
        return ps.make_manywell(32)
    raise ValueError(name)


@torch.no_grad()
def _value(vm, x, t):
    return vm(x, t.reshape(-1))


def _sliced_w2(gen, ref, n_proj=200, seed=0):
    """Sliced 2-Wasserstein: mean over random 1-D projections of the sorted-L2
    distance. Bounded by the coordinate scale (unlike energy-W1, which the
    sharp modes dominate)."""
    g = torch.Generator().manual_seed(seed)
    d = gen.shape[1]
    dirs = torch.randn(n_proj, d, generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    pg = torch.sort((gen @ dirs.T), dim=0).values           # (n, n_proj)
    pr = torch.sort((ref @ dirs.T), dim=0).values
    m = min(pg.shape[0], pr.shape[0])
    return float(((pg[:m] - pr[:m]) ** 2).mean().sqrt())


def rollout(vm, dim, n=N_GEN, n_step=N_STEP):
    """Euler-Maruyama on the controlled SDE from X_0=0. u = 2a grad V."""
    x = torch.zeros(n, dim, device=DEVICE)
    dt = 1.0 / n_step
    sig = math.sqrt(2 * A * dt)
    for i in range(n_step):
        t = torch.full((n, 1), i * dt, device=DEVICE)
        xr = x.detach().requires_grad_(True)
        with torch.enable_grad():
            V = vm(xr, t.reshape(-1)).sum()
            gradV = torch.autograd.grad(V, xr)[0]
        x = x + (2 * A * gradV) * dt + sig * torch.randn_like(x)
    return x.detach()


def evaluate(prob, vm, dim):
    out = {}
    # log-Z estimate: V(0,0) = log Z by construction
    v00 = float(_value(vm, torch.zeros(1, dim, device=DEVICE),
                       torch.zeros(1, device=DEVICE)).item())
    out["logZ_hat"] = v00
    out["logZ_true"] = float(prob["log_Z"])
    out["logZ_err"] = abs(v00 - float(prob["log_Z"]))

    gen = rollout(vm, dim).cpu()
    finite = torch.isfinite(gen).all(1)
    gen = gen[finite]
    out["frac_finite"] = float(finite.float().mean())
    ref = torch.as_tensor(prob["true_sample"](gen.shape[0])).float()
    out["mode_cov"] = float(prob["mode_coverage"](gen.numpy()))
    out["mode_cov_ref"] = float(prob["mode_coverage"](ref.numpy()))
    out["energy_w1"] = float(prob["energy_w1"](gen.numpy(), ref.numpy()))
    out["sliced_w2"] = _sliced_w2(gen, ref)          # robust sample distance
    out["gen_energy_mean"] = float(prob["energy_fn"](gen).mean())
    out["ref_energy_mean"] = float(prob["energy_fn"](ref).mean())

    # GMM: learned V vs analytic oracle on a grid of (y,t)
    if prob["anal_fn"] is not None:
        yy = torch.randn(4000, dim, device=DEVICE) * 0.6
        tt = torch.rand(4000, device=DEVICE) * 0.98
        va = prob["anal_fn"](yy, tt)
        vl = _value(vm, yy, tt)
        out["V_rmse_vs_oracle"] = float(((va - vl) ** 2).mean().sqrt())
        out["V_bias_vs_oracle"] = float((vl - va).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=["gmm40", "manywell"])
    ap.add_argument("--method", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recipe", action="store_true",
                    help="use law-v2 build (expansion + guided proposals)")
    ap.add_argument("--guidance", type=float, default=None,
                    help="override guidance_scale (ema source) on the v1 build")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    prob = make_problem(args.problem, args.seed)
    dim = prob["diag"]["d"]
    hidden = min(256, max(64, 32 * dim))
    L.seed_everything(args.seed, workers=True)
    if args.recipe:
        import sweep_lawv2 as sw2
        from fit_lawv2 import hparams_for_dim as hp2
        params = hp2(args.method, dim)
        model, vm, ds, loader = sw2.build(args.method, params, prob, dim,
                                          hidden, args.seed)
        tag = args.method + "_recipe"
    else:
        params = hparams_for_dim(args.method, dim)
        tag = args.method
        if args.guidance is not None:
            params["use_guidance"] = True
            params["guidance_scale"] = args.guidance
            params["guidance_source"] = "ema"
            params.setdefault("ema_decay", 0.99)
            tag = f"{args.method}_g{args.guidance:g}"
        model, vm, ds, loader = sw.build(args.method, params, prob, dim, hidden,
                                         args.seed)
    t0 = time.time()
    tr = L.Trainer(max_steps=args.steps, limit_val_batches=0, logger=False,
                   enable_checkpointing=False, enable_progress_bar=False,
                   num_sanity_val_steps=0)
    err = None
    try:
        tr.fit(model, loader)
    except (RuntimeError, ValueError) as e:
        err = f"{type(e).__name__}: {str(e)[:100]}"
    vm = model.value_module.to(DEVICE).eval()
    metrics = evaluate(prob, vm, dim)
    rec = {"problem": args.problem, "method": args.method, "steps": args.steps,
           "seed": args.seed, "dim": dim, "n_modes": prob["n_modes"],
           "train_min": (time.time() - t0) / 60,
           "nonfinite_skips": int(getattr(model, "_nonfinite_count_total", 0)),
           "error": err, **metrics}
    rec["method"] = tag
    out_path = f"{RESULTS}/{args.problem}_{tag}_s{args.seed}.json"
    json.dump(rec, open(out_path, "w"), indent=1)
    print(f"=== {args.problem} / {args.method} (seed {args.seed}, {args.steps} steps) ===")
    print(f"  log Z: hat={rec['logZ_hat']:.2f} true={rec['logZ_true']:.2f} "
          f"err={rec['logZ_err']:.2f}")
    print(f"  mode_cov={rec['mode_cov']:.3f} (ref {rec['mode_cov_ref']:.3f})  "
          f"energy_W1={rec['energy_w1']:.3f}  finite={rec['frac_finite']:.3f}")
    if "V_rmse_vs_oracle" in rec:
        print(f"  V vs oracle: RMSE={rec['V_rmse_vs_oracle']:.3f} "
              f"bias={rec['V_bias_vs_oracle']:.3f}")
    if err:
        print(f"  ERROR: {err}")
    print(f"  ({rec['train_min']:.1f} min)  -> {out_path}")


if __name__ == "__main__":
    main()
