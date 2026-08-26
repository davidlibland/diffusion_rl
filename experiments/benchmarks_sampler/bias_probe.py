#!/usr/bin/env python3
"""Is the GMM-40 log-Z result inherited from the value head's initialisation?

GMM-40 has true log Z = 0 and the harness initialises the value head's bias at
bias_val = 0.0.  The head therefore starts AT the answer, so "recovers log Z to
0.08 nats" could be measuring how little the network moved rather than what it
learned.  This probe offsets the head initialisation by a known amount and asks
whether V(0,0) still converges to 0.

  offset  0 : the original configuration (control)
  offset +/-4 : head starts 4 nats away from the truth

If the learned V(0,0) returns to ~0 from all three, the value is genuinely
learned.  If it stays near its initialisation, the headline number is an
artifact of initialisation and must be retracted.

Usage: python bias_probe.py --offset 4 --seed 0 [--steps 8000]
"""
import argparse, json, os, sys, time
import torch
import lightning as L

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "dim_scaling_bs4"))
sys.path.insert(0, os.path.join(EXP, "dim_scaling_consth"))
sys.path.insert(0, os.path.join(EXP, "dim_scaling_lawv2"))
sys.path.insert(0, HERE)
os.environ.setdefault("OPT_SKIP_NONFINITE", "1")

import sweep_consth as sw                                     # noqa: E402
from fit_consth import hparams_for_dim                         # noqa: E402
# NOTE: fit_consth (law-v1) is what run_benchmark.py uses for the reported
# GMM-40 numbers; fit_lawv2 is a DIFFERENT configuration.  Match the target.
import run_benchmark as RB                                     # noqa: E402
from run_benchmark import make_problem                         # noqa: E402

RESULTS = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--method", default="single_seed_mc")
    a = ap.parse_args()

    prob = make_problem("gmm40", a.seed)
    dim = prob["diag"]["d"]
    hidden = min(256, max(64, 32 * dim))
    true_logZ = float(prob["log_Z"])
    # Offset BOTH the head bias and the loss shift: they are the same constant
    # in the harness, and the point is to move the network's starting value.
    prob = dict(prob)
    prob["bias_val"] = prob["bias_val"] + a.offset

    L.seed_everything(a.seed, workers=True)
    params = hparams_for_dim(a.method, dim)
    model, vm, ds, loader = sw.build(a.method, params, prob, dim, hidden, a.seed)

    v0_init = float(vm(torch.zeros(1, dim), torch.zeros(1)).item())
    t0 = time.time()
    tr = L.Trainer(max_steps=a.steps, limit_val_batches=0, logger=False,
                   enable_checkpointing=False, enable_progress_bar=False,
                   num_sanity_val_steps=0)
    err = None
    try:
        tr.fit(model, loader)
    except (RuntimeError, ValueError) as e:
        err = f"{type(e).__name__}: {str(e)[:90]}"
    vm = model.value_module.to(RB.DEVICE).eval()
    with torch.no_grad():
        v00 = float(vm(torch.zeros(1, dim, device=RB.DEVICE),
                       torch.zeros(1, device=RB.DEVICE)).item())
    rec = {"offset": a.offset, "seed": a.seed, "steps": a.steps,
           "method": a.method, "v00_init": v0_init, "logZ_hat": v00,
           "logZ_true": true_logZ, "logZ_err": abs(v00 - true_logZ),
           "moved": v00 - v0_init, "train_min": (time.time()-t0)/60,
           "error": err}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(rec, open(f"{RESULTS}/biasprobe_off{a.offset:+g}_s{a.seed}.json","w"), indent=1)
    print(f"offset={a.offset:+.1f} seed={a.seed}: V(0,0) init={v0_init:+.3f} "
          f"-> final={v00:+.3f}  (true {true_logZ:+.1f}, err {abs(v00-true_logZ):.3f}, "
          f"moved {v00-v0_init:+.3f})  [{(time.time()-t0)/60:.1f}m] {err or ''}", flush=True)


if __name__ == "__main__":
    main()
