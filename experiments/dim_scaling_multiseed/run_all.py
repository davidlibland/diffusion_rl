#!/usr/bin/env python3
"""Concurrent driver: run every (method, dimension) cell of the multi-seed study.

Each cell (run_cell.py) is a subprocess running all MS_N_SEEDS seeds in-process
(amortising the torch import); cells run MS_CONCURRENCY at a time on the one GPU.
Cells already fully done (summary with n == MS_N_SEEDS) are skipped; partially
done cells resume their remaining seeds.  Per-cell stdout -> results/log_<cell>.out.

Env: MS_DIMS="2,8,16,32,64,128,256,512"  MS_METHODS=...  MS_CONCURRENCY=3
     MS_N_SEEDS=30  (+ per-cell budget vars consumed by run_cell.py)
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CELL = os.path.join(HERE, "run_cell.py")

DIMS = [int(x) for x in os.environ.get("MS_DIMS", "2,8,16,32,64,128,256,512").split(",")]
METHODS = os.environ.get(
    "MS_METHODS",
    "off_policy,single_seed_mc,single_seed_td_lambda,ancestral_mc_td_lambda",
).split(",")
CONC = int(os.environ.get("MS_CONCURRENCY", 3))
N_SEEDS = int(os.environ.get("MS_N_SEEDS", 30))


def is_done(method, dim):
    p = f"{RESULTS}/{method}_d{dim}.json"
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p))
        return d.get("summary", {}).get("n", 0) >= N_SEEDS
    except Exception:
        return False


cells = [(d, m) for d in DIMS for m in METHODS]
pending = [(d, m) for (d, m) in cells if not is_done(m, d)]
print(f"Driver: {len(cells)} cells, {len(cells)-len(pending)} done, "
      f"{len(pending)} to run, concurrency={CONC}, seeds={N_SEEDS}", flush=True)

t0 = time.time()
running = {}
done, failed = [], []


def launch(d, m):
    log = open(f"{RESULTS}/log_{m}_d{d}.out", "a")
    log.write(f"\n\n===== launch {time.strftime('%H:%M:%S')} =====\n"); log.flush()
    p = subprocess.Popen([sys.executable, CELL, "--method", m, "--dim", str(d)],
                         stdout=log, stderr=subprocess.STDOUT)
    running[p] = (d, m, log)
    print(f"  launch {m}_d{d}  ({len(running)} running, {len(pending)} pending, "
          f"{(time.time()-t0)/60:.1f}m)", flush=True)


while pending or running:
    while pending and len(running) < CONC:
        d, m = pending.pop(0)
        launch(d, m); time.sleep(8)
    time.sleep(20)
    for p in list(running):
        if p.poll() is None:
            continue
        d, m, log = running.pop(p); log.close()
        ok = is_done(m, d)
        (done if ok else failed).append(f"{m}_d{d}")
        summ = ""
        try:
            js = json.load(open(f"{RESULTS}/{m}_d{d}.json")).get("summary", {})
            summ = f"regret {js.get('regret_mean'):.3f}±{js.get('regret_sem'):.3f}"
        except Exception:
            pass
        print(f"  finish {m}_d{d} ok={ok} {summ}  ({(time.time()-t0)/60:.1f}m)",
              flush=True)

print(f"\nDriver done in {(time.time()-t0)/60:.1f} min. "
      f"completed={len(done)} failed={len(failed)}", flush=True)
if failed:
    print("FAILED/incomplete: " + ", ".join(failed), flush=True)
try:
    subprocess.run([sys.executable, os.path.join(HERE, "summarize.py")])
except Exception as e:  # noqa: BLE001
    print(f"(summarize skipped: {e})", flush=True)
