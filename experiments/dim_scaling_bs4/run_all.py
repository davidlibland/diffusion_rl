#!/usr/bin/env python3
"""Driver: run the bs4-style Optuna pipeline for every (method, dimension) cell.

Each cell runs as a fresh subprocess (``sweep.py``) so GPU memory is isolated
across cells and a crash in one doesn't kill the rest.  Cells whose results
JSON already exists are skipped, so the whole run is resumable — re-launching
picks up where it left off (and a partially-swept cell resumes from its
persistent optuna study DB).

Concurrency: with BS=4 and a small net a single cell only uses ~35-40% of the
GPU (it is launch/Python-overhead bound), so we run up to DSB_CONCURRENCY cells
at once on the one GPU.  Each cell's stdout goes to results/log_<cell>.out to
avoid interleaving; the driver prints concise launch/finish lines.

Env: DSB_DIMS="2,8,32,128"  DSB_METHODS="off_policy,single_seed_mc,..."
     DSB_CONCURRENCY=3   (+ the per-cell budget vars consumed by sweep.py)
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SWEEP = os.path.join(HERE, "sweep.py")

DIMS = [int(x) for x in os.environ.get("DSB_DIMS", "2,8,32,128").split(",")]
METHODS = os.environ.get(
    "DSB_METHODS",
    "off_policy,single_seed_mc,single_seed_td_lambda,ancestral_mc_td_lambda",
).split(",")
CONC = int(os.environ.get("DSB_CONCURRENCY", 1))

cells = [(d, m) for d in DIMS for m in METHODS]
pending = [(d, m) for (d, m) in cells if not os.path.exists(f"{RESULTS}/{m}_d{d}.json")]
n_skip = len(cells) - len(pending)
print(f"Driver: {len(cells)} cells ({len(DIMS)}d × {len(METHODS)}m), "
      f"{n_skip} already done, {len(pending)} to run, concurrency={CONC}", flush=True)

t0 = time.time()
running = {}   # Popen -> (d, m, logfile)
done, failed = [], []


def launch(d, m):
    log = open(f"{RESULTS}/log_{m}_d{d}.out", "w")
    p = subprocess.Popen([sys.executable, SWEEP, "--method", m, "--dim", str(d)],
                         stdout=log, stderr=subprocess.STDOUT)
    running[p] = (d, m, log)
    print(f"  launch {m}_d{d}  ({len(running)} running, {len(pending)} pending, "
          f"elapsed {(time.time()-t0)/60:.1f}m)", flush=True)


while pending or running:
    while pending and len(running) < CONC:
        d, m = pending.pop(0)
        launch(d, m)
        time.sleep(8)  # stagger to avoid simultaneous calibration/build spikes
    time.sleep(15)
    for p in list(running):
        if p.poll() is None:
            continue
        d, m, log = running.pop(p); log.close()
        ok = os.path.exists(f"{RESULTS}/{m}_d{d}.json")
        (done if (p.returncode == 0 and ok) else failed).append(f"{m}_d{d}")
        tail = ""
        try:
            with open(f"{RESULTS}/log_{m}_d{d}.out") as fh:
                lines = [ln for ln in fh if "CONVERGED" in ln]
                tail = lines[-1].strip() if lines else ""
        except Exception:
            pass
        print(f"  finish {m}_d{d}  rc={p.returncode} ok={ok}  {tail}  "
              f"(elapsed {(time.time()-t0)/60:.1f}m)", flush=True)

print(f"\nDriver done in {(time.time()-t0)/60:.1f} min. "
      f"completed={len(done)} failed={len(failed)}", flush=True)
if failed:
    print("FAILED cells: " + ", ".join(failed), flush=True)

try:
    subprocess.run([sys.executable, os.path.join(HERE, "summarize.py")])
except Exception as e:  # noqa: BLE001
    print(f"(summarize skipped: {e})", flush=True)
