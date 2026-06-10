#!/usr/bin/env bash
# Parallel driver for optuna_fbrrt_bs4_sweep.py on a single multi-GPU-core box
# (32-core 9950X + RTX 3090 Ti: one BS=4 run uses ~2 GB VRAM / ~1 core, so we
# run K workers at once; study state lives in a multi-process-safe optuna
# JournalStorage).
#
# Usage:  bash experiments/bs4_moons/2026-06-10/run_parallel.sh
# Env:    K (workers, default 8), OPT_N_TRIALS (default 160), plus all the
#         OPT_* knobs of the sweep script.
set -uo pipefail
REPO=${REPO:-"$(dirname "$0")/../../.."}
cd "$REPO"

K=${K:-8}
export OPT_N_TRIALS=${OPT_N_TRIALS:-160}
SCRIPT=${SCRIPT:-experiments/bs4_moons/2026-06-10/optuna_fbrrt_bs4_sweep.py}
LOGD=${LOGD:-/tmp/fbrrt2_workers}
mkdir -p "$LOGD"

echo "[$(date +%H:%M:%S)] stage 1: $K sweep workers (budget $OPT_N_TRIALS trials)"
for i in $(seq 0 $((K - 1))); do
  sleep 2  # stagger study creation / GMM fits
  OMP_NUM_THREADS=3 OPT_EXIT_AFTER=sweep OPT_SAMPLER_SEED=$((42 + i)) \
    python "$SCRIPT" > "$LOGD/sweep_w$i.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] stage 1 done"

echo "[$(date +%H:%M:%S)] stage 2: $K confirm workers"
for i in $(seq 0 $((K - 1))); do
  sleep 1
  OMP_NUM_THREADS=3 OPT_EXIT_AFTER=confirm OPT_WORKER_ID=$i OPT_N_WORKERS=$K \
    python "$SCRIPT" > "$LOGD/confirm_w$i.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] stage 2 done"

echo "[$(date +%H:%M:%S)] stage 3: aggregate + parallel per-method converges"
OMP_NUM_THREADS=8 python "$SCRIPT" > "$LOGD/final.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] ALL DONE (exit=$rc)"
exit $rc
