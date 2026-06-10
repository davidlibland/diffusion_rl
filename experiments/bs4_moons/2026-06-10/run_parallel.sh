#!/usr/bin/env bash
# Parallel driver for optuna_fbrrt_bs4_sweep.py on a single box
# (32-core 9950X + RTX 3090 Ti: one BS=4 run uses ~2 GB VRAM / ~1 core).
#
# Each FBRRT method gets its OWN optuna study with its OWN full trial budget
# (OPT_N_TRIALS per method), so TPE cannot starve a method that looks weak
# early.  Per method, KPM sweep workers share the study (JournalStorage,
# multi-process safe); all methods run concurrently:
#
#   stage 1   3 methods x KPM sweep workers   (3*KPM concurrent trainings)
#   stage 2   3 methods x KPM confirm workers (idempotent, partitioned)
#   stage 3   3 final passes in parallel, each converging its method's
#             confirmed winner to OPT_CONV_STEPS
#
# Usage: bash experiments/bs4_moons/2026-06-10/run_parallel.sh
# Env:   KPM (workers per method, default 3), OPT_N_TRIALS (per-method trial
#        budget, default 80), plus all OPT_* knobs of the sweep script.
set -uo pipefail
REPO=${REPO:-"$(dirname "$0")/../../.."}
cd "$REPO"

KPM=${KPM:-3}
export OPT_N_TRIALS=${OPT_N_TRIALS:-80}
SCRIPT=${SCRIPT:-experiments/bs4_moons/2026-06-10/optuna_fbrrt_bs4_sweep.py}
LOGD=${LOGD:-/tmp/fbrrt2_workers}
METHODS=${METHODS:-"fbrrt fbrrt_td_lambda fbrrt_cv"}
mkdir -p "$LOGD"

echo "[$(date +%H:%M:%S)] stage 1: sweep — $KPM workers/method, " \
     "$OPT_N_TRIALS trials/method, methods: $METHODS"
mi=0
for m in $METHODS; do
  for i in $(seq 0 $((KPM - 1))); do
    sleep 2  # stagger study creation / GMM fits
    OMP_NUM_THREADS=3 OPT_METHOD=$m OPT_EXIT_AFTER=sweep \
      OPT_SAMPLER_SEED=$((42 + i + mi * 100)) \
      python "$SCRIPT" > "$LOGD/sweep_${m}_w$i.log" 2>&1 &
  done
  mi=$((mi + 1))
done
wait
echo "[$(date +%H:%M:%S)] stage 1 done"

echo "[$(date +%H:%M:%S)] stage 2: confirm — $KPM workers/method"
for m in $METHODS; do
  for i in $(seq 0 $((KPM - 1))); do
    sleep 1
    OMP_NUM_THREADS=3 OPT_METHOD=$m OPT_EXIT_AFTER=confirm \
      OPT_WORKER_ID=$i OPT_N_WORKERS=$KPM \
      python "$SCRIPT" > "$LOGD/confirm_${m}_w$i.log" 2>&1 &
  done
done
wait
echo "[$(date +%H:%M:%S)] stage 2 done"

echo "[$(date +%H:%M:%S)] stage 3: per-method final pass + converge (parallel)"
for m in $METHODS; do
  OMP_NUM_THREADS=6 OPT_METHOD=$m \
    python "$SCRIPT" > "$LOGD/final_${m}.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] ALL DONE"
