#!/usr/bin/env bash
# Recipe dimension-scaling grid: 3 arms x 8 dims x 30 nested seeds.
#   arms: ssmc x {expand_ns60, expand_ns60_sub}, ssmc-td x expand_ns60
# Idempotent/resumable (per-seed JSONs); big dims first for load balance.
set -uo pipefail
REPO=${REPO:-"$(dirname "$0")/../.."}
cd "$REPO"
HERE=experiments/dim_scaling_recipe
LOGD=${LOGD:-"$HERE/logs"}
mkdir -p "$LOGD"
POOL=${POOL:-5}
GRID_DIMS=${GRID_DIMS:-"512 256 128 64 32 16 8 2"}
export OPT_SKIP_NONFINITE=1

echo "[$(date +%H:%M:%S)] recipe grid (pool=$POOL)"
for d in $GRID_DIMS; do
  echo "single_seed_mc $d expand_ns60"
  echo "single_seed_mc $d expand_ns60_sub"
  echo "single_seed_td_lambda $d expand_ns60"
done | xargs -P "$POOL" -L1 bash -c \
    'OMP_NUM_THREADS=4 python experiments/dim_scaling_recipe/run_recipe_cell.py \
       --method $0 --dim $1 --arm $2 \
       > experiments/dim_scaling_recipe/logs/cell_$2_$0_d$1.log 2>&1; \
     echo "  done $2 $0 d=$1"'
echo "[$(date +%H:%M:%S)] ALL CELLS DONE"
python "$HERE/plot_recipe.py"
echo "[$(date +%H:%M:%S)] PLOT DONE"
