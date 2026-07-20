#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
H=experiments/benchmarks_sampler
for prob in gmm40 manywell; do for meth in off_policy single_seed_mc; do for s in 0 1 2; do
  echo "$prob $meth $s"
done; done; done | xargs -P 4 -L1 bash -c \
  'OMP_NUM_THREADS=4 python experiments/benchmarks_sampler/run_benchmark.py \
     --problem $0 --method $1 --seed $2 --steps 15000 \
     > experiments/benchmarks_sampler/logs/${0}_${1}_s${2}.log 2>&1; echo "done $0 $1 s$2"'
echo ALL_DONE
