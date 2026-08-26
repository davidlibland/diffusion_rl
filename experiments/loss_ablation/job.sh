#!/bin/bash
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_THREADS=1
python run_loss_cell.py --loss "$1" --dim "$2" --lr "$3" --seeds "$4" \
  --steps "$5" --tag "$6" >> "logs/${6}_${1}_d${2}.log" 2>&1
