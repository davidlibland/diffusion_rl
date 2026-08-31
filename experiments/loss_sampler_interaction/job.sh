#!/bin/bash
cd /home/dlibland/dev/diffusion_rl/experiments/loss_sampler_interaction
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
python run_cell.py --method "$1" --dim "$2" --loss "$3" --seeds 10 --steps 8000 \
  >> "logs/${1}_${3}_d${2}.log" 2>&1
