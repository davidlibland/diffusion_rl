#!/bin/bash
# $1=loss $2=dim $3=lr $4=seeds $5=steps $6=tag $7=clip
cd /home/dlibland/dev/diffusion_rl/experiments/loss_ablation
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_THREADS=1
python run_loss_cell.py --loss "$1" --dim "$2" --lr "$3" --seeds "$4" \
  --steps "$5" --tag "$6" --clip "$7" >> "logs/${6}_${1}_d${2}.log" 2>&1
