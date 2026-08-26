#!/bin/bash
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_THREADS=1
python bias_probe.py --offset "$1" --seed "$2" --steps 8000 >> logs_bp.txt 2>&1
