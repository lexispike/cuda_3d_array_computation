#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=transpose
#SBATCH --reservation=eece5640
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=transpose.%j.out

time ./Q1_tiled

#nvprof evaluation
echo "nvprof stats"

nvprof --print-gpu-trace ./Q1_tiled
