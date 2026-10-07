#!/bin/bash
#SBATCH --job-name=lensing_cuda
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load cuda

echo "=== Assignment 4 CUDA GPU Lensing Job ==="
echo "Started : $(date)"
echo "Node    : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "--- Running CUDA gravitational lensing ---"
./lens_cuda

echo ""
echo "Finished: $(date)"
