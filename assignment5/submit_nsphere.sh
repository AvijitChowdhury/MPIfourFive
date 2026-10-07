#!/bin/bash
#SBATCH --job-name=nsphere_cuda
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load cuda

echo "=== Assignment 5: N-sphere CUDA Job ==="
echo "Started : $(date)"
echo "Node    : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "--- Test 1: 1D sphere r=25.5 (expected 51) ---"
./nsphere_cuda 1 25.5

echo ""
echo "--- Test 2: 2D sphere r=2.05 (expected 13) ---"
./nsphere_cuda 2 2.05

echo ""
echo "--- Test 3: 3D sphere r=1.5 (expected 19) ---"
./nsphere_cuda 3 1.5

echo ""
echo "--- Test 4: 4D sphere r=10.0 ---"
./nsphere_cuda 4 10.0

echo ""
echo "--- Test 5: 5D sphere r=8.0 ---"
./nsphere_cuda 5 8.0

echo ""
echo "--- Test 6: 6D sphere r=6.0 ---"
./nsphere_cuda 6 6.0

echo ""
echo "Finished: $(date)"
