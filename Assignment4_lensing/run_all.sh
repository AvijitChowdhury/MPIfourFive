#!/bin/bash
# Build and run everything for Assignment 4 on hpc-parallelgpu01.  Usage: ./run_all.sh | tee a4_results.txt
set -e
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
make clean >/dev/null 2>&1 || true
make
echo "=== sequential baseline ==="; ./lens_demo
for s in 0.01 0.005 0.0025 0.00125; do echo "=== GPU example 3, lens_scale $s ==="; ./lens_cuda 3 $s; done
for n in 10 50 200 1000; do echo "=== GPU example n, nlenses=$n ==="; ./lens_cuda n 0.005 $n; done
for t in 8 16 32; do echo "=== GPU block ${t}x${t} ==="; ./lens_cuda 3 0.0025 50 $t; done
