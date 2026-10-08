#!/bin/bash
# Build and run everything for Assignment 5.  Usage: ./run_all.sh | tee a5_results.txt
set -e
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
make
echo "=== correctness (expect 51, 13, 19) ==="
./nsphere 1 25.5; ./nsphere 2 2.05; ./nsphere 3 1.5
echo "=== dimension scaling ==="
for n in 4 5 6 7 8; do ./nsphere $n 6.0; done
echo "=== threads per block (7D, r=6) ==="
for t in 32 64 128 256 512 1024; do ./nsphere 7 6.0 $t 1; done
echo "=== radius scaling (8D) ==="
for r in 4.0 5.0 6.0 7.0; do ./nsphere 8 $r 256 1; done
