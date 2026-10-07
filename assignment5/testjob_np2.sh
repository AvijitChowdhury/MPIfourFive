#!/bin/bash
#SBATCH --job-name=mpi_sum_np2
#SBATCH --partition=test            # use test partition for quick checks
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2          # 2 processes total
#SBATCH --time=00:02:00
#SBATCH --output=slurm-%j-np2.out
#SBATCH --error=slurm-%j-np2.err

module load openmpi4/intel-openapi/64/4.1.8-with-ucx

echo "=== Job np=2 started: $(date) ==="
echo "Total tasks: $SLURM_NTASKS"

srun ./first 1000000

echo "=== Job np=2 finished: $(date) ==="
