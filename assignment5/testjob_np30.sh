#!/bin/bash
#SBATCH --job-name=mpi_sum_np30
#SBATCH --partition=short           # use 'test' for initial testing
#SBATCH --nodes=3                   # 3 nodes
#SBATCH --tasks-per-node=10         # 10 tasks per node = 30 total processes
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%j-np30.out
#SBATCH --error=slurm-%j-np30.err

# Load the same OpenMPI module used at compile time
module load openmpi4/intel-openapi/64/4.1.8-with-ucx

echo "=== Job np=30 started: $(date) ==="
echo "Nodes allocated: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"

# Run with 30 MPI processes, summing integers 1..10000000
srun ./second 10000000

echo "=== Job np=30 finished: $(date) ==="
