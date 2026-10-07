#!/bin/bash
#SBATCH --job-name=mpi_sum_np20
#SBATCH --partition=short           # use 'test' for initial testing
#SBATCH --nodes=2                   # 2 nodes
#SBATCH --tasks-per-node=10         # 10 tasks per node = 20 total processes
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%j-np20.out
#SBATCH --error=slurm-%j-np20.err

# Load the same OpenMPI module used at compile time
module load openmpi4/intel-openapi/64/4.1.8-with-ucx

echo "=== Job np=20 started: $(date) ==="
echo "Nodes allocated: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"

# Run with 20 MPI processes, summing integers 1..10000000
srun ./second 10000000

echo "=== Job np=20 finished: $(date) ==="
