# Makefile - 159.735 Assignment 5: Lattice Points in a Hypersphere (CUDA)
#
# Before compiling, load CUDA:
#   module load cuda
#
# Usage:
#   make           -> build both nsphere (CPU) and nsphere_cuda (GPU)
#   make clean     -> remove binaries

CPP     = g++
NVCC    = nvcc
CFLAGS  = -O3
# Adjust -arch to match your GPU (sm_60, sm_70, sm_75, sm_80, sm_86, sm_89)
NVFLAGS = -O3 -arch=sm_70

all: nsphere nsphere_cuda

nsphere: nsphere.cpp
	$(CPP) $(CFLAGS) -o nsphere nsphere.cpp -lm

nsphere_cuda: nsphere_cuda.cu
	$(NVCC) $(NVFLAGS) -o nsphere_cuda nsphere_cuda.cu -lm

clean:
	rm -f nsphere nsphere_cuda
