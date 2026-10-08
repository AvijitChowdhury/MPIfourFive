# hpc-parallelgpu01 has Tesla T4 GPUs (compute capability 7.5). Override: make ARCH=sm_XX
ARCH ?= sm_75
all: nsphere
nsphere: nsphere.cu
	nvcc -O3 -w -arch=$(ARCH) -o $@ nsphere.cu
clean:
	rm -f nsphere
