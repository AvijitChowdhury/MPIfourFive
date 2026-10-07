# Makefile - 159.735 Assignment 4: Gravitational Lensing (CUDA)
#
# Usage:
#   make          -> build both lens_demo (CPU) and lens_cuda (GPU)
#   make clean    -> remove binaries and objects

CPP      = g++
NVCC     = nvcc
CFLAGS   = -O3 -w
# -arch=sm_70  targets Volta (V100). Change to match your GPU:
#   sm_60 -> Pascal (P100)   sm_75 -> Turing (T4, RTX 2080)
#   sm_80 -> Ampere (A100)   sm_86 -> RTX 3080/3090
NVFLAGS  = -O3 -arch=sm_70
LIBS     = -lcfitsio -lm
OBJS     = fitsfile.o lenses.o

all: lens_demo lens_cuda

# Sequential CPU version
lens_demo: lens_demo.cpp $(OBJS)
	$(CPP) $(CFLAGS) -o lens_demo lens_demo.cpp $(OBJS) $(LIBS)

# CUDA GPU version
lens_cuda: lens_cuda.cu $(OBJS)
	$(NVCC) $(NVFLAGS) -o lens_cuda lens_cuda.cu $(OBJS) $(LIBS)

# Shared object modules
fitsfile.o: fitsfile.cpp fitsfile.h
	$(CPP) $(CFLAGS) -c fitsfile.cpp

lenses.o: lenses.cpp lenses.h
	$(CPP) $(CFLAGS) -c lenses.cpp

clean:
	rm -f lens_demo lens_cuda $(OBJS) *.fit
