# Makefile for 159.735 Assignment 4 on hpc-parallelgpu01 (CFITSIO is installed system-wide;
# the GPU there is a Tesla T4 = compute capability 7.5 = sm_75)
#
# CFITSIO location: edit CFITSIO_DIR, or override on the command line
#     make CFITSIO_DIR=$HOME/cfitsio-4.6.2
# (if cfitsio is installed system-wide, e.g. hpc-parallelgpu01, use:  make CFITSIO_DIR=)
CFITSIO_DIR ?=

ifneq ($(CFITSIO_DIR),)
INCP = -I$(CFITSIO_DIR)
LIBP = -L$(CFITSIO_DIR)/.libs -Wl,-rpath,$(CFITSIO_DIR)/.libs
endif

CPP    = g++
NVCC   = nvcc
CFLAGS = -w -O3
# Set to your card's compute capability, e.g. sm_75 (RTX 20xx), sm_86 (RTX 30xx), sm_61 (GTX 10xx)
# Find it with: nvidia-smi --query-gpu=name,compute_cap --format=csv
ARCH  ?= sm_75
NVFLAGS = -O3 -w -arch=$(ARCH)

LIBS = -lcfitsio -lm
OBJS = fitsfile.o lenses.o

all : lens_demo lens_cuda

clean :
	rm -f lens_demo lens_cuda *.o

lens_demo : lens_demo.cpp $(OBJS)
	$(CPP) $(CFLAGS) $(INCP) -o $@ lens_demo.cpp $(OBJS) $(LIBP) $(LIBS)

# CUDA program. Host-only modules (fitsfile, lenses) are built with g++ as before.
lens_cuda : lens_cuda.cu $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCP) -o $@ lens_cuda.cu $(OBJS) $(LIBP) $(LIBS)

# Same program, no CFITSIO needed (writes lens_cuda.pgm)
lens_cuda_nofits : lens_cuda.cu lenses.o
	$(NVCC) $(NVFLAGS) -DNO_FITS -o $@ lens_cuda.cu lenses.o

fitsfile.o : fitsfile.cpp fitsfile.h
	$(CPP) $(CFLAGS) $(INCP) -c fitsfile.cpp

lenses.o : lenses.cpp lenses.h
	$(CPP) $(CFLAGS) -c lenses.cpp
