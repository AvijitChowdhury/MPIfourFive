/*
   159.735 Parallel Programming - Assignment 5
   CUDA GPU implementation: count lattice points inside an N-dimensional hypersphere.

   Strategy:
   -  The total number of candidate integer coordinate points to test is
      (2*floor(r)+1)^ndim.  We assign one CUDA thread to each candidate point.
   -  Each thread converts its global thread index to an ndim-digit base-B number
      (using the base-conversion trick from the assignment hints).  This gives the
      integer coordinate vector for that point.
   -  The thread then computes the squared distance from the origin and, if the
      point lies strictly inside the sphere, atomically increments a shared counter.
   -  For large ndim or large r the candidate count can overflow a 32-bit integer,
      so we use unsigned long long (ULONGINT) throughout.

   Compile:
     nvcc -O3 -arch=sm_70 -o nsphere_cuda nsphere_cuda.cu
     (adjust -arch to match your GPU compute capability)

   Run:
     ./nsphere_cuda [ndim] [radius]
   Examples:
     ./nsphere_cuda 1 25.5     -> 51
     ./nsphere_cuda 2 2.05     -> 13
     ./nsphere_cuda 3 1.5      -> 19
     ./nsphere_cuda 4 10.0     -> (large number - verifiable with CPU version)
*/

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

typedef unsigned long long ULONGINT;

// ----------------------------------------------------------------
// CUDA error-check macro
// ----------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__         \
                      << "  " << cudaGetErrorString(_e) << std::endl;         \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ----------------------------------------------------------------
// Device helper: compute base^exp for ULONGINT values
// ----------------------------------------------------------------
__device__ ULONGINT powull(ULONGINT base, int exp) {
    ULONGINT result = 1;
    for (int i = 0; i < exp; ++i) result *= base;
    return result;
}

// ----------------------------------------------------------------
// Kernel: each thread tests one candidate lattice point.
//
// Parameters:
//   d_count  - device pointer to the running count (atomically updated)
//   ntotal   - total number of candidate points
//   ndim     - number of spatial dimensions
//   halfb    - floor(radius); the "centre" of each digit axis
//   base     - 2*halfb+1 (number of integers per dimension axis)
//   rsquare  - radius^2 (threshold for inside test)
// ----------------------------------------------------------------
__global__ void count_kernel(ULONGINT* d_count,
                              ULONGINT  ntotal,
                              int       ndim,
                              long      halfb,
                              long      base,
                              double    rsquare)
{
    ULONGINT n = (ULONGINT)blockDim.x * blockIdx.x + threadIdx.x;
    ULONGINT stride = (ULONGINT)blockDim.x * gridDim.x;

    while (n < ntotal) {
        // Convert n to its base-B representation (digit = coordinate + halfb)
        double rtestsq = 0.0;
        ULONGINT num = n;
        for (int k = 0; k < ndim; ++k) {
            long digit = (long)(num % (ULONGINT)base);
            num /= (ULONGINT)base;
            double xk = (double)(digit - halfb);
            rtestsq += xk * xk;
        }

        if (rtestsq < rsquare)
            atomicAdd(d_count, (ULONGINT)1);

        n += stride;
    }
}

// ----------------------------------------------------------------
// Host helper: sequential count (for verification)
// ----------------------------------------------------------------
ULONGINT cpu_count(int ndim, double radius) {
    long   halfb   = (long)floor(radius);
    long   base    = 2 * halfb + 1;
    double rsquare = radius * radius;

    // Use recursion to avoid overflow on very large ntotal
    ULONGINT count = 0;
    ULONGINT ntotal = 1;
    for (int d = 0; d < ndim; ++d) ntotal *= (ULONGINT)base;

    for (ULONGINT n = 0; n < ntotal; ++n) {
        double rtestsq = 0.0;
        ULONGINT num = n;
        for (int k = 0; k < ndim; ++k) {
            long digit = (long)(num % (ULONGINT)base);
            num /= (ULONGINT)base;
            double xk = (double)(digit - halfb);
            rtestsq += xk * xk;
        }
        if (rtestsq < rsquare) ++count;
    }
    return count;
}

// ----------------------------------------------------------------
// main
// ----------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " [ndim] [radius]" << std::endl;
        return 1;
    }

    const int    ndim   = std::atoi(argv[1]);
    const double radius = std::atof(argv[2]);

    // ---- Print GPU info ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "# GPU             : " << prop.name << std::endl;
    std::cout << "# SMs             : " << prop.multiProcessorCount << std::endl;
    std::cout << "# MaxThreads/Block: " << prop.maxThreadsPerBlock  << std::endl;
    std::cout << "# GlobalMem (MB)  : " << prop.totalGlobalMem / (1024*1024) << std::endl;

    // ---- Problem parameters ----
    const long      halfb   = (long)floor(radius);
    const long      base    = 2 * halfb + 1;
    const double    rsquare = radius * radius;

    // Total candidate points = base^ndim
    ULONGINT ntotal = 1;
    for (int d = 0; d < ndim; ++d) {
        // Overflow guard: if ntotal would exceed device memory, warn user
        if (ntotal > (ULONGINT)4e12 / (ULONGINT)base) {
            std::cerr << "WARNING: ntotal too large for this device. "
                      << "Reduce ndim or radius." << std::endl;
            return 1;
        }
        ntotal *= (ULONGINT)base;
    }

    std::cout << "# ndim            : " << ndim    << std::endl;
    std::cout << "# radius          : " << radius  << std::endl;
    std::cout << "# base (2r+1)     : " << base    << std::endl;
    std::cout << "# ntotal (pts)    : " << ntotal  << std::endl;

    // ---- Allocate counter on device ----
    ULONGINT* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(ULONGINT)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(ULONGINT)));

    // ---- Launch configuration ----
    // 256 threads per block is a reliable default.
    const int TPB = 256;
    // Cap blocks at GPU limit; kernel uses grid-stride loop to cover ntotal.
    ULONGINT needed_blocks = (ntotal + TPB - 1) / (ULONGINT)TPB;
    int BPG = (needed_blocks > 65535ULL) ? 65535 : (int)needed_blocks;

    std::cout << "# Launch          : " << BPG << " blocks x " << TPB
              << " threads" << std::endl;

    // ---- Time with CUDA events ----
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    count_kernel<<<BPG, TPB>>>(d_count, ntotal, ndim, halfb, base, rsquare);

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaGetLastError());

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));

    // ---- Retrieve result ----
    ULONGINT h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(ULONGINT), cudaMemcpyDeviceToHost));

    std::cout << "# GPU kernel time : " << kernel_ms << " ms" << std::endl;
    std::cout << "# Count (GPU)     : " << h_count << std::endl;

    // ---- Verify with CPU for small problems ----
    if (ntotal <= 50000000ULL) {
        ULONGINT cpu_result = cpu_count(ndim, radius);
        std::cout << "# Count (CPU ref) : " << cpu_result << std::endl;
        std::cout << "# Match           : " << (h_count == cpu_result ? "YES" : "NO")
                  << std::endl;
    } else {
        std::cout << "# (CPU verify skipped - ntotal too large)" << std::endl;
    }

    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return 0;
}
