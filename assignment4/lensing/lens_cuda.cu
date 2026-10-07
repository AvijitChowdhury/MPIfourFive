/*
   159.735 Parallel Programming - Assignment 4
   CUDA GPU implementation of gravitational lensing by inverse ray tracing.

   Each CUDA thread handles one pixel on the lens plane independently —
   an embarrassingly parallel problem perfectly suited to the GPU.

   Compile:  nvcc -O3 -arch=sm_70 -o lens_cuda lens_cuda.cu fitsfile.o lenses.o -lcfitsio -lm
             (adjust -arch to match your GPU, e.g. sm_75, sm_86, sm_89)
   Run:      ./lens_cuda
*/

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "lenses.h"
#include "arrayff.hxx"

// Lens plane physical boundaries (same as sequential version)
const float WL  = 2.0f;
const float XL1 = -WL;
const float XL2 =  WL;
const float YL1 = -WL;
const float YL2 =  WL;

// ----------------------------------------------------------------
// CUDA error checking macro
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
// Device (GPU) version of the lens equation shoot().
// Cannot call host functions from device code, so this is a
// separate __device__ implementation.
// ----------------------------------------------------------------
__device__ void shoot_device(float& xs, float& ys,
                              float xl, float yl,
                              const float* xlens, const float* ylens,
                              const float* eps, int nlenses)
{
    float dx, dy, dr;
    xs = xl;
    ys = yl;
    for (int p = 0; p < nlenses; ++p) {
        dx = xl - xlens[p];
        dy = yl - ylens[p];
        dr = dx * dx + dy * dy;
        xs -= eps[p] * dx / dr;
        ys -= eps[p] * dy / dr;
    }
}

// ----------------------------------------------------------------
// Kernel: one thread per pixel, grid-stride loop for large images
// ----------------------------------------------------------------
__global__ void lensing_kernel(float* lensim,
                                int npixx, int npixy,
                                float lens_scale,
                                float xsrc, float ysrc,
                                float rsrc2, float ldc,
                                const float* xlens,
                                const float* ylens,
                                const float* eps,
                                int nlenses)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = npixx * npixy;

    while (n < total) {
        int ix = n % npixx;
        int iy = n / npixx;

        // Physical coordinates on the lens plane
        float xl = XL1 + ix * lens_scale;
        float yl = YL1 + iy * lens_scale;

        // Ray trace to source plane
        float xs, ys;
        shoot_device(xs, ys, xl, yl, xlens, ylens, eps, nlenses);

        // Test if ray hits source star and apply limb darkening
        float xd   = xs - xsrc;
        float yd   = ys - ysrc;
        float sep2 = xd * xd + yd * yd;

        if (sep2 < rsrc2) {
            float mu = sqrtf(1.0f - sep2 / rsrc2);
            lensim[iy * npixx + ix] = 1.0f - ldc * (1.0f - mu);
        } else {
            lensim[iy * npixx + ix] = 0.0f;
        }

        n += stride;
    }
}

// ----------------------------------------------------------------
// main
// ----------------------------------------------------------------
int main(int argc, char* argv[])
{
    // --- Print GPU info ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "# GPU: " << prop.name << std::endl;
    std::cout << "# SMs: " << prop.multiProcessorCount
              << "   MaxThreadsPerBlock: " << prop.maxThreadsPerBlock
              << "   GlobalMem: " << prop.totalGlobalMem / (1024*1024) << " MB"
              << std::endl;

    // --- Lens configuration ---
    // Change to set_example_1, _2, or _n(N,...) for different systems
    float *xlens, *ylens, *eps;
    const int nlenses = set_example_3(&xlens, &ylens, &eps);
    std::cout << "# Simulating " << nlenses << " lens system" << std::endl;

    // --- Source star ---
    const float rsrc  = 0.1f;
    const float ldc   = 0.5f;
    const float xsrc  = 0.0f;
    const float ysrc  = 0.0f;
    const float rsrc2 = rsrc * rsrc;

    // --- Image dimensions ---
    // Reduce lens_scale (e.g. 0.002) for a finer/larger image
    const float lens_scale = 0.005f;
    const int npixx = static_cast<int>(floorf((XL2 - XL1) / lens_scale)) + 1;
    const int npixy = static_cast<int>(floorf((YL2 - YL1) / lens_scale)) + 1;
    const int total = npixx * npixy;
    std::cout << "# Image: " << npixx << " x " << npixy
              << " = " << total << " pixels" << std::endl;

    // --- Allocate device memory ---
    float *d_xlens, *d_ylens, *d_eps, *d_lensim;
    CUDA_CHECK(cudaMalloc(&d_xlens,  nlenses * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ylens,  nlenses * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_eps,    nlenses * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lensim, total   * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_xlens, xlens, nlenses*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ylens, ylens, nlenses*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eps,   eps,   nlenses*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_lensim, 0, total * sizeof(float)));

    // --- Launch configuration ---
    const int TPB = 256;                                    // threads per block
    int BPG = (total + TPB - 1) / TPB;                     // blocks per grid
    if (BPG > 65535) BPG = 65535;                          // hard GPU limit
    std::cout << "# Launch: " << BPG << " blocks x " << TPB << " threads" << std::endl;

    // --- Time kernel with CUDA events (accurate for GPU code) ---
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));

    lensing_kernel<<<BPG, TPB>>>(d_lensim, npixx, npixy,
                                  lens_scale, xsrc, ysrc, rsrc2, ldc,
                                  d_xlens, d_ylens, d_eps, nlenses);

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaGetLastError());

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));
    std::cout << "# GPU kernel time: " << kernel_ms << " ms" << std::endl;

    // --- Copy result to host ---
    float* h_lensim = new float[total]();
    CUDA_CHECK(cudaMemcpy(h_lensim, d_lensim, total*sizeof(float), cudaMemcpyDeviceToHost));

    // --- Transfer into Array<> wrapper and count lit pixels ---
    Array<float, 2> lensim(npixy, npixx);
    int numuse = 0;
    for (int iy = 0; iy < npixy; ++iy)
    for (int ix = 0; ix < npixx; ++ix) {
        lensim(iy, ix) = h_lensim[iy * npixx + ix];
        if (h_lensim[iy * npixx + ix] > 0.0f) ++numuse;
    }
    std::cout << "# Pixels in lens image: " << numuse << std::endl;

    // --- Write FITS file (view with ds9) ---
    dump_array<float, 2>(lensim, "lens.fit");
    std::cout << "# Written to lens.fit" << std::endl;

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_xlens));
    CUDA_CHECK(cudaFree(d_ylens));
    CUDA_CHECK(cudaFree(d_eps));
    CUDA_CHECK(cudaFree(d_lensim));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    delete[] h_lensim;
    delete[] xlens; delete[] ylens; delete[] eps;
    return 0;
}
