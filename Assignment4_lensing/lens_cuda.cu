/* 159.735 Assignment 4 - gravitational lensing by inverse ray tracing on a GPU.

   Usage:  ./lens_cuda [example=1|2|3|n] [lens_scale=0.005] [nlenses(for n)=50] [threads/dim=16]
   e.g.    ./lens_cuda 3 0.002
           ./lens_cuda n 0.005 100

   One CUDA thread = one lens-plane pixel.  A 2D grid of 2D blocks maps
   directly onto the image, so neighbouring threads in a warp (threadIdx.x)
   write neighbouring floats (coalesced stores).

   Build with CFITSIO  (default)      : see makefile / README
   Build WITHOUT CFITSIO (-DNO_FITS)  : writes lens_cuda.pgm instead (Windows-friendly)
*/
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>

#include "lenses.h"          // host shoot(), set_example_*()
#ifndef NO_FITS
#include "arrayff.hxx"       // Array<> + dump_array() (needs cfitsio)
#else
#include "array.hxx"
#endif

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
  std::exit(1);} } while (0)

const float WL  = 2.0f;
const float XL1 = -WL, XL2 = WL, YL1 = -WL, YL2 = WL;

// Device version of shoot() (lenses.cpp's is host-only)
__device__ __forceinline__ void d_shoot(float& xs, float& ys, float xl, float yl,
                                        const float* __restrict__ xlens,
                                        const float* __restrict__ ylens,
                                        const float* __restrict__ eps, int nlenses)
{
  xs = xl; ys = yl;
  for (int p = 0; p < nlenses; ++p) {
    float dx = xl - xlens[p];
    float dy = yl - ylens[p];
    float dr = dx * dx + dy * dy;
    xs -= eps[p] * dx / dr;
    ys -= eps[p] * dy / dr;
  }
}

__global__ void lens_kernel(float* __restrict__ lensim, int npixx, int npixy,
                            float lens_scale,
                            const float* __restrict__ xlens,
                            const float* __restrict__ ylens,
                            const float* __restrict__ eps, int nlenses,
                            float xsrc, float ysrc, float rsrc2, float ldc)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix >= npixx || iy >= npixy) return;          // edge blocks overhang the image

  const float xl = XL1 + ix * lens_scale;
  const float yl = YL1 + iy * lens_scale;
  float xs, ys;
  d_shoot(xs, ys, xl, yl, xlens, ylens, eps, nlenses);

  const float xd = xs - xsrc, yd = ys - ysrc;
  const float sep2 = xd * xd + yd * yd;
  float val = 0.0f;
  if (sep2 < rsrc2) {
    const float mu = sqrtf(1.0f - sep2 / rsrc2);
    val = 1.0f - ldc * (1.0f - mu);
  }
  lensim[(size_t)iy * npixx + ix] = val;           // every pixel written: no memset needed
}

#ifdef NO_FITS
static void write_pgm(const float* im, int w, int h, const char* name)
{
  FILE* f = std::fopen(name, "wb");
  std::fprintf(f, "P5\n%d %d\n255\n", w, h);
  for (int y = h - 1; y >= 0; --y)                 // flip so +y is up, like ds9
    for (int x = 0; x < w; ++x) {
      float v = im[(size_t)y * w + x];
      unsigned char c = (unsigned char)(v < 0 ? 0 : (v > 1 ? 255 : v * 255.0f));
      std::fputc(c, f);
    }
  std::fclose(f);
}
#endif

int main(int argc, char* argv[])
{
  const std::string ex = (argc > 1) ? argv[1] : "1";
  const float lens_scale = (argc > 2) ? (float)std::atof(argv[2]) : 0.005f;
  const int   nuse = (argc > 3) ? std::atoi(argv[3]) : 50;
  const int   tdim = (argc > 4) ? std::atoi(argv[4]) : 16;   // 16x16 = 256 threads/block

  float *xlens, *ylens, *eps;
  int nlenses;
  if      (ex == "2") nlenses = set_example_2(&xlens, &ylens, &eps);
  else if (ex == "3") nlenses = set_example_3(&xlens, &ylens, &eps);
  else if (ex == "n") nlenses = set_example_n(nuse, &xlens, &ylens, &eps);
  else                nlenses = set_example_1(&xlens, &ylens, &eps);
  std::cout << "# Simulating " << nlenses << " lens system" << std::endl;

  const float rsrc = 0.1f, ldc = 0.5f, xsrc = 0.0f, ysrc = 0.0f;
  const float rsrc2 = rsrc * rsrc;

  const int npixx = (int)std::floor((XL2 - XL1) / lens_scale) + 1;
  const int npixy = (int)std::floor((YL2 - YL1) / lens_scale) + 1;
  std::cout << "# Building " << npixx << "X" << npixy << " lens image" << std::endl;

  Array<float, 2> lensim(npixy, npixx);

  // ---- device memory ----
  const size_t nbytes = (size_t)npixx * npixy * sizeof(float);
  float *d_im, *d_x, *d_y, *d_e;
  CUDA_CHECK(cudaMalloc(&d_im, nbytes));
  CUDA_CHECK(cudaMalloc(&d_x, nlenses * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, nlenses * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_e, nlenses * sizeof(float)));

  cudaEvent_t t0, t1, t2, t3;
  cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventCreate(&t2); cudaEventCreate(&t3);

  cudaEventRecord(t0);
  CUDA_CHECK(cudaMemcpy(d_x, xlens, nlenses * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, ylens, nlenses * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_e, eps,   nlenses * sizeof(float), cudaMemcpyHostToDevice));

  dim3 tpb(tdim, tdim);
  dim3 bpg((npixx + tpb.x - 1) / tpb.x, (npixy + tpb.y - 1) / tpb.y);
  cudaEventRecord(t1);
  lens_kernel<<<bpg, tpb>>>(d_im, npixx, npixy, lens_scale, d_x, d_y, d_e, nlenses,
                            xsrc, ysrc, rsrc2, ldc);
  CUDA_CHECK(cudaGetLastError());
  cudaEventRecord(t2);
  CUDA_CHECK(cudaMemcpy(lensim.buffer, d_im, nbytes, cudaMemcpyDeviceToHost));
  cudaEventRecord(t3);
  CUDA_CHECK(cudaEventSynchronize(t3));

  float h2d, kern, d2h, total;
  cudaEventElapsedTime(&h2d, t0, t1);
  cudaEventElapsedTime(&kern, t1, t2);
  cudaEventElapsedTime(&d2h, t2, t3);
  cudaEventElapsedTime(&total, t0, t3);

  long numuse = 0;
  for (int n = 0; n < lensim.ntotal; ++n) if (lensim.buffer[n] > 0.0f) ++numuse;
  std::cout << "# Block " << tdim << "x" << tdim << ", grid " << bpg.x << "x" << bpg.y << "\n"
            << "# Kernel: " << kern << " ms   H2D: " << h2d << " ms   D2H: " << d2h
            << " ms   Total: " << total << " ms   hit pixels: " << numuse << std::endl;

#ifndef NO_FITS
  dump_array<float, 2>(lensim, "lens_cuda.fit");
  std::cout << "# Wrote lens_cuda.fit" << std::endl;
#else
  write_pgm(lensim.buffer, npixx, npixy, "lens_cuda.pgm");
  std::cout << "# Wrote lens_cuda.pgm" << std::endl;
#endif

  cudaFree(d_im); cudaFree(d_x); cudaFree(d_y); cudaFree(d_e);
  delete[] xlens; delete[] ylens; delete[] eps;
  return 0;
}
