/* 159.735 Assignment 5 - lattice points inside an N-dimensional hypersphere (CUDA).

   Build :  nvcc -O3 -arch=native -o nsphere nsphere.cu       (Linux / CTCP)
            nvcc -O3 -arch=sm_75 -o nsphere.exe nsphere.cu    (Windows, x64 Native Tools prompt)
   Run   :  ./nsphere <ndim> <radius> [threads_per_block=256] [skipcpu=0|1]
   e.g.    ./nsphere 2 2.05      -> 13
           ./nsphere 3 1.5       -> 19
           ./nsphere 1 25.5      -> 51
           ./nsphere 6 8.0

   Algorithm
   ---------
   Every integer point lies in the cube [-h,h]^N, h = floor(r), so there are
   base^N candidates, base = 2h+1.  A point is counted if sum(x_k^2) < r^2.

   GPU decomposition: one thread owns a "row" = a fixed choice of the upper
   N-1 coordinates (base^(N-1) rows).  The thread decodes its row number into
   N-1 base-`base` digits once (N-1 integer divisions), accumulates their
   squares, and (if already < r^2) sweeps the last coordinate in a plain loop
   with no division.  That cuts the expensive 64-bit div/mod work by a factor
   `base` compared with one thread per point, and gives each thread a good
   amount of arithmetic.  Rows are handled with a grid-stride loop, so the
   grid size is fixed and any problem size works.  Counting: thread-local
   counter -> shared-memory tree reduction -> ONE 64-bit atomicAdd per block.
*/
#include <cstdio>
#include <ctime>
#include <string>
#include <cstdlib>
#include <cmath>
#include <iostream>

typedef unsigned long long ULONGINT;

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
  std::exit(1);} } while (0)

/* base^k, or 0 if it would overflow 64 bits */
static ULONGINT powl_checked(ULONGINT b, int k)
{
  ULONGINT p = 1;
  for (int i = 0; i < k; ++i) {
    if (p > ~0ULL / b) return 0;
    p *= b;
  }
  return p;
}

/* ------------------------------ sequential ------------------------------ */
/* Straightforward odometer version (same idea as count_in_v2 in the handout). */
ULONGINT h_count_in(double radius, int ndim)
{
  const long halfb = (long)std::floor(radius);
  const long base  = 2 * halfb + 1;
  const double rsq = radius * radius;
  const ULONGINT ntotal = powl_checked(base, ndim);
  ULONGINT count = 0;
  long* idx = new long[ndim]();
  for (ULONGINT n = 0; n < ntotal; ++n) {
    double s = 0;
    for (int k = 0; k < ndim; ++k) { double x = idx[k] - halfb; s += x * x; }
    if (s < rsq) ++count;
    for (int k = 0; k < ndim; ++k) {            // add one, with carry
      if (++idx[k] < base) break;
      idx[k] = 0;
    }
  }
  delete[] idx;
  return count;
}

/* -------------------------------- device -------------------------------- */
__global__ void count_kernel(ULONGINT nrows, int ndim, long base, long halfb,
                             double rsq, ULONGINT* total)
{
  extern __shared__ ULONGINT sh[];               // one slot per thread

  ULONGINT local = 0;
  const ULONGINT stride = (ULONGINT)gridDim.x * blockDim.x;
  for (ULONGINT row = (ULONGINT)blockIdx.x * blockDim.x + threadIdx.x;
       row < nrows; row += stride) {

    // decode the upper (ndim-1) coordinates of this row
    ULONGINT r = row;
    double s = 0.0;
    for (int k = 1; k < ndim; ++k) {
      long d = (long)(r % (ULONGINT)base);
      r /= (ULONGINT)base;
      double x = (double)(d - halfb);
      s += x * x;
    }
    // sweep the last coordinate (no division needed)
    if (s < rsq) {
      for (long d = 0; d < base; ++d) {
        double x = (double)(d - halfb);
        if (s + x * x < rsq) ++local;
      }
    }
  }

  // block reduction (blockDim.x must be a power of two)
  sh[threadIdx.x] = local;
  __syncthreads();
  for (unsigned int half = blockDim.x / 2; half > 0; half >>= 1) {
    if (threadIdx.x < half) sh[threadIdx.x] += sh[threadIdx.x + half];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(total, sh[0]);  // unsigned long long overload
}

ULONGINT d_count_in(double radius, int ndim, int tpb, float* ms_kernel)
{
  const long halfb = (long)std::floor(radius);
  const long base  = 2 * halfb + 1;
  const double rsq = radius * radius;
  const ULONGINT nrows = powl_checked(base, ndim - 1);
  if (nrows == 0) { std::fprintf(stderr, "Problem too large: %ld^%d rows overflows 64 bits\n", base, ndim - 1); std::exit(1); }

  ULONGINT needed = (nrows + tpb - 1) / tpb;
  int bpg = (int)(needed < 65535ULL * 4 ? needed : 65535ULL * 4);   // cap; grid-stride covers the rest

  ULONGINT* d_total;
  CUDA_CHECK(cudaMalloc(&d_total, sizeof(ULONGINT)));
  CUDA_CHECK(cudaMemset(d_total, 0, sizeof(ULONGINT)));

  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a);
  count_kernel<<<bpg, tpb, tpb * sizeof(ULONGINT)>>>(nrows, ndim, base, halfb, rsq, d_total);
  CUDA_CHECK(cudaGetLastError());
  cudaEventRecord(b);
  CUDA_CHECK(cudaEventSynchronize(b));
  cudaEventElapsedTime(ms_kernel, a, b);

  ULONGINT result = 0;
  CUDA_CHECK(cudaMemcpy(&result, d_total, sizeof(ULONGINT), cudaMemcpyDeviceToHost));
  cudaFree(d_total);
  std::cout << "# rows=" << nrows << "  blocks=" << bpg << "  threads/block=" << tpb << std::endl;
  return result;
}

int main(int argc, char* argv[])
{
  if (argc < 3) { std::fprintf(stderr, "usage: %s ndim radius [threads_per_block] [skipcpu]\n", argv[0]); return 1; }
  const int    ndim = std::atoi(argv[1]);
  const double r    = std::atof(argv[2]);
  int tpb           = (argc > 3) ? std::atoi(argv[3]) : 256;
  const int skipcpu = (argc > 4) ? std::atoi(argv[4]) : 0;
  if (ndim < 1 || r <= 0) { std::fprintf(stderr, "need ndim>=1 and radius>0\n"); return 1; }
  if (tpb < 1 || (tpb & (tpb - 1))) { std::fprintf(stderr, "threads_per_block must be a power of 2\n"); return 1; }

  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  if (tpb > p.maxThreadsPerBlock) tpb = p.maxThreadsPerBlock;
  std::cout << "# GPU: " << p.name << "  SMs=" << p.multiProcessorCount
            << "  CC " << p.major << "." << p.minor << std::endl;

  float ms = 0;
  const ULONGINT gpu = d_count_in(r, ndim, tpb, &ms);
  std::cout << "GPU count = " << gpu << "   kernel time = " << ms << " ms" << std::endl;

  const long base = 2 * (long)std::floor(r) + 1;
  const ULONGINT ntotal = powl_checked(base, ndim);
  if (!skipcpu && ntotal != 0 && ntotal <= 4000000000ULL) {
    clock_t t0 = clock();
    const ULONGINT cpu = h_count_in(r, ndim);
    double cms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC;
    std::cout << "CPU count = " << cpu << "   time = " << cms << " ms   "
              << (cpu == gpu ? "MATCH" : "*** MISMATCH ***") << "   speedup = " << cms / (ms > 0 ? ms : 1e-3) << "x" << std::endl;
  } else {
    std::cout << "(CPU check skipped)" << std::endl;
  }
  return 0;
}
