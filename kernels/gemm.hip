#include <hip/hip_runtime.h>

/*
  @param A: pointer to input float16 array.
  @param B: pointer to input float16 array.
  @param C: pointer to output float16 array.
  @param m: rows of A / rows of C - divisible by 64.
  @param n: cols of B / cols of C - divisible by 64.
  @param k: cols of A / rows of B - divisible by 64.
*/
__global__ void gemm_kernel(const unsigned short *__restrict__ A,
                            const unsigned short *__restrict__ B,
                            unsigned short *__restrict__ C,
                            const unsigned int m, const unsigned int n,
                            const unsigned int k) {
  const _Float16 *A_f16 = reinterpret_cast<const _Float16 *>(A);
  const _Float16 *B_f16 = reinterpret_cast<const _Float16 *>(B);
  _Float16 *C_f16 = reinterpret_cast<_Float16 *>(C);
  // 0..=31.
  const unsigned int lane = threadIdx.x;
  // 0..=7.
  const unsigned int wave = threadIdx.y;
  // 0..=255.
  const unsigned int thread = (wave << 5) + lane;
  const unsigned int block_row = blockIdx.y << 6;
  const unsigned int block_col = blockIdx.x << 5;
  // [2(0..=3)] * 16.
  const unsigned int wave_row = (wave >> 1) << 4;
  // (0..=1) * 16.
  const unsigned int wave_col = (wave & 1) << 4;
  __shared__ _Float16 A_lds[64][16];
  __shared__ _Float16 B_lds[16][32];
}

/*
  @param A: pointer to input float16 array.
  @param B: pointer to input float16 array.
  @param C: pointer to output float16 array.
  @param m: rows of A / rows of C - divisible by 64.
  @param n: cols of B / cols of C - divisible by 64.
  @param k: cols of A / rows of B - divisible by 64.
*/
extern "C" int lufloat_gemm(const unsigned short *A, const unsigned short *B,
                            unsigned short *C, const unsigned int m,
                            const unsigned int n, const unsigned int k) {
  /*
    @param gridDim: 16x4 x 16x2 tiling.
    @param blockDim: wave32 * 8 wavefronts.
  */
  hipLaunchKernelGGL(gemm_kernel, dim3(n >> 5, m >> 6), dim3(32, 8), 0, 0, A, B,
                     C, m, n, k);
  return static_cast<int>(hipGetLastError());
}