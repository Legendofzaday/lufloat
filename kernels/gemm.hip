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
  // 0..=15.
  const unsigned int lane16 = lane & 15;
  // 0..=1.
  const unsigned int half = lane >> 4;
  // [2(0..=3)] * 16.
  const unsigned int wave_row = (wave >> 1) << 4;
  // (0..=1) * 16.
  const unsigned int wave_col = (wave & 1) << 4;
  const unsigned int block_row = blockIdx.y << 6;
  const unsigned int block_col = blockIdx.x << 5;
  __shared__ _Float16 A_lds[64][16];
  __shared__ _Float16 B_lds[16][32];
  _Float16 __attribute__((ext_vector_type(16))) c_frag = {};
  for (unsigned int ki = 0; ki < k; ki += 16) {
    // 0..=63.
    const unsigned int a_row = thread >> 2;
    // 0..=3.
    const unsigned int a_col = (thread & 3) << 2;
    // 64bit vectorized load (4 float16).
    const _Float16 *a_src = A_f16 + (block_row + a_row) * k + (ki + a_col);
    *reinterpret_cast<unsigned int __attribute__((ext_vector_type(2))) *>(
        &A_lds([a_row][a_col]) =
        *reinterpret_cast<const unsigned int
                          __attribute__((ext_vector_type(2))) *>(a_src);
    if (thread < 128) {
      // 0..=15.
      const unsigned int b_row = thread >> 3;
      // 0..=31.
      const unsigned int b_col = (thread & 7) << 2;
      // 64bit vectorized load (4 float16).
      const _Float16 *b_src = B_f16 + (ki + b_row) * n + (block_col + b_col);
      *reinterpret_cast<unsigned int __attribute__((ext_vector_type(2))) *>(
        &B_lds([b_row][b_col]) =
        *reinterpret_cast<const unsigned int
                          __attribute__((ext_vector_type(2))) *>(b_src);
    }
    __syncthreads();
    _Float16 __attribute__((ext_vector_type(16))) a_frag;
    _Float16 __attribute__((ext_vector_type(16))) b_frag;
#pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
      a_frag[i] = A_lds[wave_row + i][lane16];
      b_frag[i] = B_lds[i][wave_col + lane16];
    }
    c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);
    __syncthreads();
  }
#pragma unroll
  for (unsigned int i = 0; i < 8; ++i) {
    const unsigned int row = macro_row + wave_row + (i << 1) + half;
    const unsigned int col = macro_col + wave_col + lane16;
    C_f16[row * n + col] = c_frag[i << 1];
  }
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