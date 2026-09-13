#include <hip/hip_runtime.h>

using half8 = _Float16 __attribute__((ext_vector_type(8)));
using half16 = _Float16 __attribute__((ext_vector_type(16)));

/*
  @param A: pointer to input float16 array.
  @param B: pointer to input float16 array.
  @param C: pointer to output float16 array.
  @param n: cols of B || cols of C.
  @param k: cols of A || rows of B.
*/
__global__ void __attribute__((amdgpu_flat_work_group_size(512, 512)))
gemm_kernel(const unsigned short *__restrict__ A,
            const unsigned short *__restrict__ B,
            unsigned short *__restrict__ C, const unsigned int n,
            const unsigned int k) {
  // pointer to input float16 array.
  const _Float16 *const A_f16 = reinterpret_cast<const _Float16 *>(A);
  // pointer to input float16 array.
  const _Float16 *const B_f16 = reinterpret_cast<const _Float16 *>(B);
  // pointer to output float16 array.
  _Float16 *const C_f16 = reinterpret_cast<_Float16 *>(C);
  // blockIdx.x * 256.
  const unsigned int bx = blockIdx.x << 8;
  // blockIdx.y * 256.
  const unsigned int by = blockIdx.y << 8;
  // 0..=1.
  const unsigned int wave_half = threadIdx.x >> 4 & 1;
  // 0..=3.
  const unsigned int wave_n = threadIdx.x >> 5 & 3;
  // 0..=127.
  const unsigned int a_m_base = threadIdx.x >> 2;
  // 0 || 8 || 16 || 24.
  const unsigned int a_k_col = threadIdx.x << 3 & 24;
  // 0..=15.
  const unsigned int b_k_base = threadIdx.x >> 5;
  // 0 || 8 || 16 || 24 || … || 248.
  const unsigned int b_n_col = threadIdx.x << 3 & 248;
  alignas(16) __shared__ half8 A_lds_t[2][1024];
  alignas(16) __shared__ half8 B_lds_t[2][1024];
  half8 ra[2];
  half8 rb[2];
#pragma unroll
  for (unsigned int i = 0; i < 2; ++i) {
    // 128 bit vectorized load.
    ra[i] = *reinterpret_cast<const half8 *>(
        &A_f16[(by + (a_m_base + (i << 7))) * k + a_k_col]);
  }
#pragma unroll
  for (unsigned int i = 0; i < 2; ++i) {
    // 128 bit vectorized load.
    rb[i] = *reinterpret_cast<const half8 *>(
        &B_f16[(b_k_base + (i << 4)) * n + (bx + b_n_col)]);
  }
#pragma unroll
  for (unsigned int i = 0; i < 2; ++i) {
    // [0..=127] + 128 * i.
    const unsigned int a_m = a_m_base + (i << 7);
    const unsigned int swizzled_k = (a_k_col >> 3) ^ ((a_m >> 1) & 3);
    A_lds_t[0][a_m * 4 + swizzled_k] = ra[i];
  }
#pragma unroll
  for (unsigned int i = 0; i < 2; ++i) {
    // [0..=15] + 16 * i.
    const unsigned int b_k = b_k_base + (i << 4);
    const unsigned int swizzled_n = (b_n_col >> 3) ^ b_k;
    B_lds_t[0][b_k * 32 + swizzled_n] = rb[i];
  }
  __syncthreads();
  unsigned int phase = 0;
  half16 a_frag[4];
  half16 b_frag[4];
  half8 c_frag[4][4] = {};
  for (unsigned int i = 32; i < k; i += 32) {
#pragma unroll
    for (unsigned int j = 0; j < 2; ++j) {
      // 128 bit vectorized load.
      ra[j] = *reinterpret_cast<const half8 *>(
          &A_f16[(by + (a_m_base + (j << 7))) * k + a_k_col + i]);
    }
#pragma unroll
    for (unsigned int i = 0; i < 2; ++i) {
      // 128 bit vectorized load.
      rb[j] = *reinterpret_cast<const half8 *>(
          &B_f16[(b_k_base + (j << 4)) * n + (bx + b_n_col) + i * n]);
    }
  }
}

/*
  @param A: pointer to input float16 array.
  @param B: pointer to input float16 array.
  @param C: pointer to output float16 array.
  @param m: rows of A || rows of C.
  @param n: cols of B || cols of C.
  @param k: cols of A || rows of B.
*/
extern "C" int lufloat_gemm(const unsigned short *A, const unsigned short *B,
                            unsigned short *C, const unsigned int m,
                            const unsigned int n, const unsigned int k) {
  /*
    @param gridDim: 256x256 tiling.
    @param blockDim: 16 wave 32.
  */
  hipLaunchKernelGGL(gemm_kernel, dim3(n >> 8, m >> 8), dim3(512), 0, 0, A, B,
                     C, n, k);
  // kernel launch error.
  return static_cast<int>(hipGetLastError());
}