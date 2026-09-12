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
__global__ void __attribute__((amdgpu_flat_work_group_size(256, 256)))
gemm_kernel(const unsigned short *__restrict__ A,
            const unsigned short *__restrict__ B,
            unsigned short *__restrict__ C, const unsigned int n,
            const unsigned int k) {
  const _Float16 *const A_f16 = reinterpret_cast<const _Float16 *>(A);
  const _Float16 *const B_f16 = reinterpret_cast<const _Float16 *>(B);
  _Float16 *const C_f16 = reinterpret_cast<_Float16 *>(C);
  alignas(16) __shared__ _Float16 A_lds_t[2][32][256];
  alignas(16) __shared__ _Float16 B_lds_t[2][256][32];
  const unsigned int bx = blockDim.x << 8;
  const unsigned int by = blockDim.y << 8;
  const unsigned int lane_16 = threadIdx.x & 15;
  const unsigned int wave_half = threadIdx.x >> 4 & 1;
  const unsigned int wave_m = threadIdx.x >> 6;
  const unsigned int wave_n = threadIdx.x >> 5 & 1;
  const unsigned int a_m_base = threadIdx.x >> 2;
  const unsigned int a_k_col = threadIdx.x << 3 & 24;
  const unsigned int b_k_base = threadIdx.x >> 5;
  const unsigned int b_n_col = threadIdx.x << 3 & 248;
  half16 c_frag[4][4] = {};
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
    @param blockDim: 8 wave 32.
  */
  hipLaunchKernelGGL(gemm_kernel, dim3(n >> 8, m >> 8), dim3(256), 0, 0, A, B,
                     C, n, k);
  // kernel launch error.
  return static_cast<int>(hipGetLastError());
}