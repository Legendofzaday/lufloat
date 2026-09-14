#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

using DATA_LAYOUT_A = rocwmma::row_major;
using DATA_LAYOUT_B = rocwmma::col_major;
using DATA_LAYOUT_C = rocwmma::row_major;

constexpr auto ROCWMMA_M = 16u;
constexpr auto ROCWMMA_N = 16u;
constexpr auto ROCWMMA_K = 16u;

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