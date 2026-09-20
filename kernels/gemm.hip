#include <hip/hip_runtime.h>

using half16 = _Float16 __attribute__((ext_vector_type(16)));
using float8 = float __attribute__((ext_vector_type(8)));

/*
  @param A: pointer to input float16 array.
  @param B: pointer to input float16 array.
  @param C: pointer to output float16 array.
  @param n: cols of B || cols of C.
  @param k: cols of A || rows of B.
*/
__global__ void __attribute__((amdgpu_flat_work_group_size(256, 256)))
gemm_kernel(const _Float16 *__restrict__ A, const _Float16 *__restrict__ B,
            _Float16 *__restrict__ C, const unsigned int n,
            const unsigned int k) {
  __shared__ uint4 a_lds[256][8];
  __shared__ uint4 b_lds[64][16];
  auto block_m = blockIdx.y;
  auto block_n = blockIdx.x;
  if ((gridDim.x & 1) == 0 && (gridDim.y & 1) == 0) {
    const auto block_id = blockIdx.y * gridDim.x + blockIdx.x;
    const auto s_outer = block_id >> 2;
    const auto s_inner = block_id & 3;
    const auto num_s_n = gridDim.x >> 1;
    const auto s_m = s_outer / num_s_n;
    auto s_n = s_outer % num_s_n;
    if (s_m & 1) {
      s_n = (num_s_n - 1) - s_n;
    }
    const auto local_m = s_inner >> 1;
    const auto local_n = (s_inner & 1) ^ local_m;
    block_m = (s_m << 1) + local_m;
    block_n = (s_n << 1) + local_n;
  }
  const auto block_row = block_m << 8;
  const auto block_col = block_n << 7;
}

/*
  @param A: pointer to input float16 array.
  @param B: pointer to input float16 array.
  @param C: pointer to output float16 array.
  @param m: rows of A || rows of C.
  @param n: cols of B || cols of C.
  @param k: cols of A || rows of B.
*/
extern "C" int lufloat_gemm(const _Float16 *A, const _Float16 *B, _Float16 *C,
                            const unsigned int m, const unsigned int n,
                            const unsigned int k) {
  /*
    @param gridDim: 256x128 tiling.
    @param blockDim: 8 wave 32.
  */
  hipLaunchKernelGGL(gemm_kernel, dim3(n >> 7, m >> 8), dim3(256), 0, 0, A, B,
                     C, n, k);

  return static_cast<int>(hipGetLastError());
}