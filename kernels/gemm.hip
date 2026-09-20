#include <hip/hip_runtime.h>

using half16 = _Float16 __attribute__((ext_vector_type(16)));
using float8 = float __attribute__((ext_vector_type(8)));

__global__ void __attribute__((amdgpu_flat_work_group_size(256, 256)))
gemm_kernel(const _Float16 *__restrict__ A, const _Float16 *__restrict__ B,
            _Float16 *__restrict__ C, const unsigned int n,
            const unsigned int k) {
  __shared__ uint4 a_lds[128][8];
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
  const auto block_row = block_m << 7;
  const auto block_col = block_n << 7;
  const auto wave_m = threadIdx.x >> 6;
  const auto wave_n = (threadIdx.x >> 5) & 1;
  const auto lane = threadIdx.x & 31;
  const auto a_row = threadIdx.x >> 3;
  const auto a_col = (threadIdx.x & 7) << 3;
  const auto b_row = threadIdx.x >> 4;
  const auto b_col = (threadIdx.x & 15) << 3;
  const auto *a_ptr =
      A + (block_row + a_row) * static_cast<unsigned long long>(k) + a_col;
  const auto *b_ptr =
      B + b_row * static_cast<unsigned long long>(n) + (block_col + b_col);
  const auto a_step = k << 5;
  const auto b_step = n << 4;
  {
    const auto c = threadIdx.x & 7;
#pragma unroll
    for (auto i = 0; i < 4; ++i) {
      const auto r = a_row + (i << 5);
      a_lds[r][c ^ (r & 7)] =
          *reinterpret_cast<const uint4 *>(a_ptr + i * a_step);
    }
  }
  {
    const auto c = threadIdx.x & 15;
#pragma unroll
    for (auto i = 0; i < 4; ++i) {
      const auto r = b_row + (i << 4);
      b_lds[r][c ^ (r & 15)] =
          *reinterpret_cast<const uint4 *>(b_ptr + i * b_step);
    }
  }
  __syncthreads();
  uint4 a_prefetch[4];
  uint4 b_prefetch[4];
  float8 c_frag[2][4] = {};
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
    @param gridDim: 128x128 tiling.
    @param blockDim: 8 wave 32.
  */
  hipLaunchKernelGGL(gemm_kernel, dim3(n >> 7, m >> 7), dim3(256), 0, 0, A, B,
                     C, n, k);
  // kernel launch error.
  return static_cast<int>(hipGetLastError());
}