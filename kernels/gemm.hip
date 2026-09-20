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
__global__ void __attribute__((amdgpu_flat_work_group_size(512, 512)))
gemm_kernel(const _Float16 *__restrict__ A, const _Float16 *__restrict__ B,
            _Float16 *__restrict__ C, const unsigned int n,
            const unsigned int k) {
  __shared__ uint4 a_lds[256][8];
  __shared__ uint4 b_lds[64][32];
  auto block_m = blockIdx.y;
  auto block_n = blockIdx.x;
  if ((gridDim.x & 3) == 0 && (gridDim.y & 3) == 0) {
    const auto block_id = blockIdx.y * gridDim.x + blockIdx.x;
    const auto s_outer = block_id >> 4;
    const auto s_inner = block_id & 15;
    const auto num_s_n = gridDim.x >> 2;
    const auto s_m = s_outer / num_s_n;
    const auto s_n = s_outer % num_s_n;
    block_m = (s_m << 2) + (s_inner >> 2);
    block_n = (s_n << 2) + (s_inner & 3);
  }
  const auto block_row = block_m << 8;
  const auto block_col = block_n << 8;
  const auto wave = threadIdx.x >> 5;
  const auto lane = threadIdx.x & 31;
  const auto wave_m = threadIdx.x >> 7;
  const auto wave_n = threadIdx.x >> 5 & 31;
  const auto a_r_base = threadIdx.x >> 3;
  const auto a_c = threadIdx.x & 7;
  const auto a_col = a_c << 3;
  const auto b_r_base = threadIdx.x >> 5;
  const auto b_c = threadIdx.x & 31;
  const auto b_col = b_c << 3;
  const auto *const a_ptr0 =
      A + static_cast<unsigned long long>(block_row + a_r_base + 0) * k + a_col;
  const auto *const a_ptr1 =
      A + static_cast<unsigned long long>(block_row + a_r_base + 64) * k +
      a_col;
  const auto *const a_ptr2 =
      A + static_cast<unsigned long long>(block_row + a_r_base + 128) * k +
      a_col;
  const auto *const a_ptr3 =
      A + static_cast<unsigned long long>(block_row + a_r_base + 192) * k +
      a_col;
  const auto *const b_ptr0 = B +
                             static_cast<unsigned long long>(b_r_base + 0) * n +
                             (block_col + b_col);
  const auto *const b_ptr1 =
      B + static_cast<unsigned long long>(b_r_base + 16) * n +
      (block_col + b_col);
  const auto *const b_ptr2 =
      B + static_cast<unsigned long long>(b_r_base + 32) * n +
      (block_col + b_col);
  const auto *const b_ptr3 =
      B + static_cast<unsigned long long>(b_r_base + 48) * n +
      (block_col + b_col);
  auto *const a_lds_flat = reinterpret_cast<uint4 *>(a_lds);
  float8 c_frag[4][4] = {};
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
    @param gridDim: 256x256 tiling.
    @param blockDim: 16 wave 32.
  */
  hipLaunchKernelGGL(gemm_kernel, dim3(n >> 8, m >> 8), dim3(512), 0, 0, A, B,
                     C, n, k);

  return static_cast<int>(hipGetLastError());
}