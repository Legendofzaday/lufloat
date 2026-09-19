#include <hip/hip_runtime.h>

using half16 = _Float16 __attribute__((ext_vector_type(16)));

constexpr auto S_M = 4;
constexpr auto S_N = 4;
constexpr auto S_BLOCK = S_M * S_N;

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
  __shared__ uint4 a_lds[32][32];
  __shared__ uint4 b_lds[32][32];
  const auto block_id = blockIdx.y * gridDim.x + blockIdx.x;
  const auto s_outer = block_id / S_BLOCK;
  const auto s_inner = block_id % S_BLOCK;
  const auto num_s_n = (gridDim.x + S_N - 1) / S_N;
  const auto s_m = s_outer / num_s_n;
  const auto s_n = s_outer % num_s_n;
  const auto block_m = s_m * S_M + s_inner / S_N;
  const auto block_n = s_n * S_N + s_inner % S_N;
  const auto block_row = block_m << 8;
  const auto block_col = block_n << 8;
  const auto wave = threadIdx.x >> 5;
  const auto lane = threadIdx.x & 31;
  const auto wave_m = threadIdx.x >> 7;
  const auto wave_n = threadIdx.x >> 5 & 3;
  const auto a_vec_id0 = threadIdx.x;
  const auto a_vec_id1 = threadIdx.x + 512;
  const auto a_row0 = a_vec_id0 >> 2;
  const auto a_col0 = (a_vec_id0 & 3) << 3;
  const auto a_row1 = a_vec_id1 >> 2;
  const auto a_col1 = (a_vec_id1 & 3) << 3;
  const auto b_row0 = a_vec_id0 >> 5;
  const auto b_col0 = (a_vec_id0 & 31) << 3;
  const auto b_row1 = a_vec_id1 >> 5;
  const auto b_col1 = (a_vec_id1 & 31) << 3;
  half16 c_frag[4][4] = {};
  const auto *const a_base_ptr0 =
      A + static_cast<unsigned long long>(block_row + a_row0) * k + a_col0;
  const auto *const a_base_ptr1 =
      A + static_cast<unsigned long long>(block_row + a_row1) * k + a_col1;
  const auto *const b_base_ptr0 =
      B + static_cast<unsigned long long>(b_row0) * n + (block_col + b_col0);
  const auto *const b_base_ptr1 =
      B + static_cast<unsigned long long>(b_row1) * n + (block_col + b_col1);
  auto *const a_lds_flat = reinterpret_cast<uint4 *>(a_lds);
  for (auto i = 0; i < k; i += 32) {
    const auto k_stride_b = static_cast<unsigned long long>(i) * n;
    const auto *const a_ptr0 = reinterpret_cast<const uint4 *>(a_base_ptr0 + i);
    const auto *const a_ptr1 = reinterpret_cast<const uint4 *>(a_base_ptr1 + i);
    const auto *const b_ptr0 =
        reinterpret_cast<const uint4 *>(b_base_ptr0 + k_stride_b);
    const auto *const b_ptr1 =
        reinterpret_cast<const uint4 *>(b_base_ptr1 + k_stride_b);
    const uint4 a_val0 = *a_ptr0;
    const uint4 a_val1 = *a_ptr1;
    const uint4 b_val0 = *b_ptr0;
    const uint4 b_val1 = *b_ptr1;
    a_lds_flat[a_vec_id0 ^ (a_row0 & 31)] = a_val0;
    a_lds_flat[a_vec_id1 ^ (a_row1 & 31)] = a_val1;
    b_lds[b_row0][(b_col0 >> 3) ^ (b_row0 & 31)] = b_val0;
    b_lds[b_row1][(b_col1 >> 3) ^ (b_row1 & 31)] = b_val1;
    __syncthreads();
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
extern "C" int lufloat_gemm(const _Float16 *A, const _Float16 *B, _Float16 *C,
                            const unsigned int m, const unsigned int n,
                            const unsigned int k) {
  /*
    @param gridDim: 256x256 tiling.
    @param blockDim: 16 wave 32.
  */
  hipLaunchKernelGGL(gemm_kernel, dim3(n >> 8, m >> 8), dim3(512), 0, 0, A, B,
                     C, n, k);
  // kernel launch error.
  return static_cast<int>(hipGetLastError());
}