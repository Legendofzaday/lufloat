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
#pragma unroll
    for (auto j = 0; j < 2; ++j) {
      const auto k_offset = j << 4;
      half16 a_frag[4];
#pragma unroll
      for (auto l = 0; l < 4; ++l) {
        const auto r = (wave_m << 6) + (l << 4) + (lane & 15);
        const auto c_base = k_offset >> 3;
        auto *const a_dst = reinterpret_cast<uint4 *>(&a_frag[l]);
        a_dst[0] = a_lds_flat[((r << 2) + c_base + 0) ^ (r & 31)];
        a_dst[1] = a_lds_flat[((r << 2) + c_base + 1) ^ (r & 31)];
      }
      half16 b_frag[4];
#pragma unroll
      for (auto e = 0; e < 4; ++e) {
        const auto r = k_offset + (lane & 15);
        const auto c_base = (wave_n << 3) + (e << 1);
        auto *const b_dst = reinterpret_cast<uint4 *>(&b_frag[e]);
        b_dst[0] = b_lds[r][(c_base + 0) ^ (r & 31)];
        b_dst[1] = b_lds[r][(c_base + 1) ^ (r & 31)];
      }
#pragma unroll
      for (auto l = 0; l < 4; ++l) {
#pragma unroll
        for (auto e = 0; e < 4; ++e) {
          c_frag[l][e] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
              a_frag[l], b_frag[e], c_frag[l][e], false);
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (auto l = 0; l < 4; ++l) {
    const auto c_row = block_row + (wave_m << 6) + (l << 4) + (lane & 15);
    _Float16 *const c_row_ptr = C + static_cast<unsigned long long>(c_row) * n;
#pragma unroll
    for (auto e = 0; e < 4; ++e) {
      const auto c_col =
          block_col + (wave_n << 6) + (e << 4) + ((lane >> 4) << 3);
      auto *const c_dst = reinterpret_cast<uint4 *>(c_row_ptr + c_col);
      *c_dst = *reinterpret_cast<const uint4 *>(&c_frag[l][e]);
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