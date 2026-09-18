#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using rocwmma::accumulator;
using rocwmma::col_major;
using rocwmma::fill_fragment;
using rocwmma::fragment;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::row_major;
using rocwmma::fragment_scheduler::default_schedule;

constexpr auto ROCWMMA_M = 16u;
constexpr auto ROCWMMA_N = 16u;
constexpr auto ROCWMMA_K = 16u;

using DataTypeA = _Float16;
using DataTypeB = _Float16;
using DataTypeAcc = float;

using DataLayoutA = col_major;
using DataLayoutB = row_major;
using DataLayoutAcc = row_major;

using SchedulerA = default_schedule;
using SchedulerB = default_schedule;
using SchedulerAcc = default_schedule;

using FragA = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, _Float16,
                       DataLayoutA, default_schedule>;
using FragB = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, _Float16,
                       DataLayoutB, default_schedule>;
using FragAcc = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float,
                         DataLayoutAcc, default_schedule>;
using FragOut = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, _Float16,
                         DataLayoutAcc, default_schedule>;

/*
  @param A: pointer to input float16 array.
  @param B: pointer to input float16 array.
  @param C: pointer to output float16 array.
  @param m: rows of A || rows of C.
  @param n: cols of B || cols of C.
  @param k: cols of A || rows of B.
*/
__global__ void __attribute__((amdgpu_flat_work_group_size(512, 512)))
gemm_kernel(const _Float16 *__restrict__ A, const _Float16 *__restrict__ B,
            _Float16 *__restrict__ C, const unsigned int m,
            const unsigned int n, const unsigned int k) {
  /*
    -------------------------------------------------------------
    | col 0..=63 | col 64..=127 | col 128..=191 | col 192..=255 |
    -------------------------------------------------------------
    | 64x64 | row 0..=63    | (0, 0) | (0, 1) | (0, 2) | (0, 3) |
    | 64x64 | row 64..=127  | (1, 0) | (1, 1) | (1, 2) | (1, 3) |
    | 64x64 | row 128..=191 | (2, 0) | (2, 1) | (2, 2) | (2, 3) |
    | 64x64 | row 192..=255 | (3, 0) | (3, 1) | (3, 2) | (3, 3) |
    -------------------------------------------------------------
  */
  // 0..=15.
  const auto wave = threadIdx.x >> 5;
  // 0..=31.
  const auto lane = threadIdx.x & 31;
  // input float16 array pointers.
  const uint4 *a_ptr =
      reinterpret_cast<const uint4 *>(A + wave * m + (blockIdx.y << 8)) + lane;
  const uint4 *b_ptr =
      reinterpret_cast<const uint4 *>(B + wave * n + (blockIdx.x << 8)) + lane;
  // col ^ row xor swizzle.
  const auto swizzled_lane_lo = lane ^ wave;
  const auto swizzled_lane_hi = lane ^ (wave | 16u);
  // 256x256x32 double buffering.
  __shared__ uint4 Alds[2][32][32];
  __shared__ uint4 Blds[2][32][32];
  Alds[0][wave][swizzled_lane_lo] = a_ptr[0];
  Alds[0][wave | 16u][swizzled_lane_hi] = a_ptr[m << 1];
  Blds[0][wave][swizzled_lane_lo] = b_ptr[0];
  Blds[0][wave | 16u][swizzled_lane_hi] = b_ptr[n << 1];
  // prefetch registers.
  uint4 ra[2];
  uint4 rb[2];
  // accumulator.
  FragAcc c_frag[4][4];
#pragma unroll
  for (auto i = 0u; i < 4; ++i) {
#pragma unroll
    for (auto j = 0u; j < 4; ++j) {
      fill_fragment(c_frag[i][j], 0.0f);
    }
  }
  // double buffering mode.
  auto phase = 0u;
  __syncthreads();
  for (auto i = 0u; i < k; i += 32) {
  }
#pragma unroll
  for (auto i = 0u; i < 4; ++i) {
    const auto c_row = (blockIdx.y << 8) + (threadIdx.x >> 1) + (i << 4);
#pragma unroll
    for (auto j = 0u; j < 4u; ++j) {
      const auto c_col =
          (blockIdx.x << 8) + ((threadIdx.x >> 5 & 3) << 6) + (j << 4);
      _Float16 *const c_ptr = C + c_row * n + c_col;
      FragOut out_frag;
#pragma unroll
      for (auto e = 0u; e < 8u; ++e) {
        out_frag.x[e] = static_cast<_Float16>(c_frag[i][j].x[e]);
      }
      rocwmma::store_matrix_sync(c_ptr, out_frag, n);
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
                     C, m, n, k);
  // kernel launch error.
  return static_cast<int>(hipGetLastError());
}