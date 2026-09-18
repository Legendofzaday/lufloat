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

constexpr unsigned int ROCWMMA_M = 16u;
constexpr unsigned int ROCWMMA_N = 16u;
constexpr unsigned int ROCWMMA_K = 16u;

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
using FragC = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float,
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
  // blockIdx * tile size.
  const unsigned int block_m = blockIdx.y << 8;
  const unsigned int block_n = blockIdx.x << 8;
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
  const unsigned int wave_m = threadIdx.x >> 7;
  const unsigned int wave_n = threadIdx.x >> 5 & 3;
  // 0..=15.
  const unsigned int wave = threadIdx.x >> 5;
  // 0..=31.
  const unsigned int lane = threadIdx.x & 31;
  // 16 element jump from low to high.
  const unsigned int a_k_offset = m << 1;
  // 16 element jump from low to high.
  const unsigned int b_k_offset = n << 1;
  // 32 element jump at end of tile.
  const unsigned int a_k_step = m << 2;
  // 32 element jump at end of tile.
  const unsigned int b_k_step = n << 2;
  // 256x256x32 double buffering.
  __shared__ uint4 Alds[2][32][32];
  __shared__ uint4 Blds[2][32][32];
  // accumulator.
  FragC c_frag[4][4];
#pragma unroll
  for (unsigned int i = 0u; i < 4; ++i) {
#pragma unroll
    for (unsigned int j = 0u; j < 4; ++j) {
      fill_fragment(c_frag[i][j], 0.0f);
    }
  }
  // double buffering mode.
  unsigned int phase = 0u;
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