#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using rocwmma::accumulator;
using rocwmma::col_major;
using rocwmma::fill_fragment;
using rocwmma::fragment;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::row_major;
using rocwmma::store_matrix_sync;
using rocwmma::fragment_scheduler::default_schedule;

constexpr auto ROCWMMA_M = 16u;
constexpr auto ROCWMMA_N = 16u;
constexpr auto ROCWMMA_K = 16u;

using DataTypeA = _Float16;
using DataTypeB = _Float16;
using DataTypeAcc = float;
using DataTypeOut = _Float16;

using DataLayoutA = col_major;
using DataLayoutB = row_major;
using DataLayoutAcc = row_major;
using DataLayoutOut = row_major;

using SchedulerA = default_schedule;
using SchedulerB = default_schedule;
using SchedulerAcc = default_schedule;
using SchedulerOut = default_schedule;

using FragA = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, DataTypeA,
                       DataLayoutA, SchedulerA>;
using FragB = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, DataTypeB,
                       DataLayoutB, SchedulerB>;
using FragAcc = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                         DataTypeAcc, DataLayoutAcc, SchedulerAcc>;
using FragOut = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                         DataTypeOut, DataLayoutOut, SchedulerOut>;

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
  // 256x256x32 double buffering.
  __shared__ uint4 Alds[2][32][32];
  __shared__ uint4 Blds[2][32][32];
  Alds[0][wave][lane ^ wave] = a_ptr[0];
  Alds[0][wave | 16][lane ^ (wave | 16)] = a_ptr[m << 1];
  Blds[0][wave][lane ^ wave] = b_ptr[0];
  Blds[0][wave | 16][lane ^ (wave | 16)] = b_ptr[n << 1];
  // prefetch registers.
  uint4 ra[2];
  uint4 rb[2];
  // accumulator.
  FragAcc acc_frag[4][4];
#pragma unroll
  for (auto i = 0u; i < 4; ++i) {
#pragma unroll
    for (auto j = 0u; j < 4; ++j) {
      fill_fragment(acc_frag[i][j], 0.0f);
    }
  }
  // double buffering mode.
  auto phase = 0u;
  __syncthreads();
  for (auto i = 0u; i < k; i += 32) {
    const auto next = i + 32;
    // start prefetch.
    if (next < k) {
      a_ptr += (m << 2);
      b_ptr += (n << 2);
      ra[0] = a_ptr[0];
      ra[1] = a_ptr[m << 1];
      rb[0] = b_ptr[0];
      rb[1] = b_ptr[n << 1];
    }
    // finish prefetch.
    if (next < k) {
      const auto write_phase = phase ^ 1u;
      Alds[write_phase][wave][lane ^ wave] = ra[0];
      Alds[write_phase][wave | 16u][lane ^ (wave | 16)] = ra[1];
      Blds[write_phase][wave][lane ^ wave] = rb[0];
      Blds[write_phase][wave | 16u][lane ^ (wave | 16)] = rb[1];
      phase = write_phase;
      __syncthreads();
    }
  }
  // vectorized downcast store.
#pragma unroll
  for (auto i = 0u; i < 4; ++i) {
    const auto row = (blockIdx.y << 8) + (threadIdx.x >> 7 << 6) + (i << 4);
#pragma unroll
    for (auto j = 0u; j < 4; ++j) {
      const auto col =
          (blockIdx.x << 8) + ((threadIdx.x >> 5 & 3) << 6) + (j << 4);
      _Float16 *const c_ptr = C + row * n + col;
      FragOut out_frag;
#pragma unroll
      for (auto k = 0u; k < 8; ++k) {
        out_frag.x[k] = static_cast<_Float16>(acc_frag[i][j].x[k]);
      }
      store_matrix_sync(c_ptr, out_frag, n);
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