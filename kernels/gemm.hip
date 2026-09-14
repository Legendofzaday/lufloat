#include <__clang_hip_runtime_wrapper.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

using rocwmma::accumulator;
using rocwmma::fill_fragment;
using rocwmma::fragment;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::fragment_scheduler::default_schedule;

using DATA_LAYOUT_A = rocwmma::row_major;
using DATA_LAYOUT_B = rocwmma::col_major;
using DATA_LAYOUT_C = rocwmma::row_major;

constexpr auto ROCWMMA_M = 16u;
constexpr auto ROCWMMA_N = 16u;
constexpr auto ROCWMMA_K = 16u;

using FragA = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, _Float16,
                       DATA_LAYOUT_A, default_schedule>;
using FragB = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, _Float16,
                       DATA_LAYOUT_B, default_schedule>;
using FragC = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float,
                       DATA_LAYOUT_C, default_schedule>;

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
  const auto *const r_A = reinterpret_cast<const uint4 *>(A);
  const auto *const r_B = reinterpret_cast<const uint4 *>(B);
  auto *const rw_C = reinterpret_cast<uint4 *>(C);
  const auto wave = threadIdx.x >> 5;
  const auto lane = threadIdx.x & 31;
  const auto wave_m = wave << 4 & ~63;
  const auto wave_n = wave << 6 & 192;
  __shared__ uint4 Alds[2][256][4];
  __shared__ uint4 Blds[2][256][4];
  FragC fragC[4][4];
#pragma unroll
  for (auto i = 0u; i < 4; ++i) {
#pragma unroll
    for (auto j = 0u; j < 4; ++j) {
      fill_fragment(fragC[i][j], 0.0f);
    }
  }
  const auto load_a_row_or_b_col = threadIdx.x >> 1;
  const auto load_a_col_or_b_row = (threadIdx.x << 1) & 2;
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