#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using rocwmma::accumulator;
using rocwmma::col_major;
using rocwmma::fragment;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::row_major;
using rocwmma::fragment_scheduler::coop_col_slice_2d;
using rocwmma::fragment_scheduler::coop_row_major_2d;
using rocwmma::fragment_scheduler::coop_row_slice_2d;
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

constexpr auto TBLOCK_X = 128u;
constexpr auto TBLOCK_Y = 4u;

using SchedulerA = coop_row_slice_2d<TBLOCK_X, TBLOCK_Y>;
using SchedulerB = coop_col_slice_2d<TBLOCK_X, TBLOCK_Y>;
using SchedulerAcc = default_schedule;
using SchedulerOut = coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;

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
            const unsigned int n, const unsigned int k) {}

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