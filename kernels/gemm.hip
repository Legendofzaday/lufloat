#include <hip/hip_runtime.h>

using half16 = _Float16 __attribute__((ext_vector_type(16)));

/*
  @param A: pointer to input float16 array.
  @param B: pointer to input float16 array.
  @param C: pointer to output float16 array.
  @param n: cols of B || cols of C.
  @param k: cols of A || rows of B.
*/
__global__ void __launch_bounds__(128)
    gemm_kernel(const unsigned short *__restrict__ A,
                const unsigned short *__restrict__ B,
                unsigned short *__restrict__ C, const unsigned int n,
                const unsigned int k) {
  const _Float16 *const A_f16 = reinterpret_cast<const _Float16 *>(A);
  const _Float16 *const B_f16 = reinterpret_cast<const _Float16 *>(B);
  _Float16 *const C_f16 = reinterpret_cast<_Float16 *>(C);
  // 0..=127.
  const unsigned int tid = threadIdx.x;
  // 0..=3.
  const unsigned int wave = tid >> 5;
  // 0..=31.
  const unsigned int lane = tid & 31;
  // 0 | 1
  // =====
  // 2 | 3
  const unsigned int wave_row = wave >> 1 << 5;
  const unsigned int wave_col = (wave & 1) << 5;
  // 0..=1.
  const unsigned int half = lane >> 4;
  // 0..=15.
  const unsigned int lane16 = lane & 15;
  // 0..=m-1.64.
  const unsigned int block_row = blockIdx.y << 6;
  // 0..=n-1.64.
  const unsigned int block_col = blockIdx.x << 6;
  __shared__ _Float16 A_lds_t[2][16][64];
  __shared__ _Float16 B_lds_t[2][64][16];
  half16 cfrag[2][2] = {};
  // 0..=63.
  const unsigned int a_load_row = tid >> 1;
  // 0 || 8.
  const unsigned int a_load_col = (tid & 1) << 3;
  // 0..=15.
  const unsigned int b_load_row = tid >> 3;
  // 0 || 32.
  const unsigned int b_load_col = (tid & 7) << 3;
  // 128bit vectorized load.
  const uint4 a_vec = *reinterpret_cast<const uint4 *>(
      &A_f16[(block_row + a_load_row) * k + a_load_col]);
  // uint4 as 8 float16.
  const _Float16 *const a_vals = reinterpret_cast<const _Float16 *>(&a_vec);
#pragma unroll
  // swizzled store.
  for (int i = 0; i < 8; ++i) {
    A_lds_t[0][a_load_col + i][a_load_row] = a_vals[i];
  }
  // 128bit vectorized load.
  const uint4 b_vec = *reinterpret_cast<const uint4 *>(
      &B_f16[b_load_row * n + (block_col + b_load_col)]);
  // uint4 as 8 float16.
  const _Float16 *const b_vals = reinterpret_cast<const _Float16 *>(&b_vec);
#pragma unroll
  // swizzled store.
  for (int i = 0; i < 8; ++i) {
    B_lds_t[0][b_load_col + i][b_load_row] = b_vals[i];
  }
  // block level sync.
  __syncthreads();
  // ping pong write.
  unsigned int write_phase = 0;
  // ping pong read.
  unsigned int read_phase = 0;
  for (unsigned int kk = 0; kk < k; kk += 16) {
    if (kk + 16 < k) {
      // 0..=1.
      write_phase ^= 1;
      const unsigned int next_kk = kk + 16;
      // 128bit vectorized load.
      const uint4 a_next = *reinterpret_cast<const uint4 *>(
          &A_f16[(block_row + a_load_row) * k + (next_kk + a_load_col)]);
      // uint4 as 8 float16.
      const _Float16 *const a_nvals =
          reinterpret_cast<const _Float16 *>(&a_next);
#pragma unroll
      // swizzled store.
      for (int i = 0; i < 8; ++i) {
        A_lds_t[write_phase][a_load_col + i][a_load_row] = a_nvals[i];
      }
      // 128bit vectorized load.
      const uint4 b_next = *reinterpret_cast<const uint4 *>(
          &B_f16[(next_kk + b_load_row) * n + (block_col + b_load_col)]);
      // uint4 as 8 float16.
      const _Float16 *const b_nvals =
          reinterpret_cast<const _Float16 *>(&b_next);
#pragma unroll
      // swizzled store.
      for (int i = 0; i < 8; ++i) {
        B_lds_t[write_phase][b_load_col + i][b_load_row] = b_nvals[i];
      }
    }
    // 256bit vectorized load.
    const half16 a_frag0 = *reinterpret_cast<const half16 *>(
        &A_lds_t[read_phase][lane16][wave_row]);
    // 256bit vectorized load.
    const half16 a_frag1 = *reinterpret_cast<const half16 *>(
        &A_lds_t[read_phase][lane16][wave_row + 16]);
    // 256bit vectorized load.
    const half16 b_frag0 = *reinterpret_cast<const half16 *>(
        &B_lds_t[read_phase][wave_col + lane16][0]);
    // 256bit vectorized load.
    const half16 b_frag1 = *reinterpret_cast<const half16 *>(
        &B_lds_t[read_phase][wave_col + 16 + lane16][0]);
    cfrag[0][0] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
        a_frag0, b_frag0, cfrag[0][0], false);
    cfrag[0][1] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
        a_frag0, b_frag1, cfrag[0][1], false);
    cfrag[1][0] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
        a_frag1, b_frag0, cfrag[1][0], false);
    cfrag[1][1] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
        a_frag1, b_frag1, cfrag[1][1], false);
    // 0..=1.
    read_phase ^= 1;
    // block level sync.
    __syncthreads();
  }
#pragma unroll
  for (int tm = 0; tm < 2; ++tm) {
#pragma unroll
    for (int tn = 0; tn < 2; ++tn) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const unsigned int row =
            block_row + wave_row + (tm << 4) + (e << 1) + half;
        const unsigned int col = block_col + wave_col + (tn << 4) + lane16;
        C_f16[row * n + col] = cfrag[tm][tn][e << 1];
      }
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
extern "C" int lufloat_gemm(const unsigned short *A, const unsigned short *B,
                            unsigned short *C, const unsigned int m,
                            const unsigned int n, const unsigned int k) {
  /*
    @param gridDim: 64x64 tiling.
    @param blockDim: 4 wave 32.
  */
  hipLaunchKernelGGL(gemm_kernel, dim3(n >> 6, m >> 6), dim3(128), 0, 0, A, B,
                     C, n, k);
  // kernel launch error.
  return static_cast<int>(hipGetLastError());
}