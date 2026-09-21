#include <hip/hip_runtime.h>

using u32x4 = unsigned int __attribute__((ext_vector_type(4)));
using half16 = _Float16 __attribute__((ext_vector_type(16)));
using float8 = float __attribute__((ext_vector_type(8)));

__global__ void __attribute__((amdgpu_flat_work_group_size(512, 512)))
gemm_kernel(const _Float16 *__restrict__ A, const _Float16 *__restrict__ B,
            _Float16 *__restrict__ C, const unsigned int n,
            const unsigned int k) {
  __shared__ uint4 a_lds[256][8];
  __shared__ uint4 b_lds[64][32];
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
  const auto block_row = block_m << 8;
  const auto block_col = block_n << 8;
  const auto wave_m = threadIdx.x >> 5 >> 2;
  const auto wave_n = threadIdx.x >> 5 & 3;
  const auto lane = threadIdx.x & 31;
  const auto a_row = threadIdx.x >> 3;
  const auto a_col = (threadIdx.x & 7) << 3;
  const auto b_row = threadIdx.x >> 5;
  const auto b_col = (threadIdx.x & 31) << 3;
  const auto *a_ptr =
      A + (block_row + a_row) * static_cast<unsigned long long>(k) + a_col;
  const auto *b_ptr =
      B + b_row * static_cast<unsigned long long>(n) + (block_col + b_col);
  const auto a_step = k << 6;
  const auto bc_step = n << 4;
  {
    const auto c = threadIdx.x & 7;
#pragma unroll
    for (auto i = 0; i < 4; ++i) {
      const auto r = a_row + (i << 6);
      a_lds[r][c ^ (r & 7)] =
          *reinterpret_cast<const uint4 *>(a_ptr + i * a_step);
    }
  }
  {
    const auto c = threadIdx.x & 31;
#pragma unroll
    for (auto i = 0; i < 4; ++i) {
      const auto r = b_row + (i << 4);
      b_lds[r][c ^ (r & 31)] =
          *reinterpret_cast<const uint4 *>(b_ptr + i * bc_step);
    }
  }
  __syncthreads();
  uint4 a_prefetch[4];
  uint4 b_prefetch[4];
  asm volatile(""
               : "+v"(*reinterpret_cast<u32x4 *>(&a_prefetch[0])),
                 "+v"(*reinterpret_cast<u32x4 *>(&a_prefetch[1])),
                 "+v"(*reinterpret_cast<u32x4 *>(&a_prefetch[2])),
                 "+v"(*reinterpret_cast<u32x4 *>(&a_prefetch[3])));

  asm volatile(""
               : "+v"(*reinterpret_cast<u32x4 *>(&b_prefetch[0])),
                 "+v"(*reinterpret_cast<u32x4 *>(&b_prefetch[1])),
                 "+v"(*reinterpret_cast<u32x4 *>(&b_prefetch[2])),
                 "+v"(*reinterpret_cast<u32x4 *>(&b_prefetch[3])));

  float8 c_frag[4][4] = {};
  const auto b_step = static_cast<unsigned long long>(bc_step) << 2;
  for (auto i = 0; i < k; i += 64) {
    const auto next = i + 64;
    if (next < k) {
#pragma unroll
      for (auto j = 0; j < 4; ++j) {
        a_prefetch[j] =
            *reinterpret_cast<const uint4 *>(a_ptr + 64 + j * a_step);
      }
#pragma unroll
      for (auto j = 0; j < 4; ++j) {
        b_prefetch[j] =
            *reinterpret_cast<const uint4 *>(b_ptr + b_step + j * bc_step);
      }
    }
    half16 a_frag[4];
    half16 b_frag;
#pragma unroll
    for (auto j = 0; j < 4; ++j) {
      const auto k_offset = j << 4;
      const auto c_base = j << 1;
#pragma unroll
      for (auto e = 0; e < 4; ++e) {
        const auto r = (wave_m << 6) + (e << 4) + (lane & 15);
        auto *const a_dst = reinterpret_cast<uint4 *>(&a_frag[e]);
        a_dst[0] = a_lds[r][(c_base + 0) ^ (r & 7)];
        a_dst[1] = a_lds[r][(c_base + 1) ^ (r & 7)];
      }
#pragma unroll
      for (auto l = 0; l < 4; ++l) {
        const auto r = k_offset + (lane & 15);
        const auto c = (wave_n << 3) + (l << 1);
        auto *const b_dst = reinterpret_cast<uint4 *>(&b_frag);
        b_dst[0] = b_lds[r][(c + 0) ^ (r & 31)];
        b_dst[1] = b_lds[r][(c + 1) ^ (r & 31)];
#pragma unroll
        for (auto e = 0; e < 4; ++e) {
          c_frag[e][l] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              a_frag[e], b_frag, c_frag[e][l]);
        }
      }
    }
    if (next < k) {
      __syncthreads();
      a_ptr += 64;
      b_ptr += b_step;
      {
        const auto c = threadIdx.x & 7;
#pragma unroll
        for (auto j = 0; j < 4; ++j) {
          const auto r = a_row + (j << 6);
          a_lds[r][c ^ (r & 7)] = a_prefetch[j];
        }
      }
      {
        const auto c = threadIdx.x & 31;
#pragma unroll
        for (auto j = 0; j < 4; ++j) {
          const auto r = b_row + (j << 4);
          b_lds[r][c ^ (r & 31)] = b_prefetch[j];
        }
      }
      __syncthreads();
    }
  }
  const auto c_r_base = block_row + (wave_m << 6) + (lane & 15);
  const auto c_c_base = block_col + (wave_n << 6) + (lane >> 4 << 3);
  _Float16 *const c_base =
      C + static_cast<unsigned long long>(c_r_base) * n + c_c_base;
#pragma unroll
  for (auto e = 0; e < 4; ++e) {
    _Float16 *const c_row_ptr = c_base + e * bc_step;
#pragma unroll
    for (auto l = 0; l < 4; ++l) {
      uint4 out;
      out.x = __builtin_bit_cast(
          unsigned int,
          __builtin_amdgcn_cvt_pkrtz(c_frag[e][l][0], c_frag[e][l][1]));
      out.y = __builtin_bit_cast(
          unsigned int,
          __builtin_amdgcn_cvt_pkrtz(c_frag[e][l][2], c_frag[e][l][3]));
      out.z = __builtin_bit_cast(
          unsigned int,
          __builtin_amdgcn_cvt_pkrtz(c_frag[e][l][4], c_frag[e][l][5]));
      out.w = __builtin_bit_cast(
          unsigned int,
          __builtin_amdgcn_cvt_pkrtz(c_frag[e][l][6], c_frag[e][l][7]));
      *reinterpret_cast<uint4 *>(c_row_ptr + (l << 4)) = out;
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
  // kernel launch error.
  return static_cast<int>(hipGetLastError());
}