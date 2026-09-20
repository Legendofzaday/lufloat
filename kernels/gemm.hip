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
  const auto block_row = blockIdx.y << 8;
  const auto block_col = blockIdx.x << 8;
  const auto lane = threadIdx.x & 31;
  const auto wave_m = threadIdx.x >> 7;
  const auto wave_n = threadIdx.x >> 5 & 3;
  const auto a_r_base = threadIdx.x >> 3;
  const auto a_c = threadIdx.x & 7;
  const auto a_col = a_c << 3;
  const auto b_r_base = threadIdx.x >> 5;
  const auto b_c = threadIdx.x & 31;
  const auto b_col = b_c << 3;
  const auto *const a_ptr0 =
      A + static_cast<unsigned long long>(block_row + a_r_base) * k + a_col;
  const auto *const a_ptr1 =
      A + static_cast<unsigned long long>(block_row + a_r_base + 64) * k +
      a_col;
  const auto *const a_ptr2 =
      A + static_cast<unsigned long long>(block_row + a_r_base + 128) * k +
      a_col;
  const auto *const a_ptr3 =
      A + static_cast<unsigned long long>(block_row + a_r_base + 192) * k +
      a_col;
  const auto *const b_ptr0 =
      B + static_cast<unsigned long long>(b_r_base) * n + (block_col + b_col);
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
  for (unsigned int i = 0; i < k; i += 64) {
    const auto k_stride_b = static_cast<unsigned long long>(i) * n;
    a_lds_flat[(threadIdx.x) ^ (a_r_base & 7)] =
        *reinterpret_cast<const uint4 *>(a_ptr0 + i);
    a_lds_flat[(threadIdx.x + 512) ^ (a_r_base & 7)] =
        *reinterpret_cast<const uint4 *>(a_ptr1 + i);
    a_lds_flat[(threadIdx.x + 1024) ^ (a_r_base & 7)] =
        *reinterpret_cast<const uint4 *>(a_ptr2 + i);
    a_lds_flat[(threadIdx.x + 1536) ^ (a_r_base & 7)] =
        *reinterpret_cast<const uint4 *>(a_ptr3 + i);
    b_lds[b_r_base][b_c ^ ((b_r_base) & 31)] =
        *reinterpret_cast<const uint4 *>(b_ptr0 + k_stride_b);
    b_lds[b_r_base + 16][b_c ^ ((b_r_base + 16) & 31)] =
        *reinterpret_cast<const uint4 *>(b_ptr1 + k_stride_b);
    b_lds[b_r_base + 32][b_c ^ ((b_r_base + 32) & 31)] =
        *reinterpret_cast<const uint4 *>(b_ptr2 + k_stride_b);
    b_lds[b_r_base + 48][b_c ^ ((b_r_base + 48) & 31)] =
        *reinterpret_cast<const uint4 *>(b_ptr3 + k_stride_b);
    __syncthreads();
#pragma unroll
    for (auto j = 0; j < 4; ++j) {
      const auto k_offset = j << 4;
      half16 a_frag[4];
#pragma unroll
      for (auto l = 0; l < 4; ++l) {
        const auto r = (wave_m << 6) + (l << 4) + (lane & 15);
        const auto c_base = k_offset >> 3;
        auto *const a_dst = reinterpret_cast<uint4 *>(&a_frag[l]);
        a_dst[0] = a_lds_flat[((r << 3) + c_base) ^ (r & 7)];
        a_dst[1] = a_lds_flat[((r << 3) + c_base + 1) ^ (r & 7)];
      }
#pragma unroll
      for (auto l = 0; l < 4; ++l) {
        half16 b_frag;
        const auto r = k_offset + (lane & 15);
        const auto c = (wave_n << 3) + (l << 1);
        auto *const b_dst = reinterpret_cast<uint4 *>(&b_frag);
        b_dst[0] = b_lds[r][(c) ^ (r & 31)];
        b_dst[1] = b_lds[r][(c + 1) ^ (r & 31)];
#pragma unroll
        for (auto e = 0; e < 4; ++e) {
          c_frag[e][l] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              a_frag[e], b_frag, c_frag[e][l]);
        }
      }
    }
    __syncthreads();
  }
  const auto c_row_offset = block_row + (wave_m << 6) + (lane & 15);
  const auto c_col_offset = block_col + (wave_n << 6) + (lane >> 4 << 3);
  uint4 out_vec;
#pragma unroll
  for (auto i = 0; i < 4; ++i) {
    _Float16 *const c_row_ptr =
        C + n * static_cast<unsigned long long>(c_row_offset + (i << 4));
#pragma unroll
    for (auto l = 0; l < 4; ++l) {
      out_vec.x = __builtin_bit_cast(
          unsigned int,
          __builtin_amdgcn_cvt_pkrtz(c_frag[i][l][0], c_frag[i][l][1]));
      out_vec.y = __builtin_bit_cast(
          unsigned int,
          __builtin_amdgcn_cvt_pkrtz(c_frag[i][l][2], c_frag[i][l][3]));
      out_vec.z = __builtin_bit_cast(
          unsigned int,
          __builtin_amdgcn_cvt_pkrtz(c_frag[i][l][4], c_frag[i][l][5]));
      out_vec.w = __builtin_bit_cast(
          unsigned int,
          __builtin_amdgcn_cvt_pkrtz(c_frag[i][l][6], c_frag[i][l][7]));
      *reinterpret_cast<uint4 *>(c_row_ptr + c_col_offset + (l << 4)) = out_vec;
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