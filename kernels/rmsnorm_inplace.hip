#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

/*
  @param data: pointer to input and output float16 array.
  @param weight: pointer to input float16 array.
  @param cols: length of rows.
  @param eps: epsilon.
*/
template <unsigned int BLOCK_THREADS>
__global__ void
rmsnorm_inplace_kernel(unsigned short *__restrict__ data,
                       const unsigned short *__restrict__ weight,
                       const unsigned int cols, const float eps) {
  // input float16 array.
  auto *const rw_data = reinterpret_cast<_Float16 *>(data) +
                        static_cast<unsigned long long>(blockIdx.x) * cols;
  // read only input float16 array.
  const auto *const r_weight = reinterpret_cast<const _Float16 *>(weight);
  // accumulator.
  auto thread_sum = 0.0f;
  // stride loop.
  for (auto i = threadIdx.x << 3u; i < cols; i += BLOCK_THREADS << 3) {
    // 128bit vectorized load.
    const auto vec = *reinterpret_cast<const uint4 *>(&rw_data[i]);
    // uint4 as 8 float16.
    const auto *const h = reinterpret_cast<const _Float16 *>(&vec);
#pragma unroll
    // \Sigma_{i=0}^7x_i^2
    for (auto j = 0u; j < 8; ++j) {
      thread_sum = fmaf(h[j], h[j], thread_sum);
    }
  }
#pragma unroll
  /*
    thread_sum[lane0..=lane15] += thread_sum[lane16..=lane31]
    thread_sum[lane0..=lane7] += thread_sum[lane8..=lane15]
    thread_sum[lane0..=lane3] += thread_sum[lane4..=lane7]
    thread_sum[lane0..=lane1] += thread_sum[lane2..=lane3]
    thread_sum[lane0] += thread_sum[lane1]
  */
  for (auto i = 16u; i > 0; i >>= 1) {
    thread_sum += __shfl_down(thread_sum, i);
  }
  // 0..=7.
  const auto wave = threadIdx.x >> 5u;
  // 0..=31.
  const auto lane = threadIdx.x & 31u;
  __shared__ float wave_sums[32u];
  if (lane == 0)
    wave_sums[wave] = thread_sum;
  __syncthreads();
  if (wave == 0) {
    auto block_sum = (lane < BLOCK_THREADS >> 5) ? wave_sums[lane] : 0.0f;
#pragma unroll
    /*
      block_sum[0..=3] += block_sum[4..=7]
      block_sum[0..=1] += block_sum[2..=3]
      block_sum[0] += block_sum[1]
    */
    for (auto i = 4u; i > 0; i >>= 1) {
      block_sum += __shfl_down(block_sum, i);
    }
    if (lane == 0) {
      // \frac{1}{\sqrt{\frac{block\_sum}{cols} + eps}}
      wave_sums[0u] = rsqrtf(block_sum / cols + eps);
    }
  }
  __syncthreads();
  const auto rrms = wave_sums[0u];
  // stride loop.
  for (auto i = threadIdx.x << 3u; i < cols; i += BLOCK_THREADS << 3) {
    // 128bit vectorized load.
    const auto vec = *reinterpret_cast<const uint4 *>(&rw_data[i]);
    // 128bit vectorized load.
    const auto vec_weight = *reinterpret_cast<const uint4 *>(&r_weight[i]);
    // uint4 as 8 float16.
    const auto *const h = reinterpret_cast<const _Float16 *>(&vec);
    // uint4 as 8 float16.
    const auto *const h_weight =
        reinterpret_cast<const _Float16 *>(&vec_weight);
    uint4 out_vec;
    // uint4 as 8 float16.
    auto *const h_out = reinterpret_cast<_Float16 *>(&out_vec);
#pragma unroll
    for (auto j = 0u; j < 8; ++j) {
      h_out[j] = h[j] * h_weight[j] * rrms;
    }
    // 128bit vectorized store.
    *reinterpret_cast<uint4 *>(&rw_data[i]) = out_vec;
  }
}

/*
  @param data: pointer to input and output float16 array.
  @param weight: pointer to input float16 array.
  @param cols: length of rows.
  @param rows: length of cols.
  @param eps: epsilon.
*/
extern "C" int lufloat_rmsnorm(unsigned short *data,
                               const unsigned short *weight,
                               const unsigned int cols, const unsigned int rows,
                               const float eps) {
  if (cols % 2048 == 0) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rmsnorm_inplace_kernel<256>), dim3(rows),
                       dim3(256), 0, 0, data, weight, cols, eps);
  } else if (cols % 1024 == 0) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rmsnorm_inplace_kernel<128>), dim3(rows),
                       dim3(128), 0, 0, data, weight, cols, eps);
  } else if (cols % 512 == 0) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rmsnorm_inplace_kernel<64>), dim3(rows),
                       dim3(64), 0, 0, data, weight, cols, eps);
  } else if (cols % 256 == 0) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rmsnorm_inplace_kernel<32>), dim3(rows),
                       dim3(32), 0, 0, data, weight, cols, eps);
  } else {
    return -1;
  }
  // kernel launch error.
  return static_cast<int>(hipGetLastError());
}