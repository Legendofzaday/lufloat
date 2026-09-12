#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

/*
  @param data: pointer to input and float16 array.
  @param probabilities: pointer to output float16 array.
  @param cols: length of rows.
  @param rows: total number of rows.
*/
template <unsigned int BLOCK_THREADS, unsigned int ROWS_PER_BLOCK>
__global__ void __attribute__((amdgpu_flat_work_group_size(256, 256)))
softmax_kernel(const unsigned short *__restrict__ data,
               unsigned short *__restrict__ probabilities,
               const unsigned int cols, const unsigned int rows) {
  const auto row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
  const auto active = (row < rows);
  const auto block_offset =
      static_cast<unsigned long long>(active ? row : 0u) * cols;
  // input float16 array.
  const auto *const r_data =
      reinterpret_cast<const _Float16 *>(data) + block_offset;
  // output float16 array.
  auto *const w_probabilities =
      reinterpret_cast<_Float16 *>(probabilities) + block_offset;
  float thread_max = -65504.0f;
  // stride loop.
  for (auto i = threadIdx.x << 3u; i < cols; i += BLOCK_THREADS << 3) {
    // 128bit vectorized load.
    const auto vec = *reinterpret_cast<const uint4 *>(&r_data[i]);
    // uint4 as 8 float16.
    const auto *const h = reinterpret_cast<const _Float16 *>(&vec);
#pragma unroll
    for (auto j = 0u; j < 8; ++j) {
      thread_max = fmaxf(thread_max, h[j]);
    }
  }
#pragma unroll
  /*
    thread_max[lane0..=lane15] = fmaxf(thread_max[lane0..=lane15],
    thread_max[lane16..=lane31]) thread_max[lane0..=lane7] =
    fmaxf(thread_max[lane0..=lane7], thread_max[lane8..=lane15])
    thread_max[lane0..=lane3] = fmaxf(thread_max[lane0..=lane3],
    thread_max[lane4..=lane7]) thread_max[lane0..=lane1] =
    fmaxf(thread_max[lane0..=lane1], thread_max[lane2..=lane3])
    thread_max[lane0] = fmaxf(thread_max[lane0], thread_max[lane1])
  */
  for (auto i = 16u; i > 0; i >>= 1) {
    thread_max = fmaxf(thread_max, __shfl_down(thread_max, i));
  }
  // 0..=7.
  const auto wave = threadIdx.x >> 5u;
  // 0..=31.
  const auto lane = threadIdx.x & 31u;
  __shared__ float wave_shared[ROWS_PER_BLOCK][8u];
  if (lane == 0)
    wave_shared[threadIdx.y][wave] = thread_max;
  __syncthreads();
  if (wave == 0) {
    auto block_max = (lane < (BLOCK_THREADS >> 5))
                         ? wave_shared[threadIdx.y][lane]
                         : -65504.0f;
#pragma unroll
    for (auto i = 4u; i > 0; i >>= 1) {
      block_max = fmaxf(block_max, __shfl_down(block_max, i));
    }
    if (lane == 0) {
      wave_shared[threadIdx.y][0u] = block_max;
    }
  }
  __syncthreads();
  const auto rmax = wave_shared[threadIdx.y][0u];
  float thread_sum = 0.0f;
  // stride loop.
  for (auto i = threadIdx.x << 3u; i < cols; i += BLOCK_THREADS << 3) {
    // 128bit vectorized load.
    const auto vec = *reinterpret_cast<const uint4 *>(&r_data[i]);
    // uint4 as 8 float16.
    const auto *const h = reinterpret_cast<const _Float16 *>(&vec);
#pragma unroll
    for (auto j = 0u; j < 8; ++j) {
      thread_sum += exp2f((h[j] - rmax) * 1.44269504f);
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
  if (lane == 0)
    wave_shared[threadIdx.y][wave] = thread_sum;
  __syncthreads();
  if (wave == 0) {
    auto block_sum =
        (lane < (BLOCK_THREADS >> 5)) ? wave_shared[threadIdx.y][lane] : 0.0f;
#pragma unroll
    /*
      block_sum[0..=3] += block_sum[4..=7]
      block_sum[0..=1] += block_sum[2..=3]
      block_sum[0] += block_sum[1]
    */
    for (auto i = 4u; i > 0; i >>= 1) {
      block_sum += __shfl_down(block_sum, i);
    }
    if (lane == 0)
      wave_shared[threadIdx.y][0u] = __frcp_rn(block_sum);
  }
  __syncthreads();
  const auto inv_sum = wave_shared[threadIdx.y][0u];
  // stride loop.
  for (auto i = threadIdx.x << 3u; i < cols; i += BLOCK_THREADS << 3) {
    // 128bit vectorized load.
    const auto vec = *reinterpret_cast<const uint4 *>(&r_data[i]);
    // uint4 as 8 float16.
    const auto *const h = reinterpret_cast<const _Float16 *>(&vec);
    uint4 out_vec;
    auto *const h_out = reinterpret_cast<_Float16 *>(&out_vec);
#pragma unroll
    for (auto j = 0u; j < 8; ++j) {
      h_out[j] =
          static_cast<_Float16>(exp2f((h[j] - rmax) * 1.44269504f) * inv_sum);
    }
    // 128bit vectorized store.
    if (active)
      *reinterpret_cast<uint4 *>(&w_probabilities[i]) = out_vec;
  }
}

/*
  @param data: pointer to input and float16 array.
  @param probabilities: pointer to output float16 array.
  @param cols: length of rows.
  @param rows: total number of rows.
*/
extern "C" int lufloat_softmax(const unsigned short *data,
                               unsigned short *probabilities,
                               const unsigned int cols,
                               const unsigned int rows) {
  if (cols % 2048 == 0) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(softmax_kernel<256, 1>), dim3(rows),
                       dim3(256, 1), 0, 0, data, probabilities, cols, rows);
  } else if (cols % 1024 == 0) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(softmax_kernel<128, 2>),
                       dim3((rows + 1) >> 1), dim3(128, 2), 0, 0, data,
                       probabilities, cols, rows);
  } else if (cols % 512 == 0) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(softmax_kernel<64, 4>),
                       dim3((rows + 3) >> 2), dim3(64, 4), 0, 0, data,
                       probabilities, cols, rows);
  } else if (cols % 8 == 0) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(softmax_kernel<32, 8>),
                       dim3((rows + 7) >> 3), dim3(32, 8), 0, 0, data,
                       probabilities, cols, rows);
  } else {
    return -1;
  }
  // kernel launch error.
  return static_cast<int>(hipGetLastError());
}