#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

/*
  @param data: pointer to input float16 array.
  @param reduced: pointer to output float16 array.
*/
__global__ void sum_kernel(const unsigned short *__restrict__ data,
                           unsigned short *__restrict__ reduced) {
  // global vector index.
  const unsigned long long idx =
      (static_cast<unsigned long long>(blockIdx.x) *
           static_cast<unsigned long long>(blockDim.x) +
       static_cast<unsigned long long>(threadIdx.x))
      << 3;
  // 128bit vectorized load (8 float16 as uint4).
  const uint4 vec = *reinterpret_cast<const uint4 *>(&data[idx]);
  // uint4 as 4 half2.
  const half2 *h = reinterpret_cast<const half2 *>(&vec);
  // 4 half2 sum as half2.
  const half2 thread_tmp = __hadd2(__hadd2(h[0], h[1]), __hadd2(h[2], h[3]));
  // half2 sum as __half.
  __half thread_sum = __hadd(thread_tmp.x, thread_tmp.y);
#pragma unroll
  /*
    thread_sum[0..=15] += thread_sum[16..=31]
    thread_sum[0..=7] += thread_sum[7..=15]
    thread_sum[0..=3] += thread_sum[3..=7]
    thread_sum[0..=1] += thread_sum[1..=3]
    thread_sum[0] += thread_sum[1]
  */
  for (unsigned int i = 16; i > 0; i >>= 1) {
    thread_sum += __shfl_down(thread_sum, i);
  }
  const unsigned int wave = threadIdx.x >> 5;
  const unsigned int lane = threadIdx.x & 31;
  __shared__ __half wave_sums[8];
  if (lane == 0)
    wave_sums[wave] = thread_sum;
  __syncthreads();
  if (wave == 0) {
    __half block_sum = (lane < 8) ? wave_sums[lane] : __float2half(0.0f);
#pragma unroll
    for (unsigned int i = 4; i > 0; i >>= 1) {
      block_sum += __shfl_down(block_sum, i);
    }
    if (lane == 0) {
      reinterpret_cast<__half *>(reduced)[blockIdx.x] = block_sum;
    }
  }
}

/*
  @param data: pointer to input float16 array.
  @param reduced: pointer to output float16 array.
  @param size: length of input float16 array (must be a multiple of 2048).
*/
extern "C" int lufloat_sum(const unsigned short *data, unsigned short *reduced,
                           const unsigned long long size) {
  /*
    @param gridDim: size / (blockDim * sizeof(uint4)/sizeof(float16)).
    @param blockDim: wave32 * 8 wavefronts = 256 threads.
  */
  hipLaunchKernelGGL(sum_kernel, dim3(size >> 11), dim3(256), 0, 0, data,
                     reduced);
  // kernel launch error.
  return static_cast<int>(hipGetLastError());
}
