#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void sum_kernel(const unsigned short *__restrict__ data,
                           unsigned short *__restrict__ reduced) {
  const unsigned long long idx =
      (static_cast<unsigned long long>(blockIdx.x) *
           static_cast<unsigned long long>(blockDim.x) +
       static_cast<unsigned long long>(threadIdx.x))
      << 3;
  const uint4 vec = *reinterpret_cast<const uint4 *>(&data[idx]);
  const half2 *h = reinterpret_cast<const half2 *>(&vec);
  const half2 thread_tmp = __hadd2(__hadd2(h[0], h[1]), __hadd2(h[2], h[3]));
  __half thread_sum = thread_tmp.x + thread_tmp.y;
#pragma unroll
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

extern "C" int lufloat_sum(const unsigned short *data,
                           unsigned short *reduced,
                           const unsigned long long size) {
  hipLaunchKernelGGL(sum_kernel, dim3(size >> 11), dim3(256), 0, 0, data,
                     reduced);
  return static_cast<int>(hipGetLastError());
}
