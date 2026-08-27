#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

/*
  @param data: same as lufloat_sum @param data.
  @param sum: same as lufloat_sum @param sum.
*/
__global__ void sum_kernel(const unsigned short *__restrict__ data,
                           double *__restrict__ sum) {
  // global vector index.
  const unsigned long long idx =
      ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  // 128bit vectorized load (8 float16 as uint4).
  const uint4 vec = *reinterpret_cast<const uint4 *>(&data[idx]);
  // uint4 as 4 half2.
  const half2 *h = reinterpret_cast<const half2 *>(&vec);
  // half2 as float2.
  const float2 f0 = __half22float2(h[0]);
  const float2 f1 = __half22float2(h[1]);
  const float2 f2 = __half22float2(h[2]);
  const float2 f3 = __half22float2(h[3]);
  float thread_sum = f0.x + f0.y + f1.x + f1.y + f2.x + f2.y + f3.x + f3.y;
  // wavefront (0-7).
  const unsigned int wave = threadIdx.x >> 5;
  // wave32 (0-31).
  const unsigned int lane = threadIdx.x & 31;
  /*
    threads 0..=15 += threads 16..=31
    threads 0..=7 += threads 8..=15
    threads 0..=3 += threads 4..=7
    threads 0..=1 += threads 2..=3
    thread 0 += thread 1
  */
  for (unsigned int offset = 16; offset > 0; offset >>= 1) {
    thread_sum += __shfl_down(thread_sum, offset);
  }
  __shared__ float wave_sums[8];
  if (lane == 0)
    wave_sums[wave] = thread_sum;
  __syncthreads();
  if (wave == 0) {
    float block_sum = (lane < 8) ? wave_sums[lane] : 0.0f;
    for (unsigned int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_down(block_sum, offset);
    }
    if (lane == 0) {
      atomicAdd(sum, (double)block_sum);
    }
  }
}

/*
  @param data: pointer to input float16 array.
  @param size: length of input float16 array (must be a multiple of 2048).
  @param sum: pointer to output float64.
*/
extern "C" int lufloat_sum(const unsigned short *data,
                           const unsigned long long size, double *sum) {
  /*
    @param gridDim: size / (blockDim * sizeof(uint4)/sizeof(float16)).
    @param blockDim: wave32 * 8 wavefronts = 256 threads.
  */
  hipLaunchKernelGGL(sum_kernel, dim3(size >> 11), dim3(256), 0, 0, data, sum);
  // kernel launch error.
  return (int)hipGetLastError();
}