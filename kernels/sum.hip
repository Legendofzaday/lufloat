#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void sum_pass1_kernel(const unsigned short *__restrict__ data,
                                 double *__restrict__ partials) {
  const unsigned long long idx =
      ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  const uint4 vec = *reinterpret_cast<const uint4 *>(&data[idx]);
  const half2 *h = reinterpret_cast<const half2 *>(&vec);
  const float2 f0 = __half22float2(h[0]);
  const float2 f1 = __half22float2(h[1]);
  const float2 f2 = __half22float2(h[2]);
  const float2 f3 = __half22float2(h[3]);
  float thread_sum = f0.x + f0.y + f1.x + f1.y + f2.x + f2.y + f3.x + f3.y;
  const unsigned int wave = threadIdx.x >> 5;
  const unsigned int lane = threadIdx.x & 31;
  for (unsigned int offset = 16; offset > 0; offset >>= 1) {
    thread_sum += __shfl_down(thread_sum, offset);
  }
  __shared__ double wave_sums[8];
  if (lane == 0)
    wave_sums[wave] = (double)thread_sum;
  __syncthreads();
  if (wave == 0) {
    double block_sum = (lane < 8) ? wave_sums[lane] : 0.0;
    for (unsigned int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_down(block_sum, offset);
    }
    if (lane == 0) {
      partials[blockIdx.x] = block_sum;
    }
  }
}

__global__ void sum_pass2_kernel(const double *__restrict__ partials,
                                 double *__restrict__ final_sum,
                                 const unsigned long long size) {
  const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride = blockDim.x * gridDim.x;
  double thread_sum = 0.0;
  for (unsigned long long i = idx; i < size; i += stride) {
    thread_sum += partials[i];
  }
  const unsigned int wave = threadIdx.x >> 5;
  const unsigned int lane = threadIdx.x & 31;
  for (unsigned int offset = 16; offset > 0; offset >>= 1) {
    thread_sum += __shfl_down(thread_sum, offset);
  }
  __shared__ double wave_sums[8];
  if (lane == 0)
    wave_sums[wave] = thread_sum;
  __syncthreads();
  if (wave == 0) {
    double block_sum = (lane < 8) ? wave_sums[lane] : 0.0;
    for (unsigned int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_down(block_sum, offset);
    }
    if (lane == 0) {
      atomicAdd(final_sum, block_sum);
    }
  }
}

extern "C" int lufloat_sum(const unsigned short *data,
                           const unsigned long long size, double *partials,
                           double *final_sum) {

  unsigned long long pass1_blocks = size >> 11;
  hipLaunchKernelGGL(sum_pass1_kernel, dim3(pass1_blocks), dim3(256), 0, 0,
                     data, partials);
  unsigned int pass2_blocks =
      (pass1_blocks > 1024) ? 1024 : (unsigned int)pass1_blocks;
  if (pass2_blocks == 0)
    pass2_blocks = 1;
  hipLaunchKernelGGL(sum_pass2_kernel, dim3(pass2_blocks), dim3(256), 0, 0,
                     partials, final_sum, pass1_blocks);
  return (int)hipGetLastError();
}
