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