#include <hip/hip_runtime.h>

/*
  @param data: same as abs @param data.
  @param magnitude: same as abs @param magnitude.
*/
__global__ void abs_kernel(const unsigned short *__restrict__ data,
                           unsigned short *__restrict__ magnitude) {
  // global vector index.
  const unsigned long long idx =
      ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  // 128bit vectorized load (8 float16 as uint4).
  uint4 vec = *reinterpret_cast<const uint4 *>(&data[idx]);
  // sign bit isolation for unsigned integer as 2 float16 (simd within
  // register).
  vec.x = vec.x & 0x7FFF7FFF;
  vec.y = vec.y & 0x7FFF7FFF;
  vec.z = vec.z & 0x7FFF7FFF;
  vec.w = vec.w & 0x7FFF7FFF;
  // 128bit vectorized store (uint4 as 8 float16).
  *reinterpret_cast<uint4 *>(&magnitude[idx]) = vec;
}

/*
  @param data: pointer to input float16 array.
  @param size: length of input float16 array (must be a multiple of 2048).
  @param magnitude: pointer to output float16 array.
*/
extern "C" int lufloat_abs(const unsigned short *data,
                           const unsigned long long size,
                           unsigned short *magnitude) {
  /*
    @param gridDim: size / (blockDim * sizeof(uint4)/sizeof(float16)).
    @param blockDim: wave32 * 8 wavefronts = 256 threads.
  */
  hipLaunchKernelGGL(abs_kernel, dim3(size >> 11), dim3(256), 0, 0, data,
                     magnitude);
  // kernel launch error.
  return (int)hipGetLastError();
}