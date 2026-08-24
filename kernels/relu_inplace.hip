#include <hip/hip_runtime.h>

// @param data: same as lufloat_relu_inplace @param data.
__global__ void relu_inplace_kernel(unsigned short *data) {
  // global vector index.
  const unsigned long long idx =
      ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  // 128bit vectorized load (8 float16 as uint4).
  uint4 vec = *reinterpret_cast<const uint4 *>(&data[idx]);
  // sign bit isolation for unsigned integer as 2 float16 (simd within
  // register).
  vec.x = vec.x & ~(((vec.x >> 15) & 0x00010001) * 0xFFFF);
  vec.y = vec.y & ~(((vec.y >> 15) & 0x00010001) * 0xFFFF);
  vec.z = vec.z & ~(((vec.z >> 15) & 0x00010001) * 0xFFFF);
  vec.w = vec.w & ~(((vec.w >> 15) & 0x00010001) * 0xFFFF);
  // 128bit vectorized store (uint4 as 8 float16).
  *reinterpret_cast<uint4 *>(&data[idx]) = vec;
}

/*
  @param data: pointer to input float16 array.
  @param size: length of input float16 array (must be a multiple of 2048).
*/
extern "C" int lufloat_relu_inplace(unsigned short *data,
                                    const unsigned long long size) {
  /*
    @param gridDim: size / (blockDim * sizeof(uint4)/sizeof(float16)).
    @param blockDim: wave32 * 8 wavefronts = 256 threads.
  */
  hipLaunchKernelGGL(relu_inplace_kernel, dim3(size >> 11), dim3(256), 0, 0,
                     data);
  // kernel launch error.
  return (int)hipGetLastError();
}