#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

/*
  @param data: same as lufloat_mul_inplace @param data.
  @param other: same as lufloat_mul_inplace @param other.
*/
__global__ void mul_inplace_kernel(unsigned short *__restrict__ data,
                                   const unsigned short *__restrict__ other) {
  // global vector index.
  const unsigned long long idx =
      ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  // 128bit vectorized load (8 float16 as uint4).
  uint4 vec = *reinterpret_cast<const uint4 *>(&data[idx]);
  const uint4 vec_other = *reinterpret_cast<const uint4 *>(&other[idx]);
  // uint4 as 4 half2.
  half2 *h = reinterpret_cast<half2 *>(&vec);
  const half2 *h_other = reinterpret_cast<const half2 *>(&vec_other);
  h[0] = __hmul2(h[0], h_other[0]);
  h[1] = __hmul2(h[1], h_other[1]);
  h[2] = __hmul2(h[2], h_other[2]);
  h[3] = __hmul2(h[3], h_other[3]);
  // 128bit vectorized store (uint4 as 8 float16).
  *reinterpret_cast<uint4 *>(&data[idx]) = vec;
}

/*
  @param data: pointer to input float16 array.
  @param other: pointer to input float16 other.
  @param size: length of input float16 array (must be a multiple of 2048).
*/
extern "C" int lufloat_mul_inplace(unsigned short *data,
                                   const unsigned short *other,
                                   const unsigned long long size) {
  /*
    @param gridDim: size / (blockDim * sizeof(uint4)/sizeof(float16)).
    @param blockDim: wave32 * 8 wavefronts = 256 threads.
  */
  hipLaunchKernelGGL(mul_inplace_kernel, dim3(size >> 11), dim3(256), 0, 0,
                     data, other);
  // kernel launch error.
  return (int)hipGetLastError();
}