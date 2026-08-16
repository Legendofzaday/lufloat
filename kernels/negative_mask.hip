#include <hip/hip_runtime.h>

/*
  @param data: same as negative_mask @param data.
  @param mask: same as negative_mask @param mask.
*/
__global__ void negative_mask_kernel(const unsigned short *__restrict__ data,
                                     unsigned short *__restrict__ mask) {
  // global vector index.
  const unsigned long long idx =
      ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  // 128bit vectorized load (8 float16 as uint4).
  const uint4 vec = *reinterpret_cast<const uint4 *>(&data[idx]);
  // temporary output container.
  uint4 out_vec;
  // sign bit isolation for unsigned integer as 2 float16 (simd within
  // register).
  out_vec.x = (vec.x >> 15) & 0x00010001;
  out_vec.y = (vec.y >> 15) & 0x00010001;
  out_vec.z = (vec.z >> 15) & 0x00010001;
  out_vec.w = (vec.w >> 15) & 0x00010001;
  // 128bit vectorized store (uint4 as 8 float16).
  *reinterpret_cast<uint4 *>(&mask[idx]) = out_vec;
}

/*
  @param data: pointer to input float16 array.
  @param size: length of input float16 array (must be a multiple of 2048).
  @param mask: pointer to output float16 array.
*/
extern "C" int negative_mask(const unsigned short *data,
                             const unsigned long long size,
                             unsigned short *mask) {
  /*
    @param gridDim: size / (blockDim * sizeof(uint4)/sizeof(float16)).
    @param blockDim: wave32 * 8 wavefronts = 256 threads.
  */
  hipLaunchKernelGGL(negative_mask_kernel, dim3(size >> 11), dim3(256), 0, 0,
                     data, mask);
  // kernel launch error.
  return (int)hipGetLastError();
}