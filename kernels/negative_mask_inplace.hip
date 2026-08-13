#include <hip/hip_runtime.h>


// @param data: same as negative_mask_inplace @param data.
__global__ void negative_mask_inplace_kernel(unsigned short *data) {
  const unsigned long long idx =
      ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  const uint4 vec = *reinterpret_cast<const uint4 *>(&data[idx]);
  uint4 out_vec;
  out_vec.x = (vec.x >> 15) & 0x00010001;
  out_vec.y = (vec.y >> 15) & 0x00010001;
  out_vec.z = (vec.z >> 15) & 0x00010001;
  out_vec.w = (vec.w >> 15) & 0x00010001;
  *reinterpret_cast<uint4 *>(&data[idx]) = out_vec;
}

/*
  @param data: pointer to input float16 array.
  @param size: length of input float16 array (must be a multiple of 2048).
*/
extern "C" void negative_mask_inplace(unsigned short *data,
                                      const unsigned long long size) {
  /*
    @param gridDim: size / (blockDim * sizeof(uint4)/sizeof(float16)).
    @param blockDim: wave32 * 8 wavefronts = 256 threads.
  */
  hipLaunchKernelGGL(negative_mask_inplace_kernel, dim3(size >> 11), dim3(256),
                     0, 0, data);
}