#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

/*
  @param data: same as lufloat_swiglu @param data.
  @param gate: same as lufloat_swiglu @param gate.
  @param activation: same as lufloat_swiglu @param activation.
*/
__global__ void swiglu_kernel(const unsigned short *__restrict__ data,
                              const unsigned short *__restrict__ gate,
                              unsigned short *__restrict__ activation) {
  // global vector index.
  const unsigned long long idx =
      ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  // 128bit vectorized load (8 float16 as uint4).
  uint4 vec = *reinterpret_cast<const uint4 *>(&data[idx]);
  const uint4 vec_gate = *reinterpret_cast<const uint4 *>(&gate[idx]);
  // uint4 as 4 half2.
  half2 *h = reinterpret_cast<half2 *>(&vec);
  const half2 *h_gate = reinterpret_cast<const half2 *>(&vec_gate);
  // -log2(e).
  const half2 exp_coef = __builtin_bit_cast(half2, 0xBDC5BDC5U);
  // 1.0.
  const half2 one = __builtin_bit_cast(half2, 0x3C003C00U);
  const half2 y0 = __hmul2(h[0], exp_coef);
  const half2 y1 = __hmul2(h[1], exp_coef);
  const half2 y2 = __hmul2(h[2], exp_coef);
  const half2 y3 = __hmul2(h[3], exp_coef);
  const half2 e0 = h2exp2(y0);
  const half2 e1 = h2exp2(y1);
  const half2 e2 = h2exp2(y2);
  const half2 e3 = h2exp2(y3);
  const half2 d0 = __hadd2(e0, one);
  const half2 d1 = __hadd2(e1, one);
  const half2 d2 = __hadd2(e2, one);
  const half2 d3 = __hadd2(e3, one);
  const half2 r0 = h2rcp(d0);
  const half2 r1 = h2rcp(d1);
  const half2 r2 = h2rcp(d2);
  const half2 r3 = h2rcp(d3);
  const half2 s0 = __hmul2(h[0], r0);
  const half2 s1 = __hmul2(h[1], r1);
  const half2 s2 = __hmul2(h[2], r2);
  const half2 s3 = __hmul2(h[3], r3);
  h[0] = __hmul2(h_gate[0], s0);
  h[1] = __hmul2(h_gate[1], s1);
  h[2] = __hmul2(h_gate[2], s2);
  h[3] = __hmul2(h_gate[3], s3);
  // 128bit vectorized store (uint4 as 8 float16).
  *reinterpret_cast<uint4 *>(&activation[idx]) = vec;
}

/*
  @param data: pointer to input float16 array.
  @param gate: pointer to input float16 array.
  @param size: length of input float16 array (must be a multiple of 2048).
  @param activation: pointer to output float16 array.
*/
extern "C" int lufloat_swiglu(const unsigned short *data,
                              const unsigned short *gate,
                              const unsigned long long size,
                              unsigned short *activation) {
  /*
    @param gridDim: size / (blockDim * sizeof(uint4)/sizeof(float16)).
    @param blockDim: wave32 * 8 wavefronts = 256 threads.
  */
  hipLaunchKernelGGL(swiglu_kernel, dim3(size >> 11), dim3(256), 0, 0, data,
                     gate, activation);
  // kernel launch error.
  return (int)hipGetLastError();
}