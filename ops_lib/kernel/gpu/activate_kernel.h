//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_KERNEL_GPU_LEAKYRELU_H
#define OPSLIB_KERNEL_GPU_LEAKYRELU_H
#include <cuda_fp16.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void leaky_relu_forward_gpu(cudaStream_t stream,const T* input,const T* gamma,
  T* output,int batchsize,int channel,int height,int width);

template<typename T>
void gelu_forward_gpu(cudaStream_t stream,const T* src,T* dst,int ele_size);

void gelu_forward_gpu(cudaStream_t stream,const __half* src,__half* dst,int ele_size);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_LEAKYRELU_H
