//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_KERNEL_GPU_GROUPNORM_H
#define OPSLIB_KERNEL_GPU_GROUPNORM_H
#include <cuda_fp16.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void groupnorm_forward_gpu(cudaStream_t stream,const T* input,
  const T* gamma,const T* beta,
  T* buffer,T* mean,T* var,T* output,
  int group,int batchsize,int channel,int input_h,int input_w,T eps);

void groupnorm_forward_gpu(cudaStream_t stream,const __half* input,
  const __half* gamma,const __half* beta,
  __half* buffer,__half* mean,__half* var,__half* output,
  int group,int batchsize,int channel,int input_h,int input_w,__half eps);

template<typename T>
void layernorm_forward_gpu(cudaStream_t stream,const T* input,
  const T* gamma,const T* beta,
  T* buffer,T* mean,T* var,T* output,
  int batchsize,int layer_len,int layer_dim,T eps);

void layernorm_forward_gpu(cudaStream_t stream,const __half* input,
  const __half* gamma,const __half* beta,
  __half* buffer,__half* mean,__half* var,__half* output,
  int batchsize,int layer_len,int layer_dim,__half eps);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_GROUPNORM_H
