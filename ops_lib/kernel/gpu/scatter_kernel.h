//
// Created by chengjin on 2020-06-30.
//

#ifndef OPSLIB_KERNEL_GPU_SCATTER_H
#define OPSLIB_KERNEL_GPU_SCATTER_H
#include <cuda_fp16.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void scatter_ne_forward_gpu(cudaStream_t stream,const T* cond_input,
  const T* input,T* output,T value,int n);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_SCATTER_H
