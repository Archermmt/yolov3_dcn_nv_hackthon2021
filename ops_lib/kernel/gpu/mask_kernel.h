//
// Created by chengjin on 2021-02-24.
//

#ifndef OPSLIB_KERNEL_GPU_MASK_H
#define OPSLIB_KERNEL_GPU_MASK_H
#include <cuda_fp16.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void mask_ne_forward_gpu(cudaStream_t stream,const T* input,T* output,T value,int n);

template<typename T>
void mask_gt_forward_gpu(cudaStream_t stream,const T* input,T* output,T value,int n);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_MASK_H
