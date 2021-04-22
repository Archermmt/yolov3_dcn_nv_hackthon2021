//
// Created by chengjin on 2021-03-15.
//

#ifndef OPSLIB_KERNEL_GPU_NONLINEAR_PRED_BOX_H
#define OPSLIB_KERNEL_GPU_NONLINEAR_PRED_BOX_H
#include <cuda_fp16.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void nonlinear_pred_box(cudaStream_t stream,
  const T* in_data,const T* delta_data,T* output,
  int batchsize,int box_num,int box_dim,int image_h,int image_w);

void nonlinear_pred_box(cudaStream_t stream,
  const __half* in_data,const __half* delta_data,__half* output,
  int batchsize,int box_num,int box_dim,int image_h,int image_w);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_NONLINEAR_PRED_BOX_H
