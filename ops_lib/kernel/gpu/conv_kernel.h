//
// Created by chengjin on 2021-03-08.
//

#ifndef OPSLIB_KERNEL_GPU_CONV_H
#define OPSLIB_KERNEL_GPU_CONV_H
#include <cuda_fp16.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void deformable_conv_dmc(cudaStream_t stream,
  const T* input,const T* offset,const T* mask,T* output, \
  int batchsize,int channel,int height,int width,int out_height,int out_width, \
  int kernel_h,int kernel_w,int stride_h,int stride_w,int pad_h,int pad_w, \
  int dilation_h,int dilation_w);

void deformable_conv_dmc(cudaStream_t stream,
  const __half* input,const __half* offset,const __half* mask,__half* output, \
  int batchsize,int channel,int height,int width,int out_height,int out_width, \
  int kernel_h,int kernel_w,int stride_h,int stride_w,int pad_h,int pad_w, \
  int dilation_h,int dilation_w);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_CONV_H
