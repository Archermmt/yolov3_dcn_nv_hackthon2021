//
// Created by chengjin on 2021-03-16.
//

#ifndef OPSLIB_KERNEL_GPU_UPSAMPLE_H
#define OPSLIB_KERNEL_GPU_UPSAMPLE_H
#include <cuda_fp16.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void bilinear_upsample(cudaStream_t stream,const T* input,T* output,
  int batchsize,int channel,int height,int width,int out_h,int out_w,bool align_corners);

void bilinear_upsample(cudaStream_t stream,const __half* input,__half* output,
  int batchsize,int channel,int height,int width,int out_h,int out_w,bool align_corners);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_UPSAMPLE_H
