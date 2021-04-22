//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_KERNEL_GPU_TRANSIMAGE_H
#define OPSLIB_KERNEL_GPU_TRANSIMAGE_H
#include <cuda_fp16.h>

namespace quake {
namespace framework {
namespace ops_lib {

void trans_image_NCHW_RGB_gpu(cudaStream_t stream,const char* input,float* output,
  int batchsize,int channel,int input_h,int input_w);

void trans_image_NCHW_RGB_gpu(cudaStream_t stream,const char* input,__half* output,
  int batchsize,int channel,int input_h,int input_w);

void trans_image_NCHW_BGR_gpu(cudaStream_t stream,const char* input,float* output,
  int batchsize,int channel,int input_h,int input_w);

void trans_image_NCHW_BGR_gpu(cudaStream_t stream,const char* input,__half* output,
  int batchsize,int channel,int input_h,int input_w);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_TRANSIMAGE_H
