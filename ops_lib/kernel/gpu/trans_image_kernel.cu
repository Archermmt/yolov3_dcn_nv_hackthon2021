//
// Created by chengjin on 2020-06-02.
//

#include "cu_utils.h"
#include "trans_image_kernel.h"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__global__ static void _trans_image_NCHW_RGB(int batch_size,int channel,int height,int width,
  const char* input,T* output)
{
  int w_idx = blockIdx.x * blockDim.x + threadIdx.x;  // width index
  int h = blockIdx.y * blockDim.y + threadIdx.y;  // batchsize*channel*height index
  int flatten_c=h/height;
  int h_idx=h%height;
  int bz_idx=flatten_c/channel;
  int dst_c=flatten_c%channel;
  if(bz_idx<batch_size && dst_c<channel && h_idx<height && w_idx<width){
    //input [b,h,w,c]->output [b,c,h,w], channel BGR->RGB
    int src_c=2-dst_c;
    int src_idx=bz_idx*(height*width*channel)+h_idx*(width*channel)+w_idx*channel+src_c;
    int dst_idx=bz_idx*(channel*height*width)+dst_c*(height*width)+h_idx*width+w_idx;
    output[dst_idx]=T(uint8_t(input[src_idx]));
  }
}

template<typename T>
__global__ static void _trans_image_NCHW_BGR(int batch_size,int channel,int height,int width,
  const char* input,T* output)
{
  int w_idx = blockIdx.x * blockDim.x + threadIdx.x;  // width index
  int h = blockIdx.y * blockDim.y + threadIdx.y;  // batchsize*channel*height index
  int flatten_c=h/height;
  int h_idx=h%height;
  int bz_idx=flatten_c/channel;
  int c_idx=flatten_c%channel;
  if(bz_idx<batch_size && c_idx<channel && h_idx<height && w_idx<width){
    int src_idx=bz_idx*(height*width*channel)+h_idx*(width*channel)+w_idx*channel+c_idx;
    int dst_idx=bz_idx*(channel*height*width)+c_idx*(height*width)+h_idx*width+w_idx;
    output[dst_idx]=T(uint8_t(input[src_idx]));
  }
}

//implements
void trans_image_NCHW_RGB_gpu(cudaStream_t stream,const char* input,float* output,
  int batchsize,int channel,int input_h,int input_w)
{
  //int flatten_channel=batchsize*channel;
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(input_w,CU2DBLOCK),n_blocks(input_h*batchsize*channel,CU2DBLOCK));
  _trans_image_NCHW_RGB<<<Gr,Bl,0,stream>>>(batchsize,channel,input_h,input_w,input,output);
}

void trans_image_NCHW_RGB_gpu(cudaStream_t stream,const char* input,__half* output,
  int batchsize,int channel,int input_h,int input_w)
{
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(input_w,CU2DBLOCK),n_blocks(input_h*batchsize*channel,CU2DBLOCK));
  _trans_image_NCHW_RGB<<<Gr,Bl,0,stream>>>(batchsize,channel,input_h,input_w,input,output);
}

void trans_image_NCHW_BGR_gpu(cudaStream_t stream,const char* input,float* output,
  int batchsize,int channel,int input_h,int input_w)
{
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(input_w,CU2DBLOCK),n_blocks(input_h*batchsize*channel,CU2DBLOCK));
  _trans_image_NCHW_BGR<<<Gr,Bl,0,stream>>>(batchsize,channel,input_h,input_w,input,output);
}

void trans_image_NCHW_BGR_gpu(cudaStream_t stream,const char* input,__half* output,
  int batchsize,int channel,int input_h,int input_w)
{
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(input_w,CU2DBLOCK),n_blocks(input_h*batchsize*channel,CU2DBLOCK));
  _trans_image_NCHW_BGR<<<Gr,Bl,0,stream>>>(batchsize,channel,input_h,input_w,input,output);
}

} // namespace ops_lib
} // namespace framework
} // namespace quake
