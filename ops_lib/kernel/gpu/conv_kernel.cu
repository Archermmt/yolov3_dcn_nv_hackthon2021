//
// Created by chengjin on 2021-03-08.
//

#include "cu_utils.h"
#include "conv_kernel.h"
#include "cu_device.cuh"
#include <math.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__device__ inline T dmc_bilinear(const T* input,int height,int width,T im_h,T im_w){
  T out=T(0);
  if(im_h>T(-1) && im_w>T(-1) && im_h<T(height) && im_w<T(width)){
    int h_low = floor(im_h);
    int w_low = floor(im_w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    T lh = im_h - T(h_low);
    T lw = im_w - T(w_low);
    T hh = T(1) - lh;
    T hw = T(1) - lw;
  
    T v1 = (h_low>=0 && w_low>=0) ? input[h_low*width+w_low] : T(0);
    T v2 = (h_low>=0 && w_high<=(width-1)) ? input[h_low*width+w_high] : T(0);
    T v3 = (h_high<=(height-1) && w_low>=0) ? input[h_high*width+w_low] : T(0);
    T v4 = (h_high<=(height-1) && w_high<=(width-1)) ? input[h_high*width+w_high] : T(0);

    out = hh*hw*v1 + hh*lw*v2 + lh*hw*v3 + lh*lw*v4;
  }
  return out;
}

template<typename T>
__global__ static void _deformable_conv_dmc(
  const T* input,const T* offset,const T* mask,T* output, \
  int batchsize,int channel,int height,int width,int out_height,int out_width, \
  int kernel_h,int kernel_w,int stride_h,int stride_w,int pad_h,int pad_w, \
  int dilation_h,int dilation_w){
  int w = blockIdx.x * blockDim.x + threadIdx.x;  // width index kerne_h*kernel_w*out_h*out_w
  int h = blockIdx.y * blockDim.y + threadIdx.y;  // height index bz*channel
  int index=h*out_height*out_width+w;
  //run kernel
  if(w<(out_height*out_width) && h<(batchsize*channel*kernel_h*kernel_w)){
    //h dim [bz*channel] where channel=in_channel*kernel_h*kernel_w
    int stride=channel*kernel_h*kernel_w;
    int stride_len=h%stride;
    int idx_n=h/stride;
    int idx_c=stride_len/(kernel_h*kernel_w);
    stride_len=stride_len%(kernel_h*kernel_w);
    int idx_kh=stride_len/kernel_w;
    int idx_kw=stride_len%kernel_w;
    // w dim out_height*out_width
    int idx_outh=w/out_width;
    int idx_outw=w%out_width;
    //offset index by [n,kh*kw*2,out_h,out_w]
    int offset_h_idx=idx_n*kernel_h*kernel_w*2*out_height*out_width+(idx_kh*kernel_w+idx_kw)*2*out_height*out_width+idx_outh*out_width+idx_outw;
    int offset_w_idx=offset_h_idx+out_height*out_width;
    //mask index by [n,kh*kw,out_h,out_w]
    int mask_idx=idx_n*kernel_h*kernel_w*out_height*out_width+(idx_kh*kernel_w+idx_kw)*out_height*out_width+idx_outh*out_width+idx_outw;
    T im_h = T(idx_outh*stride_h+idx_kh*dilation_h-pad_h)+offset[offset_h_idx];
    T im_w = T(idx_outw*stride_w+idx_kw*dilation_w-pad_w)+offset[offset_w_idx];
    T out=dmc_bilinear(input+idx_n*channel*height*width+idx_c*height*width,height,width,im_h,im_w);
    output[index]=out*mask[mask_idx];
  }
}

template<typename T>
__device__ inline T dmc_bilinearH(const T* input,int height,int width,T im_h,T im_w){
  T out=T(0);
  if(im_h>T(-1) && im_w>T(-1) && im_h<T(height) && im_w<T(width)){
    int h_low = hfloor(im_h);
    int w_low = hfloor(im_w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    T lh = im_h - T(h_low);
    T lw = im_w - T(w_low);
    T hh = T(1) - lh;
    T hw = T(1) - lw;
  
    T v1 = (h_low>=0 && w_low>=0) ? input[h_low*width+w_low] : T(0);
    T v2 = (h_low>=0 && w_high<=(width-1)) ? input[h_low*width+w_high] : T(0);
    T v3 = (h_high<=(height-1) && w_low>=0) ? input[h_high*width+w_low] : T(0);
    T v4 = (h_high<=(height-1) && w_high<=(width-1)) ? input[h_high*width+w_high] : T(0);

    out = hh*hw*v1 + hh*lw*v2 + lh*hw*v3 + lh*lw*v4;
  }
  return out;
}

template<typename T>
__global__ static void _deformable_conv_dmcH(
  const T* input,const T* offset,const T* mask,T* output, \
  int batchsize,int channel,int height,int width,int out_height,int out_width, \
  int kernel_h,int kernel_w,int stride_h,int stride_w,int pad_h,int pad_w, \
  int dilation_h,int dilation_w){
  int w = blockIdx.x * blockDim.x + threadIdx.x;  // width index kerne_h*kernel_w*out_h*out_w
  int h = blockIdx.y * blockDim.y + threadIdx.y;  // height index bz*channel
  int index=h*out_height*out_width+w;
  //run kernel
  if(w<(out_height*out_width) && h<(batchsize*channel*kernel_h*kernel_w)){
    //h dim [bz*channel] where channel=in_channel*kernel_h*kernel_w
    int stride=channel*kernel_h*kernel_w;
    int stride_len=h%stride;
    int idx_n=h/stride;
    int idx_c=stride_len/(kernel_h*kernel_w);
    stride_len=stride_len%(kernel_h*kernel_w);
    int idx_kh=stride_len/kernel_w;
    int idx_kw=stride_len%kernel_w;
    // w dim out_height*out_width
    int idx_outh=w/out_width;
    int idx_outw=w%out_width;
    //offset index by [n,kh*kw*2,out_h,out_w]
    int offset_h_idx=idx_n*kernel_h*kernel_w*2*out_height*out_width+(idx_kh*kernel_w+idx_kw)*2*out_height*out_width+idx_outh*out_width+idx_outw;
    int offset_w_idx=offset_h_idx+out_height*out_width;
    //mask index by [n,kh*kw,out_h,out_w]
    int mask_idx=idx_n*kernel_h*kernel_w*out_height*out_width+(idx_kh*kernel_w+idx_kw)*out_height*out_width+idx_outh*out_width+idx_outw;
    T im_h = T(idx_outh*stride_h+idx_kh*dilation_h-pad_h)+offset[offset_h_idx];
    T im_w = T(idx_outw*stride_w+idx_kw*dilation_w-pad_w)+offset[offset_w_idx];
    T out=dmc_bilinearH(input+idx_n*channel*height*width+idx_c*height*width,height,width,im_h,im_w);
    output[index]=out*mask[mask_idx];
  }
}

template<typename T>
void deformable_conv_dmc(cudaStream_t stream, \
  const T* input,const T* offset,const T* mask,T* output, \
  int batchsize,int channel,int height,int width,int out_height,int out_width, \
  int kernel_h,int kernel_w,int stride_h,int stride_w,int pad_h,int pad_w, \
  int dilation_h,int dilation_w){
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(out_height*out_width,CU2DBLOCK),n_blocks(batchsize*channel*kernel_h*kernel_w,CU2DBLOCK));
  _deformable_conv_dmc<<<Gr,Bl,0,stream>>>(
    input,offset,mask,output, \
    batchsize,channel,height,width,out_height,out_height, \
    kernel_h,kernel_w,stride_h,stride_w,pad_h,pad_w, \
    dilation_h,dilation_w);
}

void deformable_conv_dmc(cudaStream_t stream, \
  const __half* input,const __half* offset,const __half* mask,__half* output, \
  int batchsize,int channel,int height,int width,int out_height,int out_width, \
  int kernel_h,int kernel_w,int stride_h,int stride_w,int pad_h,int pad_w, \
  int dilation_h,int dilation_w){
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(out_height*out_width,CU2DBLOCK),n_blocks(batchsize*channel*kernel_h*kernel_w,CU2DBLOCK)); 
  _deformable_conv_dmcH<<<Gr,Bl,0,stream>>>(
    input,offset,mask,output, \
    batchsize,channel,height,width,out_height,out_height, \
    kernel_h,kernel_w,stride_h,stride_w,pad_h,pad_w, \
    dilation_h,dilation_w);
}

template
void deformable_conv_dmc<float>(cudaStream_t stream, \
  const float* input,const float* offset,const float* mask,float* output, \
  int batchsize,int channel,int height,int width,int out_height,int out_width, \
  int kernel_h,int kernel_w,int stride_h,int stride_w,int pad_h,int pad_w, \
  int dilation_h,int dilation_w);

} // namespace ops_lib
} // namespace framework
} // namespace quake
