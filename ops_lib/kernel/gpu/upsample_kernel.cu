//
// Created by chengjin on 2021-03-16.
//

#include "cu_utils.h"
#include "upsample_kernel.h"
#include "cu_device.cuh"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__global__ static void _upsample(const T* input,T* output,int flatten_channel,int height,int width,int out_h,int out_w,T ratio_h,T ratio_w,int align_corners){
  int hw_idx = blockIdx.x * blockDim.x + threadIdx.x;  // out_h
  int c_idx = blockIdx.y * blockDim.y + threadIdx.y;  // flatten_channel
  int h_idx=hw_idx/out_w;
  int w_idx=hw_idx%out_w;
  if(c_idx<flatten_channel && h_idx<out_h && w_idx<out_w){
    //h dimension
    int h = align_corners==0 ? int(ratio_h*(T(h_idx)+T(0.5))-T(0.5)) : int(ratio_h*T(h_idx));
    h=max(0,h);
    int hid = h<(height-1) ? 1 : 0;
    T h1lambda;
    if(align_corners==0){
      T idx_src_h = max(ratio_h * (T(h_idx) + T(0.5)) - T(0.5), T(0));
      h1lambda = idx_src_h - T(h);
    }else{
      h1lambda = ratio_h * T(h_idx) - T(h);
    }
    T h2lambda=T(1)-h1lambda;
    //w dimension
    int w = align_corners==0 ? int(ratio_w*(T(w_idx)+T(0.5))-T(0.5)) : int(ratio_w*T(w_idx));
    w=max(0,w);
    int wid = w<(width-1) ? 1 : 0;
    T w1lambda;
    if(align_corners==0){
      T idx_src_w = max(ratio_w * (T(w_idx) + T(0.5)) - T(0.5), T(0));
      w1lambda = idx_src_w - T(w);
    }else{
      w1lambda = ratio_w * T(w_idx) - T(w);
    }
    T w2lambda=T(1)-w1lambda;
    
    int in_idx1=c_idx*height*width+h*width+w;
    int in_idx2=c_idx*height*width+(h+hid)*width+w;
    output[c_idx*out_h*out_w+h_idx*out_w+w_idx]= \
      h2lambda*(w2lambda*input[in_idx1]+w1lambda*input[in_idx1+wid]) + \
      h1lambda*(w2lambda*input[in_idx2]+w1lambda*input[in_idx2+wid]);
  }
}

template<typename T>
__global__ static void _upsampleH(const T* input,T* output,int flatten_channel,int height,int width,int out_h,int out_w,T ratio_h,T ratio_w,int align_corners){
  int hw_idx = blockIdx.x * blockDim.x + threadIdx.x;  // out_h
  int c_idx = blockIdx.y * blockDim.y + threadIdx.y;  // flatten_channel
  int h_idx=hw_idx/out_w;
  int w_idx=hw_idx%out_w;
  if(c_idx<flatten_channel && h_idx<out_h && w_idx<out_w){
    //h dimension
    int h = align_corners==0 ? int(ratio_h*(T(h_idx)+T(0.5))-T(0.5)) : int(ratio_h*T(h_idx));
    h=max(0,h);
    int hid = h<(height-1) ? 1 : 0;
    T h1lambda;
    if(align_corners==0){
      T idx_src_h = hmax(ratio_h * (T(h_idx) + T(0.5)) - T(0.5), T(0));
      h1lambda = idx_src_h - T(h);
    }else{
      h1lambda = ratio_h * T(h_idx) - T(h);
    }
    T h2lambda=T(1)-h1lambda;
    //w dimension
    int w = align_corners==0 ? int(ratio_w*(T(w_idx)+T(0.5))-T(0.5)) : int(ratio_w*T(w_idx));
    w=max(0,w);
    int wid = w<(width-1) ? 1 : 0;
    T w1lambda;
    if(align_corners==0){
      T idx_src_w = hmax(ratio_w * (T(w_idx) + T(0.5)) - T(0.5), T(0));
      w1lambda = idx_src_w - T(w);
    }else{
      w1lambda = ratio_w * T(w_idx) - T(w);
    }
    T w2lambda=T(1)-w1lambda;
    
    int in_idx1=c_idx*height*width+h*width+w;
    int in_idx2=c_idx*height*width+(h+hid)*width+w;
    output[c_idx*out_h*out_w+h_idx*out_w+w_idx]= \
      h2lambda*(w2lambda*input[in_idx1]+w1lambda*input[in_idx1+wid]) + \
      h1lambda*(w2lambda*input[in_idx2]+w1lambda*input[in_idx2+wid]);
  }
}

//implements
template<typename T>
void bilinear_upsample(cudaStream_t stream,const T* input,T* output,
  int batchsize,int channel,int height,int width,int out_h,int out_w,bool align_corners)
{
  T ratio_h,ratio_w;
  if(align_corners){
    ratio_h=T(height-1)/T(out_h-1);
    ratio_w=T(height-1)/T(out_h-1);
  }else{
    ratio_h=T(height)/T(out_h);
    ratio_w=T(width)/T(out_w);
  }

  int flatten_channel=batchsize * channel;
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(out_h*out_w,CU2DBLOCK),n_blocks(flatten_channel,CU2DBLOCK));
  _upsample<<<Gr,Bl,0,stream>>>(input,output,flatten_channel,height,width,out_h,out_w,ratio_h,ratio_w,int(align_corners));
}

void bilinear_upsample(cudaStream_t stream,const __half* input,__half* output,
  int batchsize,int channel,int height,int width,int out_h,int out_w,bool align_corners){
  float ratio_h,ratio_w;
  if(align_corners){
    ratio_h=float(height-1)/float(out_h-1);
    ratio_w=float(height-1)/float(out_h-1);
  }else{
    ratio_h=float(height)/float(out_h);
    ratio_w=float(width)/float(out_w);
  }

  int flatten_channel=batchsize * channel;
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(out_h*out_w,CU2DBLOCK),n_blocks(flatten_channel,CU2DBLOCK));
  _upsampleH<<<Gr,Bl,0,stream>>>(input,output,flatten_channel,height,width,out_h,out_w,\
    dlr_float2half(ratio_h),dlr_float2half(ratio_w),int(align_corners));
}

template
void bilinear_upsample<float>(cudaStream_t stream,const float* input,float* output,
  int batchsize,int channel,int height,int width,int out_h,int out_w,bool align_corners);

} // namespace ops_lib
} // namespace framework
} // namespace quake
