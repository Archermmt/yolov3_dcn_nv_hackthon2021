//
// Created by chengjin on 2020-06-02.
//

#include "cu_utils.h"
#include "activate_kernel.h"
#include "cu_device.cuh"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__global__ static void _leaky_relu(
  int batchsize,int channel,int height,int width,
  const T* input,const T* gamma,T* output) 
{
  int w = blockIdx.x * blockDim.x + threadIdx.x;  // width index
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // height*batchsize*channel index
  int h=y%height;
  int c=y/height;

  int gamma_idx=c%channel;
  int idx = w+h*width+c*height*width;
  if (c<channel*batchsize && w < width && h < height)
  {
    if (input[idx]<T(0))
      output[idx]=input[idx]*gamma[gamma_idx];
    else
      output[idx]=input[idx];
  }
}

template<typename T>
__global__ static void _gelu(const int n,const T* src,T* dst){
  KERNEL_LOOP(index,n){
    //0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
    T act=src[index]*T(0.7978845608)*(T(1.0)+T(0.044715)*src[index]*src[index]);
    dst[index]=T(0.5)*src[index]*(T(1)+_tanh(act));
  }
}

template<typename T>
__global__ static void _geluH(const int n,const T* src,T* dst){
  KERNEL_LOOP(index,n){
    //0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
    T act=src[index]*T(0.7978845608)*(T(1.0)+T(0.044715)*src[index]*src[index]);
    dst[index]=T(0.5)*src[index]*(T(1)+_tanhH(act));
  }
}

//implements
template<typename T>
void leaky_relu_forward_gpu(cudaStream_t stream,const T* input,const T* gamma,
  T* output,int batchsize,int channel,int height,int width)
{
  //int flatten_channel=batchsize*channel;
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(width,CU2DBLOCK),n_blocks(batchsize*channel*height,CU2DBLOCK));
  _leaky_relu<<<Gr,Bl,0,stream>>>(batchsize,channel,height,width,input,gamma,output);
}

template<typename T>
void gelu_forward_gpu(cudaStream_t stream,const T* src,T* dst,int ele_size)
{
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(ele_size,CU1DBLOCK));
  _gelu<<<Gr,Bl,0,stream>>>(ele_size,src,dst);
}

void gelu_forward_gpu(cudaStream_t stream,const __half* src,__half* dst,int ele_size)
{
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(ele_size,CU1DBLOCK));
  _geluH<<<Gr,Bl,0,stream>>>(ele_size,src,dst);
}

template
void leaky_relu_forward_gpu<float>(cudaStream_t stream,const float* input,const float* gamma,
  float* output,int batchsize,int channel,int height,int width);

template
void leaky_relu_forward_gpu<__half>(cudaStream_t stream,const __half* input,const __half* gamma,
  __half* output,int batchsize,int channel,int height,int width);

template
void gelu_forward_gpu<float>(cudaStream_t stream,const float* src,float* dst,int ele_size);

} // namespace ops_lib
} // namespace framework
} // namespace quake
