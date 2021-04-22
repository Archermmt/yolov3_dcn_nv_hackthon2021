//
// Created by chengjin on 2021-02-24.
//

#include "cu_utils.h"
#include "mask_kernel.h"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__global__ static void _ne_mask(const int n,const T* input,T* output,T val){
  KERNEL_LOOP(index,n){
    output[index] = input[index]!=val ? 1:0;
  }
}

template<typename T>
__global__ static void _gt_mask(const int n,const T* input,T* output,T val){
  KERNEL_LOOP(index,n){
    output[index] = input[index]>val ? 1:0;
  }
}

//implements
template<typename T>
void mask_ne_forward_gpu(cudaStream_t stream,const T* input,T* output,T value,int n)
{
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(n,CU1DBLOCK));
  _ne_mask<<<Gr,Bl,0,stream>>>(n,input,output,value);
}

template<typename T>
void mask_gt_forward_gpu(cudaStream_t stream,const T* input,T* output,T value,int n)
{
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(n,CU1DBLOCK));
  _gt_mask<<<Gr,Bl,0,stream>>>(n,input,output,value);
}

template
void mask_ne_forward_gpu<float>(cudaStream_t stream,const float* input,float* output,float value,int n);

template
void mask_ne_forward_gpu<__half>(cudaStream_t stream,const __half* input,__half* output,__half value,int n);

template
void mask_gt_forward_gpu<float>(cudaStream_t stream,const float* input,float* output,float value,int n);

template
void mask_gt_forward_gpu<__half>(cudaStream_t stream,const __half* input,__half* output,__half value,int n);

} // namespace ops_lib
} // namespace framework
} // namespace quake
