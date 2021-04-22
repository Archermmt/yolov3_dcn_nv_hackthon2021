//
// Created by chengjin on 2020-06-02.
//

#include "cu_utils.h"
#include "scatter_kernel.h"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__global__ static void _ne_set(const int n,const T* cond_input,const T* input,
  T* output,T val){
  KERNEL_LOOP(index,n){
    output[index] = cond_input[index]==val ? cond_input[index]:input[index];
  }
}

//implements
template<typename T>
void scatter_ne_forward_gpu(cudaStream_t stream,const T* cond_input,
  const T* input,T* output,T value,int n)
{
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(n,CU1DBLOCK));
  _ne_set<<<Gr,Bl,0,stream>>>(n,cond_input,input,output,value);
}

template
void scatter_ne_forward_gpu<float>(cudaStream_t stream,const float* cond_input,
  const float* input,float* output,float value,int n);

template
void scatter_ne_forward_gpu<__half>(cudaStream_t stream,const __half* cond_input,
  const __half* input,__half* output,__half value,int n);

} // namespace ops_lib
} // namespace framework
} // namespace quake
