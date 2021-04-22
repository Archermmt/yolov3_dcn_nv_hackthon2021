//
// Created by chengjin on 2020-06-02.
//

#include "cu_utils.h"
#include "quantize_kernel.h"
#include "quantize_kernel.cuh"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__global__ static void _normal_quantize(const int n,const T* src,T* dst,
  int val_max,T val_amp,int keep_scale,int method){
  KERNEL_LOOP(index,n){
    //method: 0: do nothing, 1: do floor with mod, 2: do special round, 3: do floor with clip
    if(0==method){
      dst[index]=src[index]*val_amp;
    }else{
      int result_=0;
      _normal_quantize_device(src[index],result_,val_max,val_amp,method);
      if(0!=keep_scale){
        dst[index]=T(result_)*(1/val_amp);
      }else{
        dst[index]=result_;
      }
    }
  }
}

template<typename T>
void normal_quantize_gpu(const int n,const T* src,T* dst,int val_max,T val_amp,int keep_scale,int method){
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(n,CU1DBLOCK));
  _normal_quantize<<<Gr,Bl>>>(n,src,dst,val_max,val_amp,keep_scale,method);
}

template
void normal_quantize_gpu<float>(const int n,const float* src,float* dst,int val_max,float val_amp,int keep_scale,int method);

} // namespace ops_lib
} // namespace framework
} // namespace quake
