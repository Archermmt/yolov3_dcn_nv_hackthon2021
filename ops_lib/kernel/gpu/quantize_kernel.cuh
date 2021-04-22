//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_KERNEL_GPU_QUANTIZE_CUH
#define OPSLIB_KERNEL_GPU_QUANTIZE_CUH

#include <math.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__device__ void _normal_quantize_device(const T& src,int& res,
  int val_max,T val_amp,int method){
  T res_real_= src*val_amp;
  //method: 1: do floor with mod, 2: do special round, 3: do ceil with clip
  if(1==method){
    res=floor(res_real_);
    if (res > val_max-1) {
      //res=res%val_max-val_max;
      res = val_max-1;
    } else if (res < -val_max) {
      //res=val_max+res%(-val_max);
      res=-val_max;
    }
  }else if(2==method){  
    if(res_real_ > val_max-1) {
      res = val_max-1;
    }else if(res_real_ < -val_max) {
      res=-val_max;
    }else if(res_real_<0 && (res_real_-floor(res_real_))==0.5) {
      res = ceil(res_real_);
    }else{
      res = round(res_real_);
    }
  }else if(3==method){
    res=ceil(res_real_);
    if (res > val_max-1) {
      res=val_max-1;
    } else if (res < -val_max) {
      res=-val_max;
    }
  }
}

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_QUANTIZE_CUH
