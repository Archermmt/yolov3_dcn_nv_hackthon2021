//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_KERNEL_GPU_CUDEVICE_CUH
#define OPSLIB_KERNEL_GPU_CUDEVICE_CUH
#include <cuda_fp16.h>
#include <math.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__device__ T inline _sigmoid(const T val){
  return T(1)/(T(1)+T(exp(-T(1)*val)));
}

__device__ __half inline _sigmoidH(const __half val){
  __half div=__hadd(__half(1),hexp(__hmul(__half(-1),val)));
  return __hdiv(__half(1),div);
}

template<typename T>
__device__ void _clip(T& val,const T min, const T max){
  if(val<=min)
    val=min;
  if(val>=max)
    val=max;
}

__device__ inline __half hmax(__half a, __half b){
  if(__hge(a,b))
    return a;
  else
    return b;
}

__device__ inline __half hmin(__half a, __half b){
  if(__hle(a,b))
    return a;
  else
    return b;
}

template<typename T>
__device__ inline T _tanh(const T& x){
  T exp_2x = exp(T(2.0)*x);
  T res;
  if(isinf(exp_2x)) {
    res = 1.0;
  } else {
    res = (exp_2x - T(1.0)) / (exp_2x + T(1.0));
  }
  return res;
}

__device__ inline __half _tanhH(const __half& x){
  __half exp_2x = hexp(__hmul(__half(2),x));
  __half res;
  if(__hge(exp_2x,65535)) {
    res = __half(1);
  } else {
    res = __hdiv(__hsub(exp_2x,__half(1)),__hadd(exp_2x,__half(1)));
  }
  return res;
}

template<typename T>
__inline__ __device__ int _binary_search(const T* data,int length,T target){
  int left=0;
  int right = length-1;
  int middle=0;
  if(target<data[length-1]){
    return length;
  }
  if(target>data[0]){
    return 0;
  }
  while(left < right){
    middle = (left+right)/2;
    if(target<=data[middle-1] && target>data[middle]){
      return middle;
    }else if(target < data[middle]){
      left = middle+1;
    }else{
      right = middle-1;
    }
  }
  return left;
}

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_CUDEVICE_CUH
