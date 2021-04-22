//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_KERNEL_GPU_QUANTIZE_H
#define OPSLIB_KERNEL_GPU_QUANTIZE_H

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void normal_quantize_gpu(const int n,const T* src,T* dst,int val_max,T val_amp,int keep_scale,int method);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_QUANTIZE_H