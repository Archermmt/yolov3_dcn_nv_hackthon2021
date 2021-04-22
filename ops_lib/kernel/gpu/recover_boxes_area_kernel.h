//
// Created by chengjin on 2021-03-15.
//

#ifndef OPSLIB_KERNEL_GPU_RECOVER_BOXEES_AREA_H
#define OPSLIB_KERNEL_GPU_RECOVER_BOXEES_AREA_H
#include <cuda_fp16.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void recover_boxes_area(cudaStream_t stream,
  const T* in_data,const T* resize_data,T* out_boxes,T* out_areas,
  int batchsize,int box_num,int box_dim,int pad_h,int pad_w);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_RECOVER_BOXEES_AREA_H
