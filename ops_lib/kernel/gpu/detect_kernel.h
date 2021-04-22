//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_KERNEL_GPU_DETECT_H
#define OPSLIB_KERNEL_GPU_DETECT_H
#include <cuda_fp16.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void delta2bbox_forward_gpu(cudaStream_t stream,const T* rois,const T* deltas,
  const T* scores,T* bboxes,int row,int max_height,int max_width,T max_ratio);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_DETECT_H
