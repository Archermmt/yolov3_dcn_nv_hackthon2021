//
// Created by chengjin on 2021-03-09.
//

#ifndef OPSLIB_KERNEL_GPU_YOLO_BOX_H
#define OPSLIB_KERNEL_GPU_YOLO_BOX_H
#include <cuda_fp16.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void yolo_box(cudaStream_t stream,
  const T* input,const T* imgsize_data,const T* anchors,T* boxes,T* scores,T* buffer, \
  int batchsize,int channel,int height,int width, \
  int anchor_num,int boxes_num,int class_num,T conf_thresh, \
  T scale_x_y,bool clip_box);

void yolo_box(cudaStream_t stream,
  const __half* input,const __half* imgsize_data,const __half* anchors,__half* boxes,__half* scores,__half* buffer, \
  int batchsize,int channel,int height,int width, \
  int anchors_num,int boxes_num,int class_num,__half conf_thresh, \
  __half scale_x_y,bool clip_box);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_YOLO_BOX_H
