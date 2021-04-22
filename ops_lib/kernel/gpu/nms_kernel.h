//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_KERNEL_GPU_NMS_H
#define OPSLIB_KERNEL_GPU_NMS_H
#include <cuda_fp16.h>

#define NMS_THREADS_PER_BLOCK 512

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void nms_gpu(cudaStream_t stream,const T* bboxes,T* output_boxes,T* num_outs,
  int* int_num_outs,int* mask,int* index_buffer,
  int batchsize,int bbox_num,int bbox_dim,int max_to_keep,T overlap_thresh,T score_thresh);

void nms_gpu(cudaStream_t stream,const __half* bboxes,__half* output_boxes,__half* num_outs,
  int* int_num_outs,int* mask,int* index_buffer,
  int batchsize,int bbox_num,int bbox_dim,int max_to_keep,__half overlap_thresh,__half score_thresh);

template<typename T>
void multiclass_nms_gpu(cudaStream_t stream,const T* in_boxes,const T* in_scores,T* out_boxes,T* out_nums,
  int* int_num_outs,int* mask,int* index_buffer,
  int batchsize,int box_num,int class_num,int box_dim,int max_to_keep,T overlap_thresh,T score_thresh);

void multiclass_nms_gpu(cudaStream_t stream,const __half* in_boxes,const __half* in_scores,__half* out_boxes,__half* out_nums,
  int* int_num_outs,int* mask,int* index_buffer,
  int batchsize,int box_num,int class_num,int box_dim,int max_to_keep,__half overlap_thresh,__half score_thresh);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_NMS_H
