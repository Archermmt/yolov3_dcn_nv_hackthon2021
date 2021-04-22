//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_KERNEL_GPU_CUMATH_H
#define OPSLIB_KERNEL_GPU_CUMATH_H
#include <cuda_fp16.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename Tin,typename Tout>
void gpu_cast(cudaStream_t stream,const int n,const Tin* src,Tout* dst);

template<typename Tin,typename Tout>
void gpu_round_cast(cudaStream_t stream,const int n,const Tin* src,Tout* dst);

void gpu_set(cudaStream_t stream,const int n,float* src,float val);

void gpu_set(cudaStream_t stream,const int n,__half* src,__half val);

void gpu_stride_set(cudaStream_t stream,int n,int stride,int pos,const float* src,float* dst);

void gpu_stride_set(cudaStream_t stream,int n,int stride,int pos,const __half* src,__half* dst);

void gpu_mat_set(cudaStream_t stream,int row,int col,int src_stride,int dst_stride,const unsigned char* src,unsigned char* dst);

void gpu_mat_set(cudaStream_t stream,int row,int col,int src_stride,int dst_stride,const float* src,float* dst);

void gpu_step_mat_transpose(cudaStream_t stream,int row,int col,int dst_width,int dst_step,const float* src,float* dst);

void gpu_step_mat_transpose(cudaStream_t stream,int row,int col,int dst_width,int dst_step,const unsigned char* src,unsigned char* dst);

int gpu_vec_max(int dim,int* vec);

void gpu_sum_mat_cols(cudaStream_t stream,int row,int col,const float* mat,float* vec);

void gpu_sum_mat_cols(cudaStream_t stream,int row,int col,const __half* mat,__half* vec);

void gpu_mean_mat_cols(cudaStream_t stream,int row,int col,const float* mat,float* vec);

void gpu_mean_mat_cols(cudaStream_t stream,int row,int col,const __half* mat,__half* vec);

void gpu_moment_mat_cols(cudaStream_t stream,int row,int col,const float* mat,float* buffer,float* mean,float* var);

void gpu_moment_mat_cols(cudaStream_t stream,int row,int col,const __half* mat,__half* buffer,__half* mean,__half* var);

void gpu_search_mat_cols(cudaStream_t stream,int row,int col,const float* data,int* pos,float target);

void gpu_search_mat_cols(cudaStream_t stream,int row,int col,const int* data,int* pos,int target);

void gpu_search_mat_cols(cudaStream_t stream,int row,int col,const __half* data,int* pos,__half target);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_CUMATH_H
