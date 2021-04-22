//
// Created by chengjin on 2020-06-30.
//

#ifndef OPSLIB_KERNEL_GPU_EMBEDDING_H
#define OPSLIB_KERNEL_GPU_EMBEDDING_H
#include <cuda_fp16.h>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T,typename Tidx>
void embedding_forward_gpu(cudaStream_t stream,const Tidx* input,const T* weight,
  T* output,int batchsize,int idx_num,int emb_num,int emb_dim);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_EMBEDDING_H
