//
// Created by chengjin on 2020-06-30.
//

#include "cu_utils.h"
#include "embedding_kernel.h"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T,typename Tidx>
__global__ static void _gather(int idx_num,int row,int col,
  const T* data,const Tidx* index,T* output) 
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // row index*batchsize
  int id=y%idx_num;
  int b=y/idx_num;

  //assign numbers
  if(id<idx_num){
    int row_idx=int(index[b*idx_num+id]);
    if(row_idx<row){
      output[b*idx_num*col+id*col+c]=data[row_idx*col+c];
    }
  }
}

//implements
template<typename T,typename Tidx>
void embedding_forward_gpu(cudaStream_t stream,const Tidx* input,
  const T* weight,T* output,int batchsize,int idx_num,int emb_num,int emb_dim)
{
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(emb_dim,CU2DBLOCK),n_blocks(idx_num*batchsize,CU2DBLOCK));
  _gather<<<Gr,Bl,0,stream>>>(idx_num,emb_num,emb_dim,weight,input,output);
}

template
void embedding_forward_gpu<float,float>(cudaStream_t stream,const float* input,
  const float* weight,float* output,int batchsize,int idx_num,int emb_num,int emb_dim);

template
void embedding_forward_gpu<__half,__half>(cudaStream_t stream,const __half* input,
  const __half* weight,__half* output,int batchsize,int idx_num,int emb_num,int emb_dim);

template
void embedding_forward_gpu<float,int>(cudaStream_t stream,const int* input,
  const float* weight,float* output,int batchsize,int idx_num,int emb_num,int emb_dim);

} // namespace ops_lib
} // namespace framework
} // namespace quake
