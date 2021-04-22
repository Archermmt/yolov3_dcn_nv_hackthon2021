//
// Created by chengjin on 2020-06-02.
//

#include "cu_utils.h"
#include "cu_math.h"
#include "normalize_kernel.h"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__global__ static void _cal_group_norm(int row,int col,int channel,int stride,
  T eps,const T* src,const T* gamma,const T* beta,const T* mean,const T* var,T* dst)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;//col
  int j = blockIdx.y * blockDim.y + threadIdx.y;//row
  int idx = i + j * col;
  int param_idx = (idx/stride)%(channel);
  if (i < col && j < row){
    //inputs=(inputs-mean)*gamma/np.sqrt(varience+eps)+beta
    dst[idx]=beta[param_idx]+(src[idx]-mean[j])*gamma[param_idx]/sqrt(var[j]+eps);
  }
}

__global__ static void _cal_group_normH(int row,int col,int channel,int stride,
  __half eps,const __half* src,const __half* gamma,const __half* beta,
  const __half* mean,const __half* var,__half* dst)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;//col
  int j = blockIdx.y * blockDim.y + threadIdx.y;//row
  int idx = i + j * col;
  int param_idx = (idx/stride)%(channel);
  if (i < col && j < row){
    //inputs=(inputs-mean)*gamma/np.sqrt(varience+eps)+beta
    dst[idx]=beta[param_idx]+(src[idx]-mean[j])*gamma[param_idx]/hsqrt(var[j]+eps);
  }
}

template<typename T>
__global__ static void _cal_norm(int row,int col,T eps,const T* src,
  const T* gamma,const T* beta,const T* mean,const T* var,T* dst)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;//col
  int j = blockIdx.y * blockDim.y + threadIdx.y;//row
  int idx = i + j * col;
  if (i < col && j < row){
    //inputs=(inputs-mean)*gamma/np.sqrt(varience+eps)+beta
    dst[idx]=beta[i]+(src[idx]-mean[j])*gamma[i]/sqrt(var[j]+eps);
  }
}

__global__ static void _cal_normH(int row,int col,__half eps,const __half* src,
  const __half* gamma,const __half* beta,const __half* mean,const __half* var,__half* dst)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;//col
  int j = blockIdx.y * blockDim.y + threadIdx.y;//row
  int idx = i + j * col;
  if (i < col && j < row){
    //inputs=(inputs-mean)*gamma/np.sqrt(varience+eps)+beta
    dst[idx]=beta[i]+(src[idx]-mean[j])*gamma[i]/hsqrt(var[j]+eps);
  }
}

//implements
template<typename T>
void groupnorm_forward_gpu(cudaStream_t stream,const T* input,
  const T* gamma,const T* beta,
  T* buffer,T* mean,T* var,T* output,
  int group,int batchsize,int channel,int input_h,int input_w,T eps)
{
  //input shape [N,C,H,W],reshape to [N*G,C//G*H*W]
  int mat_row=batchsize*group;
  int mat_col=channel/group*input_h*input_w;
  int stride=input_h*input_w;
  //get mean,var
  gpu_moment_mat_cols(stream,mat_row,mat_col,input,buffer,mean,var);
  //get result
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(mat_col,CU2DBLOCK),n_blocks(mat_row,CU2DBLOCK));
  _cal_group_norm<<<Gr,Bl>>>(mat_row,mat_col,channel,stride,eps,input,gamma,beta,mean,var,output);
}

void groupnorm_forward_gpu(cudaStream_t stream,const __half* input,
  const __half* gamma,const __half* beta,
  __half* buffer,__half* mean,__half* var,__half* output,
  int group,int batchsize,int channel,int input_h,int input_w,__half eps)
{
  //input shape [N,C,H,W],reshape to [N*G,C//G*H*W]
  int mat_row=batchsize*group;
  int mat_col=channel/group*input_h*input_w;
  int stride=input_h*input_w;
  //get mean,var
  gpu_moment_mat_cols(stream,mat_row,mat_col,input,buffer,mean,var);
  //get result
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(mat_col,CU2DBLOCK),n_blocks(mat_row,CU2DBLOCK));
  _cal_group_normH<<<Gr,Bl>>>(mat_row,mat_col,channel,stride,eps,input,gamma,beta,mean,var,output);
}

template<typename T>
void layernorm_forward_gpu(cudaStream_t stream,const T* input,
  const T* gamma,const T* beta,
  T* buffer,T* mean,T* var,T* output,
  int batchsize,int layer_len,int layer_dim,T eps)
{
  //input shape [N,layer_len,layer_dim],reshape to [N*layer_len,layer_dim]
  int mat_row=batchsize*layer_len;
  int mat_col=layer_dim;
  //get mean,var
  gpu_moment_mat_cols(stream,mat_row,mat_col,input,buffer,mean,var);
  //get result
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(mat_col,CU2DBLOCK),n_blocks(mat_row,CU2DBLOCK));
  _cal_norm<<<Gr,Bl>>>(mat_row,mat_col,eps,input,gamma,beta,mean,var,output);
}

void layernorm_forward_gpu(cudaStream_t stream,const __half* input,
  const __half* gamma,const __half* beta,
  __half* buffer,__half* mean,__half* var,__half* output,
  int batchsize,int layer_len,int layer_dim,__half eps)
{
  //input shape [N,layer_len,layer_dim],reshape to [N*layer_len,layer_dim]
  int mat_row=batchsize*layer_len;
  int mat_col=layer_dim;
  //get mean,var
  gpu_moment_mat_cols(stream,mat_row,mat_col,input,buffer,mean,var);
  //get result
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(mat_col,CU2DBLOCK),n_blocks(mat_row,CU2DBLOCK));
  _cal_normH<<<Gr,Bl>>>(mat_row,mat_col,eps,input,gamma,beta,mean,var,output);
}

template
void groupnorm_forward_gpu<float>(cudaStream_t stream,const float* input,
  const float* gamma,const float* beta,
  float* buffer,float* mean,float* var,float* output,
  int group,int batchsize,int channel,int input_h,int input_w,float eps);

template
void layernorm_forward_gpu<float>(cudaStream_t stream,const float* input,
  const float* gamma,const float* beta,
  float* buffer,float* mean,float* var,float* output,
  int batchsize,int layer_len,int layer_dim,float eps);

} // namespace ops_lib
} // namespace framework
} // namespace quake
