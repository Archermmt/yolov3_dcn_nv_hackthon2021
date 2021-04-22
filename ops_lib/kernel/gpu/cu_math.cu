//
// Created by chengjin on 2020-06-02.
//

#include <math.h>
#include <algorithm>

#include "cu_utils.h"
#include "cu_device.cuh"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename Tin,typename Tout>
__global__ static void _cast(const int n,const Tin* src,Tout* dst){
  KERNEL_LOOP(index,n){
    dst[index]=Tout(src[index]);
  }
}

template<typename Tin,typename Tout>
__global__ static void _round_cast(const int n,const Tin* src,Tout* dst){
  KERNEL_LOOP(index,n){
    dst[index]=Tout(src[index]+0.5);
  }
}

template<typename T>
__global__ static void _mul(const int n,const T* src,T* dst, T factor){
  KERNEL_LOOP(index,n){
    dst[index]=factor*src[index];
  }
}

template<typename T>
__global__ static void _mul_inplace(const int n,T* src,T factor){
  KERNEL_LOOP(index,n){
    src[index]*=factor;
  }
}

template<typename T>
__global__ static void _set(const int n,T* src,T val){
  KERNEL_LOOP(index,n){
    src[index]=val;
  }
}

template<typename T>
__global__ static void _stride_set(int n,int stride,int pos,const T* src,T* dst){
  KERNEL_LOOP(index,n){
    dst[index]=src[index*stride+pos];
  }
}

template<typename T>
__global__ static void _mat_set(int row,int col,int src_stride,int dst_stride,const T* src,T* dst){
  int i = blockIdx.x * blockDim.x + threadIdx.x;//col
  int j = blockIdx.y * blockDim.y + threadIdx.y;//row
  if (i < col && j < row){
    dst[j*dst_stride+i]=src[j*src_stride+i];
  }
}

template<typename T>
__global__ static void _diff_power_cols(int row,int col,const T* src,const T* mean,T* dst)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;//col
  int j = blockIdx.y * blockDim.y + threadIdx.y;//row
  int idx = i + j * col;
  if (i < col && j < row){
    dst[idx]=pow(src[idx]-mean[j],2);
  }
}

__global__ static void _diff_power_div_colsH(int row,int col,const __half* src,const __half* mean,__half* dst)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;//col
  int j = blockIdx.y * blockDim.y + threadIdx.y;//row
  int idx = i + j * col;
  if (i < col && j < row){
    __half diff=__hsub(src[idx],mean[j]);
    dst[idx]=__hdiv(__hmul(diff,diff),col);
  }
}

template<typename T>
__global__ static void _simple_search(int N,int len,
  const T* data,int* pos,T target){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid<N){
    pos[tid]=_binary_search(data+tid*len,len,target);
  }
}
/*
__global__ static void _simple_searchH(int N,int len,
  const __half* data,int* pos,__half target){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid<N){
    pos[tid]=_binary_searchH(data+tid*len,len,target);
  }
}
*/

template<typename T>
__global__ static void _step_mat_transpose(int row,int col,int dst_width,int dst_step,
  const T* src,T* dst){
  int i = blockIdx.x * blockDim.x + threadIdx.x;//col
  int j = blockIdx.y * blockDim.y + threadIdx.y;//row
  if (i < col && j < row){
    int src_idx=i*row+j;
    int dst_idx=(src_idx/dst_width)*dst_step+(src_idx%dst_width); //dst index
    dst[dst_idx]=src[i+j*col];
  }
}

//reduce kernels, from kaldi, reduction without device handle
enum EnumTransformReduce {
  SUMAB, SUM, MAX, MIN, LINFNORM, L2NORM, L1NORM, L0NORM, LPNORM, SUMH
};

template<EnumTransformReduce TransReduceType, typename T>
struct TransReduceOp {
  __forceinline__
  __device__ T InitValue() const {
    return T(0);
  }
  __forceinline__
  __device__ T Transform(const T& x) const {
    return T(0);
  }
  __forceinline__
  __device__ T Reduce(const T& a, const T& b) const {
    return T(0);
  }
  __forceinline__
  __device__ T PostReduce(const T& x, const T& output) const {
    return T(0);
  }
};

template<typename T>
struct TransReduceOp<SUM, T> {
  __forceinline__
  __device__ T InitValue() const {
    return T(0);
  }
  __forceinline__
  __device__ T Transform(const T& x) const {
    return x;
  }
  __forceinline__
  __device__ T Reduce(const T& a, const T& b) const {
    return a + b;
  }
  __forceinline__
  __device__ T PostReduce(const T& x, const T& output) const {
    return x;
  }
};

template<>
struct TransReduceOp<SUMH, __half> {
  __forceinline__
  __device__ __half InitValue() const {
    return __half(0);
  }
  __forceinline__
  __device__ __half Transform(const __half& x) const {
    return x;
  }
  __forceinline__
  __device__ __half Reduce(const __half& a, const __half& b) const {
    __half sum=__hadd(a,b);
    if(__hge(sum,65535)){
      return 65535;
    }else if(__hle(sum,-65535)){
      return -65535;
    }else{
      return sum;
    }
  }
  __forceinline__
  __device__ __half PostReduce(const __half& x, const __half& output) const {
    return x;
  }
};

template<typename T>
struct TransReduceOp<MAX, T> {
  __forceinline__
  __device__ T InitValue() const {
    return sizeof(T) == sizeof(float) ? -CUDART_INF_F : -CUDART_INF;
  }
  __forceinline__
  __device__ T Transform(const T& x) const {
    return x;
  }
  __forceinline__
  __device__ T Reduce(const T& a, const T& b) const {
    return fmax(a, b);
  }
  __forceinline__
  __device__ T PostReduce(const T& x, const T& output) const {
    return x;
  }
};

template<typename T>
struct TransReduceOp<MIN, T> {
  __forceinline__
  __device__ T InitValue() const {
    return sizeof(T) == sizeof(float) ? CUDART_INF_F : CUDART_INF;
  }
  __forceinline__
  __device__ T Transform(const T& x) const {
    return x;
  }
  __forceinline__
  __device__ T Reduce(const T& a, const T& b) const {
    return min(a, b);
  }
  __forceinline__
  __device__ T PostReduce(const T& x, const T& output) const {
    return x;
  }
};

template<EnumTransformReduce TransReduceType, typename T>
__global__
static void _vec_transform_reduce(const int dim,const T* src, T* dst,
  const TransReduceOp<TransReduceType, T> op) {
  
  __shared__ T sdata[CU1DBLOCK];
  T tdata = op.InitValue();

  const int tid = threadIdx.x;
  const int vec_len = dim;
  const int grid_stride = gridDim.x * blockDim.x;
  int i = (blockIdx.x * blockDim.x + tid);
  
  // Grid reduce. Loop over the whole vector v.
  for (; i < vec_len; i += grid_stride) {
    tdata = op.Reduce(tdata, op.Transform(src[i]));
  }
  
  sdata[tid] = tdata;
  __syncthreads();

  // Tree reduce
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift) {
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
    }
    __syncthreads();
  }

  // Reduce last warp. Threads implicitly synchronized within a warp.
  if (tid < warpSize) {
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
    }
  }
  
  // Output to vector dst.
  if (tid == 0)
    dst[blockIdx.x] = op.PostReduce(sdata[0], dst[blockIdx.x]);
}

template<EnumTransformReduce TransReduceType, typename T>
__global__
static void _vec_transform_reduce_inplace(const int dim,T* data,
  const TransReduceOp<TransReduceType, T> op) {
  
  __shared__ T sdata[CU1DBLOCK];
  T tdata = op.InitValue();

  const int tid = threadIdx.x;
  const int vec_len = dim;
  const int grid_stride = gridDim.x * blockDim.x;
  int i = (blockIdx.x * blockDim.x + tid);
  
  // Grid reduce. Loop over the whole vector v.
  for (; i < vec_len; i += grid_stride) {
    tdata = op.Reduce(tdata, op.Transform(data[i]));
    data[i]=0;
  }
  
  sdata[tid] = tdata;
  __syncthreads();

  // Tree reduce
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift) {
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
    }
    __syncthreads();
  }

  // Reduce last warp. Threads implicitly synchronized within a warp.
  if (tid < warpSize) {
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
    }
  }
  
  // Output to vector dst.
  if (tid == 0)
    data[blockIdx.x] = op.PostReduce(sdata[0], data[blockIdx.x]);
}

// Reduce a matrix 'mat' to a row vector 'result'
template<EnumTransformReduce TransReduceType, typename T>
__global__
static void _transform_reduce_mat_rows(
    T *result, const T *mat, const MatrixDim d,
    const TransReduceOp<TransReduceType, T> op) {

  __shared__ T sdata[CU1DBLOCK];
  const int tid = threadIdx.x;
  const int j = blockIdx.x;

  T tdata = op.InitValue();
  for (int i = tid; i < d.rows; i += CU1DBLOCK) {
    //Note the loads of mat are uncoalesced.  We could eliminate these
    //with shared memory but at the matrix sizes we are currently looking 
    //at it probably would not help much and would add a lot of complexity.
    //Alternatively we could look at something like trov to help loads.
    tdata = op.Reduce(tdata, op.Transform(mat[i * d.stride + j]));
  }
  sdata[tid] = tdata;
  __syncthreads();

  // Tree reduce
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift)
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
    __syncthreads();
  }

  // Reduce last warp. Threads implicitly synchronized within a warp.
  if (tid < warpSize) {
    for (int shift = warpSize; shift > 0; shift >>= 1)
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
  }

  // Output to vector result.
  if (tid == 0) {
    result[j] = op.PostReduce(sdata[0], result[j]);
  }
}

// Reduce a matrix 'mat' to a column vector 'result'
template<EnumTransformReduce TransReduceType, typename T>
__global__
static void _transform_reduce_mat_cols(
    T *result, const T *mat, const MatrixDim d,
    const TransReduceOp<TransReduceType, T> op) {

  __shared__ T sdata[CU1DBLOCK];
  const int tid = threadIdx.x;
  const int i = blockIdx.x;
  const int row_start = i * d.stride;

  T tdata = op.InitValue();
  for (int j = tid; j < d.cols; j += CU1DBLOCK) {
    tdata = op.Reduce(tdata, op.Transform(mat[row_start + j]));
  }
  sdata[tid] = tdata;
  __syncthreads();

  // Tree reduce
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift)
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
    __syncthreads();
  }

  // Reduce last warp. Threads implicitly synchronized within a warp.
  if (tid < warpSize) {
    for (int shift = warpSize; shift > 0; shift >>= 1)
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
  }

  // Output to vector result.
  if (tid == 0) {
    result[i] = op.PostReduce(sdata[0], result[i]);
  }
}

template<EnumTransformReduce TransReduceType, typename T>
__global__ static void _single_reduce(const int dim,T* dst,
  const TransReduceOp<TransReduceType, T> op){
  for(int i=1;i<dim;i++){
    dst[0]=op.Reduce(dst[0],dst[i]);
    dst[i]=0;
  }
}

template<typename Tin,typename Tout>
void gpu_cast(cudaStream_t stream,const int n,const Tin* src,Tout* dst){
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(n,CU1DBLOCK));
  _cast<<<Gr,Bl,0,stream>>>(n,src,dst);
}

template
void gpu_cast<int,float>(cudaStream_t stream,const int n,const int* src,float* dst);

template
void gpu_cast<int64_t,float>(cudaStream_t stream,const int n,const int64_t* src,float* dst);

template
void gpu_cast<int64_t,int>(cudaStream_t stream,const int n,const int64_t* src,int* dst);

template<typename Tin,typename Tout>
void gpu_round_cast(cudaStream_t stream,const int n,const Tin* src,Tout* dst){
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(n,CU1DBLOCK));
  _round_cast<<<Gr,Bl,0,stream>>>(n,src,dst);
}

template
void gpu_round_cast<float,int>(cudaStream_t stream,const int n,const float* src,int* dst);

template
void gpu_round_cast<float,int64_t>(cudaStream_t stream,const int n,const float* src,int64_t* dst);

void gpu_mul(cudaStream_t stream,const int n,const float* src,float* dst,float scale){
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(n,CU1DBLOCK));
  _mul<<<Gr,Bl,0,stream>>>(n,src,dst,scale);
}

void gpu_mul_inplace(cudaStream_t stream,const int n,float* src,float val){
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(n,CU1DBLOCK));
  _mul_inplace<<<Gr,Bl,0,stream>>>(n,src,val);
}

void gpu_set(cudaStream_t stream,const int n,float* src,float val){
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(n,CU1DBLOCK));
  _set<<<Gr,Bl,0,stream>>>(n,src,val);
}

void gpu_set(cudaStream_t stream,const int n,__half* src,__half val){
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(n,CU1DBLOCK));
  _set<<<Gr,Bl,0,stream>>>(n,src,val);
}

void gpu_stride_set(cudaStream_t stream,int n,int stride,int pos,
  const float* src,float* dst){
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(n,CU1DBLOCK));
  _stride_set<<<Gr,Bl,0,stream>>>(n,stride,pos,src,dst);
}

void gpu_stride_set(cudaStream_t stream,int n,int stride,int pos,
  const __half* src,__half* dst){
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(n,CU1DBLOCK));
  _stride_set<<<Gr,Bl,0,stream>>>(n,stride,pos,src,dst);
}

void gpu_mat_set(cudaStream_t stream,int row,int col,int src_stride,int dst_stride,
  const unsigned char* src,unsigned char* dst){
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(col,CU2DBLOCK),n_blocks(row,CU2DBLOCK));
  _mat_set<<<Gr,Bl,0,stream>>>(row,col,src_stride,dst_stride,src,dst);
}

void gpu_mat_set(cudaStream_t stream,int row,int col,int src_stride,int dst_stride,
  const float* src,float* dst){
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(col,CU2DBLOCK),n_blocks(row,CU2DBLOCK));
  _mat_set<<<Gr,Bl,0,stream>>>(row,col,src_stride,dst_stride,src,dst);
}

void gpu_step_mat_transpose(cudaStream_t stream,int row,int col,int dst_width,int dst_step,
  const unsigned char* src,unsigned char* dst){
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(col,CU2DBLOCK),n_blocks(row,CU2DBLOCK));
  _step_mat_transpose<<<Gr,Bl,0,stream>>>(row,col,dst_width,dst_step,src,dst);
}

void gpu_step_mat_transpose(cudaStream_t stream,int row,int col,int dst_width,int dst_step,
  const float* src,float* dst){
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(col,CU2DBLOCK),n_blocks(row,CU2DBLOCK));
  _step_mat_transpose<<<Gr,Bl,0,stream>>>(row,col,dst_width,dst_step,src,dst);
}

int gpu_vec_max(int dim,int* vec){
  int max_ele=0;
  if(dim<128){
    int host_vec[dim];
    cudaMemcpy(host_vec,vec,dim*sizeof(int),cudaMemcpyDeviceToHost);
    max_ele=host_vec[0];
    for(int i=1;i<dim;i++){
      if(host_vec[i]>max_ele){
        max_ele=host_vec[i];
      }
    }
  }else{
    std::cout<<"calling vec max for gpu"<<std::endl;
  }
  return max_ele;
}

void gpu_sum_mat_cols(cudaStream_t stream,int row,int col,const float* mat,float* vec){
  MatrixDim d=get_matrix_dim(row,col);
  _transform_reduce_mat_cols<<<row,CU1DBLOCK,0,stream>>>(vec,mat,d,TransReduceOp<SUM,float>());
}

void gpu_sum_mat_cols(cudaStream_t stream,int row,int col,const __half* mat,__half* vec){
  MatrixDim d=get_matrix_dim(row,col);
  _transform_reduce_mat_cols<<<row,CU1DBLOCK,0,stream>>>(vec,mat,d,TransReduceOp<SUMH,__half>());
}

void gpu_mean_mat_cols(cudaStream_t stream,int row,int col,const float* mat,float* vec){
  CUDA_CHECK(cudaMemset(vec,0,col*sizeof(float)));
  gpu_sum_mat_cols(stream,row,col,mat,vec);
  //scle to get mean
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(row,CU1DBLOCK));
  _mul_inplace<<<Gr,Bl,0,stream>>>(row,vec,1/float(col));
}

void gpu_mean_mat_cols(cudaStream_t stream,int row,int col,const __half* mat,__half* vec){
  CUDA_CHECK(cudaMemset(vec,0,col*sizeof(__half)));
  gpu_sum_mat_cols(stream,row,col,mat,vec);
  //scle to get mean
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(row,CU1DBLOCK));
  float factor=float(1)/col;
  _mul_inplace<<<Gr,Bl,0,stream>>>(row,vec,__float2half(factor));
}

void gpu_moment_mat_cols(cudaStream_t stream,int row,int col,const float* mat,float* buffer,float* mean,float* var){
  //mean=inputs.mean(axis=col)
  gpu_mean_mat_cols(stream,row,col,mat,mean);
  //power=(inputs-mean)**2
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(col,CU2DBLOCK),n_blocks(row,CU2DBLOCK));
  _diff_power_cols<<<Gr,Bl,0,stream>>>(row,col,mat,mean,buffer);
  //varience=power.mean(axis=col)
  gpu_mean_mat_cols(stream,row,col,buffer,var);
}

void gpu_moment_mat_cols(cudaStream_t stream,int row,int col,const __half* mat,__half* buffer,__half* mean,__half* var){
  //mean=inputs.sum(axis=col)/col
  dim3 Bl_mul(CU1DBLOCK);
  dim3 Gr_mul(n_blocks(row*col,CU1DBLOCK));
  float factor=float(1)/col;
  _mul<<<Gr_mul,Bl_mul,0,stream>>>(row*col,mat,buffer,__float2half(factor));
  gpu_sum_mat_cols(stream,row,col,buffer,mean);
  //div_power=(inputs-mean)**2/col
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(col,CU2DBLOCK),n_blocks(row,CU2DBLOCK));
  _diff_power_div_colsH<<<Gr,Bl,0,stream>>>(row,col,mat,mean,buffer);
  //varience=div_power.sum(axis=col)
  gpu_sum_mat_cols(stream,row,col,buffer,var);
}

void gpu_search_mat_cols(cudaStream_t stream,int row,int col,
  const int* data,int* pos,int target){
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(row,CU1DBLOCK));
  _simple_search<<<Gr,Bl,0,stream>>>(row,col,data,pos,target);
}

void gpu_search_mat_cols(cudaStream_t stream,int row,int col,
  const float* data,int* pos,float target){
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(row,CU1DBLOCK));
  _simple_search<<<Gr,Bl,0,stream>>>(row,col,data,pos,target);
}

void gpu_search_mat_cols(cudaStream_t stream,int row,int col,
  const __half* data,int* pos,__half target){
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(row,CU1DBLOCK));
  _simple_search<<<Gr,Bl,0,stream>>>(row,col,data,pos,target);
}

} // namespace ops_lib
} // namespace framework
} // namespace quake

