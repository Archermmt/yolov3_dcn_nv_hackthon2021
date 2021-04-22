//
// Created by chengjin on 2021-03-15.
//

#include "cu_utils.h"
#include "nonlinear_pred_box_kernel.h"
#include "cu_device.cuh"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__global__ static void _nonlinear_pred(const T* in_data,const T* delta_data,T* output,
  int batchsize,int box_num,int box_dim,int image_h,int image_w){
  int h = blockIdx.x * blockDim.x + threadIdx.x;  // batchsize*box_num
  int idx_box = blockIdx.y * blockDim.y + threadIdx.y;  // box_dim
  if(h<(batchsize*box_num) && idx_box<box_dim){
    int out_idx=h*box_dim+idx_box;
    if(idx_box==0 || idx_box==2){
      T width=in_data[h*box_dim+2]-in_data[h*box_dim+0]+T(1);
      T ctr_x=in_data[h*box_dim]+T(0.5)*(width-T(1));
      T pred_ctr_x=delta_data[h*box_dim]*width+ctr_x;
      T pred_w=T(exp(delta_data[h*box_dim+2]))*width;
      //calculate output
      output[out_idx]=idx_box==0 ? pred_ctr_x-T(0.5)*(pred_w-T(1)) : pred_ctr_x+T(0.5)*(pred_w-T(1));
      _clip(output[out_idx],T(0),T(image_w)-T(1));
    }else{
      T height=in_data[h*box_dim+3]-in_data[h*box_dim+1]+T(1);
      T ctr_y=in_data[h*box_dim+1]+T(0.5)*(height-T(1));
      T pred_ctr_y=delta_data[h*box_dim+1]*height+ctr_y;
      T pred_h=T(exp(delta_data[h*box_dim+3]))*height;
      //calculate output
      output[out_idx]=idx_box==1 ? pred_ctr_y-T(0.5)*(pred_h-T(1)) : pred_ctr_y+T(0.5)*(pred_h-T(1));
      _clip(output[out_idx],T(0),T(image_h)-T(1));
    }
  }
}

template<typename T>
__global__ static void _nonlinear_predH(const T* in_data,const T* delta_data,T* output,
  int batchsize,int box_num,int box_dim,int image_h,int image_w){
  int h = blockIdx.x * blockDim.x + threadIdx.x;  // batchsize*box_num
  int idx_box = blockIdx.y * blockDim.y + threadIdx.y;  // box_dim
  if(h<(batchsize*box_num) && idx_box<box_dim){
    int out_idx=h*box_dim+idx_box;
    if(idx_box==0 || idx_box==2){
      T width=in_data[h*box_dim+2]-in_data[h*box_dim+0]+T(1);
      T ctr_x=in_data[h*box_dim]+T(0.5)*(width-T(1));
      T pred_ctr_x=delta_data[h*box_dim]*width+ctr_x;
      T pred_w=T(hexp(delta_data[h*box_dim+2]))*width;
      //calculate output
      output[out_idx]=idx_box==0 ? pred_ctr_x-T(0.5)*(pred_w-T(1)) : pred_ctr_x+T(0.5)*(pred_w-T(1));
      _clip(output[out_idx],T(0),T(image_w)-T(1));
    }else{
      T height=in_data[h*box_dim+3]-in_data[h*box_dim+1]+T(1);
      T ctr_y=in_data[h*box_dim+1]+T(0.5)*(height-T(1));
      T pred_ctr_y=delta_data[h*box_dim+1]*height+ctr_y;
      T pred_h=T(hexp(delta_data[h*box_dim+3]))*height;
      //calculate output
      output[out_idx]=idx_box==1 ? pred_ctr_y-T(0.5)*(pred_h-T(1)) : pred_ctr_y+T(0.5)*(pred_h-T(1));
      _clip(output[out_idx],T(0),T(image_h)-T(1));
    }
  }
}

template<typename T>
void nonlinear_pred_box(cudaStream_t stream,
  const T* in_data,const T* delta_data,T* output,
  int batchsize,int box_num,int box_dim,int image_h,int image_w){
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(batchsize*box_num,CU2DBLOCK),n_blocks(box_dim,CU2DBLOCK));
  _nonlinear_pred<<<Gr,Bl,0,stream>>>(in_data,delta_data,output, \
    batchsize,box_num,box_dim,image_h,image_w);
}

void nonlinear_pred_box(cudaStream_t stream,
  const __half* in_data,const __half* delta_data,__half* output,
  int batchsize,int box_num,int box_dim,int image_h,int image_w){
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(batchsize*box_num,CU2DBLOCK),n_blocks(box_dim,CU2DBLOCK));
  _nonlinear_predH<<<Gr,Bl,0,stream>>>(in_data,delta_data,output, \
    batchsize,box_num,box_dim,image_h,image_w);
}

template
void nonlinear_pred_box<float>(cudaStream_t stream,
  const float* in_data,const float* delta_data,float* output,
  int batchsize,int box_num,int box_dim,int image_h,int image_w);

} // namespace ops_lib
} // namespace framework
} // namespace quake
