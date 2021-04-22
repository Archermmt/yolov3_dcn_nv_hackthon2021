//
// Created by chengjin on 2021-03-15.
//

#include "cu_utils.h"
#include "recover_boxes_area_kernel.h"
#include "cu_device.cuh"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__global__ static void _recover_boxes(const T* in_data,const T* resize_data,T* out_boxes,
  int batchsize,int box_num,int box_dim,int pad_h,int pad_w){
  int h = blockIdx.x * blockDim.x + threadIdx.x;  // batchsize*box_num
  int idx_box = blockIdx.y * blockDim.y + threadIdx.y;  // box_dim
  if(h<(batchsize*box_num) && idx_box<box_dim){
    int out_idx=h*box_dim+idx_box;
    int bz_idx=h/box_num;
    if(idx_box==0 || idx_box==2){
      out_boxes[out_idx]=(in_data[out_idx]-resize_data[bz_idx*5+2])/resize_data[bz_idx*5]-T(pad_w);
      _clip(out_boxes[out_idx],T(0),T(resize_data[bz_idx*5+4])-T(1));
    }else if (idx_box==1 || idx_box==3){
      out_boxes[out_idx]=(in_data[out_idx]-resize_data[bz_idx*5+1])/resize_data[bz_idx*5]-T(pad_h);
      _clip(out_boxes[out_idx],T(0),T(resize_data[bz_idx*5+3])-T(1));
    }else{
      out_boxes[out_idx]=in_data[out_idx];
    }
  }
}

template<typename T>
__global__ static void _calculate_areas(const T* out_boxes,T* out_areas,
  int batchsize,int box_num,int box_dim){
  int h = blockIdx.x * blockDim.x + threadIdx.x;  // batchsize*box_num
  int idx_box = blockIdx.y * blockDim.y + threadIdx.y;  // box_dim
  if(h<(batchsize*box_num) && idx_box<box_dim){
    out_areas[h]=(out_boxes[h*box_dim+3]-out_boxes[h*box_dim+1])*(out_boxes[h*box_dim+2]-out_boxes[h*box_dim]);
  }
}

template<typename T>
void recover_boxes_area(cudaStream_t stream,
  const T* in_data,const T* resize_data,T* out_boxes,T* out_areas,
  int batchsize,int box_num,int box_dim,int pad_h,int pad_w){
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(batchsize*box_num,CU2DBLOCK),n_blocks(box_dim,CU2DBLOCK));
  _recover_boxes<<<Gr,Bl,0,stream>>>(in_data,resize_data,out_boxes, \
    batchsize,box_num,box_dim,pad_h,pad_w);
  dim3 Bl_area(CU1DBLOCK);
  dim3 Gr_area(n_blocks(batchsize*box_num,CU1DBLOCK));
  _calculate_areas<<<Gr,Bl,0,stream>>>(out_boxes,out_areas, \
    batchsize,box_num,box_dim);
}

template
void recover_boxes_area<float>(cudaStream_t stream,
  const float* in_data,const float* resize_data,float* out_boxes,float* out_areas,
  int batchsize,int box_num,int box_dim,int pad_h,int pad_w);

template
void recover_boxes_area<__half>(cudaStream_t stream,
  const __half* in_data,const __half* resize_data,__half* out_boxes,__half* out_areas,
  int batchsize,int box_num,int box_dim,int pad_h,int pad_w);

} // namespace ops_lib
} // namespace framework
} // namespace quake
