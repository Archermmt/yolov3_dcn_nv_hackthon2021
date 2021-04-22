//
// Created by chengjin on 2020-06-02.
//

#include "cu_utils.h"
#include "cu_device.cuh"
#include "detect_kernel.h"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__global__ static void _delta2bbox(int row,
  const T* rois,const T* deltas,const T* scores,T* bboxes,
  int max_height,int max_width,T max_ratio){
  
  int r_id = blockIdx.x * blockDim.x + threadIdx.x;//row id
  int type_id = blockIdx.y * blockDim.y + threadIdx.y;//type

  if (type_id < 5 && r_id < row){    
    if(type_id==4){
      bboxes[r_id*5+type_id]=scores[r_id];
    }else{
      T roi_ctr_x = rois[r_id*4];
      T roi_ctr_y = rois[r_id*4+1];
      T roi_width = rois[r_id*4+2];
      T roi_height = rois[r_id*4+3];
      T dx = deltas[r_id*4];
      T dy = deltas[r_id*4+1];
      T dw = deltas[r_id*4+2];
      T dh = deltas[r_id*4+3];
      T gx,gy,gw,gh,res;
      if(type_id%2==0){
        if(max_ratio>T(0)){
          _clip(dw,-max_ratio,max_ratio);
        }
        gw = roi_width * T(expf(dw));
        gx = roi_ctr_x+roi_width*dx;
      }else{
        if(max_ratio>T(0)){
          _clip(dh,-max_ratio,max_ratio);
        }
        gh = roi_height * T(expf(dh));
        gy = roi_ctr_y+roi_height*dy;
      }
      //assign the bbox
      if(type_id==0){
        res=gx - gw * T(0.5) + T(0.5);
        _clip(res,T(0),T(max_width));
        bboxes[r_id*5+type_id]=res;
      }else if(type_id==1){
        res=gy - gh * T(0.5) + T(0.5);
        _clip(res,T(0),T(max_height));
        bboxes[r_id*5+type_id]=res;
      }else if(type_id==2){
        res=gx + gw * T(0.5) - T(0.5);
        _clip(res,T(0),T(max_width));
        bboxes[r_id*5+type_id]=res;
      }else if(type_id==3){
        res=gy + gh * T(0.5) - T(0.5);
        _clip(res,T(0),T(max_height));
        bboxes[r_id*5+type_id]=res;
      }
    }
  }
}

template<typename T>
void delta2bbox_forward_gpu(cudaStream_t stream,const T* rois,const T* deltas,
  const T* scores,T* bboxes,int row,int max_height,int max_width,T max_ratio)
{
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(row,CU2DBLOCK),n_blocks(5,CU2DBLOCK));
  _delta2bbox<<<Gr,Bl,0,stream>>>(row,rois,deltas,scores,bboxes,max_height,max_width,max_ratio);
}

template
void delta2bbox_forward_gpu<float>(cudaStream_t stream,const float* rois,const float* deltas,
  const float* scores,float* bboxes,int row,int max_height,int max_width,float max_ratio);

template
void delta2bbox_forward_gpu<__half>(cudaStream_t stream,const __half* rois,const __half* deltas,
  const __half* scores,__half* bboxes,int row,int max_height,int max_width,__half max_ratio);

} // namespace ops_lib
} // namespace framework
} // namespace quake
