//
// Created by chengjin on 2021-03-09.
//

#include "cu_utils.h"
#include "yolo_box_kernel.h"
#include "cu_device.cuh"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__global__ static void _yolo_box_activate(
  const T* input,const T* anchors,T* boxes,T* scores,
  int batchsize,int channel,int height,int width,
  int anchor_num,int boxes_num,int class_num,T conf_thresh,T scale_x_y)
{
  int h = blockIdx.x * blockDim.x + threadIdx.x;  // batchsize*boxes_num
  int idx_box = blockIdx.y * blockDim.y + threadIdx.y;  // 4+class_num
  if(h<(batchsize*boxes_num) && idx_box<4+class_num){
    int idx_n=h/boxes_num;
    int stride=height*width;
    int stride_len=h%boxes_num;
    int idx_anchor=stride_len/stride;
    stride_len=stride_len%stride;
    int idx_h=stride_len/width;
    int idx_w=stride_len%width;
    //get conf thresh
    int conf_idx=idx_n*anchor_num*(5+class_num)*height*width+ \
      idx_anchor*(5+class_num)*height*width+ \
      4*height*width+ \
      idx_h*width+idx_w;
    T pred_conf=_sigmoid(input[conf_idx]);
    pred_conf=pred_conf<conf_thresh ? T(0) : pred_conf;
    idx_box=idx_box>=4 ? idx_box+1 : idx_box;
    int in_idx=idx_n*anchor_num*(5+class_num)*height*width+ \
      idx_anchor*(5+class_num)*height*width+ \
      idx_box*height*width+ \
      idx_h*width+idx_w;
    if(idx_box<4){
      int idx_out_box=idx_n*boxes_num*4+idx_anchor*height*width*4+idx_h*width*4+idx_w*4+idx_box;
      T bias_x_y=-T(0.5)*(scale_x_y-T(1));
      if(pred_conf==T(0)){
        boxes[idx_out_box]=T(0);
      }else if(idx_box==0){
        boxes[idx_out_box]=(T(idx_w)+_sigmoid(input[in_idx])*scale_x_y+bias_x_y)/T(width);
      }else if(idx_box==1){
        boxes[idx_out_box]=(T(idx_h)+_sigmoid(input[in_idx])*scale_x_y+bias_x_y)/T(height);
      }else if(idx_box==2){
        boxes[idx_out_box]=T(exp(input[in_idx]))*anchors[idx_n*anchor_num*2+idx_anchor*2];
      }else{
        boxes[idx_out_box]=T(exp(input[in_idx]))*anchors[idx_n*anchor_num*2+idx_anchor*2+1];
      }
    }else{
      int idx_out_score=idx_n*boxes_num*class_num+idx_anchor*height*width*class_num+idx_h*width*class_num+idx_w*class_num+idx_box-5;
      scores[idx_out_score]=_sigmoid(input[in_idx])*pred_conf;
    }
  }
}

template<typename T>
__global__ static void _yolo_box_activateH(
  const T* input,const T* anchors,T* boxes,T* scores,
  int batchsize,int channel,int height,int width,
  int anchor_num,int boxes_num,int class_num,T conf_thresh,T scale_x_y)
{
  int h = blockIdx.x * blockDim.x + threadIdx.x;  // batchsize*boxes_num
  int idx_box = blockIdx.y * blockDim.y + threadIdx.y;  // 4+class_num
  if(h<(batchsize*boxes_num) && idx_box<4+class_num){
    int idx_n=h/boxes_num;
    int stride=height*width;
    int stride_len=h%boxes_num;
    int idx_anchor=stride_len/stride;
    stride_len=stride_len%stride;
    int idx_h=stride_len/width;
    int idx_w=stride_len%width;
    //get conf thresh
    int conf_idx=idx_n*anchor_num*(5+class_num)*height*width+ \
      idx_anchor*(5+class_num)*height*width+ \
      4*height*width+ \
      idx_h*width+idx_w;
    T pred_conf=_sigmoidH(input[conf_idx]);
    pred_conf=pred_conf<conf_thresh ? T(0) : pred_conf;
    idx_box=idx_box>=4 ? idx_box+1 : idx_box;
    int in_idx=idx_n*anchor_num*(5+class_num)*height*width+ \
      idx_anchor*(5+class_num)*height*width+ \
      idx_box*height*width+ \
      idx_h*width+idx_w;
    if(idx_box<4){
      int idx_out_box=idx_n*boxes_num*4+idx_anchor*height*width*4+idx_h*width*4+idx_w*4+idx_box;
      T bias_x_y=-T(0.5)*(scale_x_y-T(1));
      if(pred_conf==T(0)){
        boxes[idx_out_box]=T(0);
      }else if(idx_box==0){
        boxes[idx_out_box]=(T(idx_w)+_sigmoidH(input[in_idx])*scale_x_y+bias_x_y)/T(width);
      }else if(idx_box==1){
        boxes[idx_out_box]=(T(idx_h)+_sigmoidH(input[in_idx])*scale_x_y+bias_x_y)/T(height);
      }else if(idx_box==2){
        boxes[idx_out_box]=T(hexp(input[in_idx]))*anchors[idx_n*anchor_num*2+idx_anchor*2];
      }else{
        boxes[idx_out_box]=T(hexp(input[in_idx]))*anchors[idx_n*anchor_num*2+idx_anchor*2+1];
      }
    }else{
      int idx_out_score=idx_n*boxes_num*class_num+idx_anchor*height*width*class_num+idx_h*width*class_num+idx_w*class_num+idx_box-5;
      scores[idx_out_score]=_sigmoidH(input[in_idx])*pred_conf;
    }
  }
}

template<typename T>
__global__ static void _yolo_box_clip(const T* buffer,const T* imgsize_data,T* boxes,
  int batchsize,int boxes_num,bool clip_box)
{
  int h = blockIdx.x * blockDim.x + threadIdx.x;  // box_num
  int idx_col = blockIdx.y * blockDim.y + threadIdx.y;  // 4
  if(h<batchsize*boxes_num && idx_col<4){
    int idx_n=h/boxes_num;
    int idx_out=h*4+idx_col;
    boxes[idx_out]=buffer[idx_out];
    if(idx_col==0){
      boxes[idx_out]=(buffer[idx_out]-buffer[idx_out+2]/T(2))*imgsize_data[idx_n*2+1];
    }else if(idx_col==1){
      boxes[idx_out]=(buffer[idx_out]-buffer[idx_out+2]/T(2))*imgsize_data[idx_n*2];
    }else if(idx_col==2){
      boxes[idx_out]=(buffer[idx_out-2]+buffer[idx_out]/T(2))*imgsize_data[idx_n*2+1];
    }else{
      boxes[idx_out]=(buffer[idx_out-2]+buffer[idx_out]/T(2))*imgsize_data[idx_n*2];
    }
    if(clip_box){
      if(idx_col==0 || idx_col==2){
        _clip(boxes[idx_out],T(0),T(imgsize_data[idx_n*2+1])-T(1));
      }else{
        _clip(boxes[idx_out],T(0),T(imgsize_data[idx_n*2])-T(1));
      }
    }
  }
}

template<typename T>
void yolo_box(cudaStream_t stream,
  const T* input,const T* imgsize_data,const T* anchors,T* boxes,T* scores,T* buffer, \
  int batchsize,int channel,int height,int width, \
  int anchor_num,int boxes_num,int class_num,T conf_thresh, \
  T scale_x_y,bool clip_box)
{
  //output box shape [N,boxes_num,4]
  dim3 Bl_act(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr_act(n_blocks(batchsize*boxes_num,CU2DBLOCK),n_blocks(4+class_num,CU2DBLOCK));
  _yolo_box_activate<<<Gr_act,Bl_act,0,stream>>>(input,anchors,buffer,scores, \
    batchsize,channel,height,width,
    anchor_num,boxes_num,class_num,conf_thresh,scale_x_y);
  cudaStreamSynchronize(stream);
  //clip the boxes
  dim3 Bl_clip(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr_clip(n_blocks(batchsize*boxes_num,CU2DBLOCK),n_blocks(4,CU2DBLOCK));
  _yolo_box_clip<<<Gr_clip,Bl_clip,0,stream>>>(buffer,imgsize_data,boxes, \
    batchsize,boxes_num,clip_box);
}

void yolo_box(cudaStream_t stream,
  const __half* input,const __half* imgsize_data,const __half* anchors,__half* boxes,__half* scores,__half* buffer, \
  int batchsize,int channel,int height,int width, \
  int anchor_num,int boxes_num,int class_num,__half conf_thresh, \
  __half scale_x_y,bool clip_box)
{
  //output box shape [N,boxes_num,4]
  dim3 Bl_act(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr_act(n_blocks(batchsize*boxes_num,CU2DBLOCK),n_blocks(4+class_num,CU2DBLOCK));
  _yolo_box_activateH<<<Gr_act,Bl_act,0,stream>>>(input,anchors,buffer,scores, \
    batchsize,channel,height,width,
    anchor_num,boxes_num,class_num,conf_thresh,scale_x_y);
  cudaStreamSynchronize(stream);
  //clip the boxes
  dim3 Bl_clip(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr_clip(n_blocks(batchsize*boxes_num,CU2DBLOCK),n_blocks(4,CU2DBLOCK));
  _yolo_box_clip<<<Gr_clip,Bl_clip,0,stream>>>(buffer,imgsize_data,boxes, \
    batchsize,boxes_num,clip_box);
}

template
void yolo_box<float>(cudaStream_t stream,
  const float* input,const float* imgsize_data,const float* anchors,float* boxes,float* scores,float* buffer, \
  int batchsize,int channel,int height,int width, \
  int anchor_num,int boxes_num,int class_num,float conf_thresh, \
  float scale_x_y,bool clip_box);

} // namespace ops_lib
} // namespace framework
} // namespace quake
