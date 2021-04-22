//
// Created by chengjin on 2020-06-02.
//

#include "cu_utils.h"
#include "cu_math.h"
#include "cu_device.cuh"
#include "nms_kernel.h"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
__device__ inline T devIoU(T const * const a, T const * const b) {
  T left = max(a[0], b[0]), right = min(a[2], b[2]);
  T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  T width = max(right - left + 1, T(0)), height = max(bottom - top + 1, T(0));
  T interS = width * height;
  T Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  T Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  //printf("(float)compare a[%f,%f,%f,%f] with b[%f,%f,%f,%f] --> Sa:%f,Sb:%f --> %f vs %f\n",a[0],a[1],a[2],a[3],b[0],b[1],b[2],b[3],Sa,Sb,interS,Sa + Sb - interS);
  return interS / (Sa + Sb - interS);
}

__device__ inline __half devIoUH(__half const * const a, __half const * const b) {
  __half left = hmax(a[0], b[0]), right = hmin(a[2], b[2]);
  __half top = hmax(a[1], b[1]), bottom = hmin(a[3], b[3]);
  __half width = hmax(__hadd(__hsub(right,left),__half(1)),__half(0));
  __half height = hmax(__hadd(__hsub(bottom,top),__half(1)),__half(0));
  __half width_a=__hadd(__hsub(a[2],a[0]),__half(1));
  __half height_a=__hadd(__hsub(a[3],a[1]),__half(1));
  __half width_b=__hadd(__hsub(b[2],b[0]),__half(1));
  __half height_b=__hadd(__hsub(b[3],b[1]),__half(1));
  __half interS = __hmul(width,height);
  __half Sa = __hmul(width_a,height_a);
  __half Sb = __hmul(width_b,height_b);
  __half decrease=__half(1);
  for(int i=0;i<10;i++){
    if(__hge(interS,65535) || __hge(Sa,65535) || __hge(Sb,65535)){
      decrease=__hmul(decrease,2);
      width=__hdiv(width,decrease);
      height=__hdiv(height,decrease);
      width_a=__hdiv(width_a,decrease);
      height_a=__hdiv(height_a,decrease);
      width_b=__hdiv(width_b,decrease);
      height_b=__hdiv(height_b,decrease);
      interS = __hmul(width,height);
      Sa = __hmul(width_a,height_a);
      Sb = __hmul(width_b,height_b);
    }else{
      break;
    }
  }
  /*
  printf("(half)(decrease %f)compare a[%f,%f,%f,%f] with b[%f,%f,%f,%f] --> Sa:%f,Sb:%f --> %f vs %f\n",
    __half2float(decrease),
    __half2float(a[0]),__half2float(a[1]),__half2float(a[2]),__half2float(a[3]),
    __half2float(b[0]),__half2float(b[1]),__half2float(b[2]),__half2float(b[3]),
    __half2float(Sa),__half2float(Sb),
    __half2float(interS),__half2float(__hadd(Sa,__hsub(Sb,interS))));
  */
  return __hdiv(interS,__hadd(Sa,__hsub(Sb,interS)));
}

template<typename T>
__global__ static void _filter_bboxes_by_score(int batchsize,int bbox_num,
  int bbox_dim,const T* bboxes,int* mask,T score_thresh){
  int x = blockIdx.x * blockDim.x + threadIdx.x; //batchsize*bbox_num
  int idx_bz=x/bbox_num;
  int idx_box=x%bbox_num;
  if (idx_box<bbox_num && idx_bz<batchsize && \
    bboxes[idx_bz*bbox_num*bbox_dim+idx_box*bbox_dim+4]<score_thresh) {
    mask[idx_bz*bbox_num+idx_box]=1;
    //printf("%d,%d th score %f < %f, set %d\n",idx_bz,idx_box,bboxes[idx_bz*bbox_num*bbox_dim+idx_box*bbox_dim+4],score_thresh,idx_bz*bbox_num+idx_box);
  }
}

template<typename T>
__global__ static void _filter_bboxes_by_class_score(int batchsize,int bbox_num,
  int class_num,const T* scores,int* mask,T score_thresh){
  int x = blockIdx.x * blockDim.x + threadIdx.x; //batchsize*class_num
  int idx_box = blockIdx.y * blockDim.y + threadIdx.y; //bbox_num
  int idx_bz=x/class_num;
  int idx_cls=x%class_num;
  //mask shape [bz,class_num,bbox_num]
  //score shape [bz,bbox_num,class_num]
  int score_index=idx_bz*class_num*bbox_num+idx_box*class_num+idx_cls;
  if (idx_box<bbox_num && idx_bz<batchsize && idx_cls<class_num && \
    scores[score_index]<score_thresh) {
    mask[idx_bz*class_num*bbox_num+idx_cls*bbox_num+idx_box]=1;
  }
}

template<typename T>
__global__ static void _fast_nms(int batchsize,int bbox_num,int bbox_dim,
  const T* bboxes,int* mask,T overlap_thresh){
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // id of self
  int self_id = blockIdx.y * blockDim.y + threadIdx.y;  // id of other bboxes
  int bz_id=x/bbox_num;
  int compare_id=x%bbox_num;
  int compare_box=bz_id*bbox_num*bbox_dim+compare_id*bbox_dim;
  int self_box=bz_id*bbox_num*bbox_dim+self_id*bbox_dim;
  if(bz_id<batchsize && self_id<bbox_num && mask[bz_id*bbox_num+self_id]==0 && \
    compare_id<self_id && mask[bz_id*bbox_num+compare_id]==0){
    if (devIoU(&bboxes[compare_box],&bboxes[self_box]) > overlap_thresh){
      mask[bz_id*bbox_num+self_id]=1;
      //printf("(float) %d,%d supressed by %d with float %f \n",bz_id,self_id,compare_id,devIoU(&bboxes[compare_box],&bboxes[self_box]));
    }
  }
}

__global__ static void _fast_nmsH(int batchsize,int bbox_num,int bbox_dim,
  const __half* bboxes,int* mask,__half overlap_thresh){
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // id of self
  int self_id = blockIdx.y * blockDim.y + threadIdx.y;  // id of other bboxes
  int bz_id=x/bbox_num;
  int compare_id=x%bbox_num;
  int compare_box=bz_id*bbox_num*bbox_dim+compare_id*bbox_dim;
  int self_box=bz_id*bbox_num*bbox_dim+self_id*bbox_dim;
  if(bz_id<batchsize && self_id<bbox_num && mask[bz_id*bbox_num+self_id]==0 && \
    compare_id<self_id && mask[bz_id*bbox_num+compare_id]==0){
    __half overlap=devIoUH(&bboxes[compare_box],&bboxes[self_box]);
    if (__hgt(overlap,overlap_thresh)){
      mask[bz_id*bbox_num+self_id]=1;
      //printf("(__half) %d supressed by %d with float %f \n",self_id,compare_id,devIoU(&bboxes[compare_box],&bboxes[self_box]));
    }
  }
}

template<typename T>
__global__ static void _fast_class_nms(int batchsize,int bbox_num,int class_num,int bbox_dim,
  const T* bboxes,int* mask,T overlap_thresh){
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // id of self, batchsize*bbox_num
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // class_num*bbox_num  
  //define idx
  int cls_idx=y/bbox_num;
  int self_idx = y%bbox_num;  // id of other bboxes
  int bz_idx=x/bbox_num;
  int compare_idx=x%bbox_num;
  //bboxes shape [bz,bbox_nums,class_num,bbox_dim]
  //mask shape [bz,class_num,bbox_num]
  int compare_box=bz_idx*bbox_num*class_num*bbox_dim+compare_idx*class_num*bbox_dim+cls_idx*bbox_dim;
  int self_box=bz_idx*bbox_num*class_num*bbox_dim+self_idx*class_num*bbox_dim+cls_idx*bbox_dim;
  int self_mask_index=bz_idx*class_num*bbox_num+cls_idx*bbox_num+self_idx;
  int compare_mask_index=bz_idx*class_num*bbox_num+cls_idx*bbox_num+compare_idx;
  //do nms
  if(bz_idx<batchsize && cls_idx<class_num && self_idx<bbox_num && \
    mask[self_mask_index]==0 && compare_idx<self_idx && mask[compare_mask_index]==0){
    if (devIoU(&bboxes[compare_box],&bboxes[self_box]) > overlap_thresh){
      mask[self_mask_index]=1;
    }
  }
}

template<typename T>
__global__ static void _fast_class_nmsH(int batchsize,int bbox_num,int class_num,int bbox_dim,
  const T* bboxes,int* mask,T overlap_thresh){
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // id of self, batchsize*bbox_num
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // class_num*bbox_num  
  //define idx
  int cls_idx=y/bbox_num;
  int self_idx = y%bbox_num;  // id of other bboxes
  int bz_idx=x/bbox_num;
  int compare_idx=x%bbox_num;
  //bboxes shape [bz,bbox_nums,class_num,bbox_dim]
  //mask shape [bz,class_num,bbox_num]
  int compare_box=bz_idx*bbox_num*class_num*bbox_dim+compare_idx*class_num*bbox_dim+cls_idx*bbox_dim;
  int self_box=bz_idx*bbox_num*class_num*bbox_dim+self_idx*class_num*bbox_dim+cls_idx*bbox_dim;
  int self_mask_index=bz_idx*class_num*bbox_num+cls_idx*bbox_num+self_idx;
  int compare_mask_index=bz_idx*class_num*bbox_num+cls_idx*bbox_num+compare_idx;
  //do nms
  if(bz_idx<batchsize && cls_idx<class_num && self_idx<bbox_num && \
    mask[self_mask_index]==0 && compare_idx<self_idx && mask[compare_mask_index]==0){
    __half overlap=devIoUH(&bboxes[compare_box],&bboxes[self_box]);
    if (__hgt(overlap,overlap_thresh)){
      mask[self_mask_index]=1;
    }
  }
}

__global__ static void _sum_boxes(int batchsize,int boxes_num,int max_to_keep,
  const int* mask,int* index,int* int_num_outs){
  int box_idx = blockIdx.x * blockDim.x + threadIdx.x; // boxes_num
  int bz_idx = blockIdx.y * blockDim.y + threadIdx.y; // batchsize

  if (box_idx<boxes_num && bz_idx<batchsize && mask[bz_idx*boxes_num+box_idx]==0) {
    const int boxId = atomicAdd(&int_num_outs[bz_idx],1);
    if (boxId < max_to_keep){
      //printf("add a box at %d(%d,%d) with %d\n",bz_idx*max_to_keep+boxId,bz_idx,boxId,box_idx);
      index[bz_idx*max_to_keep+boxId]=box_idx;
    }
  }
}

__global__ static void _sum_class_boxes(int batchsize,int boxes_num,int class_num,int max_to_keep,
  const int* mask,int* index,int* int_num_outs){
  int box_idx = blockIdx.x * blockDim.x + threadIdx.x; // boxes_num
  int y = blockIdx.y * blockDim.y + threadIdx.y; // batchsize*class_num
  int bz_idx=y/class_num;
  int cls_idx=y%class_num;
  //index shape[bz,class_num,max_to_keep]
  if (box_idx<boxes_num && bz_idx<batchsize && cls_idx<class_num){
    int mask_idx=bz_idx*class_num*boxes_num+cls_idx*boxes_num+box_idx;
    if(!mask[mask_idx]){
      const int boxId = atomicAdd(&int_num_outs[bz_idx*class_num+cls_idx],1);
      if (boxId < max_to_keep){
        index[bz_idx*class_num*max_to_keep+cls_idx*max_to_keep+boxId]=box_idx;
      }
    }
  }
}

template<typename T>
__global__ static void _clip_num_outs(int batchsize,int max_to_keep,int* int_num_outs,T* num_outs){
  int bz_id = blockIdx.x * blockDim.x + threadIdx.x;  // batchsize index
  if (bz_id<batchsize) {
    if(int_num_outs[bz_id]>max_to_keep)
      int_num_outs[bz_id]=max_to_keep;
    num_outs[bz_id]=T(int_num_outs[bz_id]);
  }
}

template<typename T>
__global__ static void _clip_class_num_outs(int batchsize,int class_num,int max_to_keep,int* int_num_outs,T* num_outs){
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // batchsize*class_num index
  int bz_idx=x/class_num;
  int cls_idx=x%class_num;

  if (bz_idx<batchsize && cls_idx<class_num) {
    if(int_num_outs[bz_idx*class_num+cls_idx]>max_to_keep)
      int_num_outs[bz_idx*class_num+cls_idx]=max_to_keep;
    num_outs[bz_idx*class_num+cls_idx]=T(int_num_outs[bz_idx*class_num+cls_idx]);
  }
}

template<typename T>
__global__ static void _gather_boxes(int batchsize,int total_num,int boxes_dim,int max_to_keep,
  const T* boxes,int* index,T* output_boxes,int* int_num_outs){
  int x=blockIdx.x * blockDim.x + threadIdx.x; //batchsize*max_to_keep
  int type_id=blockIdx.y * blockDim.y + threadIdx.y;
  int bz_id=x/max_to_keep;
  int box_index=x%max_to_keep;
  
  if(type_id<boxes_dim && bz_id<batchsize && box_index<int_num_outs[bz_id]){
    int cur_box_id=bz_id*max_to_keep*boxes_dim+box_index*boxes_dim+type_id;
    output_boxes[cur_box_id]=boxes[bz_id*total_num*boxes_dim+index[bz_id*max_to_keep+box_index]*boxes_dim+type_id];
    //printf("output_box[%d,%d,%d](%d)->%d th box ->%f\n",bz_id,box_index,type_id,cur_box_id,index[bz_id*max_to_keep+box_index],output_boxes[cur_box_id]);
  }
}

template<typename T>
__global__ static void _gather_class_boxes(int batchsize,int class_num,
  int total_num,int boxes_dim,int max_to_keep,
  const T* boxes,const T* scores,int* index,T* output_boxes,int* int_num_outs){
  int x=blockIdx.x * blockDim.x + threadIdx.x; //batchsize*class_num
  int y=blockIdx.y * blockDim.y + threadIdx.y; //max_to_keep*(boxes_dim+1)
  //define the idx
  int bz_idx=x/class_num;
  int cls_idx=x%class_num;
  int box_idx=y/(boxes_dim+1);
  int type_idx=y%(boxes_dim+1);
  //in_boxes shape [bz,total_num,class_num,boxes_dim],score shape [bz,total_num,class_num]
  //index buffer shape[bz,class_num,max_to_keep]
  //out_boxes shape [bz,class_num,max_to_keep,boxes_dim+1]
  int numout_idx=bz_idx*class_num+cls_idx;
  if(type_idx<=boxes_dim && bz_idx<batchsize && cls_idx<class_num && \
    box_idx<int_num_outs[numout_idx]){
    int cur_box_idx=bz_idx*class_num*max_to_keep*(boxes_dim+1)+cls_idx*max_to_keep*(boxes_dim+1)+box_idx*(boxes_dim+1)+type_idx;
    int index_val=index[bz_idx*class_num*max_to_keep+cls_idx*max_to_keep+box_idx];
    if(type_idx==boxes_dim){
      output_boxes[cur_box_idx]=scores[bz_idx*total_num*class_num+index_val*class_num+cls_idx];
    }else{
      output_boxes[cur_box_idx]=boxes[bz_idx*total_num*class_num*boxes_dim+index_val*class_num*boxes_dim+cls_idx*boxes_dim+type_idx];
    }
  }
}

template<typename T>
void nms_gpu(cudaStream_t stream,const T* bboxes,T* output_boxes,T* num_outs,
  int* int_num_outs,int* mask,int* index_buffer,
  int batchsize,int bbox_num,int bbox_dim,int max_to_keep,T overlap_thresh,T score_thresh)
{
  //filter with score_thresh
  dim3 Bl_filter(CU1DBLOCK);
  dim3 Gr_filter(n_blocks(batchsize*bbox_num,CU1DBLOCK));
  _filter_bboxes_by_score<<<Gr_filter,Bl_filter,0,stream>>>(batchsize,bbox_num,bbox_dim,
    bboxes,mask,score_thresh);

  //set the mask
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(batchsize*bbox_num,CU2DBLOCK),n_blocks(bbox_num,CU2DBLOCK));
  _fast_nms<<<Gr,Bl,0,stream>>>(batchsize,bbox_num,bbox_dim,
    bboxes,mask,overlap_thresh);

  //sum the boxes 
  CUDA_CHECK(cudaMemset(int_num_outs,0,batchsize*sizeof(int)));
  dim3 Bl_sum(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr_sum(n_blocks(bbox_num,CU2DBLOCK),n_blocks(batchsize,CU2DBLOCK));
  _sum_boxes<<<Gr_sum,Bl_sum,0,stream>>>(batchsize,bbox_num,max_to_keep,mask,index_buffer,int_num_outs);

  dim3 Bl_clip(CU1DBLOCK);
  dim3 Gr_clip(n_blocks(batchsize,CU1DBLOCK));
  _clip_num_outs<<<Gr_clip,Bl_clip,0,stream>>>(batchsize,max_to_keep,int_num_outs,num_outs);

  CUDA_CHECK(cudaMemset(output_boxes,0,batchsize*max_to_keep*bbox_dim*sizeof(float)));
  dim3 Bl_gather(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr_gather(n_blocks(batchsize*max_to_keep,CU2DBLOCK),n_blocks(bbox_dim,CU2DBLOCK));
  _gather_boxes<<<Bl_gather,Gr_gather,0,stream>>>(batchsize,bbox_num,bbox_dim,
    max_to_keep,bboxes,index_buffer,output_boxes,int_num_outs);
}

void nms_gpu(cudaStream_t stream,const __half* bboxes,__half* output_boxes,
  __half* num_outs,
  int* int_num_outs,int* mask,int* index_buffer,
  int batchsize,int bbox_num,int bbox_dim,int max_to_keep,__half overlap_thresh,__half score_thresh)
{
  //filter with score_thresh
  dim3 Bl_filter(CU1DBLOCK);
  dim3 Gr_filter(n_blocks(batchsize*bbox_num,CU1DBLOCK));
  _filter_bboxes_by_score<<<Gr_filter,Bl_filter,0,stream>>>(batchsize,bbox_num,bbox_dim,
    bboxes,mask,score_thresh);

  //set the mask
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(batchsize*bbox_num,CU2DBLOCK),n_blocks(bbox_num,CU2DBLOCK));
  _fast_nmsH<<<Gr,Bl,0,stream>>>(batchsize,bbox_num,bbox_dim,
    bboxes,mask,overlap_thresh);

  //sum the boxes 
  CUDA_CHECK(cudaMemset(int_num_outs,0,batchsize*sizeof(int)));
  dim3 Bl_sum(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr_sum(n_blocks(bbox_num,CU2DBLOCK),n_blocks(batchsize,CU2DBLOCK));
  _sum_boxes<<<Gr_sum,Bl_sum,0,stream>>>(batchsize,bbox_num,max_to_keep,mask,index_buffer,int_num_outs);
  
  dim3 Bl_clip(CU1DBLOCK);
  dim3 Gr_clip(n_blocks(batchsize,CU1DBLOCK));
  _clip_num_outs<<<Gr_clip,Bl_clip,0,stream>>>(batchsize,max_to_keep,int_num_outs,num_outs);

  CUDA_CHECK(cudaMemset(output_boxes,0,batchsize*max_to_keep*bbox_dim*sizeof(__half)));
  dim3 Bl_gather(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr_gather(n_blocks(batchsize*max_to_keep,CU2DBLOCK),n_blocks(bbox_dim,CU2DBLOCK));
  _gather_boxes<<<Bl_gather,Gr_gather,0,stream>>>(batchsize,bbox_num,bbox_dim,
    max_to_keep,bboxes,index_buffer,output_boxes,int_num_outs);
}

template<typename T>
void multiclass_nms_gpu(cudaStream_t stream,const T* in_boxes,const T* in_scores,T* out_boxes,T* out_nums,
  int* int_num_outs,int* mask,int* index_buffer,
  int batchsize,int box_num,int class_num,int box_dim,int max_to_keep,T overlap_thresh,T score_thresh)
{
  //in_boxes shape [bz,box_nums,class_num,box_dim],score shape [bz,box_nums,class_num]
  //std::cout<<"calling multiclass_nms_gpu with batchsize "<<batchsize<<", box_num "<<box_num<<", class_num "<<class_num<<", box_dim "<<box_dim<<", max_to_keep "<<max_to_keep<<", overlap_thresh "<<overlap_thresh<<", score_thresh "<<score_thresh<<std::endl;

  //filter with score_thresh
  dim3 Bl_filter(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr_filter(n_blocks(batchsize*class_num,CU2DBLOCK),n_blocks(box_num,CU2DBLOCK));
  _filter_bboxes_by_class_score<<<Gr_filter,Bl_filter,0,stream>>>(batchsize,box_num,
    class_num,in_scores,mask,score_thresh);

  //set the mask
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(batchsize*box_num,CU2DBLOCK),n_blocks(class_num*box_num,CU2DBLOCK));
  _fast_class_nms<<<Gr,Bl,0,stream>>>(batchsize,box_num,class_num,box_dim,
    in_boxes,mask,overlap_thresh);

  //sum the boxes 
  //num_outs shape[bz,class_num]
  CUDA_CHECK(cudaMemset(int_num_outs,0,batchsize*class_num*sizeof(int)));
  dim3 Bl_sum(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr_sum(n_blocks(box_num,CU2DBLOCK),n_blocks(batchsize*class_num,CU2DBLOCK));
  _sum_class_boxes<<<Gr_sum,Bl_sum,0,stream>>>(batchsize,box_num,class_num,max_to_keep,mask,index_buffer,int_num_outs);

  //clip and assign output nums
  dim3 Bl_clip(CU1DBLOCK);
  dim3 Gr_clip(n_blocks(batchsize*class_num,CU1DBLOCK));
  _clip_class_num_outs<<<Gr_clip,Bl_clip,0,stream>>>(batchsize,class_num,max_to_keep,int_num_outs,out_nums);

  //out_boxes shape [bz,class_num,max_to_keep,5]
  CUDA_CHECK(cudaMemset(out_boxes,0,batchsize*class_num*max_to_keep*(box_dim+1)*sizeof(T)));
  dim3 Bl_gather(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr_gather(n_blocks(batchsize*class_num,CU2DBLOCK),n_blocks(max_to_keep*(box_dim+1),CU2DBLOCK));
  _gather_class_boxes<<<Bl_gather,Gr_gather,0,stream>>>(batchsize,class_num,
    box_num,box_dim,max_to_keep,in_boxes,in_scores,index_buffer,out_boxes,int_num_outs);
}

void multiclass_nms_gpu(cudaStream_t stream,const __half* in_boxes, \
  const __half* in_scores,__half* out_boxes,__half* out_nums,
  int* int_num_outs,int* mask,int* index_buffer,
  int batchsize,int box_num,int class_num,int box_dim,int max_to_keep, \
  __half overlap_thresh,__half score_thresh)
{
  //in_boxes shape [bz,box_nums,class_num,box_dim],score shape [bz,box_nums,class_num]

  //filter with score_thresh
  dim3 Bl_filter(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr_filter(n_blocks(batchsize*class_num,CU2DBLOCK),n_blocks(box_num,CU2DBLOCK));
  _filter_bboxes_by_class_score<<<Gr_filter,Bl_filter,0,stream>>>(batchsize,box_num,
    class_num,in_scores,mask,score_thresh);

  //set the mask
  dim3 Bl(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr(n_blocks(batchsize*box_num,CU2DBLOCK),n_blocks(class_num*box_num,CU2DBLOCK));
  _fast_class_nmsH<<<Gr,Bl,0,stream>>>(batchsize,box_num,class_num,box_dim,
    in_boxes,mask,overlap_thresh);

  //sum the boxes 
  //num_outs shape[bz,class_num]
  CUDA_CHECK(cudaMemset(int_num_outs,0,batchsize*class_num*sizeof(int)));
  dim3 Bl_sum(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr_sum(n_blocks(box_num,CU2DBLOCK),n_blocks(batchsize*class_num,CU2DBLOCK));
  _sum_class_boxes<<<Gr_sum,Bl_sum,0,stream>>>(batchsize,box_num,class_num,max_to_keep,mask,index_buffer,int_num_outs);

  //clip and assign output nums
  dim3 Bl_clip(CU1DBLOCK);
  dim3 Gr_clip(n_blocks(batchsize*class_num,CU1DBLOCK));
  _clip_class_num_outs<<<Gr_clip,Bl_clip,0,stream>>>(batchsize,class_num,max_to_keep,int_num_outs,out_nums);

  //out_boxes shape [bz,class_num,max_to_keep,5]
  CUDA_CHECK(cudaMemset(out_boxes,0,batchsize*class_num*max_to_keep*(box_dim+1)*sizeof(__half)));
  dim3 Bl_gather(CU2DBLOCK,CU2DBLOCK);
  dim3 Gr_gather(n_blocks(batchsize*class_num,CU2DBLOCK),n_blocks(max_to_keep*(box_dim+1),CU2DBLOCK));
  _gather_class_boxes<<<Bl_gather,Gr_gather,0,stream>>>(batchsize,class_num,
    box_num,box_dim,max_to_keep,in_boxes,in_scores,index_buffer,out_boxes,int_num_outs);
}

template
void nms_gpu<float>(cudaStream_t stream,const float* bboxes,float* output_boxes,float* num_outs,
  int* int_num_outs,int* mask,int* index_buffer,
  int batchsize,int bbox_num,int bbox_dim,int max_to_keep,float overlap_thresh,float score_thresh);

template
void multiclass_nms_gpu<float>(cudaStream_t stream,const float* in_boxes,const float* in_scores,float* out_boxes,float* out_nums,
  int* int_num_outs,int* mask,int* index_buffer,
  int batchsize,int box_num,int class_num,int box_dim,int max_to_keep,float overlap_thresh,float score_thresh);

} // namespace ops_lib
} // namespace framework
} // namespace quake
