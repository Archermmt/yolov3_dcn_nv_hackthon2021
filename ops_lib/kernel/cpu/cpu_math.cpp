//
// Created by chengjin on 2020-06-02.
//

#include <cstring>
#include <iostream>
#include <vector>
#include <stdexcept>
#include "cpu_math.h"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void set_cpu(T* data,int size,T val){
  for(int i=0;i<size;i++){
    data[i]=val;
  }
}

template<typename T, typename Tidx>
void gather(const T* input,const Tidx* indices,T* output,Tidx input_dim,Tidx indice_dim){
  #pragma omp parallel for
  for(int i=0;i<indice_dim;i++){
    std::memcpy(output+i*input_dim,input+indices[i]*input_dim,input_dim*sizeof(T));
  }
}

template<typename T>
void Tgather(const T* input,const T* indices,T* output,int input_dim,int indice_dim){
  #pragma omp parallel for
  for(int i=0;i<indice_dim;i++){
    std::memcpy(output+i*input_dim,input+int(indices[i])*input_dim,input_dim*sizeof(T));
  }
}

template<typename T>
int condition_select(const T* input,int* output,const int cond_size,const char* select_cond,T val){
  int select_size=0;
  if(!strcmp("not_equal",select_cond)){
    for(int i=0;i<cond_size;i++){
      if(input[i]!=val){
        output[select_size]=i;
        select_size+=1;
      }
    }
  }else{
    throw std::runtime_error("only support not_equal select condition");
  }
  return select_size;
}

template<typename T, typename Tidx>
void merge(T* data,const T* y_data,const Tidx* indices,Tidx input_dim,Tidx indice_dim,const char* merge_mode){
  if(!strcmp("add",merge_mode)){
    #pragma omp parallel for
    for(int i=0;i<indice_dim;i++){
      #pragma omp parallel for
      for(int j=0;j<input_dim;j++){
        data[int(indices[i])*input_dim+j]+=y_data[int(i)*input_dim+j];
      }
    }
  }else{
    throw std::runtime_error("only support add merge mode");
  }
}

template<typename Tin, typename Tout>
void cast(const Tin* input,Tout* output,int ele_num){
  #pragma omp parallel for
  for(int i=0;i<ele_num;i++){
    output[i]=input[i];
  }
}

template<typename T>
void repeat_row(T* data,int row,int col){
  #pragma omp parallel for
  for(int i=1;i<row;i++){
    std::memcpy(data+i*col,data,col*sizeof(T));
  }
}

template<typename T>
void nms(const T* bbox,T* nms_bbox,int* box_num,T score_thresh,T nms_thresh,int channel,unsigned int max_to_keep){
  //shape of bbox [channel,5], where 5 from (x1,y1,x2,y2,score)
  bool suppressed[channel];
  std::memset(suppressed,false,channel);
  std::vector<int> keep;
  //do the nms
  for(int c=0;c<channel;c++){
    if(suppressed[c] or bbox[c*5+4]<score_thresh or keep.size()>max_to_keep){
      continue;
    }else{
      keep.emplace_back(c);
      T x1=bbox[c*5+0];
      T y1=bbox[c*5+1];
      T x2=bbox[c*5+2];
      T y2=bbox[c*5+3];
      T base_area=(x2-x1+1)*(y2-y1+1);
      for(int j=c+1;j<channel;j++){
      	if(suppressed[j] or bbox[j*5+4]<score_thresh){
      	  continue;
      	}else{
      	  T cur_x1=bbox[j*5+0];
          T cur_y1=bbox[j*5+1];
          T cur_x2=bbox[j*5+2];
          T cur_y2=bbox[j*5+3];
      	  T cur_area=(cur_x2-cur_x1+1)*(cur_y2-cur_y1+1);
      	  //interaction area
      	  T xx1=std::max(x1,cur_x1);
          T yy1=std::max(y1,cur_y1);
          T xx2=std::min(x2,cur_x2);
          T yy2=std::min(y2,cur_y2);
          T w=std::max(T(0),xx2-xx1+1);
          T h=std::max(T(0),yy2-yy1+1);
          T inter=w*h;
          T ovr=inter/(base_area+cur_area-inter);
          if(ovr>=nms_thresh)
            suppressed[j]=true;
      	}
      }
    }
  }
  box_num[0]=keep.size();
  /*
  std::cout<<"keep ("<<keep.size()<<"): ";
  for(int i=0;i<keep.size();i++)
    std::cout<<keep[i]<<",";
  std::cout<<std::endl;
  */
  gather(bbox,keep.data(),nms_bbox,5,box_num[0]);
}

template
void set_cpu<float>(float* data,int size,float val);

template
void gather<float,int64_t>(const float* input,const int64_t* indices,float* output,int64_t input_dim,int64_t indice_dim);

template
void gather<float,int>(const float* input,const int* indices,float* output,int input_dim,int indice_dim);

template
void gather<int,int>(const int* input,const int* indices,int* output,int input_dim,int indice_dim);

template
int condition_select<float>(const float* input,int* output,const int cond_size,const char* select_cond,float val);

template
int condition_select<int>(const int* input,int* output,const int cond_size,const char* select_cond,int val);

template
void merge<float,int>(float* data,const float* y_data,const int* indices,int input_dim,int indice_dim,const char* merge_mode);

template
void cast<int64_t,float>(const int64_t* input,float* output,int ele_num);

template
void repeat_row<int8_t>(int8_t* data,int row,int col);

template
void repeat_row<int16_t>(int16_t* data,int row,int col);

template
void repeat_row<int>(int* data,int row,int col);

template
void repeat_row<float>(float* data,int row,int col);

template
void Tgather<float>(const float* input,const float* indices,float* output,int input_dim,int indice_dim);

template
void nms<float>(const float* bbox,float* nms_bbox,int* box_num,float sore_thresh,float nms_thresh,int channel,unsigned int max_to_keep);

} // namespace ops_lib
} // namespace framework
} // namespace quake
