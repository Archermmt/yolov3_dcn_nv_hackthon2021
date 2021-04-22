//
// Created by chengjin on 2020-06-02.
//

#include <iostream>
#include <algorithm>
#include <exception>
#include <cstring>

#include "face.h"
#include "image_math.h"

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void trans_image(const unsigned char* input,T* output,int input_h,int input_w,int channel,
  const char* layout,const char* channel_order){
  int src_c_indices[3];

  if(!strcmp("RGB",channel_order)){
    src_c_indices[0]=2;
    src_c_indices[1]=1;
    src_c_indices[2]=0;
  }else if(!strcmp("BGR",channel_order)){
    src_c_indices[0]=0;
    src_c_indices[1]=1;
    src_c_indices[2]=2;
  }else{
    throw std::runtime_error("only support RGB and BGR");
  }
  //transpose and assign
  if(!strcmp("NCHW",layout)){
    #pragma omp parallel for
    for(int h=0;h<input_h;h++){
      for(int w=0;w<input_w;w++){
        for(int c=0;c<channel;c++){
          output[c*(input_h*input_w)+h*input_w+w]=T(input[h*(input_w*channel)+w*channel+src_c_indices[c]]);
        }
      }
    }
  }else if(!strcmp("NHWC",layout)){
    #pragma omp parallel for
    for(int h=0;h<input_h;h++){
      for(int w=0;w<input_w;w++){
        for(int c=0;c<channel;c++){
          output[h*(input_w*channel)+w*channel+c]=T(input[h*(input_w*channel)+w*channel+src_c_indices[c]]);
        }
      }
    }
  }else{
    throw std::runtime_error("only support NCHW and NHWC");
  }
}

template<typename T>
void trans_scale_image(const unsigned char* input,T* output,int input_h,int input_w,int channel,
  T scale,T offset,const char* layout,const char* channel_order){
  int src_c_indices[3];

  if(!strcmp("RGB",channel_order)){
    src_c_indices[0]=2;
    src_c_indices[1]=1;
    src_c_indices[2]=0;
  }else if(!strcmp("BGR",channel_order)){
    src_c_indices[0]=0;
    src_c_indices[1]=1;
    src_c_indices[2]=2;
  }else{
    throw std::runtime_error("only support RGB and BGR");
  }
  //transpose and assign
  if(!strcmp("NCHW",layout)){
    #pragma omp parallel for
    for(int h=0;h<input_h;h++){
      for(int w=0;w<input_w;w++){
        for(int c=0;c<channel;c++){
          output[c*(input_h*input_w)+h*input_w+w]=T(input[h*(input_w*channel)+w*channel+src_c_indices[c]])*scale+offset;
        }
      }
    }
  }else if(!strcmp("NHWC",layout)){
    #pragma omp parallel for
    for(int h=0;h<input_h;h++){
      for(int w=0;w<input_w;w++){
        for(int c=0;c<channel;c++){
          output[h*(input_w*channel)+w*channel+c]=T(input[h*(input_w*channel)+w*channel+src_c_indices[c]])*scale+offset;
        }
      }
    }
  }else{
    throw std::runtime_error("only support NCHW and NHWC");
  }
}

void resize_image(const cv::Mat &image,cv::Mat &retImage,float* resize_info,int max_resize,const int* pre_padding,const char* flavor){
  cv::Mat borderImage;
  cv::Scalar value(0,0,0);
  //pad the image if needed
  if(pre_padding[0]>0 || pre_padding[1]>0 || pre_padding[2]>0 || pre_padding[3]>0){
    cv::copyMakeBorder(image,borderImage,pre_padding[0],pre_padding[1],pre_padding[2],pre_padding[3],cv::BORDER_CONSTANT,value);
  }else{
    borderImage=image;
  }
  if(!strcmp("resize",flavor)){
    cv::Mat resizeImage;
    int top=0,bottom=0,left=0,right=0;
    float scale=1.0;
    int max_size=borderImage.cols > borderImage.rows ? borderImage.cols : borderImage.rows;
    if(max_size > max_resize){
      if (borderImage.rows < borderImage.cols){
        scale=float(max_resize)/float(borderImage.cols);
        cv::resize(borderImage,resizeImage,cv::Size(max_resize,int(scale*borderImage.rows)),0,0,cv::INTER_LINEAR);
        int height=resizeImage.rows;
        top=(max_resize-height)/2;
        bottom=top+(max_resize-height)%2;
      }else if(borderImage.rows > borderImage.cols){
        scale=float(max_resize)/float(borderImage.rows);
        cv::resize(borderImage,resizeImage,cv::Size(int(scale*borderImage.cols),max_resize),0,0,cv::INTER_LINEAR);
        int width=resizeImage.cols;
        left=(max_resize-width)/2;
        right=left+(max_resize-width)%2;
      }else if(borderImage.rows == borderImage.cols){
        scale=float(max_resize)/float(borderImage.rows);
        cv::resize(borderImage,resizeImage,cv::Size(max_resize,max_resize),0,0,cv::INTER_LINEAR);
      }
    }else{
      int height=borderImage.rows;
      int width=borderImage.cols;
      top=(max_resize-height)/2;
      left=(max_resize-width)/2;
      bottom=top+(max_resize-height)%2;
      right=left+(max_resize-width)%2;
      resizeImage=borderImage;
    }
    //pad and record resize info
    cv::copyMakeBorder(resizeImage,retImage,top,bottom,left,right,cv::BORDER_CONSTANT,value);
    resize_info[0]=scale;
    resize_info[1]=float(top);
    resize_info[2]=float(left);
    resize_info[3]=float(image.rows);
    resize_info[4]=float(image.cols);
  }else if(!strcmp("scale",flavor)){
    cv::resize(borderImage,retImage,cv::Size(max_resize,max_resize),0,0,cv::INTER_LINEAR);
    resize_info[0]=float(borderImage.cols)/max_resize;
    resize_info[1]=float(borderImage.rows)/max_resize;
    resize_info[2]=float(borderImage.cols)/max_resize;
    resize_info[3]=float(borderImage.rows)/max_resize;
    resize_info[4]=float(1);
  }
}

template<typename T>
void resize_image_trans(const cv::Mat &image,T* output,float* resize_info,
  int max_resize,const int* pre_padding,const char* layout,const char* channel_order,const char* flavor){
  cv::Mat retImage;
  resize_image(image,retImage,resize_info,max_resize,pre_padding,flavor);
  //transpose the data
  trans_image(retImage.data,output,max_resize,max_resize,retImage.channels(),layout,channel_order);
}

template<typename T>
void resize_image_trans(const unsigned char* input,T* output,float* resize_info,int input_h,int input_w,int channel,
  int max_resize,const int* pre_padding,const char* layout,const char* channel_order,const char* flavor){
  cv::Mat image;
  if(3==channel){
    image=cv::Mat(input_h,input_w,CV_8UC3);
  }else{
    throw std::runtime_error("only able to handel 3 input channels");
  }
  //copy data to mat
  std::memcpy(image.data,input,input_h*input_w*channel*sizeof(unsigned char));
  resize_image_trans(image,output,resize_info,max_resize,pre_padding,layout,channel_order,flavor);
}

template<typename T>
void crop_image_kernel(const cv::Mat& image,const T* coords,const std::vector<int>& rank,
  std::vector<cv::Mat> &cropimgs,int* crop_info,int* box_ids,int resize_h,int resize_w){
  //box ids has :<image_id,rank>
  for(unsigned int i=0;i<rank.size();i++){
    box_ids[i*2+1]=rank[i];
  }

  cropimgs.resize(rank.size());
  #pragma omp parallel for
  for(unsigned int b=0;b<rank.size();b++){
    int x1=int(coords[rank[b]*5+0]);
    int y1=int(coords[rank[b]*5+1]);
    int x2=int(coords[rank[b]*5+2]);
    int y2=int(coords[rank[b]*5+3]);
    x1=x1<0?0:x1;
    y1=y1<0?0:y1;
    x2=x2>image.cols?image.cols:x2;
    y2=y2>image.rows?image.rows:y2;

    crop_info[b*4+0]=x2-x1;
    crop_info[b*4+1]=y2-y1;
    crop_info[b*4+2]=x1;
    crop_info[b*4+3]=y1;
    cv::Mat cropimg;
    if(x2-x1<=1||y2-y1<=1){
      cropimgs[b]=cv::Mat::zeros(cv::Size(resize_h,resize_w),CV_8UC3);
    }else{
      cv::Rect roi=cv::Rect(x1,y1,x2-x1,y2-y1);
      cropimgs[b]=image(roi);
      cv::resize(cropimgs[b],cropimgs[b],cv::Size(resize_h,resize_w));
      if(!cropimgs[b].isContinuous()){
        cropimgs[b]=cropimgs[b].clone();
      }
    }
  }
}

template<typename T>
void crop_resized_image_kernel(const cv::Mat& image,const T* coords,const T* resize_info,const std::vector<int>& rank,
  std::vector<cv::Mat> &cropimgs,int* crop_info,int* box_ids,int resize_h,int resize_w){
  //box ids has :<image_id,rank>
  for(unsigned int i=0;i<rank.size();i++){
    box_ids[i*2+1]=rank[i];
  }

  cropimgs.resize(rank.size());
  #pragma omp parallel for
  for(unsigned int b=0;b<rank.size();b++){
    int x1=int((coords[rank[b]*5+0]-resize_info[2])/resize_info[0])-5;
    int y1=int((coords[rank[b]*5+1]-resize_info[1])/resize_info[0])-5;
    int x2=int((coords[rank[b]*5+2]-resize_info[2])/resize_info[0])-5;
    int y2=int((coords[rank[b]*5+3]-resize_info[1])/resize_info[0])-5;

    x1=x1<0?0:x1;
    y1=y1<0?0:y1;
    x2=x2>image.cols?image.cols:x2;
    y2=y2>image.rows?image.rows:y2;

    crop_info[b*4+0]=x2-x1;
    crop_info[b*4+1]=y2-y1;
    crop_info[b*4+2]=x1;
    crop_info[b*4+3]=y1;
    cv::Mat cropimg;
    if(x2-x1<=1||y2-y1<=1){
      cropimgs[b]=cv::Mat::zeros(cv::Size(resize_h,resize_w),CV_8UC3);
    }else{
      cv::Rect roi=cv::Rect(x1,y1,x2-x1,y2-y1);
      cropimgs[b]=image(roi);
      cv::resize(cropimgs[b],cropimgs[b],cv::Size(resize_h,resize_w));
      if(!cropimgs[b].isContinuous()){
        cropimgs[b]=cropimgs[b].clone();
      }
    }
  }
}

template<typename T>
void crop_image(const cv::Mat& image,const T* coords,std::vector<cv::Mat> &cropimgs,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w){
  //re-rank the boxes according to crop info
  std::vector<int> rank;
  for(int i=0;i<box_num;i++){
    rank.emplace_back(i);
  }
  crop_image_kernel(image,coords,rank,cropimgs,crop_info,box_ids,resize_h,resize_w);
}

template<typename T>
void crop_image_trans(const cv::Mat& image,const T* coords,T* output,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w,
  const char* layout,const char* channel_order){
  std::vector<cv::Mat> cropimgs;
  crop_image(image,coords,cropimgs,crop_info,box_ids,box_num,resize_h,resize_w);
  #pragma omp parallel for
  for(unsigned int b=0;b<cropimgs.size();b++){
    trans_image(cropimgs[b].data,output+b*resize_h*resize_w*3,resize_h,resize_w,cropimgs[b].channels(),layout,channel_order);
  }
}

template<typename T>
void crop_resized_image(const cv::Mat& image,const T* coords,const T* resize_info,std::vector<cv::Mat> &cropimgs,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w){
  //re-rank the boxes according to crop info
  std::vector<float> areas;
  std::vector<int> rank;
  areas.resize(box_num);
  for(int i=0;i<box_num;i++){
    areas[i]=(coords[i*5+3]-coords[i*5+1])*(coords[i*5+2]-coords[i*5]);
  }

  for(int i=0;i<box_num;i++){
    if(i>0 && (areas[i]>areas[i-1] || (areas[i]==areas[i-1] && coords[i*5+4]>coords[(i-1)*5+4]))){
      int step=1;
      while(i>step && (areas[i]>areas[i-step-1] || (areas[i]==areas[i-1] && coords[i*5+4]>coords[(i-step-1)*5+4])))
        step++;
      rank.insert(rank.end()-step,i);
    }else{
      rank.emplace_back(i);
    }
  }
  //std::sort(std::begin(boxs), std::end(boxs), [](std::vector<float> a, std::vector<float> b) {return (a[3]-a[1])*(a[2]-a[0]) > (b[3]-b[1])*(b[2]-b[0]); });
  /*
  std::vector<int> rank;
  for(int i=0;i<box_num;i++){
    rank.emplace_back(i);
  }*/
  crop_resized_image_kernel(image,coords,resize_info,rank,cropimgs,crop_info,box_ids,resize_h,resize_w);
}

template<typename T>
void crop_resized_image_trans(const cv::Mat& image,const T* coords,const T* resize_info,T* output,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w,
  const char* layout,const char* channel_order){
  std::vector<cv::Mat> cropimgs;
  crop_resized_image(image,coords,resize_info,cropimgs,crop_info,box_ids,box_num,resize_h,resize_w);
  #pragma omp parallel for
  for(unsigned int b=0;b<cropimgs.size();b++){
    trans_image(cropimgs[b].data,output+b*resize_h*resize_w*3,resize_h,resize_w,cropimgs[b].channels(),layout,channel_order);
  }
}

template<typename T>
void crop_resized_scaled_image_trans(const cv::Mat& image,const T* coords,const T* resize_info,T* output,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w,T scale,T offset,
  const char* layout,const char* channel_order){
  std::vector<cv::Mat> cropimgs;
  crop_resized_image(image,coords,resize_info,cropimgs,crop_info,box_ids,box_num,resize_h,resize_w);
  #pragma omp parallel for
  for(unsigned int b=0;b<cropimgs.size();b++){
    trans_scale_image(cropimgs[b].data,output+b*resize_h*resize_w*3,resize_h,resize_w,cropimgs[b].channels(),
      scale,offset,layout,channel_order);
  }
}

template<typename T>
void crop_image_trans(const unsigned char* input,const T* coords,
  T* output,int input_h,int input_w,int channel,
  int* crop_info,int* box_ids,int box_num,
  int resize_h,int resize_w,
  const char* layout,const char* channel_order){
  cv::Mat image;
  if(3==channel){
    image=cv::Mat(input_h,input_w,CV_8UC3);
  }else{
    throw std::runtime_error("only able to handel 3 input channels");
  }
  //copy data to mat
  std::memcpy(image.data,input,input_h*input_w*channel*sizeof(unsigned char));  
  crop_image_trans(image,coords,output,crop_info,box_ids,box_num,resize_h,resize_w,layout,channel_order);
}

template<typename T>
void crop_image_sort(const cv::Mat& image,const T* coords,const T* rankref,std::vector<cv::Mat> &cropimgs,
  int* crop_info,int* box_ids,int box_num,int max_num,int resize_h,int resize_w,const char* sort_by){
  //re-rank the boxes according to crop info
  std::vector<int> rank;
  if(!strcmp("area",sort_by)){
    for(int i=0;i<std::min(box_num,max_num);i++){
      if(i>0 && rankref[i]>=rankref[i-1] && coords[i*5+4]>coords[(i-1)*5+4]){
        int step=1;
        while(i>step && rankref[i]>=rankref[i-step-1] && coords[i*5+4]>coords[(i-step-1)*5+4])
          step++;
        rank.insert(rank.end()-step,i);
      }else if (i==(max_num-1) && box_num>max_num && rankref[i+1]>=rankref[i] &&
        coords[(i+1)*5+4]>coords[i*5+4]){
        int pos=i;
        while(pos<9 && rankref[pos+1]>=rankref[pos] && coords[(pos+1)*5+4]>coords[pos*5+4])
          pos++;
        rank.emplace_back(pos);
      }else{
        rank.emplace_back(i);
      }
    }
  }else if(!strcmp("score",sort_by)){
    for(int i=0;i<std::min(box_num,max_num);i++){
      rank.emplace_back(i);
    }
  }
  crop_image_kernel(image,coords,rank,cropimgs,crop_info,box_ids,resize_h,resize_w);
}

template<typename T>
void crop_image_sort_trans(const cv::Mat& image,const T* coords,const T* rankref,T* output,
  int* crop_info,int* box_ids,int box_num,int max_num,int resize_h,int resize_w,
  const char* layout,const char* channel_order,const char* sort_by){
  std::vector<cv::Mat> cropimgs;
  crop_image_sort(image,coords,rankref,cropimgs,crop_info,box_ids,box_num,max_num,resize_h,resize_w,sort_by);
  #pragma omp parallel for
  for(unsigned int b=0;b<cropimgs.size();b++){
    trans_image(cropimgs[b].data,output+b*resize_h*resize_w*3,resize_h,resize_w,cropimgs[b].channels(),layout,channel_order);
  }
}

template<typename T>
void crop_image_sort_trans(const unsigned char* input,const T* coords,const T* rankref,
  T* output,int input_h,int input_w,int channel,
  int* crop_info,int* box_ids,int box_num,int max_num,
  int resize_h,int resize_w,const char* layout,const char* channel_order,const char* sort_by){
  cv::Mat image;
  if(3==channel){
    image=cv::Mat(input_h,input_w,CV_8UC3);
  }else{
    throw std::runtime_error("only able to handel 3 input channels");
  }
  //copy data to mat
  std::memcpy(image.data,input,input_h*input_w*channel*sizeof(unsigned char));  
  crop_image_sort_trans(image,coords,rankref,output,crop_info,box_ids,box_num,max_num,resize_h,resize_w,layout,channel_order,sort_by);
}

void align_face(const cv::Mat& image,const float* kpoints,const int* scale,float* k_coords,
  cv::Mat& outputimg,int wrap_h,int wrap_w){
  const cv::Point2f a1(38.2946f,51.6963f),b1(73.5318f,51.5014f),c1(56.0252f,71.7366f),d1(41.5493f,92.3655f),e1(70.7299f,92.2041f);
  std::vector<cv::Point2f> knowPoint = {a1,b1,c1,d1,e1};
  //scale the kpoints with order [width,height,left,top]
  for(int i=0;i<5;i++){
    k_coords[i]=kpoints[i]*scale[0]+scale[2];
  }
  for(int i=5;i<10;i++){
    k_coords[i]=kpoints[i]*scale[1]+scale[3];
  }
  cv::Point2f a2(k_coords[0],k_coords[5]),b2(k_coords[1],k_coords[6]),c2(k_coords[2],k_coords[7]),d2(k_coords[3],k_coords[8]),e2(k_coords[4],k_coords[9]);
  std::vector<cv::Point2f> keyPoint = {a2,b2,c2,d2,e2};

  cv::Mat keyPointMat(static_cast<int>(keyPoint.size()),2,CV_32FC1,keyPoint.data());
  cv::Mat knowPointMat(static_cast<int>(knowPoint.size()),2,CV_32FC1,knowPoint.data());
  cv::Mat pointEstimate=similarTransform(keyPointMat,knowPointMat);
  cv::warpAffine(image,outputimg,pointEstimate(cv::Rect(0,0,3,2)),cv::Size(wrap_h,wrap_w),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(0));
  if(!outputimg.isContinuous()){
    outputimg=outputimg.clone();
  }
}

template<typename T>
void align_face_trans(const cv::Mat& image,const float* kpoints,const int* scale,
  T* output,float* k_coords,int wrap_h,int wrap_w,const char* layout,const char* channel_order){
  cv::Mat outputimg;
  align_face(image,kpoints,scale,k_coords,outputimg,wrap_h,wrap_w);
  trans_image(outputimg.data,output,wrap_h,wrap_w,outputimg.channels(),layout,channel_order);
}

template<typename T>
void align_face_trans(const unsigned char* input,const float* kpoints,const int* scale,
  T* output,float* k_coords,int input_h,int input_w,int channel,int wrap_h,int wrap_w,
  const char* layout,const char* channel_order){
  cv::Mat image;
  if(3==channel){
    image=cv::Mat(input_h,input_w,CV_8UC3);
  }else{
    throw std::runtime_error("only able to handel 3 input channels");
  }
  //copy data to mat
  std::memcpy(image.data,input,input_h*input_w*channel*sizeof(unsigned char));
  align_face_trans(image,kpoints,scale,output,k_coords,wrap_h,wrap_w,layout,channel_order);
}

template
void trans_image<float>(const unsigned char* input,float* output,int input_h,int input_w,int channel,
  const char* layout,const char* channel_order);

template
void trans_image<char>(const unsigned char* input,char* output,int input_h,int input_w,int channel,
  const char* layout,const char* channel_order);

template
void trans_scale_image(const unsigned char* input,float* output,int input_h,int input_w,int channel,
  float scale,float offset,const char* layout,const char* channel_order);

template
void resize_image_trans<float>(const cv::Mat &image,float* output,float* resize_info,
  int max_resize,const int* pre_padding,const char* layout,const char* channel_order,const char* flavor);

template
void resize_image_trans<unsigned char>(const cv::Mat &image,unsigned char* output,float* resize_info,
  int max_resize,const int* pre_padding,const char* layout,const char* channel_order,const char* flavor);

template
void resize_image_trans<float>(const unsigned char* input,float* output,float* resize_info,int input_h,int input_w,int channel,
  int max_resize,const int* pre_padding,const char* layout,const char* channel_order,const char* flavor);

template
void crop_image_sort<float>(const cv::Mat& image,const float* coords,const float* rankref,std::vector<cv::Mat> &cropimgs,
  int* crop_info,int* box_ids,int box_num,int max_num,int resize_h,int resize_w,const char* sort_by);

template
void crop_image_sort_trans<float>(const cv::Mat& image,const float* coords,const float* rankref,float* output,
  int* crop_info,int* box_ids,int box_num,int max_num,int resize_h,int resize_w,
  const char* layout,const char* channel_order,const char* sort_by);

template
void crop_image_sort_trans<float>(const unsigned char* input,const float* coords,const float* rankref,
  float* output,int input_h,int input_w,int channel,
  int* crop_info,int* box_ids,int box_num,int max_num,
  int resize_h,int resize_w,const char* layout,const char* channel_order,const char* sort_by);

template
void crop_image<float>(const cv::Mat& image,const float* coords,std::vector<cv::Mat> &cropimgs,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w);

template
void crop_image_trans<float>(const cv::Mat& image,const float* coords,float* output,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w,
  const char* layout,const char* channel_order);

template
void crop_image_trans<float>(const unsigned char* input,const float* coords,
  float* output,int input_h,int input_w,int channel,
  int* crop_info,int* box_ids,int box_num,
  int resize_h,int resize_w,
  const char* layout,const char* channel_order);

template
void crop_resized_image<float>(const cv::Mat& image,const float* coords,const float* resize_info,std::vector<cv::Mat> &cropimgs,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w);

template
void crop_resized_image_trans<float>(const cv::Mat& image,const float* coords,const float* resize_info,float* output,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w,
  const char* layout,const char* channel_order);

template
void crop_resized_scaled_image_trans(const cv::Mat& image,const float* coords,const float* resize_info,float* output,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w,float scale,float offset,
  const char* layout,const char* channel_order);

template
void align_face_trans<float>(const cv::Mat& image,const float* kpoints,const int* scale,
  float* output,float* k_coords,int wrap_h,int wrap_w,const char* layout,const char* channel_order);

template
void align_face_trans<char>(const cv::Mat& image,const float* kpoints,const int* scale,
  char* output,float* k_coords,int wrap_h,int wrap_w,const char* layout,const char* channel_order);

template
void align_face_trans<float>(const unsigned char* input,const float* kpoints,const int* scale,
  float* output,float* k_coords,int input_h,int input_w,int channel,int wrap_h,int wrap_w,
  const char* layout,const char* channel_order);

} // namespace ops_lib
} // namespace framework
} // namespace quake
