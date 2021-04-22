//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_KERNEL_FACE_H
#define OPSLIB_KERNEL_FACE_H

#include <vector>
#include <opencv2/opencv.hpp>

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void trans_image(const unsigned char* input,T* output,int input_h,int input_w,int channel,
  const char* layout,const char* channel_order);

template<typename T>
void trans_scale_image(const unsigned char* input,T* output,int input_h,int input_w,int channel,
  T scale,T offset,const char* layout,const char* channel_order);

void resize_image(const cv::Mat &image,cv::Mat &retImage,float* resize_info,int max_resize,const int* pre_padding,const char* flavor);

template<typename T>
void resize_image_trans(const cv::Mat &image,T* output,float* resize_info,
  int max_resize,const int* pre_padding,const char* layout,const char* channel_order,const char* flavor);

template<typename T>
void resize_image_trans(const unsigned char* input,T* output,float* resize_info,int input_h,int input_w,int channel,
  int max_resize,const int* pre_padding,const char* layout,const char* channel_order,const char* flavor);

template<typename T>
void crop_image(const cv::Mat& image,const T* coords,std::vector<cv::Mat> &cropimgs,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w);

template<typename T>
void crop_image_trans(const cv::Mat& image,const T* coords,T* output,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w,
  const char* layout,const char* channel_order);

template<typename T>
void crop_resized_image(const cv::Mat& image,const T* coords,const T* resize_info,std::vector<cv::Mat> &cropimgs,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w);

template<typename T>
void crop_resized_image_trans(const cv::Mat& image,const T* coords,const T* resize_info,T* output,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w,
  const char* layout,const char* channel_order);

template<typename T>
void crop_resized_scaled_image_trans(const cv::Mat& image,const T* coords,const T* resize_info,T* output,
  int* crop_info,int* box_ids,int box_num,int resize_h,int resize_w,T scale,T offset,
  const char* layout,const char* channel_order);

template<typename T>
void crop_image_trans(const unsigned char* input,const T* coords,
  T* output,int input_h,int input_w,int channel,
  int* crop_info,int* box_ids,int box_num,
  int resize_h,int resize_w,
  const char* layout,const char* channel_order);

template<typename T>
void crop_image_sort(const cv::Mat& image,const T* coords,const T* rankref,std::vector<cv::Mat> &cropimgs,
  int* crop_info,int* box_ids,int box_num,int max_num,int resize_h,int resize_w,const char* sort_by);

template<typename T>
void crop_image_sort_trans(const cv::Mat& image,const T* coords,const T* rankref,T* output,
  int* crop_info,int* box_ids,int box_num,int max_num,int resize_h,int resize_w,
  const char* layout,const char* channel_order,const char* sort_by);

template<typename T>
void crop_image_sort_trans(const unsigned char* input,const T* coords,const T* rankref,
  T* output,int input_h,int input_w,int channel,
  int* crop_info,int* box_ids,int box_num,int max_num,
  int resize_h,int resize_w,const char* layout,const char* channel_order,const char* sort_by);

void align_face(const cv::Mat& image,const float* kpoints,const int* scale,float* k_coords,
  cv::Mat& outputimg,int wrap_h,int wrap_w);

template<typename T>
void align_face_trans(const cv::Mat& image,const float* kpoints,const int* scale,
  T* output,float* k_coords,int wrap_h,int wrap_w,const char* layout,const char* channel_order);

template<typename T>
void align_face_trans(const unsigned char* input,const float* kpoints,const int* scale,
  T* output,float* k_coords,int input_h,int input_w,int channel,int wrap_h,int wrap_w,
  const char* layout,const char* channel_order);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_FACE_H
