//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_KERNEL_MATH_H
#define OPSLIB_KERNEL_MATH_H

namespace quake {
namespace framework {
namespace ops_lib {

template<typename T>
void set_cpu(T* data,int size,T val);

template<typename T, typename Tidx>
void gather(const T* input,const Tidx* indices,T* output,Tidx input_dim,Tidx indice_dim);

template<typename T>
void Tgather(const T* input,const T* indices,T* output,int input_dim,int indice_dim);

template<typename T>
int condition_select(const T* input,int* output,const int cond_size,const char* select_cond,T val);

template<typename T, typename Tidx>
void merge(T* data,const T* y_data,const Tidx* indices,Tidx input_dim,Tidx indice_dim,const char* merge_mode);

template<typename Tin, typename Tout>
void cast(const Tin* input,Tout* output,int ele_num);

template<typename T>
void repeat_row(T* data,int row,int col);

template<typename T>
void nms(const T* bbox,T* nms_bbox,int* box_num,T sore_thresh,T nms_thresh,int channel,unsigned int max_to_keep);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_MATH_H
