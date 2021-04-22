//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_UTIL_BASE_H
#define OPSLIB_UTIL_BASE_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

namespace quake {
namespace framework {
namespace ops_lib {

void print_center(const std::string& msg,bool to_cout=true);

template<typename T>
void get_shapes_from_ndims(const std::vector<T>& ndims,const std::vector<T>& dims,
  std::vector<std::vector<T>>& shapes);

template<typename T>
void show_shapes(std::vector<std::vector<T>>& shapes);

template<typename T>
T get_ele_size(std::vector<T>& shape,size_t start);

template<typename T>
static void compare_buffers(const T* golden,T* buffer,size_t size){
  for(int i=0;i<size;i++)
    std::cerr<<"## "<<i<<" th buffer "<<buffer[i]<<" to valid "<<golden[i]<<std::endl;
}

template<typename T>
static int verify_buffer(const std::string& name,const T* golden,T* buffer,size_t size,float err,bool show_detail){
  int failed_cnt=0;
  float error=0;
  print_center("Checking : "+name,false);
  for(int i=0;i<size;i++){
    error=std::isnan(buffer[i]) ? 1:fabs(float(buffer[i]-golden[i])/golden[i]);
    if (error>err/100){
      std::cerr<<"## "<<i<<" th Difference output "<<buffer[i]<<" to valid "<<golden[i]<<" err "<<error*100<<" %"<<std::endl;
      if(fabs(float(golden[i]))>0.00005 || std::isnan(buffer[i]) || (std::isinf(error) && buffer[i]>0.00001))
        failed_cnt+=1;
    }
  }
  float err_rate=float(failed_cnt)*100/size;
  std::string msg=err_rate<err? "[PASS]":"[FAIL]";
  std::string final_msg="<"+msg+">, Err"+std::to_string(failed_cnt)+"/"+std::to_string(size)+"("+std::to_string(err_rate)+"%)";
  print_center(final_msg,false);
  if(show_detail){
    print_center("<Start> Compare datas",false);
    compare_buffers(golden,buffer,size);
    print_center("<End> Compare datas",false);
  }
  return failed_cnt;
}

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_UTIL_BASE_H
