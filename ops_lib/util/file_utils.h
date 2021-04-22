//
// Created by chengjin on 2020-02-03.
//

#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <cmath>
#include "base.h"

namespace quake {
namespace framework {
namespace ops_lib {

class FileUtils {
public:
  static inline bool file_exist(const std::string& file) {
    std::ifstream in_file(file,std::ifstream::binary);
    if (in_file.is_open()){
      in_file.close();
      return true;
    }else{
      return false;
    }
  }

  template<typename T>
  static inline bool write_buffer_to_file(const std::string& file,T* buffer,int size) {
    std::ofstream out_file(file,std::ofstream::binary);
    if (out_file.is_open()){
      out_file.write(reinterpret_cast<char*>(buffer),size*sizeof(T));
      out_file.close();
      return true;
    }else{
      return false;
    }
  }

  template<typename T>
  static void compare_buffer_with_file(const std::string& file,T* buffer,size_t size){
    int pos=0;
    T valid_out;
    std::ifstream in_file(file,std::ifstream::binary);
    assert(in_file.is_open() && ("Failed to open file "+file).c_str());
    while (in_file.read(reinterpret_cast<char*>(&valid_out),sizeof(T)) && pos<size){
      std::cerr<<"## "<<pos<<" th output "<<buffer[pos]<<" to valid "<<valid_out<<std::endl;
      pos++;
    }
    in_file.close();
  }

  template<typename T>
  static int verify_buffer_with_file(const std::string& name,const std::string& file,T* buffer,size_t size,float err,bool show_detail){
    int pos=0;
    int failed_cnt=0;
    float error=0;
    T valid_out;
    std::ifstream in_file(file,std::ifstream::binary);
    if(in_file.is_open()){
      print_center("Checking : "+name,false);
      while (in_file.read(reinterpret_cast<char*>(&valid_out),sizeof(T)) && pos<size){
        if(std::isnan(buffer[pos])){
          error=1;
        }else{
          error=fabs(float(buffer[pos]-valid_out)/valid_out);
        }
        if (error>err/100){
          if(!show_detail){
            std::cerr<<"## "<<pos<<" th Difference output "<<buffer[pos]<<" to valid "<<valid_out<<" err "<<error*100<<" %"<<std::endl;
          }
          if(fabs(float(valid_out))>0.0001 || std::isnan(buffer[pos]) || (std::isinf(error) && buffer[pos]>0.00001))
            failed_cnt+=1;
        }
        pos++;
      }

      in_file.close();
      float err_rate=float(failed_cnt)*100/pos;
      std::string msg=err_rate<err? "[PASS]":"[FAIL]";
      std::string final_msg="<"+msg+">, Err"+std::to_string(failed_cnt)+"/"+std::to_string(pos)+"("+std::to_string(err_rate)+"%)";
      print_center(final_msg,false);
      if(show_detail){
        print_center("<Start> Compare datas",false);
        compare_buffer_with_file(file,buffer,size);
        print_center("<End> Compare datas",false);
      }
      return failed_cnt;
    }else{
      print_center("Can not open "+file+", show the results",false);
      for(int i=0;i<size;i++){
        std::cerr<<"## "<<i<<" th result "<<buffer[i]<<std::endl;
      }
      return 0;
    }
  }

  template<typename T>
  static bool read_file_to_buffer(const std::string& file,T* buffer,int size,bool show_detail){
    std::ifstream in_file(file,std::ifstream::binary);
    if(!in_file.is_open()){
      if(show_detail)
        std::cout<<"[INFO] Failed to open "<<file<<std::endl;
      return false;
    }
    in_file.read((char*)(&buffer[0]), sizeof(T)*size);
    in_file.close();
    if(show_detail){
      print_center("<Start> Compare datas from "+file,false);
      compare_buffer_with_file(file,buffer,size);
      print_center("<End> Compare datas from "+file,false);
    }
    return true;
  }
};

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //FILE_UTILS_H
