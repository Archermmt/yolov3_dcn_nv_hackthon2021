//
// Created by chengjin on 2020-06-02.
//

#include "base.h"

namespace quake {
namespace framework {
namespace ops_lib {

void print_center(const std::string& msg,bool to_cout){
  std::string decorate="#";
  std::string space=" ";
  for(int i=0;i<25;i++)
    decorate+="#";
  int space_len=std::max(int((78-msg.size())/2),0);
  for(int i=0;i<space_len;i++)
    space+=" ";
  if(to_cout){
    std::cout<<decorate<<space<<msg<<space<<(msg.size()%2==0 ? "":" ")<<decorate<<std::endl;
  }else{
    std::cerr<<decorate<<space<<msg<<space<<(msg.size()%2==0 ? "":" ")<<decorate<<std::endl;
  }
}

template<typename T>
void get_shapes_from_ndims(const std::vector<T>& ndims,const std::vector<T>& dims,
  std::vector<std::vector<T>>& shapes){
  int base=0;
  shapes.resize(ndims.size());
  for(size_t i=0;i<ndims.size();i++){
    shapes[i].resize(ndims[i]);
    for(int j=0;j<ndims[i];j++){
      shapes[i][j]=dims[base+j];
    }
    base+=ndims[i];
  }
}

template<typename T>
void show_shapes(std::vector<std::vector<T>>& shapes){
  for(size_t i=0;i<shapes.size();i++){
    std::cout<<"shape["<<i<<"]:";
    for(auto s:shapes[i]){
      std::cout<<s<<":";
    }
    std::cout<<std::endl;
  }
}

template<typename T>
T get_ele_size(std::vector<T>& shape,size_t start){
  T ele_size=1;
  for(size_t i=start;i<shape.size();i++){
    ele_size*=shape[i];
  }
  return ele_size;
}

template
void get_shapes_from_ndims<int>(const std::vector<int>& ndims,const std::vector<int>& dims,
  std::vector<std::vector<int>>& shapes);

template
void get_shapes_from_ndims<int64_t>(const std::vector<int64_t>& ndims,const std::vector<int64_t>& dims,
  std::vector<std::vector<int64_t>>& shapes);

template
void show_shapes<int>(std::vector<std::vector<int>>& shapes);

template
void show_shapes<int64_t>(std::vector<std::vector<int64_t>>& shapes);

template
int get_ele_size(std::vector<int>& shape,size_t start);

template
int64_t get_ele_size(std::vector<int64_t>& shape,size_t start);

} // namespace ops_lib
} // namespace framework
} // namespace quake
