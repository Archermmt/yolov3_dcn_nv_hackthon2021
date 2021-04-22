//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_KERNEL_GPU_CUUTILS_H
#define OPSLIB_KERNEL_GPU_CUUTILS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <iostream>
#include <cstdio> 

#define BLOCKSIZE_COL 64
#define BLOCKSIZE_ROW 4
#define BLOCKSIZE 256
#define CUDA_NUM_THREADS 512
#define CU2DBLOCK 16
#define CU1DBLOCK 256

namespace quake {
namespace framework {
namespace ops_lib {

#define KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

typedef struct MatrixDim_ {
  uint32_t rows;
  uint32_t cols;
  uint32_t stride;
} MatrixDim;

int n_blocks(int size, int block_size);

int get_block_grid_1d(const int size);

MatrixDim get_matrix_dim(int rows,int cols,int stride=-1);

template <typename T, typename U>
T bitwise_cast(U u){
  return *reinterpret_cast<T*>(&u);
}

__half dlr_float2half(float f);

float dlr_half2float(__half h);

template<typename T>
void show_vec_data(cudaStream_t stream,T* gpu_data,int size){
  T host_buffer[size];
  cudaMemcpyAsync(host_buffer,gpu_data,size*sizeof(T),cudaMemcpyDeviceToHost,stream);
  cudaStreamSynchronize(stream);
  for(int i=0;i<size;i++){
    if(sizeof(T)==sizeof(char)){
      printf("%d th data: %f\n",i,int(host_buffer[i]));
    }else{
      printf("%d th data: %f\n",i,host_buffer[i]);
    }
  }
}

template<typename T>
void show_mat_data(cudaStream_t stream,T* gpu_data,int row,int col){
  T host_buffer[row*col];
  cudaMemcpyAsync(host_buffer,gpu_data,row*col*sizeof(T),cudaMemcpyDeviceToHost,stream);
  cudaStreamSynchronize(stream);
  for(int r=0;r<row;r++){
    std::cout<<r<<"th data:"<<std::endl;
    for(int c=0;c<col;c++){
      if(sizeof(T)==sizeof(char)){
        std::cout<<int(host_buffer[r*col+c])<<",";
      }else{
        std::cout<<host_buffer[r*col+c]<<",";
      }
    }
    std::cout<<std::endl;
  }
}

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_GPU_CUUTILS_H
