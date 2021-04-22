message(STATUS "<OPS_LIB> Building ops_kernel under ${CMAKE_CURRENT_SOURCE_DIR}")

#add cpu kernels
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/image IMAGE_KERNEL_SRCS)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/cpu CPU_KERNEL_SRCS)
#add opencv srcs
if(OpenCV_LIBS)
  message(STATUS "<OPS_LIB> Support ops_kernel with OpenCV")
  set(CPU_KERNEL_SRCS ${CPU_KERNEL_SRCS} ${IMAGE_KERNEL_SRCS})
endif()

add_library(ops_kernel SHARED ${CPU_KERNEL_SRCS})

if(OpenCV_LIBS)
  target_link_libraries(ops_kernel ${OpenCV_LIBS})
endif()

#add gpu kernels
if(CUDA_FOUND)
  if (NOT GPU_ARCH)
    SET(GPU_ARCH "6.0,6.1,6.2,7.0")
  endif ()
  string(REPLACE "," ";" _GPU_ARCH ${GPU_ARCH})
  message(STATUS "<OPS_LIB> Support ops_kernel under with GPU arch ${_GPU_ARCH}")
  foreach(ARCH ${_GPU_ARCH})
    if (ARCH EQUAL "6.0")
      #GP100/Tesla P100  DGX-1
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60)
    elseif(ARCH EQUAL "6.1")
      #GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)
    elseif(ARCH EQUAL "6.2")
      ##For Jetson Tx2 or Drive-PX2 uncomment:
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_62,code=sm_62)
    elseif(ARCH EQUAL "7.0")
      #Tesla V100
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_70,code=sm_70)
    elseif(ARCH EQUAL "7.5")
      #Tesla T4 2080TI
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=sm_75)
    else()
      message(ERROR "unsupported ARCH:${ARCH}")
    endif ()
  endforeach()
  
  AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/gpu GPU_KERNEL_SRCS)
  #old version of cmake could not find cu files
  if(${CMAKE_MAJOR_VERSION} LESS 3 OR ${CMAKE_MINOR_VERSION} LESS 10)
    file(GLOB CU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.cu)
  endif()

  cuda_add_library(ops_kernel_gpu SHARED ${GPU_KERNEL_SRCS} ${CU_SRCS})
  target_link_libraries(ops_kernel ops_kernel_gpu)
  install(TARGETS ops_kernel_gpu DESTINATION lib)
endif()

install(TARGETS ops_kernel DESTINATION lib)
