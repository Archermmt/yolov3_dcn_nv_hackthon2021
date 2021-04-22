if(INTERGRATE)
  message(STATUS "<OPS_LIB> Building ops lib as integrated part")
else()
  message(STATUS "<OPS_LIB> Building ops lib independend")
  cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
  project(dlr_opslib)

  find_package(CUDA 8.0)
  find_package(Opencv)
endif()

set(CMAKE_CXX_STANDARD 11)

if(CUDA_FOUND)
  add_definitions(-DDLR_GPU)
endif()

include(tensorrt/find_trt_lib.cmake)
if(CUDA_FOUND AND TensorRT_FOUND)
  set(BUILD_TRT ON)
else()
  set(BUILD_TRT OFF)
endif()

set(CMAKE_CXX_FLAGS "-Wall -fPIC -O3 -msse2 -msse4")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(util)
add_subdirectory(kernel)

if(BUILD_TRT)
  add_subdirectory(tensorrt)
endif()

INSTALL(DIRECTORY ./ DESTINATION include/ops_lib FILES_MATCHING PATTERN "*.h" PATTERN "*.config")
