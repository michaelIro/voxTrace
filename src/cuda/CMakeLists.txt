#########################################################################################
# sets the minimum required version of CMake itself 
cmake_minimum_required(VERSION 3.15)

# sets the project name and version number
project(voxTrace VERSION 1.0)

# sets the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
#########################################################################################
enable_language(CUDA)
#include(CheckLanguage)
#check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)      #TODO: Change do all or all-major in CMake 3.23
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
#########################################################################################
add_library(vt.cuda GPUTracer.cu)
#########################################################################################
#set_target_properties(CudaAPI PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_include_directories(vt.cuda PUBLIC)
#target_compile_features(vt.cuda PUBLIC cxx_std_20)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lcudart")
#set_target_properties(vt.cuda PROPERTIES CUDA_ARCHITECTURES "35;50;72")
target_link_libraries(vt.cuda XRayLibAPI)
#target_link_libraries(Tests XRayLibAPI Shadow3API PlotAPI PolyCapAPI OptimizerAPI vt.base vt.tracer)