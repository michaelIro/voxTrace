#########################################################################################
# sets the minimum required version of CMake itself 
cmake_minimum_required(VERSION 3.18)

# sets the project name and version number
project(voxTrace VERSION 1.0)

# sets the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
#########################################################################################
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)      #TODO: Change do all or all-major in CMake 3.23
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_library(vt.cuda RayGPU.cu ChemElementGPU.cu MaterialGPU.cu VoxelGPU.cu SampleGPU.cu TracerGPU.cuh TracerGPU.cu) #  TracerGPU.cu
#set_target_properties(vt.cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    #target_compile_options(vt.cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    #target_compile_options(vt.cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)
    #target_compile_options(vt.cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O0>)
endif()

target_link_libraries(vt.cuda XRayLibAPI)
#########################################################################################
