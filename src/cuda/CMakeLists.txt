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


#set(CMAKE_CXX_FLAGS "-g -G")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --verbose -dlink -Xcompiler -fvisibility=hidden -g)

#function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
#    get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
#    if(NOT "${old_flags}" STREQUAL "")
#        string(REPLACE ";" "," CUDA_flags "${old_flags}")
#        set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
#            "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
#            )
#    endif()
#endfunction()
#########################################################################################
#set(CUDA_VERBOSE_BUILD ON)
#set(CUDA_NVCC_FLAGS_DEBUG "-g -G")

add_library(vt.cuda RayGPU.cu ChemElementGPU.cu MaterialGPU.cu VoxelGPU.cu SampleGPU.cu TracerGPU.cu)
#set_target_properties(vt.cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(vt.cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    target_compile_options(vt.cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)
    target_compile_options(vt.cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O0>)
endif()
#target_compile_options(vt.cuda PRIVATE -G -g)

target_link_libraries(vt.cuda XRayLibAPI)


#########################################################################################
#SET(CUDA_INCLUDE_DIRS "/usr/local/cuda/include" ${CUDA_INCLUDE_DIRS})
#SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} "/usr/local/cuda/lib64/libcudart.so" "/usr/lib64/libcuda.so" "/usr/local/cuda/lib64/libcublas.so")
#set_target_properties(CudaAPI PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_include_directories(vt.cuda PUBLIC)
#target_compile_features(vt.cuda PUBLIC cxx_std_20)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lcudart")
#set_target_properties(vt.cuda PROPERTIES CUDA_ARCHITECTURES "35;50;72")
#SET(CUDA_PROPAGATE_HOST_FLAGS ON)
#set_property(TARGET vt.cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_)
#add_compile_options()
#target_link_libraries(Tests XRayLibAPI Shadow3API PlotAPI PolyCapAPI OptimizerAPI vt.base vt.tracer)