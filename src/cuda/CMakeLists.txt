#########################################################################################
# sets the minimum required version of CMake itself 
cmake_minimum_required(VERSION 3.18)

# sets the project name and version number
project(voxTrace VERSION 1.0)

# sets the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
#########################################################################################
add_library(vt.cuda MyDummy.cu)
#########################################################################################
#set_target_properties(CudaAPI PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_include_directories(vt.cuda PUBLIC)
#target_compile_features(vt.cuda PUBLIC cxx_std_20)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lcudart")
#set_target_properties(vt.cuda PROPERTIES CUDA_ARCHITECTURES "35;50;72")
#target_link_libraries(vt.tracer XRayLibAPI Shadow3API PlotAPI PolyCapAPI OptimizerAPI vt.base)
#target_link_libraries(Tests XRayLibAPI Shadow3API PlotAPI PolyCapAPI OptimizerAPI vt.base vt.tracer)