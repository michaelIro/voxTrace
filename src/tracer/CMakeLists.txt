#########################################################################################
# sets the minimum required version of CMake itself 
cmake_minimum_required(VERSION 3.10)

# sets the project name and version number
project(voxTrace VERSION 1.0)

# sets the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
#########################################################################################
add_library(vt.tracer Tracer.cpp PrimaryBeam.cpp)
target_link_libraries(vt.tracer XRayLibAPI Shadow3API PlotAPI PolyCapAPI OptimizerAPI vt.base)