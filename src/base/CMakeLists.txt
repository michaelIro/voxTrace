#########################################################################################
# sets the minimum required version of CMake itself 
cmake_minimum_required(VERSION 3.10)

# sets the project name and version number
project(voxTrace VERSION 1.0)

# sets the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
#########################################################################################
add_library(vt.base ChemElement.cpp Material.cpp Ray.cpp Voxel.cpp Sample.cpp)
target_link_libraries(vt.base XRayLibAPI)