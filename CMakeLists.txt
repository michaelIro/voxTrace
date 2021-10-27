#########################################################################################
# sets the minimum required version of CMake itself 
cmake_minimum_required(VERSION 3.10)

# sets the project name
project(voxTrace VERSION 1.0)

# sets the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
#########################################################################################
# add the executable
add_executable(Tests src/Tests.cpp)
add_executable(Main src/Main.cpp)
#########################################################################################
target_link_libraries(Main XRayLibAPI Shadow3API PlotAPI PolyCapAPI OptimizerAPI vt.base)
target_link_libraries(Tests XRayLibAPI Shadow3API PlotAPI PolyCapAPI OptimizerAPI vt.base)
#########################################################################################
add_subdirectory(src/api)
add_subdirectory(src/base)
add_subdirectory(docs)
######################################################################################### find installed libraries
# Armadillo: a C++ library for linear algebra & scientific computing (Apache 2.0 license)
find_package(Armadillo REQUIRED)
if (ARMADILLO_FOUND)
  include_directories(${ARMADILLO_INCLUDE_DIRS})
  target_link_libraries (Tests ${ARMADILLO_LIBRARIES})
  target_link_libraries (Main ${ARMADILLO_LIBRARIES})
endif (ARMADILLO_FOUND)
if(NOT ARMADILLO_FOUND)
    message(FATAL_ERROR "Armadillo was not found! Check if it is installed on your System. If it isn't, go to http://arma.sourceforge.net/ and follow the instrucions.")
endif(NOT ARMADILLO_FOUND)
######################################################################################## future acceleration options
find_package(HDF5 REQUIRED)
#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)
#find_package(OpenACC REQUIRED)

find_package(OpenMP REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
target_link_libraries(Main OpenMP::OpenMP_CXX)
target_link_libraries(Main OpenMP::OpenMP_C)

target_compile_options(Main PRIVATE -Wall ${OpenMP_CXX_FLAGS} ${OpenMP_C_FLAGS})

#find_package(MPI REQUIRED)
#find_package(CUDA REQUIRED)
######################################################################################## end of finding libraries
