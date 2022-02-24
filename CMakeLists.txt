#########################################################################################
# sets the minimum required version of CMake itself 
cmake_minimum_required(VERSION 3.15)

# sets the project name and version
project(voxTrace VERSION 1.0)

# sets the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
#########################################################################################
link_directories(/home/fs71764/miro/Software/3rd-Party/Install/lib /opt/sw/spack-0.12.1/opt/spack/linux-centos7-x86_64/gcc-4.8.5/gcc-9.1.0-mj7s6dgfnhgi2n42fyxgmitnuslcyol3/bin/gfortran)
#list(APPEND CMAKE_PREFIX_PATH "$HOME/Software/3rd-Party/Install/xraylib")
#########################################################################################
#enable_language(CUDA)
##include(CheckLanguage)
##check_language(CUDA)

#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#  set(CMAKE_CUDA_ARCHITECTURES 75)      #TODO: Change do all or all-major in CMake 3.23
#endif()

#if(NOT DEFINED CMAKE_CUDA_STANDARD)
#    set(CMAKE_CUDA_STANDARD 11)
#    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#endif()
#########################################################################################
# add the executable
add_executable(Test-1 src/Test-1.cpp)
add_executable(Test-2 src/Test-2.cpp)
add_executable(Test-3 src/Test-3.cpp)
add_executable(Test-4 src/Test-4.cpp)
add_executable(Main src/Main.cpp)
#########################################################################################
target_link_libraries(Main XRayLibAPI Shadow3API PlotAPI PolyCapAPI OptimizerAPI vt.base vt.tracer) #vt.tracer
target_link_libraries(Test-1 XRayLibAPI Shadow3API PlotAPI PolyCapAPI OptimizerAPI vt.base vt.tracer)#vt.tracer
target_link_libraries(Test-2 XRayLibAPI Shadow3API PlotAPI PolyCapAPI OptimizerAPI vt.base vt.tracer)
target_link_libraries(Test-3 XRayLibAPI Shadow3API PlotAPI PolyCapAPI OptimizerAPI vt.base vt.tracer) #vt.tracer vt.cuda
target_link_libraries(Test-4 XRayLibAPI Shadow3API PlotAPI PolyCapAPI OptimizerAPI vt.base vt.tracer) #vt.tracer
#########################################################################################
add_subdirectory(src/api)
add_subdirectory(src/base)
add_subdirectory(src/tracer)
add_subdirectory(src/cuda)
#add_subdirectory(docs)
######################################################################################### find installed libraries
# Armadillo: a C++ library for linear algebra & scientific computing (Apache 2.0 license)
find_package(Armadillo REQUIRED)
if (ARMADILLO_FOUND)
  include_directories(${ARMADILLO_INCLUDE_DIRS})
  target_link_libraries (Test-1 ${ARMADILLO_LIBRARIES})
  target_link_libraries (Test-2 ${ARMADILLO_LIBRARIES})
  target_link_libraries (Test-3 ${ARMADILLO_LIBRARIES})
  target_link_libraries (Test-4 ${ARMADILLO_LIBRARIES})
  target_link_libraries (Main ${ARMADILLO_LIBRARIES})
endif (ARMADILLO_FOUND)
if(NOT ARMADILLO_FOUND)
    message(FATAL_ERROR "Armadillo was not found! Check if it is installed on your System. If it isn't, go to http://arma.sourceforge.net/ and follow the instrucions.")
endif(NOT ARMADILLO_FOUND)

target_compile_options(Test-1 PUBLIC -DARMA_USE_HDF5)#-O3
target_compile_options(Test-2 PUBLIC -DARMA_USE_HDF5) #-O3
target_compile_options(Test-3 PUBLIC -DARMA_USE_HDF5) #-lcuda -O3
target_compile_options(Test-4 PUBLIC -DARMA_USE_HDF5) #-O3
target_compile_options(Main PUBLIC -DARMA_USE_HDF5) #-O3
######################################################################################## future acceleration options
find_package(HDF5 REQUIRED)
if (HDF5_FOUND)
  message(STATUS "THIS IS SCIPLOT....." ${HDF5_INCLUDE_DIRS})
  include_directories(${HDF5_INCLUDE_DIRS})
  target_link_libraries (Main ${HDF5_LIBRARIES})
  target_link_libraries (Test-1 ${HDF5_LIBRARIES})
  target_link_libraries (Test-2 ${HDF5_LIBRARIES})
  target_link_libraries (Test-3 ${HDF5_LIBRARIES})
  target_link_libraries (Test-4 ${HDF5_LIBRARIES})
endif (HDF5_FOUND)
#find_package(BLAS REQUIRED)      # These 
#find_package(LAPACK REQUIRED)    #
#find_package(OpenACC REQUIRED)   #
#find_package(MPI REQUIRED)
#find_package(CUDA REQUIRED)
######################################################################################## end of finding libraries

