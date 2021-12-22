#########################################################################################
# sets the minimum required version of CMake itself 
cmake_minimum_required(VERSION 3.18)

# Tells CMake to use pkg-config (The standard installations of both X-Ray-Lib and Polycap are easier to link using this)
include(FindPkgConfig)

# sets the project name and version number
project(voxTrace VERSION 1.0)

# sets the C++ standard to C++20
set(CMAKE_CXX_STANDARD 17)
#########################################################################################
add_library(XRayLibAPI XRayLibAPI.cpp)
add_library(Shadow3API Shadow3API.cpp)
add_library(PlotAPI PlotAPI.cpp)
add_library(PolyCapAPI PolyCapAPI.cpp)
add_library(OptimizerAPI OptimizerAPI.cpp)
#########################################################################################
# XrayLib: a library for interactions of X-rays with matter => https://github.com/tschoonj/xraylib
pkg_check_modules(XRAYLIB REQUIRED libxrl)

target_link_libraries(XRayLibAPI ${XRAYLIB_LIBRARIES})
target_include_directories(XRayLibAPI PUBLIC ${XRAYLIB_INCLUDE_DIRS})
target_compile_options(XRayLibAPI PUBLIC ${XRAYLIB_CFLAGS_OTHER})
#########################################################################################
# Shadow3: an open source ray tracing code for modeling optical systems (MIT Licence) => https://github.com/PaNOSC-ViNYL/shadow3
find_path(SHADOW_INCLUDE_DIR src/c/shadow_bind_cpp.hpp PATHS /home/miro/Software/3rd-party/shadow3 /home/miro/software/3rd-party/shadow3)
find_library(SHADOW_CPP_LIBRARY libshadow3c++.a)      #PATHS /path/to/shadow3/src
find_library(SHADOW_C_LIBRARY libshadow3c.a) 
find_library(SHADOW_LIBRARY libshadow3.a) 
#message(STATUS ${SHADOW_INCLUDE_DIR} "\t" ${SHADOW_LIBRARY} "\t" ${SHADOW_C_LIBRARY} "\t" ${SHADOW_CPP_LIBRARY})

target_include_directories(Shadow3API PUBLIC ${SHADOW_INCLUDE_DIR}/src ${SHADOW_INCLUDE_DIR}/src/c ${SHADOW_INCLUDE_DIR}/src/def)
target_link_libraries(Shadow3API ${SHADOW_CPP_LIBRARY} ${SHADOW_C_LIBRARY} ${SHADOW_LIBRARY})
target_link_libraries(Shadow3API gfortran)
#########################################################################################
# SciPlot: a C++ scientific plotiing library powerd by gnupolot (MIT License)
find_package(sciplot)
if (SCIPLOT_FOUND)
  include_directories(${SCIPLOT_INCLUDE_DIRS})
  target_link_libraries (PlotAPI ${SCIPLOT_LIBRARIES})
endif (SCIPLOT_FOUND)
#########################################################################################
# PolyCap: a C library to calculate X-ray transmission through polycapillaries => https://github.com/PieterTack/polycap
pkg_check_modules(POLYCAP REQUIRED libpolycap)
target_link_libraries(PolyCapAPI ${POLYCAP_LIBRARIES})
target_include_directories(PolyCapAPI PUBLIC ${POLYCAP_INCLUDE_DIRS})
target_compile_options(PolyCapAPI PUBLIC ${XPOLYCAP_CFLAGS_OTHER})

find_package(OpenMP REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
target_link_libraries(PolyCapAPI  OpenMP::OpenMP_CXX)
target_link_libraries(PolyCapAPI  OpenMP::OpenMP_C)

target_compile_options(PolyCapAPI PRIVATE ${OpenMP_CXX_FLAGS} ${OpenMP_C_FLAGS})

target_link_libraries(PolyCapAPI Shadow3API)
#########################################################################################
# Ensmallen: a flexible C++ library for efficient numerical optimization (3-clause BSD licence)
find_package(Ensmallen REQUIRED)
if (ENSMALLEN_FOUND)
  include_directories(${ENSMALLEN_INCLUDE_DIRS})
  target_link_libraries (OptimizerAPI ${ENSMALLEN_LIBRARIES})
endif (ENSMALLEN_FOUND)

# GSL - GNU Scientific Library (mostly GNU General Public Licence)
find_package(GSL REQUIRED)
if (GSL_FOUND)
  include_directories(${GSL_INCLUDE_DIRS})
  target_link_libraries (OptimizerAPI ${GSL_LIBRARIES})
endif (GSL_FOUND)

# OptimLib: a lightweight C++ library of numerical optimization methods for nonlinear functions. (Apache 2.0)
find_path(OPTIMLIB_INCLUDE_DIR optim/optim.hpp)
find_library(OPTIMLIB_LIBRARY optim)
target_link_libraries(OptimizerAPI ${OPTIMLIB_LIBRARY})
target_include_directories(OptimizerAPI PUBLIC ${OPTIMLIB_INCLUDE_DIR}/optim)
#message(STATUS ${OPTIMLIB_INCLUDE_DIR}/optim "\t" ${OPTIMLIB_LIBRARY})
#find_library(OPTIM_LIBRARY optimlib)
#########################################################################################