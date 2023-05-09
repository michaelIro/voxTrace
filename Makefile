# Usage:
#   
########################################################### 
################# COMPILE OPTIONS ######################### 
# Default value for -hpc option
HPC = false

# Check for -hpc option
ifeq ($(filter -hpc,$(MAKECMDGOALS)),)
    HPC = false
else
    HPC = true
endif
########################################################### 
############### USER SPECIFIC DIRECTORIES ################# 
############# Location of the CUDA Toolkit ################
CUDA_PATH_LOC = /usr/local/cuda-12.1#/usr/lib/nvidia-cuda-toolkit
CUDA_PATH_HPC = 
#/gpfs/opt/sw/spack-0.17.1/opt/spack/linux-almalinux8-zen3/gcc-11.2.0/cuda-11.5.0-ao7cp7wu3mvop6eocjixhdcda25p24r5
ifeq ($(HPC),true)
    CUDA_PATH ?= $(CUDA_PATH_HPC)
else
    CUDA_PATH ?= $(CUDA_PATH_LOC)
endif
######## Documentation-Tools ########
DOXYGEN = doxygen
SPHINX = sphinx-build
############# Location for XRaylib ###############
XRAYLIB_CFLAGS = `pkg-config --cflags libxrl`
XRAYLIB_LFLAGS = `pkg-config --libs libxrl`
#XRAYLIB_INCLUDE_DIR_LOC = -I/usr/include/xraylib 
#XRAYLIB_INCLUDE_DIR_HPC = -I/home/fs71764/miro/Software/3rd-Party/Install/include/xraylib 
############# Location for Polycap ###############
POLYCAP_CFLAGS = `pkg-config --cflags libpolycap`
POLYCAP_LFLAGS = `pkg-config --libs libpolycap`
#POLYCAP_INCLUDE_DIR_LOC = -I/usr/local/include/polycap -I/usr/include/easyRNG 
#POLYCAP_INCLUDE_DIR_HPC = -I/home/fs71764/miro/Software/3rd-Party/Install/include/polycap -I/home/fs71764/miro/Software/3rd-Party/Install/include/easyRNG
############# Location for Shadow3 ############### #/home/miro/Software/3rd-party/shadow3/src/c/shadow_bind_cpp.hpp
SHADOW_INCLUDE_DIRS = -I/usr/local/include/shadow3 
SHADOW_LIBS = -L/usr/local/lib -lshadow3 -lshadow3c -lshadow3c++
#SHADOW_HOME = /home/miro/Software/3rd-party/shadow3
#SHADOW_LIB_HOME = /usr/local/lib#/home/fs71764/miro/Software/3rd-Party/Install/lib
#SHADOW_INCLUDE_DIRS = -I$(SHADOW_HOME) -I$(SHADOW_HOME)/src -I$(SHADOW_HOME)/src/c -I$(SHADOW_HOME)/src/def -L$(SHADOW_HOME)/src/c -L$(SHADOW_HOME)/src 
#SHADOW_LIBS = -L/usr/lib/gcc/x86_64-linux-gnu/11/libgfortran.a -L$(SHADOW_LIB_HOME) -L$(SHADOW_LIB_HOME)/libshadow3.a -L$(SHADOW_LIB_HOME)/libshadow3c.a -L$(SHADOW_LIB_HOME)/libshadow3c++.a 
#more INCLUDE DIRS -I/usr/local/bin -I$(SHADOW_HOME) -I$(SHADOW_HOME)/src -I$(SHADOW_HOME)/src/c -I$(SHADOW_HOME)/src/def 
############# Location for GSL ###############
GSL_CFLAGS = `pkg-config --cflags gsl`
GSL_LFLAGS = `pkg-config --libs gsl`
############# Location for SCIPLOT ###############
SCIPLOT_INCLUDE_DIRS = -I/usr/local/include/sciplot #-I/usr/local/include/sciplot/specs #/home/fs71764/miro/Software/3rd-Party/Install/share/sciplot
SCIPLOT_LDFLAGS = -lsciplot
############# Location for ENSMALLEN ###############
ENSMALLEN_INCLUDE_DIRS = -I/usr/include -I/usr/include/ensmallen_bits#/home/fs71764/miro/Software/3rd-Party/Install/lib/cmake/ensmallen
ENSMALLEN_LDFLAGS = -lensmallen
############# Location for HDF5 ###############
HDF5_INCLUDE_DIR = -I/usr/include/hdf5/serial
HDF5_LIB_DIR = -L/usr/lib/x86_64-linux-gnu/hdf5/serial
############# Location for API-Libs ###############
VOXTRACE_LIB_DIR = -L$(BUILD_DIR)/api -L$(BUILD_DIR)/base -L$(BUILD_DIR)/tracer  -L$(BUILD_DIR)/io
LOCAL_LIB_DIR = -L/usr/lib/x86_64-linux-gnu 
HPC_LIB_DIR = -L/home/fs71764/miro/Software/1st-Party/voxTrace/build/src/api -L/home/fs71764/miro/Software/1st-Party/voxTrace/build/src/base -L/home/fs71764/miro/Software/3rd-Party/Install/lib -L/gpfs/opt/sw/spack-0.17.1/opt/spack/linux-almalinux8-zen3/gcc-11.2.0/armadillo-10.5.0-zzssso6lwzgjpsuubriirjj67cf2rin6/lib64 #-L/home/miro/Software/1st-party/voxTrace/build/src/api -L/home/miro/Software/1st-party/voxTrace/build/src/base -L/usr/lib/x86_64-linux-gnu
##########################################################
################### Project file structure ###############
VOXTRACE_DIR    = $(CURDIR)

# Source file directories:
SRC_MAIN_DIR    = $(VOXTRACE_DIR)/src
SRC_API_DIR     = $(SRC_MAIN_DIR)/api
SRC_BASE_DIR    = $(SRC_MAIN_DIR)/base
SRC_CUDA_DIR    = $(SRC_MAIN_DIR)/cuda
SRC_IO_DIR      = $(SRC_MAIN_DIR)/io
SRC_TRACER_DIR  = $(SRC_MAIN_DIR)/tracer

# Build directory:
BUILD_DIR               = $(VOXTRACE_DIR)/build/src
BUILD_BASE_DIR          = $(BUILD_DIR)/base
BUILD_IO_DIR            = $(BUILD_DIR)/io
BUILD_TRACER_DIR        = $(BUILD_DIR)/tracer
BUILD_CUDA_DIR          = $(BUILD_DIR)/cuda
BUILD_API_OBJ_DIR       = $(BUILD_DIR)/api/obj
BUILD_API_LIB_DIR       = $(BUILD_DIR)/api
BUILD_SPHINX_DOC_DIR    = $(BUILD_DIR)/../doc/sphinx_doc

#SPHINX_SOURCE = build/doc/xml

############# Location for Documentation ##################
XRayLibAPI_SOURCES      = $(SRC_API_DIR)/XRayLibAPI.cpp    
XRayLibAPI_HEADERS      = $(SRC_API_DIR)/XRayLibAPI.hpp
Shadow3API_SOURCES      = $(SRC_API_DIR)/Shadow3API.cpp    
Shadow3API_HEADERS      = $(SRC_API_DIR)/Shadow3API.hpp 
PolyCapAPI_SOURCES      = $(SRC_API_DIR)/PolyCapAPI.cpp    
PolyCapAPI_HEADERS      = $(SRC_API_DIR)/PolyCapAPI.hpp 
PlotAPI_SOURCES         = $(SRC_API_DIR)/PlotAPI.cpp       
PlotAPI_HEADERS         = $(SRC_API_DIR)/PlotAPI.hpp 
OptimizerAPI_SOURCES    = $(SRC_API_DIR)/OptimizerAPI.cpp  
OptimizerAPI_HEADERS    = $(SRC_API_DIR)/OptimizerAPI.hpp 

########################################################### 
#################### SOME CUDA STUFF ###################### 
##############################
# start deprecated interface #
##############################
ifeq ($(x86_64),1)
    $(info WARNING - x86_64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=x86_64 instead)
    TARGET_ARCH ?= x86_64
endif
ifeq ($(ARMv7),1)
    $(info WARNING - ARMv7 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=armv7l instead)
    TARGET_ARCH ?= armv7l
endif
ifeq ($(aarch64),1)
    $(info WARNING - aarch64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=aarch64 instead)
    TARGET_ARCH ?= aarch64
endif
ifeq ($(ppc64le),1)
    $(info WARNING - ppc64le variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=ppc64le instead)
    TARGET_ARCH ?= ppc64le
endif
ifneq ($(GCC),)
    $(info WARNING - GCC variable has been deprecated)
    $(info WARNING - please use HOST_COMPILER=$(GCC) instead)
    HOST_COMPILER ?= $(GCC)
endif
ifneq ($(abi),)
    $(error ERROR - abi variable has been removed)
endif
############################
# end deprecated interface #
############################

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 sbsa ppc64le armv7l))
    ifneq ($(TARGET_ARCH),$(HOST_ARCH))
        ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 sbsa ppc64le))
            TARGET_SIZE := 64
        else ifneq (,$(filter $(TARGET_ARCH),armv7l))
            TARGET_SIZE := 32
        endif
    else
        TARGET_SIZE := $(shell getconf LONG_BIT)
    endif
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif

# sbsa and aarch64 systems look similar. Need to differentiate them at host level for now.
ifeq ($(HOST_ARCH),aarch64)
    ifeq ($(CUDA_PATH)/targets/sbsa-linux,$(shell ls -1d $(CUDA_PATH)/targets/sbsa-linux 2>/dev/null))
        HOST_ARCH := sbsa
        TARGET_ARCH := sbsa
    endif
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-sbsa x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

# When on native aarch64 system with userspace of 32-bit, change TARGET_ARCH to armv7l
ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_SIZE),aarch64-aarch64-32)
    TARGET_ARCH = armv7l
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif


# host compiler
ifeq ($(TARGET_OS),darwin)
    ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
        HOST_COMPILER ?= clang++
    endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        ifeq ($(TARGET_OS),linux)
            HOST_COMPILER ?= arm-linux-gnueabihf-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
        else ifeq ($(TARGET_OS),android)
            HOST_COMPILER ?= arm-linux-androideabi-g++
        endif
    else ifeq ($(TARGET_ARCH),aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/q++
        else ifeq ($(TARGET_OS), android)
            HOST_COMPILER ?= aarch64-linux-android-clang++
        endif
    else ifeq ($(TARGET_ARCH),sbsa)
        HOST_COMPILER ?= aarch64-linux-gnu-g++
    else ifeq ($(TARGET_ARCH),ppc64le)
        HOST_COMPILER ?= powerpc64le-linux-gnu-g++
    endif
endif
#################### END OF CUDA STUFF #################### 
###########################################################

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     :=

# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
    CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
    LDFLAGS += -pie
    CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif
    endif
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib -L$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib/aarch64-linux-gnu -L$(TARGET_FS)/lib/aarch64-linux-gnu
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib -L$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/aarch64-linux-gnu -L$(TARGET_FS)/usr/lib/aarch64-linux-gnu
            LDFLAGS += --unresolved-symbols=ignore-in-shared-libs
            CCFLAGS += -isystem=$(TARGET_FS)/usr/include -I$(TARGET_FS)/usr/include -I$(TARGET_FS)/usr/include/libdrm
            CCFLAGS += -isystem=$(TARGET_FS)/usr/include/aarch64-linux-gnu -I$(TARGET_FS)/usr/include/aarch64-linux-gnu
        endif
    endif
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-qnx)
        NVCCFLAGS += -D_QNX_SOURCE
        NVCCFLAGS += --qpp-config 8.3.0,gcc_ntoaarch64le
        CCFLAGS += -DWIN_INTERFACE_CUSTOM -I/usr/include/aarch64-qnx-gnu
        LDFLAGS += -lsocket
        LDFLAGS += -L/usr/lib/aarch64-qnx-gnu
        CCFLAGS += "-Wl\,-rpath-link\,/usr/lib/aarch64-qnx-gnu"
        ifdef TARGET_OVERRIDE
            LDFLAGS += -lslog2
        endif

        ifneq ($(TARGET_FS),)
            LDFLAGS += -L$(TARGET_FS)/usr/lib
            CCFLAGS += "-Wl\,-rpath-link\,$(TARGET_FS)/usr/lib"
            LDFLAGS += -L$(TARGET_FS)/usr/libnvidia
            CCFLAGS += "-Wl\,-rpath-link\,$(TARGET_FS)/usr/libnvidia"
            CCFLAGS += -I$(TARGET_FS)/../include
        endif
    endif
endif

ifdef TARGET_OVERRIDE # cuda toolkit targets override
    NVCCFLAGS += -target-dir $(TARGET_OVERRIDE)
endif

# Install directory of different arch
CUDA_INSTALL_TARGET_DIR :=
ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
    CUDA_INSTALL_TARGET_DIR = targets/armv7-linux-gnueabihf/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-linux)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-linux/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),sbsa-linux)
    CUDA_INSTALL_TARGET_DIR = targets/sbsa-linux/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-android)
    CUDA_INSTALL_TARGET_DIR = targets/armv7-linux-androideabi/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-android)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-linux-androideabi/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-qnx)
    CUDA_INSTALL_TARGET_DIR = targets/ARMv7-linux-QNX/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-qnx)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-qnx/
else ifeq ($(TARGET_ARCH),ppc64le)
    CUDA_INSTALL_TARGET_DIR = targets/ppc64le-linux/
endif

# Debug build flag
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA Local
INCLUDES  := -I$(CUDA_PATH)/include $(XRAYLIB_INCLUDE_DIR_LOC) $(POLYCAP_INCLUDE_DIR_LOC) $(HDF5_INCLUDE_DIR) $(SHADOW_INCLUDE_DIRS) -I$(SRC_MAIN_DIR)/api 
LIBRARIES := -L$(CUDA_PATH)/lib64 $(HDF5_LIB_DIR) $(VOXTRACE_LIB_DIR) $(LOCAL_LIB_DIR) $(SHADOW_LIBS) 

################################################################################
# Gencode arguments
ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),armv7l aarch64 sbsa))
SMS ?= 53 61 70 72 75 80 86 87
else
SMS ?= 50 52 60 61 70 75 80 86
endif

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM) -Wno-deprecated-gpu-targets
endif
endif
##########################################################
CCLFLAGS := -larmadillo -lhdf5 -DARMA_USE_HDF5 -lxrl -l:libXRayLibAPI.a -lpolycap -l:libPolyCapAPI.a 
CCLFLAGS += -l:libShadow3API.a -lshadow3 -lshadow3c -lshadow3c++ -l:libvt.base.a -l:libvt.tracer.a -l:libvt.io.a
ALL_CCFLAGS += --std=c++17 -lcudart -lstdc++ -Xcompiler 
ALT_CCFLAGS := --std=c++17 -lcudart -lstdc++ -Wall $(CCLFLAGS) #-fopenmp # -l:libxrl.a 
##########################################################
ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif
##########################################################
###################### Target rules ######################

###################### APIs ######################
apis: api_objects api_libs

api_objects: XRayLibAPI.o Shadow3API.o PolyCapAPI.o PlotAPI.o OptimizerAPI.o 

api_libs: libXRayLibAPI.a libShadow3API.a libPolyCapAPI.a libPlotAPI.a libOptimizerAPI.a

XRayLibAPI.o: $(XRayLibAPI_SOURCES) $(XRayLibAPI_HEADERS)
	$(HOST_COMPILER) --std=c++17 $(XRAYLIB_CFLAGS) -c $(XRayLibAPI_SOURCES) -o $(BUILD_API_OBJ_DIR)/XRayLibAPI.o $(XRAYLIB_LFLAGS)

libXRayLibAPI.a: $(BUILD_API_OBJ_DIR)/XRayLibAPI.o
	ar rcs $(BUILD_API_LIB_DIR)/$@ $< 

Shadow3API.o: $(Shadow3API_SOURCES) $(Shadow3API_HEADERS) 
	$(HOST_COMPILER) --std=c++17 $(SHADOW_INCLUDE_DIRS) $(SHADOW_LIBS) -c $(Shadow3API_SOURCES) -o $(BUILD_API_OBJ_DIR)/Shadow3API.o -lshadow3 -lshadow3c -lshadow3c++

libShadow3API.a: $(BUILD_API_OBJ_DIR)/Shadow3API.o
	ar rcs $(BUILD_API_LIB_DIR)/$@ $< 

PolyCapAPI.o: $(PolyCapAPI_SOURCES) $(PolyCapAPI_HEADERS)
	$(HOST_COMPILER) --std=c++17 $(POLYCAP_CFLAGS) -c $(PolyCapAPI_SOURCES) -o $(BUILD_API_OBJ_DIR)/PolyCapAPI.o $(POLYCAP_LFLAGS)

libPolyCapAPI.a: $(BUILD_API_OBJ_DIR)/PolyCapAPI.o
	ar rcs $(BUILD_API_LIB_DIR)/$@ $< 

PlotAPI.o: $(PlotAPI_SOURCES) $(PlotAPI_HEADERS)
	$(HOST_COMPILER) --std=c++17 $(SCIPLOT_INCLUDE_DIRS) -c $(PlotAPI_SOURCES) -o $(BUILD_API_OBJ_DIR)/PlotAPI.o $(SCIPLOT_LDFLAGS)

libPlotAPI.a: $(BUILD_API_OBJ_DIR)/PlotAPI.o
	ar rcs $(BUILD_API_LIB_DIR)/$@ $< 

OptimizerAPI.o: $(OptimizerAPI_SOURCES) $(OptimizerAPI_HEADERS)
	$(HOST_COMPILER) --std=c++17 $(ENSMALLEN_INCLUDE_DIRS) -c $(OptimizerAPI_SOURCES) -o $(BUILD_API_OBJ_DIR)/OptimizerAPI.o $(ENSMALLEN_LDFLAGS)

libOptimizerAPI.a: $(BUILD_API_OBJ_DIR)/OptimizerAPI.o
	ar rcs $(BUILD_API_LIB_DIR)/$@ $<
###################### APIs ######################
all: build

build: cuda_tracer Test-3.o Test-3

cuda_tracer: RayGPU.o ChemElementGPU.o MaterialGPU.o VoxelGPU.o TracerGPU.o

new: clean apis base tracer io

doc:
	$(DOXYGEN) docs/Doxyfile ../src/api 
	$(SPHINX) -b html docs $(BUILD_SPHINX_DOC_DIR)

RayGPU.o: $(SRC_CUDA_DIR)/RayGPU.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $(BUILD_CUDA_DIR)/$@ $<

ChemElementGPU.o: $(SRC_CUDA_DIR)/ChemElementGPU.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $(BUILD_CUDA_DIR)/$@ -dc $<

MaterialGPU.o: $(SRC_CUDA_DIR)/MaterialGPU.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $(BUILD_CUDA_DIR)/$@ -dc $<

VoxelGPU.o: $(SRC_CUDA_DIR)/VoxelGPU.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $(BUILD_CUDA_DIR)/$@ -dc $<

SampleGPU.o: $(SRC_CUDA_DIR)/SampleGPU.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $(BUILD_CUDA_DIR)/$@ -dc $<

TracerGPU.o: $(SRC_CUDA_DIR)/TracerGPU.cu $(SRC_CUDA_DIR)/TracerGPU.cuh
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $(BUILD_CUDA_DIR)/$@ -dc $<

Main.o: src/Main.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $(BUILD_DIR)/$@ -dc $<

Main: $(BUILD_DIR)/Main.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $(BUILD_DIR)/$@ $+ $(LIBRARIES) $(ALL_CCFLAGS) 

Test-1.o: src/Test-1.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $(BUILD_DIR)/$@ -dc $< -Xcompiler -fPIC 

Test-1: $(BUILD_DIR)/Test-1.o
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $(BUILD_DIR)/$@ $+ $(LIBRARIES) $(ALL_CCFLAGS) -Xcompiler -fPIC 
    
Test-2.o: src/Test-2.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $(BUILD_DIR)/$@ -dc $< 

Test-2: $(BUILD_DIR)/Test-2.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $(BUILD_DIR)/$@ $+ $(LIBRARIES) $(ALL_CCFLAGS) 

Test-3.o: src/Test-3.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $(BUILD_DIR)/$@ -dc $< 

Test-3: $(BUILD_DIR)/Test-3.o $(BUILD_CUDA_DIR)/TracerGPU.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $(BUILD_DIR)/$@ $+ $(LIBRARIES) $(ALL_CCFLAGS) 

Test-4.o: src/Test-4.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $(BUILD_DIR)/$@ -dc $< 

Test-4: $(BUILD_DIR)/Test-4.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $(BUILD_DIR)/$@ $+ $(LIBRARIES) $(ALL_CCFLAGS) 

 

clean: 
	rm -rf $(BUILD_DIR)/* *.o $(EXE)
	mkdir -p $(BUILD_DIR)/api $(BUILD_DIR)/base $(BUILD_DIR)/cuda $(BUILD_DIR)/doc $(BUILD_DIR)/io $(BUILD_DIR)/tracer $(BUILD_DIR)/api/obj $(BUILD_DIR)/api/lib

base: base_objects libvt.base.a 

base_objects: $(BUILD_BASE_DIR)/ChemElement.o $(BUILD_BASE_DIR)/Material.o $(BUILD_BASE_DIR)/Ray.o $(BUILD_BASE_DIR)/Sample.o $(BUILD_BASE_DIR)/Voxel.o $(BUILD_BASE_DIR)/XRBeam.o  

libvt.base.a: 
	ar rcs $(BUILD_BASE_DIR)/$@ $(BUILD_BASE_DIR)/ChemElement.o $(BUILD_BASE_DIR)/Material.o $(BUILD_BASE_DIR)/Ray.o $(BUILD_BASE_DIR)/Sample.o $(BUILD_BASE_DIR)/Voxel.o $(BUILD_BASE_DIR)/XRBeam.o $< 

$(BUILD_BASE_DIR)/%.o: $(SRC_BASE_DIR)/%.cpp $(SRC_BASE_DIR)/%.hpp | $(BUILD_BASE_DIR)
	$(HOST_COMPILER) `pkg-config --cflags libxrl libpolycap` $(INCLUDES) $(ALT_CCFLAGS) -Wall -Werror -c $< -o $@ `pkg-config --libs libxrl libpolycap`

tracer: tracer_objects libvt.tracer.a

tracer_objects: $(BUILD_TRACER_DIR)/PrimaryBeam.o $(BUILD_TRACER_DIR)/Tracer.o

libvt.tracer.a: 
	ar rcs $(BUILD_TRACER_DIR)/$@ $(BUILD_TRACER_DIR)/PrimaryBeam.o $(BUILD_TRACER_DIR)/Tracer.o $< 

$(BUILD_TRACER_DIR)/%.o: $(SRC_TRACER_DIR)/%.cpp $(SRC_TRACER_DIR)/%.hpp | $(BUILD_TRACER_DIR)
	$(HOST_COMPILER) `pkg-config --cflags libxrl libpolycap` $(INCLUDES) $(ALT_CCFLAGS) -Wall -Werror -c $< -o $@ `pkg-config --libs libxrl libpolycap`

io: io_objects libvt.io.a

io_objects: $(BUILD_IO_DIR)/SimulationParameter.o 

libvt.io.a: 
	ar rcs $(BUILD_IO_DIR)/$@ $(BUILD_IO_DIR)/SimulationParameter.o $< 

$(BUILD_IO_DIR)/%.o: $(SRC_IO_DIR)/%.cpp $(SRC_IO_DIR)/%.hpp | $(BUILD_IO_DIR)
	$(HOST_COMPILER) `pkg-config --cflags libxrl libpolycap` $(INCLUDES) $(ALT_CCFLAGS) -Wall -Werror -c $< -o $@ `pkg-config --libs libxrl libpolycap`

fast: new cuda_tracer Test-3.o Test-3

ffast: cuda_tracer Test-3.o Test-3
