HPC ?= false

# ── Paths ────────────────────────────────────────────────────────────────────
ifeq ($(HPC),true)
    CUDA_PATH ?= /gpfs/opt/sw/spack-0.17.1/opt/spack/linux-almalinux8-zen3/gcc-11.2.0/cuda-11.5.0-ao7cp7wu3mvop6eocjixhdcda25p24r5
    ARMA_INC  := /gpfs/opt/sw/spack-0.17.1/opt/spack/linux-almalinux8-zen3/gcc-11.2.0/armadillo-10.5.0-zzssso6lwzgjpsuubriirjj67cf2rin6/include
    ARMA_LIB  := /gpfs/opt/sw/spack-0.17.1/opt/spack/linux-almalinux8-zen3/gcc-11.2.0/armadillo-10.5.0-zzssso6lwzgjpsuubriirjj67cf2rin6/lib64
	GSL_INC   := 
	GSL_LIB   := 
    ENS_INC   := 

else
    CUDA_PATH ?= 
    ARMA_INC  := /usr/include/armadillo_bits
    ARMA_LIB  := /usr/lib
	GSL_INC   := /usr/include/gsl
	GSL_LIB   := /usr/lib/aarch64-linux-gnu
    ENS_INC   := /usr/include

endif

# ── Compilers ─────────────────────────────────────────────────────────────────
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# ── Directory layout ──────────────────────────────────────────────────────────
SRC       := $(CURDIR)/src
BUILD     := $(CURDIR)/build/src
CUDA_BLD  := $(BUILD)/cuda
API_OBJ   := $(BUILD)/api/obj
API_LIB   := $(BUILD)/api
IO_BLD    := $(BUILD)/io

# ── External dependencies ─────────────────────────────────────────────────────
XRAY_CF   := $(shell pkg-config --cflags libxrl)
XRAY_LF   := $(shell pkg-config --libs   libxrl)
HDF5_INC  := /usr/lib/aarch64-linux-gnu/
HDF5_LIB  := /usr/lib/aarch64-linux-gnu/hdf5/serial/

INCLUDES  := -I$(CUDA_PATH)/include -I$(ARMA_INC) -I$(HDF5_INC) -I$(SRC)/api
LIBRARIES := -L$(CUDA_PATH)/lib64 -L$(ARMA_LIB) -L$(HDF5_LIB) -L$(API_LIB) -L$(IO_BLD)
LINK_LIBS := -larmadillo -lhdf5 -DARMA_USE_HDF5 -lxrl \
             -l:libXRayLibAPI.a -l:libPlotAPI.a -l:libOptimizerAPI.a -l:libvt.io.a

# ── NVCC / compiler flags ─────────────────────────────────────────────────────
SMS           ?= 50 52 60 61 70 75 80 86
GENCODE_FLAGS := $(foreach sm,$(SMS),-gencode arch=compute_$(sm),code=sm_$(sm))
GENCODE_FLAGS += -gencode arch=compute_$(lastword $(sort $(SMS))),code=compute_$(lastword $(sort $(SMS))) \
                 -Wno-deprecated-gpu-targets
ifeq ($(dbg),1)
    NVCCFLAGS := -g -G
endif
NVCCFLAGS += -m64 --std=c++17 -lcudart -lstdc++ -Xcompiler -fopenmp
CCFLAGS   := --std=c++17 -fopenmp

# ── CUDA object list ──────────────────────────────────────────────────────────
CUDA_NAMES := RayGPU ChemElement MaterialGPU VoxelGPU TracerGPU
CUDA_OBJS  := $(addprefix $(CUDA_BLD)/,$(addsuffix .o,$(CUDA_NAMES)))

.PHONY: all clean test polycap-test

all: $(BUILD)/SampleTracer

test: $(BUILD)/Test

polycap-test: $(BUILD)/PolyCapTraceTest

# ── CUDA device objects (pattern rule) ───────────────────────────────────────
$(CUDA_BLD)/%.o: $(SRC)/cuda/%.cu | $(CUDA_BLD)
	$(NVCC) $(XRAY_CF) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -dc -o $@ $<

# ── API objects ───────────────────────────────────────────────────────────────
$(API_OBJ)/XRayLibAPI.o: $(SRC)/api/XRayLibAPI.cpp $(SRC)/api/XRayLibAPI.hpp | $(API_OBJ)
	$(HOST_COMPILER) $(XRAY_CF) -c $< -o $@ $(XRAY_LF)

$(API_OBJ)/PlotAPI.o: $(SRC)/api/PlotAPI.cpp $(SRC)/api/PlotAPI.hpp | $(API_OBJ)
	$(HOST_COMPILER) -I/usr/include/sciplot -c $< -o $@ -lsciplot

$(API_OBJ)/OptimizerAPI.o: $(SRC)/api/OptimizerAPI.cpp $(SRC)/api/OptimizerAPI.hpp | $(API_OBJ)
	$(HOST_COMPILER) -I/usr/include -I/usr/include/ensmallen_bits -c $< -o $@ -lensmallen

# ── API static libraries (pattern rule) ───────────────────────────────────────
$(API_LIB)/lib%.a: $(API_OBJ)/%.o | $(API_LIB)
	ar rcs $@ $<

# ── SimulationParameter (io) ──────────────────────────────────────────────────
$(IO_BLD)/SimulationParameter.o: $(SRC)/io/SimulationParameter.cpp $(SRC)/io/SimulationParameter.hpp | $(IO_BLD)
	$(HOST_COMPILER) $(XRAY_CF) $(INCLUDES) $(CCFLAGS) -Wall -Werror -c $< -o $@ $(XRAY_LF)

$(IO_BLD)/libvt.io.a: $(IO_BLD)/SimulationParameter.o
	ar rcs $@ $<

# ── Main binary ───────────────────────────────────────────────────────────────
$(BUILD)/SampleTracer.o: $(SRC)/SampleTracer.cpp | $(BUILD)
	$(NVCC) $(XRAY_CF) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -dc -o $@ $<

$(BUILD)/SampleTracer: $(BUILD)/SampleTracer.o $(CUDA_OBJS) \
    $(API_LIB)/libXRayLibAPI.a $(API_LIB)/libPlotAPI.a $(API_LIB)/libOptimizerAPI.a \
    $(IO_BLD)/libvt.io.a
	$(NVCC) $(XRAY_CF) $(NVCCFLAGS) $(GENCODE_FLAGS) \
	    -o $@ $(BUILD)/SampleTracer.o $(CUDA_OBJS) $(LIBRARIES) $(LINK_LIBS)

# ── Build directory creation ──────────────────────────────────────────────────
$(CUDA_BLD) $(API_OBJ) $(API_LIB) $(IO_BLD) $(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)

# ── TEST ──────────────────────────────────────────────────
$(BUILD)/Test.o: $(SRC)/Test.cpp | $(BUILD)
	$(HOST_COMPILER) $(XRAY_CF) $(INCLUDES) $(CCFLAGS) -c $< -o $@

$(BUILD)/Test: $(BUILD)/Test.o \
    $(API_LIB)/libXRayLibAPI.a \
    $(API_LIB)/libOptimizerAPI.a
	$(HOST_COMPILER) $(CCFLAGS) -o $@ $< \
	    -L$(API_LIB) \
	    -L$(ARMA_LIB) \
	    -L$(GSL_LIB) \
	    -l:libOptimizerAPI.a \
	    -l:libXRayLibAPI.a \
	    -larmadillo -lgsl -lgslcblas \
	    $(XRAY_LF)

# ── POLYCAP TRACE TEST ───────────────────────────────────────────────────────
$(BUILD)/PolyCapTraceTest.o: $(SRC)/tests/PolyCapTraceTest.cpp | $(BUILD)
	$(HOST_COMPILER) $(XRAY_CF) $(INCLUDES) $(CCFLAGS) -c $< -o $@

$(BUILD)/PolyCapTraceTest: $(BUILD)/PolyCapTraceTest.o \
    $(API_LIB)/libXRayLibAPI.a
	$(HOST_COMPILER) $(CCFLAGS) -o $@ $< \
	    -L$(API_LIB) \
	    -L$(ARMA_LIB) \
	    -l:libXRayLibAPI.a \
	    -larmadillo \
	    $(XRAY_LF)
