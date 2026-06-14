HPC ?= false

# ── Kokkos ─────────────────────────────────────────────────────────────────────
# Kokkos 4.x is installed via CMake (no Makefile.kokkos in 4.x).
# KOKKOS_INSTALL points to the CMake install prefix (cmake --install).
# Override on the command line:  make KOKKOS_INSTALL=/path/to/kokkos-install
# For CUDA:  make KOKKOS_INSTALL=... KOKKOS_CXX=$(HOME)/kokkos/bin/nvcc_wrapper
KOKKOS_INSTALL ?= $(HOME)/kokkos-install
KOKKOS_INC    := $(KOKKOS_INSTALL)/include
KOKKOS_LIB    := $(KOKKOS_INSTALL)/lib

# ── Paths ──────────────────────────────────────────────────────────────────────
UNAME_S := $(shell uname -s)

ifeq ($(HPC),true)
    ARMA_INC  := /gpfs/opt/sw/spack-0.17.1/opt/spack/linux-almalinux8-zen3/gcc-11.2.0/armadillo-10.5.0-zzssso6lwzgjpsuubriirjj67cf2rin6/include
    ARMA_LIB  := /gpfs/opt/sw/spack-0.17.1/opt/spack/linux-almalinux8-zen3/gcc-11.2.0/armadillo-10.5.0-zzssso6lwzgjpsuubriirjj67cf2rin6/lib64
    GSL_INC   :=
    GSL_LIB   :=
    ENS_INC   :=
    HDF5_INC_ := /usr/lib/aarch64-linux-gnu/
    HDF5_LIB_ := /usr/lib/aarch64-linux-gnu/hdf5/serial/
else ifeq ($(UNAME_S),Darwin)
    BREW      := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
    ARMA_INC  := $(BREW)/include
    ARMA_LIB  := $(BREW)/lib
    GSL_INC   := $(BREW)/include
    GSL_LIB   := $(BREW)/lib
    ENS_INC   := $(BREW)/include
    HDF5_INC_ := $(BREW)/include
    HDF5_LIB_ := $(BREW)/lib
else
    ARMA_INC  := /usr/include/armadillo_bits
    ARMA_LIB  := /usr/lib
    GSL_INC   := /usr/include/gsl
    GSL_LIB   := /usr/lib/aarch64-linux-gnu
    ENS_INC   := /usr/include
    HDF5_INC_ := /usr/lib/aarch64-linux-gnu/
    HDF5_LIB_ := /usr/lib/aarch64-linux-gnu/hdf5/serial/
endif

# ── Compilers ──────────────────────────────────────────────────────────────────
HOST_COMPILER ?= g++
# KOKKOS_CXX can be overridden to nvcc_wrapper for CUDA or hipcc for HIP.
KOKKOS_CXX    ?= $(HOST_COMPILER)
CXX           := $(KOKKOS_CXX)

# ── Directory layout ───────────────────────────────────────────────────────────
SRC       := $(CURDIR)/src
BUILD     := $(CURDIR)/build/src
CORE_BLD  := $(BUILD)/core
API_OBJ   := $(BUILD)/api/obj
API_LIB   := $(BUILD)/api
IO_BLD    := $(BUILD)/io
TESTS_BLD := $(BUILD)/tests

# ── Apple Silicon / Metal (arm64 only) ─────────────────────────────────────────
UNAME_M := $(shell uname -m)

METAL_COMPILER := $(shell xcrun --find metal 2>/dev/null)

ifeq ($(UNAME_M),arm64)
ifeq ($(METAL_COMPILER),)
    # arm64 but Xcode.app not installed — build without Metal
    METAL_FLAGS :=
    METAL_LIBS  :=
    METAL_AIR   :=
    METAL_LIB   :=
    METAL_OBJ   :=
else
    METAL_BLD   := $(BUILD)/metal
    METAL_FLAGS := -DVOXTRACE_METAL
    METAL_LIBS  := -framework Metal -framework Foundation
    METAL_AIR   := $(METAL_BLD)/Tracer.air
    METAL_LIB   := $(METAL_BLD)/Tracer.metallib
    METAL_OBJ   := $(METAL_BLD)/MetalTracer.o
endif
else
    METAL_FLAGS :=
    METAL_LIBS  :=
    METAL_AIR   :=
    METAL_LIB   :=
    METAL_OBJ   :=
endif

# ── External dependencies ──────────────────────────────────────────────────────
HDF5_INC := $(HDF5_INC_)
HDF5_LIB := $(HDF5_LIB_)
# Use pkg-config when available; otherwise fall back to bare -lxrl
PKGCFG   := $(shell command -v pkg-config 2>/dev/null)
ifneq ($(PKGCFG),)
    XRAY_CF := $(shell pkg-config --cflags libxrl 2>/dev/null)
    XRAY_LF := $(shell pkg-config --libs   libxrl 2>/dev/null)
else
    XRAY_CF :=
    XRAY_LF := -lxrl
endif

INCLUDES  := -I$(KOKKOS_INC) \
             -I$(ARMA_INC) -I$(HDF5_INC) \
             -I$(SRC) -I$(SRC)/core \
             $(XRAY_CF)

LIBRARIES := -L$(ARMA_LIB) -L$(HDF5_LIB) -L$(API_LIB)

LINK_LIBS := -larmadillo -lhdf5 -DARMA_USE_HDF5 -lxrl \
             $(API_LIB)/libXRayLibAPI.a $(API_LIB)/libOptimizerAPI.a \
             $(METAL_LIBS)

# ── OpenMP flags (macOS: Homebrew libomp; Linux: -fopenmp) ────────────────────
ifeq ($(UNAME_S),Darwin)
    LIBOMP      := $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
    OMP_CFLAGS  := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
    OMP_LFLAGS  := -L$(LIBOMP)/lib -lomp
else
    OMP_CFLAGS  := -fopenmp
    OMP_LFLAGS  := -fopenmp
endif

# ── Kokkos static libs ──────────────────────────────────────────────────────────
KOKKOS_LIBS := $(KOKKOS_LIB)/libkokkoscore.a $(KOKKOS_LIB)/libkokkoscontainers.a

# ── Compiler flags ─────────────────────────────────────────────────────────────
CXXFLAGS := --std=c++20 $(OMP_CFLAGS) $(METAL_FLAGS)
LDFLAGS  := -L$(KOKKOS_LIB) $(OMP_LFLAGS)

# ── Core objects (only Tracer.cpp needs compilation; physics is header-only) ───
CORE_OBJS := $(CORE_BLD)/Tracer.o

.PHONY: all clean test test2 polycap-test

all: $(BUILD)/Test

test: $(BUILD)/Test

test2: $(BUILD)/Test2

polycap-test: $(BUILD)/PolyCapBatchTest

# ── Core object ───────────────────────────────────────────────────────────────
$(CORE_BLD)/Tracer.o: $(SRC)/core/Tracer.cpp $(SRC)/core/Tracer.hpp | $(CORE_BLD)
	$(CXX) $(INCLUDES) --std=c++20 $(OMP_CFLAGS) -c -o $@ $<

# ── API objects ───────────────────────────────────────────────────────────────
$(API_OBJ)/XRayLibAPI.o: $(SRC)/api/XRayLibAPI.cpp $(SRC)/api/XRayLibAPI.hpp | $(API_OBJ)
	$(HOST_COMPILER) $(XRAY_CF) -c $< -o $@ $(XRAY_LF)

$(API_OBJ)/OptimizerAPI.o: $(SRC)/api/OptimizerAPI.cpp $(SRC)/api/OptimizerAPI.hpp | $(API_OBJ)
	$(HOST_COMPILER) -I$(ENS_INC) -I$(ENS_INC)/ensmallen_bits -c $< -o $@

# ── API static libraries ──────────────────────────────────────────────────────
$(API_LIB)/lib%.a: $(API_OBJ)/%.o | $(API_LIB)
	ar rcs $@ $<

# ── SimulationParameter (now header-only in core) ──────────────────────────────

# ── Metal (arm64 only) ────────────────────────────────────────────────────────
ifneq ($(METAL_AIR),)
$(METAL_BLD):
	mkdir -p $@

$(METAL_AIR): $(SRC)/metal/Tracer.metal | $(METAL_BLD)
	xcrun -sdk macosx metal -c $< -o $@ -I$(SRC)

$(METAL_LIB): $(METAL_AIR)
	xcrun -sdk macosx metallib $< -o $@

$(METAL_OBJ): $(SRC)/metal/MetalTracer.mm $(METAL_LIB) | $(METAL_BLD)
	$(HOST_COMPILER) --std=c++17 $(METAL_FLAGS) $(INCLUDES) -fobjc-arc -c $< -o $@
endif

# ── Main binary ───────────────────────────────────────────────────────────────
$(TESTS_BLD)/SampleTracer.o: $(SRC)/tests/SampleTracer.cpp | $(TESTS_BLD)
	$(CXX) $(INCLUDES) $(CXXFLAGS) -c -o $@ $<

$(BUILD)/SampleTracer: $(TESTS_BLD)/SampleTracer.o $(CORE_OBJS) \
    $(API_LIB)/libXRayLibAPI.a $(API_LIB)/libOptimizerAPI.a \
    $(METAL_OBJ) $(METAL_LIB)
	$(CXX) $(CXXFLAGS) -o $@ \
	    $(TESTS_BLD)/SampleTracer.o $(CORE_OBJS) $(METAL_OBJ) \
	    $(LIBRARIES) $(LINK_LIBS) $(KOKKOS_LIBS) $(LDFLAGS)

# ── Test binary ───────────────────────────────────────────────────────────────
$(TESTS_BLD)/Test.o: $(SRC)/tests/Test.cpp | $(TESTS_BLD)
	$(HOST_COMPILER) $(INCLUDES) --std=c++20 -DVOXTRACE_HOST_ONLY -c $< -o $@

$(BUILD)/Test: $(TESTS_BLD)/Test.o \
    $(API_LIB)/libXRayLibAPI.a \
    $(API_LIB)/libOptimizerAPI.a
	$(HOST_COMPILER) --std=c++20 -o $@ \
	    $(TESTS_BLD)/Test.o \
	    $(API_LIB)/libOptimizerAPI.a \
	    $(API_LIB)/libXRayLibAPI.a \
	    -L$(ARMA_LIB) -L$(GSL_LIB) \
	    -larmadillo -lgsl -lgslcblas \
	    $(XRAY_LF)

# ── PolyCap batch validation binary ──────────────────────────────────────────
$(TESTS_BLD)/PolyCapBatchTest.o: $(SRC)/tests/PolyCapBatchTest.cpp | $(TESTS_BLD)
	$(CXX) $(INCLUDES) $(CXXFLAGS) -c -o $@ $<

$(BUILD)/PolyCapBatchTest: $(TESTS_BLD)/PolyCapBatchTest.o \
    $(API_LIB)/libXRayLibAPI.a
	$(CXX) $(CXXFLAGS) -o $@ \
	    $(TESTS_BLD)/PolyCapBatchTest.o \
	    $(API_LIB)/libXRayLibAPI.a \
<<<<<<< HEAD
=======
	    -L$(ARMA_LIB) -larmadillo \
	    $(XRAY_LF)

# ── Test2: polycap benchmark (voxTrace Kokkos vs polycap library) ─────────────
POLYCAP_CF   := $(shell pkg-config --cflags polycap 2>/dev/null)
POLYCAP_LF   := $(shell pkg-config --libs   polycap 2>/dev/null || echo -lpolycap)
POLYCAP_RPATH := -Wl,-rpath,$(shell pkg-config --variable=libdir polycap 2>/dev/null || echo /opt/homebrew/lib)

$(TESTS_BLD)/Test2.o: $(SRC)/tests/Test-2.cpp | $(TESTS_BLD)
	$(CXX) $(INCLUDES) $(POLYCAP_CF) $(CXXFLAGS) -c -o $@ $<

$(BUILD)/Test2: $(TESTS_BLD)/Test2.o \
    $(API_LIB)/libXRayLibAPI.a
	$(CXX) $(CXXFLAGS) -o $@ $< \
        $(API_LIB)/libXRayLibAPI.a \
	    $(POLYCAP_LF) $(POLYCAP_RPATH) \
>>>>>>> cdcd280 (123)
	    $(XRAY_LF) \
	    $(KOKKOS_LIBS) $(LDFLAGS)

$(TESTS_BLD)/Test2.o: $(SRC)/tests/Test-2.cpp | $(TESTS_BLD)
	$(HOST_COMPILER) $(INCLUDES) --std=c++20 -DVOXTRACE_HOST_ONLY -c $< -o $@

$(BUILD)/Test2: $(TESTS_BLD)/Test2.o \
    $(API_LIB)/libXRayLibAPI.a
	$(HOST_COMPILER) --std=c++20 -o $@ \
	    $(TESTS_BLD)/Test2.o \
	    $(API_LIB)/libXRayLibAPI.a \
	    $(XRAY_LF)

# ── TestMuXRF: full µXRF depth-scan simulation ───────────────────────────────
$(TESTS_BLD)/TestMuXRF.o: $(SRC)/tests/Test-muXRF.cpp | $(TESTS_BLD)
	$(CXX) $(INCLUDES) $(CXXFLAGS) -c -o $@ $<

$(BUILD)/TestMuXRF: $(TESTS_BLD)/TestMuXRF.o
	$(CXX) $(CXXFLAGS) -o $@ $< \
	    $(XRAY_LF) \
	    $(KOKKOS_LIBS) $(LDFLAGS)

.PHONY: testmuxrf
testmuxrf: $(BUILD)/TestMuXRF

# ── Build directory creation ──────────────────────────────────────────────────
$(CORE_BLD) $(API_OBJ) $(API_LIB) $(IO_BLD) $(BUILD) $(TESTS_BLD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)
