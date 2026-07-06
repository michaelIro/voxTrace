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

# ── External dependencies ──────────────────────────────────────────────────────
HDF5_INC := $(HDF5_INC_)
HDF5_LIB := $(HDF5_LIB_)
# Use pkg-config when available; otherwise fall back to bare -lxrl
PKGCFG   := $(shell command -v pkg-config 2>/dev/null)
ifneq ($(PKGCFG),)
    XRAY_CF := $(shell pkg-config --cflags libxrl 2>/dev/null)
    XRAY_LF := $(shell pkg-config --libs   libxrl 2>/dev/null)
    POLYCAP_CF := $(shell pkg-config --cflags polycap 2>/dev/null)
    POLYCAP_LF := $(shell pkg-config --libs   polycap 2>/dev/null)
else
    XRAY_CF :=
    XRAY_LF := -lxrl
    POLYCAP_CF :=
    POLYCAP_LF := -lpolycap
endif

# Core is organised by instrument stage (x-ray-source → optical-elements →
# sample → detector); the stage dirs are all on the include path so headers
# keep flat #include "..." names.
CORE_DIRS := $(SRC)/core $(SRC)/core/x-ray-source $(SRC)/core/optical-elements \
             $(SRC)/core/sample $(SRC)/core/detector

INCLUDES  := -I$(KOKKOS_INC) \
             -I$(ARMA_INC) -I$(HDF5_INC) \
             -I$(SRC) $(addprefix -I,$(CORE_DIRS)) \
             $(XRAY_CF)

LIBRARIES := -L$(ARMA_LIB) -L$(HDF5_LIB) -L$(API_LIB)

LINK_LIBS := -larmadillo -lhdf5 -DARMA_USE_HDF5 -lxrl \
             $(API_LIB)/libXRayLibAPI.a $(API_LIB)/libOptimizerAPI.a

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
CXXFLAGS := --std=c++20 $(OMP_CFLAGS)
LDFLAGS  := -L$(KOKKOS_LIB) $(OMP_LFLAGS)

# ── Core objects (only Tracer.cpp needs compilation; physics is header-only) ───
CORE_OBJS := $(CORE_BLD)/Tracer.o

.PHONY: all clean test test2

all: $(BUILD)/Test

test: $(BUILD)/Test

test2: $(BUILD)/Test2

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

# ── Test2: PolyCap validation (original polycap C library vs PolyCap.hpp) ─────
$(TESTS_BLD)/Test2.o: $(SRC)/tests/Test-2.cpp | $(TESTS_BLD)
	$(HOST_COMPILER) $(INCLUDES) $(POLYCAP_CF) --std=c++20 -DVOXTRACE_HOST_ONLY -c $< -o $@

$(BUILD)/Test2: $(TESTS_BLD)/Test2.o \
    $(API_LIB)/libXRayLibAPI.a
	$(HOST_COMPILER) --std=c++20 -o $@ \
	    $(TESTS_BLD)/Test2.o \
	    $(API_LIB)/libXRayLibAPI.a \
	    $(XRAY_LF) $(POLYCAP_LF)

# ── Test3: confocal µXRF (source → primary → brass → secondary → spectrum) ────
$(TESTS_BLD)/Test3.o: $(SRC)/tests/Test-3.cpp | $(TESTS_BLD)
	$(HOST_COMPILER) $(INCLUDES) --std=c++20 -DVOXTRACE_HOST_ONLY -c $< -o $@

$(BUILD)/Test3: $(TESTS_BLD)/Test3.o \
    $(API_LIB)/libXRayLibAPI.a
	$(HOST_COMPILER) --std=c++20 -o $@ \
	    $(TESTS_BLD)/Test3.o \
	    $(API_LIB)/libXRayLibAPI.a \
	    $(XRAY_LF)

.PHONY: test3
test3: $(BUILD)/Test3

# ── Test4: X-ray reflectivity (coherent layered-stack transfer matrix) ────────
$(TESTS_BLD)/Test4.o: $(SRC)/tests/Test-4.cpp | $(TESTS_BLD)
	$(HOST_COMPILER) $(INCLUDES) --std=c++20 -DVOXTRACE_HOST_ONLY -c $< -o $@

$(BUILD)/Test4: $(TESTS_BLD)/Test4.o \
    $(API_LIB)/libXRayLibAPI.a
	$(HOST_COMPILER) --std=c++20 -o $@ \
	    $(TESTS_BLD)/Test4.o \
	    $(API_LIB)/libXRayLibAPI.a \
	    $(XRAY_LF)

.PHONY: test4
test4: $(BUILD)/Test4

# ── Test5: confocal µXRF voxel-weight reconstruction (trace → spectrum →
# χ²/weighted-χ² loss vs measured spectrum → ensmallen L-BFGS). -O2 because a
# Monte-Carlo trace plus an optimisation loop runs on top of it. ──────────────
$(TESTS_BLD)/Test5.o: $(SRC)/tests/Test-5.cpp | $(TESTS_BLD)
	$(HOST_COMPILER) $(INCLUDES) --std=c++20 -O2 -DVOXTRACE_HOST_ONLY -c $< -o $@

$(BUILD)/Test5: $(TESTS_BLD)/Test5.o \
    $(API_LIB)/libXRayLibAPI.a $(API_LIB)/libOptimizerAPI.a
	$(HOST_COMPILER) --std=c++20 -o $@ \
	    $(TESTS_BLD)/Test5.o \
	    $(API_LIB)/libOptimizerAPI.a \
	    $(API_LIB)/libXRayLibAPI.a \
	    -L$(ARMA_LIB) -L$(GSL_LIB) \
	    -larmadillo -lgsl -lgslcblas \
	    $(XRAY_LF)

.PHONY: test5
test5: $(BUILD)/Test5

# ── Test5k: the same source built WITH Kokkos (OpenMP backend). The beam trace
# and the confocal scan then run as Kokkos::parallel_for; per-ray RNG streams
# keep the results identical to the serial build at any thread count. ─────────
$(TESTS_BLD)/Test5k.o: $(SRC)/tests/Test-5.cpp | $(TESTS_BLD)
	$(CXX) $(INCLUDES) $(CXXFLAGS) -O2 -c $< -o $@

$(BUILD)/Test5k: $(TESTS_BLD)/Test5k.o \
    $(API_LIB)/libXRayLibAPI.a $(API_LIB)/libOptimizerAPI.a
	$(CXX) $(CXXFLAGS) -O2 -o $@ \
	    $(TESTS_BLD)/Test5k.o \
	    $(API_LIB)/libOptimizerAPI.a \
	    $(API_LIB)/libXRayLibAPI.a \
	    -L$(ARMA_LIB) -L$(GSL_LIB) \
	    -larmadillo -lgsl -lgslcblas \
	    $(XRAY_LF) $(KOKKOS_LIBS) $(LDFLAGS)

.PHONY: test5-kokkos
test5-kokkos: $(BUILD)/Test5k

# ── Documentation ─────────────────────────────────────────────────────────────
# Doxygen extracts the in-source API docs to XML; Sphinx + Breathe render the
# RST guide in docs/ into an HTML site that embeds them.
# Prereqs: doxygen, plus the Python deps in docs/requirements.txt
#          (pip install -r docs/requirements.txt — e.g. inside .venv).
DOXYGEN     ?= doxygen
SPHINXBUILD ?= sphinx-build

.PHONY: docs docs-clean
docs:
	$(DOXYGEN) docs/Doxyfile
	$(SPHINXBUILD) -b html docs build/doc/html
	@echo "Docs built → build/doc/html/index.html"

docs-clean:
	rm -rf build/doc

# ── Build directory creation ──────────────────────────────────────────────────
$(CORE_BLD) $(API_OBJ) $(API_LIB) $(IO_BLD) $(BUILD) $(TESTS_BLD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)
