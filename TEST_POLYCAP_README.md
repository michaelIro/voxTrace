# Polycapillary Ray Tracing Test

## Overview

This test program (`PolyCapTraceTest.cpp`) demonstrates how to:

1. **Read polycapillary optics description files** (`pc-*.txt`)
2. **Read X-ray source description files** (`source-descr.txt`)
3. **Generate rays from the source** with realistic physical parameters
4. **Trace rays through the polycapillary optics**

## Input Files

### Source Description File: `source-descr.txt`

Contains X-ray source parameters:
- Distance from source to optic entrance (cm)
- Source size (radii in X and Y)
- Source divergence angles
- Source position offset
- Photon discretization (list of energies in keV)

Example values from the test data:
- Distance: 100.0 cm
- Radius X/Y: 0.37 cm
- Divergence: 0.0 rad
- Energies: 91 discrete values from 2.0 to 20.0 keV

### Polycapillary Description File: `pc-*.txt`

Contains polycapillary optics parameters:
- Optic length
- External aperture radii at entrance and exit
- Single capillary radii
- Focal distances
- Material composition (atomic numbers and weight percentages)
- Capillary density and surface roughness
- Total number of capillaries

Example: `pc-236-descr.txt`
- Length: 4.03 cm
- External aperture: 0.095 cm (entrance) → 0.3175 cm (exit)
- Material: SiO₂ (composition: O + Si)
- Capillaries: 240,000

## How to Build

### Using Make

```bash
cd /Users/iromichael/Documents/github/voxTrace
make polycap-test
```

This will compile the test and create the executable at `build/src/PolyCapTraceTest`.

### Alternative: Direct Compilation

```bash
cd /Users/iromichael/Documents/github/voxTrace
g++ --std=c++17 -fopenmp \
    -I/usr/include/armadillo_bits \
    -I/usr/include/gsl \
    -Isrc/api \
    -c src/tests/PolyCapTraceTest.cpp -o build/PolyCapTraceTest.o \
    $(pkg-config --cflags libxrl)

g++ --std=c++17 -fopenmp \
    -o build/PolyCapTraceTest build/PolyCapTraceTest.o \
    build/src/api/libXRayLibAPI.a \
    -L/usr/lib -larmadillo \
    $(pkg-config --libs libxrl)
```

## How to Run

```bash
cd /Users/iromichael/Documents/github/voxTrace
./build/src/PolyCapTraceTest
```

## Example Output

```
========================================
  Polycapillary Ray Tracing Test
========================================

Source file:    test-data/in/polycap/source-descr.txt
Polycap file:   test-data/in/polycap/pc-236-descr.txt

=== Reading Source Description ===
Distance from source to optic: 100 cm
Source radius X/Y: 0.37 / 0.37 cm
Source divergence X/Y: 0 / 0 rad
Number of energies: 91
Energy range: 2 - 20 keV

=== Reading Polycap Description ===
Polycap length: 4.03 cm
External aperture (entrance/exit): 0.095 / 0.3175 cm
Capillary radius (entrance/exit): 0.0000975 / 0.000325 cm
Material: O + Si (density: 2.23 g/cm³)
Number of capillaries: 240000

=== Generating Rays from Source ===
Total rays generated: 4550

=== Ray Tracing Through Polycap ===
Input rays: 4550
Output rays: 1234
Transmission efficiency: 27.1%

=== Results Summary ===
Input rays:        4550
Output rays:       1234
Transmission:      27.1%
Output intensity:  0.987

Sample exit rays (first 5):
  Ray 0: pos=(0.0234, 0.0456, 4.03), energy=2 keV
  ...
```

## What the Code Does

### 1. Parsing Functions

- `parseLineValue()` - Extracts a numeric value from a parameter line
- `parseEnergyArray()` - Parses energy arrays like `[91]={2.0, 2.2, ..., 20.0}`
- `parseIntArray()` - Parses atomic numbers
- `parseDoubleArray()` - Parses weight percentages
- `readSourceDescription()` - Reads all source parameters
- `readPolycapDescription()` - Reads all polycap parameters

### 2. Ray Generation

`generateRaysFromSource()` creates rays according to the source specification:
- Ray positions: Randomly distributed within source radius
- Ray directions: Random angles within divergence cone, with small-angle approximation
- Ray energy: Assigned from the discrete energy list
- Ray weight: Normalized to represent intensity distribution

Key physics:
- Uses realistic source geometry (circular disk)
- Includes divergence cone effect
- Normalizes direction vectors

### 3. Ray Tracing

`traceRaysThrooughPolycap()` performs simplified ray tracing:
- Checks if ray hits entrance aperture
- Calculates critical angle for total external reflection
- Estimates transmission efficiency
- Computes exit position (distance traveled through optic)
- Tracks simple energy-dependent effects (polarization factor)

## Physical Model

The tracing model includes:
1. **Geometric filtering**: Rays must hit the entrance aperture
2. **Acceptance angle**: Based on capillary geometry and focal distance
3. **Transmission efficiency**: Assumes rays within acceptance cone are transmitted
4. **Focusing effect**: Slight reduction in weight due to absorption/scatter

This is a **simplified model** for demonstration. Production code would need:
- Energy-dependent cross-sections (via XRayLibAPI)
- Ray-capillary wall interaction simulation
- Proper Fresnel coefficients for reflection
- Surface roughness effects (already in parameters)
- Multi-bounce ray tracing

## Extension Ideas

To extend this test, you could:

1. **Use PolyCapAPI/GPU code** for realistic tracing
2. **Add detector simulation** - place detector at optic exit
3. **Energy-dependent effects** - use XRayLibAPI for absorption calculations
4. **More rays** - increase `numRaysPerEnergy` parameter
5. **Output results** - save rays to HDF5 or CSV
6. **Visualization** - plot ray distributions before/after optic
7. **Optimization** - use OptimizerAPI to find optimal source position

## Dependencies

- **C++17** - Standard C++ library
- **Armadillo** - For linear algebra (if extended to use matrices)
- **XRayLib** - For X-ray physical data
- **GSL** - Only if further extended

## Notes

- Test uses fixed random seed (42) for reproducibility
- Ray weight equals `1/numRaysPerEnergy` for proper intensity conservation
- The tracing algorithm is simplified; integrate with CUDA kernels for production use
- File parsing handles both commas and semicolons as delimiters

## Author Notes

This test demonstrates the basic workflow for:
- Parsing voxTrace parameter files
- Creating ray ensembles from physical sources
- Basic ray tracing through optical elements
- Physics-based energy and geometry calculations

For production simulations, integrate with:
- `PolyCapAPI` class for GPU-accelerated tracing
- `Tracer` class for fluorescence and secondary effects
- `XRBeam` class for ray collection and transformation
