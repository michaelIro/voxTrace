#pragma once

#include "Platform.hpp"
#include "Ray.hpp"
#include "Voxel.hpp"
#include "Material.hpp"
#include "ChemElement.hpp"
#include "RNG.hpp"
#include "SimulationParameter.hpp"

// ── Tracer ────────────────────────────────────────────────────────────────────
// Orchestrates the Monte-Carlo ray tracing loop.
//   • Kokkos parallel_for replaces CUDA <<<N,1>>> kernel launches.
//   • traceForward is KOKKOS_INLINE_FUNCTION → also usable from Metal.

class Tracer {
public:

    // Host entry points (Kokkos parallel_for dispatch)
    static void callTracePreBeam();
    static void callTraceNewBeam(SimulationParameter& simp);

    // Single-ray forward Monte-Carlo step — inlined into every dispatch kernel
    KOKKOS_INLINE_FUNCTION
    static void traceForward(Ray& ray, const Voxel& voxel,
                             const Material* materials,
                             const ChemElement* elements,
                             RNG& rng);
};
