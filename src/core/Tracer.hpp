#pragma once
/**
 * @file Tracer.hpp
 * @brief Dispatch layer that drives the parallel Monte-Carlo trace.
 */

#include "Platform.hpp"
#include "Ray.hpp"
#include "Voxel.hpp"
#include "Material.hpp"
#include "ChemElement.hpp"
#include "RNG.hpp"
#include "SimulationParameter.hpp"

/**
 * @brief Dispatch layer — drives the Monte-Carlo ray-tracing loop over all rays.
 *
 * Owns the host entry points that allocate the `Kokkos::View` buffers (rays,
 * voxels, materials, elements) and launch the trace with `Kokkos::parallel_for`
 * (replacing the original CUDA `<<<N,1>>>` kernel launches). The per-ray physics
 * step is @ref traceForward, a `KOKKOS_INLINE_FUNCTION` inlined into every
 * dispatch kernel so the @e same code also runs from a Metal shader.
 */
class Tracer {
public:

    /// Host entry point: trace the pre-sample beam (Kokkos parallel_for dispatch).
    static void callTracePreBeam();
    /// Host entry point: generate and trace a fresh beam through the sample.
    static void callTraceNewBeam(SimulationParameter& simp);

    /// Single-ray forward Monte-Carlo step, inlined into every dispatch kernel:
    /// draw attenuation over the voxel path, and on interaction select the
    /// element/type (photoelectric+fluorescence, Rayleigh, Compton) and rewrite
    /// the ray's energy/direction. Mutates @p ray (and advances `nextVoxel`).
    KOKKOS_INLINE_FUNCTION
    static void traceForward(Ray& ray, const Voxel& voxel,
                             const Material* materials,
                             const ChemElement* elements,
                             RNG& rng);
};
