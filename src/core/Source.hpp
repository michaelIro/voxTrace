#pragma once
#include "Platform.hpp"
#include "RNG.hpp"
#include "Ray.hpp"

// ── Source ────────────────────────────────────────────────────────────────────
// Gaussian capillary X-ray source. generate() replicates the original
// generateRayGPU logic (spatial sigma = r_out/3, direction sigma = r_f/3)
// using a portable RNG instead of curand.

class Source {
    float energyKeV_;    // photon energy
    float sourceRadius_; // r_out: spatial Gaussian sigma * 3
    float divergence_;   // r_f:   direction spread Gaussian sigma * 3
    float focalDist_;    // f:     focal-plane distance along Y

public:

    KOKKOS_INLINE_FUNCTION Source() {}

    // Factory: build from the prim_cap_geom array [r_out, f, r_f, energy_keV]
    KOKKOS_INLINE_FUNCTION static Source fromCapGeom(float r_out, float f,
                                                      float r_f, float energy_keV) {
        Source s;
        s.sourceRadius_ = r_out;
        s.divergence_   = r_f;
        s.focalDist_    = f;
        s.energyKeV_    = energy_keV;
        return s;
    }

    // Generate a ray — identical physics to original generateRayGPU
    KOKKOS_INLINE_FUNCTION Ray generate(int index, VT_THREAD RNG& rng) const VT_CONST_METH {
        float x0 = rng.normal() / 3.0f * sourceRadius_;
        float z0 = rng.normal() / 3.0f * sourceRadius_;
        float xD = rng.normal() / 3.0f * divergence_;
        float yD = focalDist_;
        float zD = rng.normal() / 3.0f * divergence_;

        xD -= x0;
        zD -= z0;

        float n = sqrtf(xD*xD + yD*yD + zD*zD);
        xD /= n; yD /= n; zD /= n;

        Ray ray;
        ray.setStartCoordinates(x0, 0.f, z0);
        ray.setEndCoordinates(xD, yD, zD);
        ray.setEnergyKeV(energyKeV_);
        ray.setOOBFlag(false);
        ray.setIAFlag(false);
        ray.setIANum(0);
        ray.setTIn(0.f);
        return ray;
    }

#ifndef __METAL_VERSION__
    inline void print() const {
        printf("Source: E=%.2fkeV r=%.4fcm div=%.4f f=%.2fcm\n",
               energyKeV_, sourceRadius_, divergence_, focalDist_);
    }
#endif
};
