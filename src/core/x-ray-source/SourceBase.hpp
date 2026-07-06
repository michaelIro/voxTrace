#pragma once
/**
 * @file SourceBase.hpp
 * @brief CRTP base for every X-ray source: shared emission geometry + Ray assembly.
 */
#include "Platform.hpp"
#include "RNG.hpp"
#include "Ray.hpp"

/**
 * @brief The emission geometry common to all source types — the "mother" class.
 *
 * A source has two jobs: *place* a photon (where it starts and which way it
 * points) and *colour* it (its energy). Placement — a 2-D Gaussian spot of
 * radius `sourceRadius_` aimed at a focal plane at distance `focalDist_` with a
 * Gaussian angular spread `divergence_` — is identical for every kind of source
 * and lives here, in `generate()`. The energy is the one thing that
 * distinguishes a monochromatic @ref Source from an @ref XRayTube, a
 * @ref Synchrotron or a @ref LiquidMetalJet, so it is deferred to the concrete
 * source through the static (CRTP) hook `Derived::sampleEnergy(rng)`.
 *
 * Using the Curiously-Recurring-Template-Pattern instead of `virtual` keeps the
 * dispatch a compile-time call with zero overhead and no vtable — so the whole
 * hierarchy stays trivially copyable into device memory and valid for Kokkos
 * (CUDA/HIP/OpenMP) and Metal MSL, exactly like the rest of voxTrace.
 *
 * A new source type is just `class Foo : public SourceBase<Foo>` with one method
 * `float sampleEnergy(RNG&)`.
 */
template <class Derived>
class SourceBase {
protected:
    float sourceRadius_ = 0.f;   // spatial Gaussian sigma * 3 (r_out)
    float divergence_   = 0.f;   // angular  Gaussian sigma * 3 (r_f)
    float focalDist_    = 0.f;   // focal-plane distance along Y (f)

    // Down-cast to the concrete source, preserving the Metal address space.
    KOKKOS_INLINE_FUNCTION const VT_CONST_METH Derived& self() const VT_CONST_METH {
        return static_cast<const VT_CONST_METH Derived&>(*this);
    }

public:
    KOKKOS_INLINE_FUNCTION SourceBase() {}

    /// Set the shared emission geometry (from the prim_cap_geom array).
    KOKKOS_INLINE_FUNCTION void setGeometry(float r_out, float f, float r_f) {
        sourceRadius_ = r_out;
        focalDist_    = f;
        divergence_   = r_f;
    }

    /// Draw one primary @ref Ray — Gaussian spot + divergence (here), and the
    /// energy delegated to the concrete source's `sampleEnergy` — identical
    /// physics to the original `generateRayGPU`.
    KOKKOS_INLINE_FUNCTION Ray generate(int index, VT_THREAD RNG& rng) const VT_CONST_METH {
        (void)index;
        float x0 = rng.normal() / 3.0f * sourceRadius_;
        float z0 = rng.normal() / 3.0f * sourceRadius_;
        float xD = rng.normal() / 3.0f * divergence_;
        float yD = focalDist_;
        float zD = rng.normal() / 3.0f * divergence_;

        xD -= x0;
        zD -= z0;

        float n = sqrtf(xD*xD + yD*yD + zD*zD);

        Ray ray;
        ray.setStartCoordinates(x0, 0.f, z0);
        ray.setEndCoordinates(xD / n, yD / n, zD / n);
        ray.setEnergyKeV(self().sampleEnergy(rng));
        ray.setOOBFlag(false);
        ray.setIAFlag(false);
        ray.setIANum(0);
        ray.setTIn(0.f);
        return ray;
    }
};
