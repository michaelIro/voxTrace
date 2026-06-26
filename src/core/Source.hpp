#pragma once
/**
 * @file Source.hpp
 * @brief Monochromatic geometric source — the default concrete @ref SourceBase.
 */
#include "SourceBase.hpp"

/**
 * @brief Ideal monochromatic point/spot source (the original geometric model).
 *
 * The simplest concrete @ref SourceBase: every photon leaves with the same
 * energy. This is the primary-optic exit-beam model validated in the reference
 * paper and the baseline that the polychromatic sources (@ref XRayTube,
 * @ref Synchrotron, @ref LiquidMetalJet) extend by overriding `sampleEnergy`.
 */
class Source : public SourceBase<Source> {
    float energyKeV_ = 0.f;

public:
    KOKKOS_INLINE_FUNCTION Source() {}

    /// Factory from the prim_cap_geom array [r_out, f, r_f, energy_keV].
    KOKKOS_INLINE_FUNCTION static Source fromCapGeom(float r_out, float f,
                                                     float r_f, float energy_keV) {
        Source s;
        s.setGeometry(r_out, f, r_f);
        s.energyKeV_ = energy_keV;
        return s;
    }

    /// Energy hook: a delta at the single line energy.
    KOKKOS_INLINE_FUNCTION float sampleEnergy(VT_THREAD RNG& rng) const VT_CONST_METH {
        (void)rng;
        return energyKeV_;
    }

#ifndef __METAL_VERSION__
    inline void print() const {
        printf("Source(mono): E=%.2fkeV r=%.4fcm div=%.4f f=%.2fcm\n",
               energyKeV_, sourceRadius_, divergence_, focalDist_);
    }
#endif
};
