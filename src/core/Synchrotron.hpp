#pragma once
/**
 * @file Synchrotron.hpp
 * @brief Monochromatized synchrotron source: narrow energy band, low divergence.
 */
#include "SourceBase.hpp"

#ifndef __METAL_VERSION__
    #include <cstdio>
#endif

/**
 * @brief Monochromatized synchrotron-beamline source.
 *
 * After a crystal monochromator the beam is an almost-parallel, almost-mono
 * line at `E0_` with a small relative bandwidth `relBW_` (ΔE/E set by the
 * monochromator). `sampleEnergy` draws a Gaussian about E0 with sigma =
 * relBW·E0; the low divergence is supplied through the shared geometry (pass a
 * small `r_f` to @ref SourceBase::setGeometry).
 */
class Synchrotron : public SourceBase<Synchrotron> {
    float E0_    = 10.f;       // monochromator energy [keV]
    float relBW_ = 1e-4f;      // ΔE/E (e.g. ~1e-4 for Si(111))

public:
    KOKKOS_INLINE_FUNCTION Synchrotron() {}

    static Synchrotron make(float r_out, float f, float r_f, float E0, float relBW) {
        Synchrotron s;
        s.setGeometry(r_out, f, r_f);
        s.E0_    = E0;
        s.relBW_ = relBW;
        return s;
    }

    /// Energy hook: a narrow Gaussian about the monochromator energy.
    KOKKOS_INLINE_FUNCTION float sampleEnergy(VT_THREAD RNG& rng) const VT_CONST_METH {
        return E0_ + rng.normal() * relBW_ * E0_;
    }

#ifndef __METAL_VERSION__
    inline void print() const {
        printf("Synchrotron: E0=%.3fkeV dE/E=%.1e\n", E0_, relBW_);
    }
#endif
};
