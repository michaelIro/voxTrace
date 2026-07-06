#pragma once
/**
 * @file LiquidMetalJet.hpp
 * @brief Liquid-metal-jet source: anode-metal Kα/Kβ lines on a bremsstrahlung floor.
 */
#include "SourceBase.hpp"

#include <cstdio>

/**
 * @brief Liquid-metal-jet source (e.g. a Ga or In jet anode).
 *
 * A regenerating liquid-metal anode tolerates very high electron-beam power, so
 * the spectrum is dominated by the jet metal's characteristic Kα/Kβ lines
 * (`kAlpha_`/`kBeta_`, branching `kBetaFrac_`) on a weak Kramers bremsstrahlung
 * continuum up to the beam voltage `kVp_`. With probability `lineFrac_` a photon
 * is a characteristic line, otherwise a bremsstrahlung photon (as in
 * @ref XRayTube). Defaults model a gallium jet (Kα 9.25 keV, Kβ 10.26 keV).
 */
class LiquidMetalJet : public SourceBase<LiquidMetalJet> {
    float kAlpha_    = 9.25f;     // jet-metal Kα [keV]  (Ga)
    float kBeta_     = 10.26f;    // jet-metal Kβ [keV]  (Ga)
    float kBetaFrac_ = 0.15f;     // P(Kβ | characteristic line)
    float lineFrac_  = 0.8f;      // fraction of photons in the characteristic lines
    float kVp_       = 70.f;      // electron-beam voltage → brem end point [keV]
    float Emin_      = 1.f;       // low-energy cut [keV]

public:
    KOKKOS_INLINE_FUNCTION LiquidMetalJet() {}

    static LiquidMetalJet make(float r_out, float f, float r_f,
                               float kAlpha, float kBeta, float kBetaFrac,
                               float lineFrac, float kVp, float Emin) {
        LiquidMetalJet j;
        j.setGeometry(r_out, f, r_f);
        j.kAlpha_    = kAlpha;
        j.kBeta_     = kBeta;
        j.kBetaFrac_ = kBetaFrac;
        j.lineFrac_  = lineFrac;
        j.kVp_       = kVp;
        j.Emin_      = Emin;
        return j;
    }

    /// Energy hook: a Kα/Kβ line (prob. `lineFrac_`) or a Kramers bremsstrahlung
    /// photon by rejection sampling.
    KOKKOS_INLINE_FUNCTION float sampleEnergy(VT_THREAD RNG& rng) const VT_CONST_METH {
        if (rng.frand() < lineFrac_)
            return (rng.frand() < kBetaFrac_) ? kBeta_ : kAlpha_;
        float fmax = (kVp_ - Emin_) / Emin_;
        for (int it = 0; it < 64; ++it) {
            float E = Emin_ + rng.frand() * (kVp_ - Emin_);
            float f = (kVp_ - E) / E;
            if (rng.frand() * fmax < f) return E;
        }
        return Emin_;
    }

    inline void print() const {
        printf("LiquidMetalJet: Ka=%.2f Kb=%.2f lineFrac=%.2f kVp=%.1f\n",
               kAlpha_, kBeta_, lineFrac_, kVp_);
    }
};
