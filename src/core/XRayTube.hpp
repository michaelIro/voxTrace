#pragma once
/**
 * @file XRayTube.hpp
 * @brief X-ray-tube source: Kramers bremsstrahlung continuum + anode K-lines.
 */
#include "SourceBase.hpp"

#ifndef __METAL_VERSION__
    #include <cstdio>
#endif

/**
 * @brief Sealed/transmission X-ray-tube source.
 *
 * A concrete @ref SourceBase whose `sampleEnergy` draws, per photon, either a
 * characteristic anode line (with probability `lineFrac_`, chosen among up to
 * `MAX_LINES` weighted lines) or a bremsstrahlung photon rejection-sampled from
 * Kramers' law  I(E) ∝ (Emax − E)/E  on the band [`Emin_`, `kVp_`]. The
 * low-energy cut `Emin_` stands in for window/inherent filtration; the tube
 * voltage `kVp_` is the bremsstrahlung end point (Duane–Hunt limit).
 */
class XRayTube : public SourceBase<XRayTube> {
public:
    VT_SCONSTEXPR int MAX_LINES = 4;

private:
    float kVp_      = 50.f;             // tube voltage → bremsstrahlung end point [keV]
    float Emin_     = 1.f;              // low-energy cut (filtration) [keV]
    float lineFrac_ = 0.f;             // fraction of photons emitted as characteristic lines
    int   nLines_   = 0;
    float lineE_[MAX_LINES]   = {};
    float lineCdf_[MAX_LINES] = {};    // normalised cumulative line weights

public:
    KOKKOS_INLINE_FUNCTION XRayTube() {}

    /// Build a tube: geometry + voltage + low-energy cut + characteristic lines.
    /// @p energies / @p weights list up to MAX_LINES anode lines; @p lineFrac is
    /// the probability that a generated photon is a characteristic line.
    static XRayTube make(float r_out, float f, float r_f,
                         float kVp, float Emin, float lineFrac,
                         int nLines, const float* energies, const float* weights) {
        XRayTube t;
        t.setGeometry(r_out, f, r_f);
        t.kVp_  = kVp;
        t.Emin_ = Emin;
        t.nLines_   = (nLines < MAX_LINES) ? nLines : MAX_LINES;
        t.lineFrac_ = (t.nLines_ > 0) ? lineFrac : 0.f;

        float sum = 0.f;
        for (int i = 0; i < t.nLines_; ++i) sum += weights[i];
        float acc = 0.f;
        for (int i = 0; i < t.nLines_; ++i) {
            t.lineE_[i]   = energies[i];
            acc          += weights[i] / sum;
            t.lineCdf_[i] = acc;
        }
        return t;
    }

    /// Energy hook: a characteristic line (prob. `lineFrac_`) or a Kramers
    /// bremsstrahlung photon by rejection sampling.
    KOKKOS_INLINE_FUNCTION float sampleEnergy(VT_THREAD RNG& rng) const VT_CONST_METH {
        if (nLines_ > 0 && rng.frand() < lineFrac_) {
            float u = rng.frand();
            for (int i = 0; i < nLines_; ++i)
                if (u <= lineCdf_[i]) return lineE_[i];
            return lineE_[nLines_ - 1];
        }
        float fmax = (kVp_ - Emin_) / Emin_;            // peak of (kVp−E)/E at E=Emin
        for (int it = 0; it < 64; ++it) {
            float E = Emin_ + rng.frand() * (kVp_ - Emin_);
            float f = (kVp_ - E) / E;
            if (rng.frand() * fmax < f) return E;
        }
        return Emin_;
    }

#ifndef __METAL_VERSION__
    inline void print() const {
        printf("XRayTube: kVp=%.1f Emin=%.1f lineFrac=%.2f nLines=%d\n",
               kVp_, Emin_, lineFrac_, nLines_);
    }
#endif
};
