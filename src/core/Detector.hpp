#pragma once
/**
 * @file Detector.hpp
 * @brief Si(Li) energy-dispersive detector response (photopeak, escape, Compton, resolution).
 */
#include "Platform.hpp"
#include "RNG.hpp"
#include "ChemElement.hpp"

/**
 * @brief Si(Li) / Si-PIN energy-dispersive detector response.
 *
 * Turns the true energy of a photon entering the detector into a *measured*
 * channel energy, reproducing the features that make a real EDXRF spectrum look
 * natural:
 *   - **detection efficiency** — Beer–Lambert transmission through the Be window
 *     and Si dead layer (low-E roll-off) and the absorption probability in the
 *     finite-thickness active crystal (high-E roll-off);
 *   - **Si escape peaks** — a K-photoabsorption can release a Si Kα photon that
 *     escapes the crystal, so E − 1.74 keV is recorded instead of E;
 *   - **Compton continuum** — incoherent scatter inside the crystal whose
 *     scattered photon escapes deposits only the recoil-electron energy, filling
 *     the shelf below each photopeak;
 *   - **finite resolution** — Gaussian broadening with the Fano + electronic
 *     noise width FWHM(E) = 2.355·√(σ_noise² + ε·F·E).
 *
 * The detector reuses the same @ref ChemElement interaction physics as the
 * sample — the crystal is just a Si medium, so its K-fluorescence, transition
 * lines and Compton kinematics all come from a shared `ChemElement` array. The
 * detector stores only the indices of its Si crystal and Be window in that
 * array, keeping it trivially copyable to the device like every other operator.
 */
class Detector {
    float thickness_ = 0.30f;    // active Si thickness [cm]
    float beWin_     = 0.0025f;  // Be window thickness [cm] (25 µm)
    float deadLayer_ = 1e-4f;    // Si dead layer [cm] (1 µm)
    float fano_      = 0.114f;   // Si Fano factor
    float noiseFWHM_ = 0.080f;   // electronic-noise FWHM [keV]
    int   siIdx_ = 0;            // Si crystal index in the shared ChemElement array
    int   beIdx_ = 1;            // Be window index in the shared ChemElement array

    VT_SCONSTEXPR float EPS         = 0.00381f;  // Si e–h pair creation energy [keV]
    VT_SCONSTEXPR int   MAX_SHELLS  = 25;        // ChemElement shell table size (guard)

    // Linear attenuation [1/cm] of element @p idx at energy @p e.
    KOKKOS_INLINE_FUNCTION float mu(int idx, float e, const VT_DEVICE ChemElement* elems) const VT_DEVICE_METH {
        return elems[idx].CS_Tot(e) * elems[idx].Rho();
    }

    // Does a secondary photon of energy @p eph born at depth @p z escape the slab?
    KOKKOS_INLINE_FUNCTION bool escapes(float z, float eph, VT_THREAD RNG& rng,
                                        const VT_DEVICE ChemElement* elems) const VT_DEVICE_METH {
        float cosA = 2.0f * rng.frand() - 1.0f;                 // isotropic emission
        if (cosA == 0.f) return false;
        float dist = (cosA > 0.f) ? (thickness_ - z) / cosA : z / (-cosA);
        return rng.frand() < expf(-mu(siIdx_, eph, elems) * dist);
    }

public:
    KOKKOS_INLINE_FUNCTION Detector() {}

    static Detector make(int siIdx, int beIdx, float thickness, float beWin,
                         float deadLayer, float fano, float noiseFWHM) {
        Detector d;
        d.siIdx_ = siIdx; d.beIdx_ = beIdx;
        d.thickness_ = thickness; d.beWin_ = beWin; d.deadLayer_ = deadLayer;
        d.fano_ = fano; d.noiseFWHM_ = noiseFWHM;
        return d;
    }

    /// Gaussian resolution sigma at energy @p e [keV].
    KOKKOS_INLINE_FUNCTION float resolutionSigma(float e) const VT_DEVICE_METH {
        float sn = noiseFWHM_ / 2.3548f;
        return sqrtf(sn*sn + EPS*fano_*fmaxf(e, 0.f));
    }

    /// Detection efficiency at energy @p e: window/dead-layer transmission times
    /// the active-crystal absorption probability.
    KOKKOS_INLINE_FUNCTION float efficiency(float e, const VT_DEVICE ChemElement* elems) const VT_DEVICE_METH {
        float tWin = expf(-(mu(beIdx_, e, elems)*beWin_ + mu(siIdx_, e, elems)*deadLayer_));
        float pInt = 1.0f - expf(-mu(siIdx_, e, elems)*thickness_);
        return tWin * pInt;
    }

    /// Process one photon of true energy @p e entering the crystal. Returns the
    /// measured (resolution-broadened) channel energy and sets @p wOut to the
    /// detection-efficiency weight; returns a negative energy if not counted.
    KOKKOS_INLINE_FUNCTION float detect(float e, VT_THREAD RNG& rng,
                                        const VT_DEVICE ChemElement* elems,
                                        VT_THREAD float& wOut) const VT_DEVICE_METH {
        wOut = efficiency(e, elems);
        if (wOut <= 0.f) return -1.f;

        const VT_DEVICE ChemElement& si = elems[siIdx_];
        float muE = mu(siIdx_, e, elems);

        // interaction depth in the active crystal (truncated exponential)
        float pInt = 1.0f - expf(-muE * thickness_);
        float z    = -logf(1.0f - rng.frand()*pInt) / muE;

        float dep;
        int type = si.getInteractionType(e, rng.frand());
        if (type == 0) {                                   // photoelectric
            dep = e;
            int shell = si.getExcitedShell(e, rng.frand());
            if (shell < MAX_SHELLS && rng.frand() < si.Fluor_Y(shell)) {
                float eline = si.Line_Energy(si.getTransition(shell, rng.frand()));
                if (eline > 0.f && escapes(z, eline, rng, elems))
                    dep = e - eline;                       // → Si escape peak
            }
        } else {                                           // scatter (Rayleigh / Compton)
            float es = (type == 1) ? e                     // coherent: same energy
                                   : si.getComptEnergy(e, si.getThetaCompt(e, rng.frand()));
            dep = e - es;                                  // recoil electron deposited
            if (!escapes(z, es, rng, elems))
                dep = e;                                   // scattered photon reabsorbed → full E
        }

        return dep + rng.normal() * resolutionSigma(dep); // apply finite resolution
    }
};
