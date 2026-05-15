#pragma once
#include "Platform.hpp"
#include "Ray.hpp"

#ifndef __METAL_VERSION__
    #include <cmath>
    #include <cstdio>
#endif

// ── PolyCap ───────────────────────────────────────────────────────────────────
// Polycapillary X-ray optic. Debye-Waller modified Fresnel transmission model.
// All methods compile for Kokkos and Metal MSL.

#define POLYCAP_MAX_ELEMENTS 8

class PolyCap {
    float posY_, length_;
    float rExtIn_, rExtOut_;
    float rCapIn_, rCapOut_;
    float focalIn_, focalOut_;

    int   nElements_;
    int   atomicNumbers_[POLYCAP_MAX_ELEMENTS];
    float weightFracs_  [POLYCAP_MAX_ELEMENTS];
    float density_;
    float roughness_;
    int   nCapillaries_;
    float critCoeff_;
    float mu_rho_;          // linear attenuation [cm⁻¹] = CS_Total * density

    KOKKOS_INLINE_FUNCTION float atomicMass(int z) const {
        const float A[] = {0,1,4,7,9,11,12,14,16,19,20,
                           23,24,27,28,31,32,35,40,39,40};
        if (z > 0 && z <= 20) return A[z];
        return (float)(2 * z);
    }

    KOKKOS_INLINE_FUNCTION void computeCritCoeff() {
        float ZoverA = 0.f;
        for (int i = 0; i < nElements_; ++i)
            ZoverA += (weightFracs_[i] / 100.f)
                    * ((float)atomicNumbers_[i] / atomicMass(atomicNumbers_[i]));
        critCoeff_ = 28.8e-3f * sqrtf(density_ * ZoverA);
    }

public:

    KOKKOS_INLINE_FUNCTION PolyCap() {}

    KOKKOS_INLINE_FUNCTION PolyCap(
        float posY, float length,
        float rExtIn, float rExtOut,
        float rCapIn, float rCapOut,
        float focalIn, float focalOut,
        int nElements, const int* atomicNumbers, const float* weightFracs,
        float density, float roughness, int nCapillaries)
        : posY_(posY), length_(length),
          rExtIn_(rExtIn), rExtOut_(rExtOut),
          rCapIn_(rCapIn), rCapOut_(rCapOut),
          focalIn_(focalIn), focalOut_(focalOut),
          nElements_(nElements), density_(density),
          roughness_(roughness), nCapillaries_(nCapillaries),
          mu_rho_(0.f)
    {
        for (int i = 0; i < nElements_; ++i) {
            atomicNumbers_[i] = atomicNumbers[i];
            weightFracs_[i]   = weightFracs[i];
        }
        computeCritCoeff();
    }

    // Set mass-attenuation × density [cm⁻¹] for the current photon energy.
    // Call on the host before each Kokkos kernel (lambda captures by value).
    KOKKOS_INLINE_FUNCTION void setMuRho(float v) { mu_rho_ = v; }

    KOKKOS_INLINE_FUNCTION bool isEntering(const Ray& ray) const {
        if (fabsf(ray.getDirY()) < 1e-9f) return false;
        float t  = (posY_ - ray.getStartY()) / ray.getDirY();
        if (t < 0.f) return false;
        float hx = ray.getStartX() + t * ray.getDirX();
        float hz = ray.getStartZ() + t * ray.getDirZ();
        return (hx*hx + hz*hz) <= rExtIn_ * rExtIn_;
    }

    KOKKOS_INLINE_FUNCTION void trace(Ray& ray) const {
        if (!isEntering(ray)) { ray.setIAFlag(false); return; }

        float e = ray.getEnergyKeV();
        if (e <= 0.f) { ray.setIAFlag(false); return; }

        float sinTheta = sqrtf(ray.getDirX()*ray.getDirX()
                             + ray.getDirZ()*ray.getDirZ());
        float thetaCrit = critCoeff_ / e;

        // ── Full complex Fresnel reflectivity per bounce ──────────────────────
        // X-ray refractive index: n = 1 − δ − iβ
        //   δ = θ_c² / 2  (from critCoeff_ = E·√(2δ))
        //   β = μρ · ħc / (4π E)  [μρ in cm⁻¹, ħc_cm ≈ 1.23984e-7 keV·cm]
        // Reflectivity: R_F = |(sinθ − w) / (sinθ + w)|²
        //   where w = √(sin²θ − θ_c² − 2iβ)
        const float HC_CM  = 1.23984e-7f;          // keV·cm
        float beta  = mu_rho_ * HC_CM / (4.f * VT_PI * e);
        float xr    = sinTheta*sinTheta - thetaCrit*thetaCrit;
        float xi    = -2.f * beta;
        float rm    = sqrtf(xr*xr + xi*xi);
        float wu    = sqrtf((rm + xr) * 0.5f);     // Re(w)
        float wv    = (xi < 0.f ? -1.f : 1.f) * sqrtf((rm - xr) * 0.5f); // Im(w)
        float numer = (sinTheta-wu)*(sinTheta-wu) + wv*wv;
        float denom = (sinTheta+wu)*(sinTheta+wu) + wv*wv;
        float R_F   = (denom > 1e-30f) ? fminf(numer / denom, 1.f) : 0.f;

        // ── Debye-Waller surface-roughness factor ─────────────────────────────
        // roughness_ in Å, ħc_Ang = 12.398 keV·Å
        const float HC_ANG = 12.398f;
        float expt = (4.f * VT_PI * roughness_ * sinTheta * e) / HC_ANG;
        float R_DW = expf(-expt * expt);

        float R = R_F * R_DW;

        float rCap  = (rCapIn_ + rCapOut_) * 0.5f;
        float nRefl = (sinTheta > 1e-6f)
                    ? (length_ * sinTheta) / (2.f * rCap * fabsf(ray.getDirY()))
                    : 0.f;

        float t  = (posY_ + length_ - ray.getStartY()) / ray.getDirY();
        float ex = ray.getStartX() + t * ray.getDirX();
        float ez = ray.getStartZ() + t * ray.getDirZ();

        if (ex*ex + ez*ez > rExtOut_ * rExtOut_) { ray.setIAFlag(false); return; }

        ray.setStartCoordinates(ex, posY_ + length_, ez);

        // Redirect exit ray toward the output focal spot.
        // For a focusing optic focalOut_ is finite (e.g. 0.49 cm);
        // for a collimating optic focalOut_ ≈ 1e8 → no meaningful redirect.
        if (focalOut_ < 1e7f) {
            float fdx = 0.f - ex;
            float fdy = focalOut_;          // exit face to focal spot
            float fdz = 0.f - ez;
            float fl  = sqrtf(fdx*fdx + fdy*fdy + fdz*fdz);
            ray.setEndCoordinates(fdx/fl, fdy/fl, fdz/fl);
        }

        ray.setIAFlag(true);
        ray.setIANum(ray.getIANum() + (int)nRefl);
        ray.setProb(ray.getProb() * powf(R, nRefl));
    }

#ifndef __METAL_VERSION__
    inline void print() const {
        printf("PolyCap: L=%.2fcm rExt=(%.4f->%.4f)cm rCap=(%.5f->%.5f)cm"
               " rho=%.2fg/cc rough=%.1fA ncap=%d\n",
               length_, rExtIn_, rExtOut_, rCapIn_, rCapOut_,
               density_, roughness_, nCapillaries_);
    }
#endif
};
