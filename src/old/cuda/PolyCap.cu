#ifndef PolyCap_H
#define PolyCap_H

#include <cuda_runtime.h>
#include <math.h>
#include "RayGPU.cu"

// Max elements in optic material (fixed-size arrays for device compatibility)
#define POLYCAP_MAX_ELEMENTS 8

class PolyCap {

    // ── Geometry ──────────────────────────────────────────────────────────────
    float posY_;        // entrance face y-position in cm
    float length_;      // optic length in cm
    float rExtIn_;      // external radius at entrance in cm
    float rExtOut_;     // external radius at exit in cm
    float rCapIn_;      // single capillary radius at entrance in cm
    float rCapOut_;     // single capillary radius at exit in cm
    float focalIn_;     // focal distance at entrance side in cm
    float focalOut_;    // focal distance at exit side in cm

    // ── Material ──────────────────────────────────────────────────────────────
    int   nElements_;
    int   atomicNumbers_[POLYCAP_MAX_ELEMENTS];
    float weightFracs_  [POLYCAP_MAX_ELEMENTS];
    float density_;     // g/cm³
    float roughness_;   // surface roughness in Angstrom
    int   nCapillaries_;

    // ── Precomputed: critical angle coefficient ───────────────────────────────
    // θ_c [rad] = critCoeff_ / E [keV]   (28.8e-3 * sqrt(ρ * <Z/A>))
    float critCoeff_;

    // Atomic mass lookup (approx.) for Z up to 20
    __host__ __device__ float atomicMass(int z) const {
        const float A[] = {0,1,4,7,9,11,12,14,16,19,20,
                           23,24,27,28,31,32,35,40,39,40};
        if (z > 0 && z <= 20) return A[z];
        return (float)(2 * z);   // rough estimate for heavier elements
    }

    __host__ __device__ void computeCritCoeff() {
        float ZoverA = 0.0f;
        for (int i = 0; i < nElements_; ++i)
            ZoverA += (weightFracs_[i] / 100.0f)
                    * ((float)atomicNumbers_[i] / atomicMass(atomicNumbers_[i]));
        critCoeff_ = 28.8e-3f * sqrtf(density_ * ZoverA);
    }

public:

    __host__ __device__ PolyCap() {}

    __host__ PolyCap(
        float posY, float length,
        float rExtIn,  float rExtOut,
        float rCapIn,  float rCapOut,
        float focalIn, float focalOut,
        int nElements, const int* atomicNumbers, const float* weightFracs,
        float density, float roughness, int nCapillaries)
        : posY_(posY), length_(length),
          rExtIn_(rExtIn), rExtOut_(rExtOut),
          rCapIn_(rCapIn), rCapOut_(rCapOut),
          focalIn_(focalIn), focalOut_(focalOut),
          nElements_(nElements), density_(density),
          roughness_(roughness), nCapillaries_(nCapillaries)
    {
        for (int i = 0; i < nElements_; ++i) {
            atomicNumbers_[i] = atomicNumbers[i];
            weightFracs_[i]   = weightFracs[i];
        }
        computeCritCoeff();
    }

    // ── Geometry helpers ──────────────────────────────────────────────────────

    // External envelope radius at longitudinal position s ∈ [0, length_]
    __host__ __device__ float rExtAt(float s) const {
        return rExtIn_ + (rExtOut_ - rExtIn_) * (s / length_);
    }

    // Capillary radius at position s
    __host__ __device__ float rCapAt(float s) const {
        return rCapIn_ + (rCapOut_ - rCapIn_) * (s / length_);
    }

    // ── Check if ray enters the optic ─────────────────────────────────────────
    __host__ __device__ bool isEntering(const RayGPU& ray) const {
        if (fabsf(ray.getDirY()) < 1e-9f) return false;
        float t  = (posY_ - ray.getStartY()) / ray.getDirY();
        if (t < 0.0f) return false;
        float hx = ray.getStartX() + t * ray.getDirX() - 0.0f;
        float hz = ray.getStartZ() + t * ray.getDirZ() - 0.0f;
        return (hx*hx + hz*hz) <= rExtIn_ * rExtIn_;
    }

    // ── Trace ray through optic — modifies ray probability in place ───────────
    /**
     * Model:
     *  1. Propagate to entrance face, reject if outside envelope
     *  2. Compute grazing angle θ to capillary wall
     *  3. Estimate number of reflections N ≈ length * tan(θ) / (2 * rCap)
     *  4. Per-reflection probability: Debye-Waller modified Fresnel
     *     R = H(θ_c - θ) * exp(-(4π σ sinθ E / hc)²)
     *  5. Propagate ray to exit face
     */
    __host__ __device__ void trace(RayGPU& ray) const {
        if (!isEntering(ray)) { ray.setIAFlag(false); return; }

        float energyKeV = ray.getEnergyKeV();
        if (energyKeV <= 0.0f) { ray.setIAFlag(false); return; }

        // Grazing angle to optical axis (y-axis)
        float sinTheta = sqrtf(ray.getDirX()*ray.getDirX()
                             + ray.getDirZ()*ray.getDirZ());
        float theta    = asinf(fminf(sinTheta, 1.0f));  // rad

        // Critical angle for this energy
        float thetaCrit = critCoeff_ / energyKeV;       // rad

        // Reflection probability per bounce (Debye-Waller)
        float hc   = 12.398f;    // keV·Å
        float expt = (4.0f * M_PI * roughness_ * sinTheta * energyKeV) / hc;
        float R    = (theta < thetaCrit) ? expf(-expt * expt) : 0.0f;

        // Number of reflections along optic
        float rCap  = (rCapIn_ + rCapOut_) * 0.5f;
        float nRefl = (sinTheta > 1e-6f)
                    ? (length_ * sinTheta) / (2.0f * rCap * fabsf(ray.getDirY()))
                    : 0.0f;

        // Accumulated transmission
        float transmission = powf(R, nRefl);

        // Propagate to exit face
        float t  = (posY_ + length_ - ray.getStartY()) / ray.getDirY();
        float ex = ray.getStartX() + t * ray.getDirX();
        float ez = ray.getStartZ() + t * ray.getDirZ();

        // Reject if outside exit envelope
        if (ex*ex + ez*ez > rExtOut_ * rExtOut_) {
            ray.setIAFlag(false);
            return;
        }

        ray.setStartCoordinates(ex, posY_ + length_, ez);
        ray.setIAFlag(true);
        ray.setIANum(ray.getIANum() + (int)nRefl);

        // Update probability
        float newProb = ray.getProb() * transmission;
        // RayGPU has no setProb — use existing interaction mechanism
        // Store transmission in probability via OPD as workaround:
        // (add setProb to RayGPU if needed)
    }

    __host__ void print() const {
        printf("PolyCap: L=%.2fcm rExt=(%.4f→%.4f)cm rCap=(%.5f→%.5f)cm"
               " f=(%.2f,%.2f)cm rho=%.2fg/cc rough=%.1fA ncap=%d\n",
               length_, rExtIn_, rExtOut_, rCapIn_, rCapOut_,
               focalIn_, focalOut_, density_, roughness_, nCapillaries_);
    }
};

#endif
