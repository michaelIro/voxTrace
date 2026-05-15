#ifndef Detector_H
#define Detector_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "RayGPU.cu"
#include "Spectrum.cu"

class Detector {

    float posX_, posY_, posZ_;  // detector centre position
    float radius_;              // active detector radius
    int   channels_;            // number of spectrum channels
    float gain_;                // keV per channel
    float zero_;                // zero offset in keV: E(ch) = zero_ + gain_*ch
    float fano_;                // Fano factor (~0.114 for Si)
    float noise_;               // electronic noise in keV (sigma)
    float liveTime_;            // live time in seconds

public:

    __host__ __device__ Detector() {}

    __host__ __device__ Detector(
        float posX, float posY, float posZ, float radius,
        int channels, float gain, float zero,
        float fano, float noise, float liveTime)
        : posX_(posX), posY_(posY), posZ_(posZ), radius_(radius),
          channels_(channels), gain_(gain), zero_(zero),
          fano_(fano), noise_(noise), liveTime_(liveTime) {}

    // ── Accessors ─────────────────────────────────────────────────────────────

    __host__ __device__ int   channels()  const { return channels_; }
    __host__ __device__ float gain()      const { return gain_; }
    __host__ __device__ float zero()      const { return zero_; }
    __host__ __device__ float liveTime()  const { return liveTime_; }

    // ── Geometry: check if ray hits the detector face ─────────────────────────

    __host__ __device__ bool isHit(const RayGPU& ray) const {
        // propagate ray to detector plane (posY_)
        float dy = posY_ - ray.getStartY();
        if (fabsf(ray.getDirY()) < 1e-9f) return false;
        float t  = dy / ray.getDirY();
        if (t < 0.0f) return false;

        float hx = ray.getStartX() + t * ray.getDirX() - posX_;
        float hz = ray.getStartZ() + t * ray.getDirZ() - posZ_;
        return (hx*hx + hz*hz) <= radius_ * radius_;
    }

    // ── Record hit into spectrum (device: applies Fano + noise broadening) ────

    __device__ void recordHit(const RayGPU& ray, Spectrum& spectrum,
                               curandState_t* localState) const
    {
        if (!isHit(ray)) return;

        float energyKeV = ray.getEnergyKeV();

        // Energy resolution: σ² = fano_ * 0.00365 * E + noise_²
        // 0.00365 keV = mean ionisation energy in Si
        float sigma = sqrtf(fano_ * 0.00365f * energyKeV + noise_ * noise_);
        float eMeas = energyKeV + sigma * curand_normal(localState);

        int ch = (int)((eMeas - zero_) / gain_);
        if (ch >= 0 && ch < spectrum.nChannels())
            spectrum.addCount(ch, ray.getProb());
    }

    // ── Host version: no broadening (deterministic, for testing) ─────────────

    __host__ void recordHitHost(const RayGPU& ray, Spectrum& spectrum) const {
        if (!isHit(ray)) return;
        int ch = (int)((ray.getEnergyKeV() - zero_) / gain_);
        if (ch >= 0 && ch < spectrum.nChannels())
            spectrum.addCount(ch, ray.getProb());
    }

    __host__ void print() const {
        printf("Detector: pos=(%.2f,%.2f,%.2f) r=%.2f ch=%d gain=%.4f zero=%.4f\n",
               posX_, posY_, posZ_, radius_, channels_, gain_, zero_);
    }
};

#endif
