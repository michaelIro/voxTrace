#ifndef Spectrum_H
#define Spectrum_H

#include <cuda_runtime.h>
#include <math.h>

// ── ROI ───────────────────────────────────────────────────────────────────────

struct ROI {
    int   chMin;
    int   chMax;
    float weight;
};

// ── Spectrum ──────────────────────────────────────────────────────────────────

class Spectrum {

    float* counts_;     // raw pointer — works on both host and device
    int    nChannels_;
    float  offset_;     // E(ch) = offset_ + gain_ * ch  [keV]
    float  gain_;

public:

    // ── Lifecycle (host only — allocate/free on host, then pass to device) ───

    __host__ Spectrum(int nChannels, float offset = 0.0f, float gain = 1.0f)
        : nChannels_(nChannels), offset_(offset), gain_(gain)
    {
        cudaMallocManaged(&counts_, nChannels_ * sizeof(float));
        for (int i = 0; i < nChannels_; ++i) counts_[i] = 0.0f;
    }

    __host__ ~Spectrum() { cudaFree(counts_); }

    // ── Accessors ─────────────────────────────────────────────────────────────

    __host__ __device__ int   nChannels()              const { return nChannels_; }
    __host__ __device__ float count(int ch)            const { return counts_[ch]; }
    __host__ __device__ float energy(int ch)           const { return offset_ + gain_ * ch; }
    __host__ __device__ int   channel(float energyKeV) const { return (int)((energyKeV - offset_) / gain_); }

    __host__ __device__ void  setCount(int ch, float val)  { counts_[ch] = val; }
    __host__ __device__ void  addCount(int ch, float val)  { counts_[ch] += val; }
    __host__ __device__ void  setCalibration(float offset, float gain) { offset_ = offset; gain_ = gain; }

    // ── Loss: chi-square over all channels Σ (obs-exp)²/exp ─────────────────

    __host__ __device__ float chiSquare(const Spectrum& expected) const {
        float chi2 = 0.0f;
        for (int ch = 0; ch < nChannels_; ++ch) {
            float exp = expected.count(ch);
            if (exp <= 0.0f) continue;
            float diff = counts_[ch] - exp;
            chi2 += (diff * diff) / exp;
        }
        return chi2;
    }

    // ── Loss: weighted chi-square — ROI channels use roi.weight, rest = 1.0 ─

    __host__ __device__ float chiSquareWeighted(const Spectrum& expected,
                                                 const ROI* rois,
                                                 int nRois) const {
        float chi2 = 0.0f;
        for (int ch = 0; ch < nChannels_; ++ch) {
            float exp = expected.count(ch);
            if (exp <= 0.0f) continue;

            float w = 1.0f;
            for (int r = 0; r < nRois; ++r)
                if (ch >= rois[r].chMin && ch <= rois[r].chMax)
                    w = rois[r].weight;

            float diff = counts_[ch] - exp;
            chi2 += w * (diff * diff) / exp;
        }
        return chi2;
    }
};

#endif
