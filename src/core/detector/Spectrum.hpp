#pragma once
/**
 * @file Spectrum.hpp
 * @brief Detector energy histogram and regions of interest.
 */

#include <cmath>
#include <new>

/// Region of interest: a channel range [chMin, chMax] with a fitting weight.
struct ROI {
    int   chMin;
    int   chMax;
    float weight;
};

/**
 * @brief Energy histogram of detected photons (the simulated detector spectrum).
 *
 * Host-only output container: photons are binned into channels by a linear
 * energy calibration (`offset`, `gain`). Unlike the device-side core it owns a
 * heap-allocated counts array, since it is only accumulated/serialised on the host.
 */
class Spectrum {
    float* counts_;
    int    nChannels_;
    float  offset_;
    float  gain_;

public:

    Spectrum(int nChannels, float offset = 0.0f, float gain = 1.0f)
        : nChannels_(nChannels), offset_(offset), gain_(gain)
    {
        counts_ = new float[nChannels_]();
    }

    ~Spectrum() { delete[] counts_; }

    // Non-copyable to avoid double-free; move only
    Spectrum(const Spectrum&)            = delete;
    Spectrum& operator=(const Spectrum&) = delete;

    int   nChannels()              const { return nChannels_; }
    float count(int ch)            const { return counts_[ch]; }
    float energy(int ch)           const { return offset_ + gain_ * (float)ch; }
    int   channel(float keV)       const { return (int)((keV - offset_) / gain_); }

    void  setCount(int ch, float v)    { counts_[ch] = v; }
    void  addCount(int ch, float v)    { counts_[ch] += v; }
    void  setCalibration(float o, float g) { offset_=o; gain_=g; }

    // Chi-square over all channels: Σ (obs − exp)² / exp
    float chiSquare(const Spectrum& expected) const {
        float chi2 = 0.f;
        for (int ch = 0; ch < nChannels_; ++ch) {
            float ex = expected.count(ch);
            if (ex <= 0.f) continue;
            float d = counts_[ch] - ex;
            chi2 += (d * d) / ex;
        }
        return chi2;
    }

    // Weighted chi-square; ROI channels use roi.weight, rest use 1.0
    float chiSquareWeighted(const Spectrum& expected,
                             const ROI* rois, int nRois) const {
        float chi2 = 0.f;
        for (int ch = 0; ch < nChannels_; ++ch) {
            float ex = expected.count(ch);
            if (ex <= 0.f) continue;
            float w = 1.f;
            for (int r = 0; r < nRois; ++r)
                if (ch >= rois[r].chMin && ch <= rois[r].chMax)
                    w = rois[r].weight;
            float d = counts_[ch] - ex;
            chi2 += w * (d * d) / ex;
        }
        return chi2;
    }
};
