#pragma once
/**
 * @file ResponseMatrix.hpp
 * @brief Sparse voxel→spectrum response: the linear forward model S(w) = B + Σ_v w_v R_v.
 */

#include <algorithm>
#include <vector>

#include "Spectrum.hpp"
#include "SpectrumLoss.hpp"

/**
 * @brief Sparse linear forward model of one scan position.
 *
 * While a Monte-Carlo trace runs, every detected photon deposits its weight
 * into a (voxel-parameter, detector-channel) cell via add(). Afterwards the
 * simulated spectrum for ANY voxel-weight vector w follows without re-tracing:
 *
 *     S_ch(w) = B_ch + Σ_p w_p R[p][ch]
 *
 * The weights scale each voxel's *emission* only; attenuation stays at the
 * nominal composition — the standard first-order linearisation of confocal-XRF
 * reconstruction, which turns the fit into a fast linear least-squares-like
 * problem with an exact gradient. R is stored sparsely as (param, channel,
 * weight) triplets (each MC event fills exactly one cell, so the matrix is as
 * sparse as the statistics); B collects the photons of voxels that are not
 * fitted (too few events to constrain a parameter — frozen at weight 1).
 * Host-only fitting-layer class, like @ref Spectrum.
 */
class ResponseMatrix {
public:
    struct Entry {
        int   param;  ///< voxel-parameter index
        int   ch;     ///< detector channel
        float w;      ///< accumulated response [expected counts at w_param = 1]
    };

private:
    std::vector<Entry>  entries_;
    std::vector<double> baseline_;
    int nParams_   = 0;
    int nChannels_ = 0;

public:
    ResponseMatrix(int nParams, int nChannels)
        : baseline_(nChannels, 0.0), nParams_(nParams), nChannels_(nChannels) {}

    int    nParams()   const { return nParams_; }
    int    nChannels() const { return nChannels_; }
    size_t nnz()       const { return entries_.size(); }

    void add(int param, int ch, float w) { entries_.push_back({param, ch, w}); }
    void addBaseline(int ch, double w)   { baseline_[ch] += w; }

    /// Merge duplicate (param, channel) cells — call once after filling.
    void finalize() {
        std::sort(entries_.begin(), entries_.end(),
                  [](const Entry& a, const Entry& b) {
                      return (a.param != b.param) ? a.param < b.param : a.ch < b.ch;
                  });
        size_t out = 0;
        for (size_t i = 0; i < entries_.size();) {
            Entry e = entries_[i];
            for (++i; i < entries_.size() &&
                      entries_[i].param == e.param && entries_[i].ch == e.ch; ++i)
                e.w += entries_[i].w;
            entries_[out++] = e;
        }
        entries_.resize(out);
    }

    /// Multiply every response/baseline cell by @p s (live-time / flux scaling).
    void scale(double s) {
        for (Entry& e : entries_) e.w = (float)(e.w * s);
        for (double& b : baseline_) b *= s;
    }

    /// Total response mass Σ_ch R[p][ch] per parameter (diagnostics and
    /// response-weighted profile averages).
    void addParamMass(std::vector<double>& mass) const {
        for (const Entry& e : entries_) mass[e.param] += e.w;
    }

    /// Assemble S(w) into @p out (a Spectrum with nChannels() channels).
    void assemble(const std::vector<double>& w, Spectrum& out) const {
        out.clear();
        for (int ch = 0; ch < nChannels_; ++ch) out.addCount(ch, (float)baseline_[ch]);
        for (const Entry& e : entries_) out.addCount(e.ch, (float)(w[e.param] * e.w));
    }

    /// Chain rule of @p loss through the model:  g_p += Σ_ch ∂L/∂S_ch · R[p][ch].
    void accumulateGrad(const SpectrumLoss& loss, const Spectrum& sim,
                        const Spectrum& meas, std::vector<double>& g) const {
        std::vector<float> dLdS(nChannels_);
        for (int ch = 0; ch < nChannels_; ++ch) dLdS[ch] = loss.grad(sim, meas, ch);
        for (const Entry& e : entries_) g[e.param] += (double)dLdS[e.ch] * e.w;
    }
};
