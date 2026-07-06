#pragma once
/**
 * @file SpectrumLoss.hpp
 * @brief Selectable χ² / ROI-weighted χ² loss between a simulated and a measured spectrum.
 */

#include "Spectrum.hpp"

/**
 * @brief Loss function between a simulated and a measured @ref Spectrum.
 *
 * Wraps the two figures of merit of @ref Spectrum — plain chi-square and
 * ROI-weighted chi-square — behind one switchable object and adds the
 * per-channel derivative ∂L/∂S_ch that a fit needs to propagate the loss
 * back through a linear forward model (see @ref ResponseMatrix):
 *
 *     L = Σ_ch  W_ch (S_ch − M_ch)² / M_ch          (Neyman χ², var = M_ch)
 *
 * with W_ch = 1 everywhere in CHI2 mode, while in CHI2_WEIGHTED mode W_ch is
 * taken from the registered ROIs, so selected fluorescence lines (e.g. trace
 * element peaks that carry few counts) can be emphasised in the fit. Channels
 * with M_ch ≤ 0 are skipped, exactly as in Spectrum::chiSquare*, and the
 * value() itself is computed by those Spectrum methods — this class only
 * selects between them and differentiates. Host-only fitting-layer class.
 */
class SpectrumLoss {
public:
    enum Type { CHI2 = 0, CHI2_WEIGHTED = 1 };

private:
    static constexpr int MAX_ROIS = 16;
    Type type_ = CHI2;
    ROI  rois_[MAX_ROIS] = {};
    int  nRois_ = 0;

public:
    explicit SpectrumLoss(Type type = CHI2) : type_(type) {}

    Type type()  const { return type_; }
    int  nRois() const { return nRois_; }

    /// Register a region of interest [chMin, chMax] with fit weight @p weight.
    /// Only used in CHI2_WEIGHTED mode; a later ROI overrides an earlier one.
    void addROI(int chMin, int chMax, float weight) {
        if (nRois_ < MAX_ROIS) rois_[nRois_++] = {chMin, chMax, weight};
    }

    /// Fit weight W_ch of channel @p ch (mirrors Spectrum::chiSquareWeighted).
    float channelWeight(int ch) const {
        if (type_ == CHI2) return 1.f;
        float w = 1.f;
        for (int r = 0; r < nRois_; ++r)
            if (ch >= rois_[r].chMin && ch <= rois_[r].chMax) w = rois_[r].weight;
        return w;
    }

    /// Loss value — delegates to the Spectrum chi-square methods.
    float value(const Spectrum& sim, const Spectrum& meas) const {
        return (type_ == CHI2) ? sim.chiSquare(meas)
                               : sim.chiSquareWeighted(meas, rois_, nRois_);
    }

    /// ∂L/∂S_ch = 2 W_ch (S_ch − M_ch) / M_ch  (0 where the channel is skipped).
    float grad(const Spectrum& sim, const Spectrum& meas, int ch) const {
        float m = meas.count(ch);
        if (m <= 0.f) return 0.f;
        return 2.f * channelWeight(ch) * (sim.count(ch) - m) / m;
    }
};
