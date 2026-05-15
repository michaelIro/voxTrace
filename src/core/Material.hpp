#pragma once
#include "Platform.hpp"
#include "ChemElement.hpp"

// ── Material ──────────────────────────────────────────────────────────────────
// Composite voxel material. Stores weight fractions for up to MAX_ELEMENTS
// elements. Element data lives in a shared ChemElement array passed at call
// sites — avoiding pointer members so the struct is MSL-compatible.

static constexpr int MAX_ELEMENTS = 8;

class Material {
    int   num_elements_  = 0;
    float weights_[MAX_ELEMENTS] = {};
    float rho_ = 0.f;

public:

    KOKKOS_INLINE_FUNCTION Material() {}

    // Host-side construction: density computed from element densities
    KOKKOS_INLINE_FUNCTION Material(int n, const float* w, const ChemElement* elems)
        : num_elements_(n) {
        for (int i = 0; i < n; ++i) {
            weights_[i] = w[i];
            rho_ += elems[i].Rho() * w[i];
        }
    }

    KOKKOS_INLINE_FUNCTION float Rho() const { return rho_; }

    KOKKOS_INLINE_FUNCTION float CS_Tot(float e, const ChemElement* elems) const {
        float tot = 0.f;
        for (int i = 0; i < num_elements_; ++i)
            tot += weights_[i] * elems[i].CS_Tot(e);
        return tot;
    }

    KOKKOS_INLINE_FUNCTION float CS_Tot_Lin(float e, const ChemElement* elems) const {
        return CS_Tot(e, elems) * rho_;
    }

    // Returns index of the interacting element in the shared elements array
    KOKKOS_INLINE_FUNCTION int getInteractingElementIdx(float e, float r,
                                                         const ChemElement* elems) const {
        float muTot = CS_Tot(e, elems);
        float sum   = 0.f;
        for (int i = 0; i < num_elements_ - 1; ++i) {
            sum += elems[i].CS_Tot(e) * weights_[i] / muTot;
            if (sum >= r) return i;
        }
        return num_elements_ - 1;
    }
};
