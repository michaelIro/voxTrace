#pragma once
#include "Platform.hpp"

// ── Portable xorshift64* RNG ──────────────────────────────────────────────────
// Works in C++, Kokkos device code (CUDA/HIP/OpenMP/SYCL), and Metal MSL.

struct RNG {
    uint64_t state_;

    KOKKOS_INLINE_FUNCTION RNG(uint64_t seed) : state_(seed | 1ULL) {}

    KOKKOS_INLINE_FUNCTION uint64_t next64() {
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        return state_ * 0x2545F4914F6CDD1DULL;
    }

    // Uniform float in [0, 1)
    KOKKOS_INLINE_FUNCTION float frand() {
        return (float)(next64() >> 11) * (1.0f / (float)(1ULL << 53));
    }

    // Standard normal via Box-Muller
    KOKKOS_INLINE_FUNCTION float normal() {
        float u1 = fmaxf(frand(), 1e-7f);
        float u2 = frand();
        return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * VT_PI * u2);
    }
};
