#pragma once
/**
 * @file RNG.hpp
 * @brief Portable per-thread pseudo-random number generator.
 */
#include "Platform.hpp"

/**
 * @brief Portable xorshift64* RNG — one lightweight stream per ray/thread.
 *
 * A self-contained 64-bit generator (8 bytes of state) seeded per thread, so it
 * works identically in host C++, Kokkos device code (CUDA/HIP/OpenMP/SYCL) and
 * Metal MSL — replacing curand from the original CUDA code. Provides uniform
 * (`frand`) and standard-normal (`normal`, Box–Muller) draws used throughout the
 * source and interaction sampling.
 */
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
