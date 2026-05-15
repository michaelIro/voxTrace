#pragma once

// ── Backend compatibility shim ────────────────────────────────────────────────
// This header is included by every shared physics header.
// It makes KOKKOS_INLINE_FUNCTION available in both Kokkos and Metal contexts.

#ifdef __METAL_VERSION__
    // Metal Shading Language: no Kokkos
    #define KOKKOS_INLINE_FUNCTION inline
    #define KOKKOS_FUNCTION        inline
#elif defined(VOXTRACE_HOST_ONLY)
    // Host-only compilation (tests, tools) — no Kokkos runtime needed
    #define KOKKOS_INLINE_FUNCTION inline
    #define KOKKOS_FUNCTION        inline
#else
    #include <Kokkos_Core.hpp>
#endif

static constexpr float VT_PI = 3.14159265358979323846f;
