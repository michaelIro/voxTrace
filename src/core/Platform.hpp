#pragma once
/**
 * @file Platform.hpp
 * @brief Backend compatibility shim — the single point where the accelerator is abstracted.
 *
 * Included by every shared physics header. It defines the portable annotation
 * macros (`KOKKOS_INLINE_FUNCTION`, `VT_SCONSTEXPR`, `VT_DEVICE_METH`, ...) so
 * the same source is valid in two regimes, selected by a predefined macro:
 *   - `VOXTRACE_HOST_ONLY` → plain host build, no Kokkos headers (tests, tools).
 *   - otherwise            → full Kokkos build (CUDA/HIP/OpenMP/Serial).
 *
 * See the "Accelerators & performance portability" documentation page for the
 * build-time details.
 */

// ── Backend compatibility shim ────────────────────────────────────────────────
// This header is included by every shared physics header.
// It makes KOKKOS_INLINE_FUNCTION available in both host-only and Kokkos builds.

#ifdef VOXTRACE_HOST_ONLY
    // Host-only compilation — no Kokkos runtime needed
    #define KOKKOS_INLINE_FUNCTION inline
    #define KOKKOS_FUNCTION        inline
#else
    #include <Kokkos_Core.hpp>
#endif

// Portable annotation macros. These carried address-space qualifiers under the
// old Metal backend; on host/Kokkos they are no-ops, kept so device methods stay
// self-documenting and a future backend can redefine them in one place.
#define VT_SCONSTEXPR   static constexpr
#define VT_THREAD
#define VT_DEVICE
#define VT_DEVICE_METH
#define VT_CONST_METH

static constexpr float VT_PI = 3.14159265358979323846f;
