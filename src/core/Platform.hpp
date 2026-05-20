#pragma once

// ── Backend compatibility shim ────────────────────────────────────────────────
// This header is included by every shared physics header.
// It makes KOKKOS_INLINE_FUNCTION available in both Kokkos and Metal contexts.

#ifdef __METAL_VERSION__
    // Metal Shading Language: no Kokkos
    #define KOKKOS_INLINE_FUNCTION inline
    #define KOKKOS_FUNCTION        inline
    // Metal uses C++ math names — no 'f' suffix
    #define cosf   cos
    #define sinf   sin
    #define sqrtf  sqrt
    #define fabsf  fabs
    #define fmaxf  fmax
    #define fminf  fmin
    #define logf   log
    #define expf   exp
    #define floorf floor
    #define ceilf  ceil
    #define atan2f atan2
    // Static class constants need 'constant constexpr' in Metal
    #define VT_SCONSTEXPR   static constant constexpr
    #define lroundf(x)      ((int)round(x))
    // Address space qualifiers: parameters and trailing method qualifiers
    #define VT_THREAD       thread    // refs/ptrs to thread-local objects (Ray, RNG)
    #define VT_DEVICE       device    // ptrs to device-buffer objects
    #define VT_DEVICE_METH  device    // trailing qualifier for device-buffer methods
    #define VT_CONST_METH   constant  // trailing qualifier for constant-buffer methods
#elif defined(VOXTRACE_HOST_ONLY) || defined(VOXTRACE_METAL)
    // Host-only / Metal host-side compilation — no Kokkos runtime needed
    #define KOKKOS_INLINE_FUNCTION inline
    #define KOKKOS_FUNCTION        inline
    #define VT_SCONSTEXPR   static constexpr
    #define VT_THREAD
    #define VT_DEVICE
    #define VT_DEVICE_METH
    #define VT_CONST_METH
#else
    #include <Kokkos_Core.hpp>
    #define VT_SCONSTEXPR   static constexpr
    #define VT_THREAD
    #define VT_DEVICE
    #define VT_DEVICE_METH
    #define VT_CONST_METH
#endif

// VT_PI: requires 'constant' address space at program scope in Metal
#ifdef __METAL_VERSION__
    constant constexpr float VT_PI = 3.14159265358979323846f;
#else
    static constexpr float VT_PI = 3.14159265358979323846f;
#endif
