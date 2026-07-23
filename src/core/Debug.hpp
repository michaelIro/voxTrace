#pragma once
/**
 * @file Debug.hpp
 * @brief Runtime ray/voxel inspection — stage-tagged one-line state dumps.
 *
 * Lets you watch what a photon actually does while the trace runs, without
 * recompiling: set the level (and optionally a single ray id) at startup —
 * in the tests via `debug=N` / `debug_ray=I` in Setup.txt or on the command
 * line — and every instrumented stage prints one grep-friendly line:
 *
 *     [dbg    1234|surface-entry ] pos 0: (0.01527 0.01496) dir=(+0.7113 ...)
 *
 * Levels: 0 = off, 1 = detected events only, 2 = + every stage a ray passes
 * (beam exit, surface entry, interaction, emission, secondary optic,
 * detector), 3 = + the voxel walk. Combine level 3 with `debug_ray` to follow
 * ONE photon through the whole instrument.
 *
 * Device-safe by construction: the settings travel in a small vtdbg::Ctx that
 * kernels capture BY VALUE (device code cannot read host globals), and every
 * line is a single plain `printf` — supported in CUDA/HIP device code — whose
 * prefix and body format are merged at compile time by @ref VT_DBG, so lines
 * stay atomic. In parallel runs the line ORDER may interleave; pin one ray
 * with `debug_ray` if that matters.
 */

#include <cstdio>

#include "Platform.hpp"
#include "Ray.hpp"
#include "Voxel.hpp"

/// One atomic debug line: `VT_DBG(id, "stage", "x=%d", x)`. The prefix and the
/// caller's format are concatenated at compile time into a single printf.
#define VT_DBG(id, stage, fmt, ...) \
    printf("[dbg %7ld|%-14s] " fmt "\n", (long)(id), (stage) __VA_OPT__(,) __VA_ARGS__)

namespace vtdbg {

/// Debug settings; kernels hold a copy (value capture), the host keeps the
/// process-wide instance in @ref cfg.
struct Ctx {
    int  level = 0;    ///< 0 off, 1 events, 2 + ray stages, 3 + voxel walk
    long only  = -1;   ///< restrict output to one ray id (-1 = all)

    /// Is stage level @p lvl active for ray @p id?
    KOKKOS_INLINE_FUNCTION bool on(int lvl, long id = -1) const {
        return level >= lvl && (only < 0 || id < 0 || only == id);
    }
};

inline Ctx cfg;   ///< process-wide settings (host side; copy into kernels)

/// One-line dump of a Ray's state (position, direction, energy, weight, flag).
KOKKOS_INLINE_FUNCTION void ray(long id, const char* stage, const Ray& r) {
    VT_DBG(id, stage, "pos=(%+.5f %+.5f %+.5f) dir=(%+.4f %+.4f %+.4f) E=%.3f keV prob=%.3e flag=%d",
           (double)r.getStartX(), (double)r.getStartY(), (double)r.getStartZ(),
           (double)r.getDirX(),   (double)r.getDirY(),   (double)r.getDirZ(),
           (double)r.getEnergyKeV(), (double)r.getProb(), (int)r.getIAFlag());
}

/// One-line dump of the voxel a ray is walking through.
KOKKOS_INLINE_FUNCTION void voxel(long id, const char* stage, int idx, const Voxel& v) {
    VT_DBG(id, stage, "voxel %d  x[%.4f,%.4f) y[%.4f,%.4f) z[%.4f,%.4f) mat=%d",
           idx, (double)v.getX0(), (double)v.getX1(), (double)v.getY0(), (double)v.getY1(),
           (double)v.getZ0(), (double)v.getZ1(), v.getMaterialIdx());
}

}  // namespace vtdbg
