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
 *     [dbg   1234|surface-entry] pos=(+0.00123 -0.00045 +0.00000) dir=(...) E=17.400 prob=2.1e-01
 *
 * Levels: 0 = off, 1 = detected events only, 2 = + every stage a ray passes
 * (beam exit, surface entry, interaction, emission, secondary optic,
 * detector), 3 = + the voxel walk. Combine level 3 with `debug_ray` to follow
 * ONE photon through the whole instrument. Plain printf so it works in serial
 * and OpenMP/Kokkos-parallel runs alike (lines stay atomic, order may
 * interleave — pin one ray if that matters).
 */

#include <cstdarg>
#include <cstdio>

#include "Ray.hpp"
#include "Voxel.hpp"

namespace vtdbg {

inline int  level = 0;    ///< 0 off, 1 events, 2 + ray stages, 3 + voxel walk
inline long only  = -1;   ///< restrict output to one ray id (-1 = all)

/// Is stage level @p lvl active for ray @p id?
inline bool on(int lvl, long id = -1) {
    return level >= lvl && (only < 0 || id < 0 || only == id);
}

/// Free-form tagged message (printf-style).
inline void msg(long id, const char* stage, const char* fmt, ...) {
    char body[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(body, sizeof body, fmt, ap);
    va_end(ap);
    std::printf("[dbg %7ld|%-14s] %s\n", id, stage, body);
}

/// One-line dump of a Ray's state (position, direction, energy, weight, flag).
inline void ray(long id, const char* stage, const Ray& r) {
    msg(id, stage, "pos=(%+.5f %+.5f %+.5f) dir=(%+.4f %+.4f %+.4f) E=%.3f keV prob=%.3e flag=%d",
        r.getStartX(), r.getStartY(), r.getStartZ(),
        r.getDirX(),   r.getDirY(),   r.getDirZ(),
        r.getEnergyKeV(), r.getProb(), (int)r.getIAFlag());
}

/// One-line dump of the voxel a ray is walking through.
inline void voxel(long id, const char* stage, int idx, const Voxel& v) {
    msg(id, stage, "voxel %d  x[%.4f,%.4f) y[%.4f,%.4f) z[%.4f,%.4f) mat=%d",
        idx, v.getX0(), v.getX1(), v.getY0(), v.getY1(), v.getZ0(), v.getZ1(),
        v.getMaterialIdx());
}

}  // namespace vtdbg
