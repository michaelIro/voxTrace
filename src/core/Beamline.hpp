#pragma once
/**
 * @file Beamline.hpp
 * @brief The configurable trace chain: geometry helpers + the two GPU-ready kernels.
 *
 * Everything a beamline simulation shares, independent of which stages are
 * mounted: double-precision vector geometry (Vec3, OpticFrame), per-ray RNG
 * streams, the voxel-grid walk (Grid, sampleInteraction, opticalDepthOut) and
 * the two Kokkos functors —
 *
 *   BeamKernel  source → [primary polycap] → focused/parallel exit ray
 *   ScanKernel  exit ray → sample → emission → [secondary polycap | detector
 *               aperture] → detector → detected Event
 *
 * Optional stages are runtime flags: `BeamKernel.hasOptic = false` emits the
 * bare parallel source beam; `ScanKernel.hasSecondary = false` aims emitted
 * photons at a bare detector aperture disk instead of the confocal optic.
 * `vr` switches importance sampling (aimed emission, multiplied weights)
 * against brute-force analog MC (isotropic emission, Bernoulli survival,
 * unit-weight counts). `usePol` toggles the scatter polarization weights.
 *
 * GPU-ready by construction (see Test-5 "Phase 1"): explicit functors (no
 * extended-lambda limits under nvcc), trivially copyable state captured by
 * value, device pointers from DeviceBuffer, KOKKOS_INLINE_FUNCTION helpers
 * with global-namespace math, device-safe debug via a value-captured
 * vtdbg::Ctx. Per-ray RNG streams make results identical at any thread count
 * on any backend.
 */

#include <cstdint>

#include "Platform.hpp"
#include "Ray.hpp"
#include "RNG.hpp"
#include "PolyCap.hpp"
#include "ChemElement.hpp"
#include "Material.hpp"
#include "Voxel.hpp"
#include "Sample.hpp"
#include "Detector.hpp"
#include "Debug.hpp"

namespace vt {

constexpr double PI_D  = 3.14159265358979323846;
constexpr double TWO_PI_D = 6.28318530717958647692;

// ── double-precision vector geometry ─────────────────────────────────────────

struct Vec3 {
    double x = 0, y = 0, z = 0;
    KOKKOS_INLINE_FUNCTION Vec3 operator+(Vec3 o) const { return {x+o.x, y+o.y, z+o.z}; }
    KOKKOS_INLINE_FUNCTION Vec3 operator-(Vec3 o) const { return {x-o.x, y-o.y, z-o.z}; }
    KOKKOS_INLINE_FUNCTION Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
};
KOKKOS_INLINE_FUNCTION double dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
KOKKOS_INLINE_FUNCTION Vec3   cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
KOKKOS_INLINE_FUNCTION Vec3   norm(Vec3 a) { double n = sqrt(dot(a,a)); return {a.x/n, a.y/n, a.z/n}; }

// Maps between the sample frame and one optic's frame (optic axis = +z); the
// optic-frame plane z = zFoc maps to the target point C in the sample frame.
// Constructed on the host, used (trivially copied) inside kernels.
struct OpticFrame {
    Vec3 C, axis, u, v;
    double zFoc;
    OpticFrame(Vec3 target, Vec3 ax, double z_focal) : C(target), axis(norm(ax)), zFoc(z_focal) {
        v = {0, 1, 0};                 // both optics tilt in the x-z plane
        u = norm(cross(v, axis));
        v = cross(axis, u);
    }
    KOKKOS_INLINE_FUNCTION void toOptic(Vec3 p, Vec3 d, Vec3& po, Vec3& doo) const {
        Vec3 r = p - C;
        po = {dot(r,u), dot(r,v), zFoc + dot(r,axis)};
        doo = {dot(d,u), dot(d,v), dot(d,axis)};
    }
    KOKKOS_INLINE_FUNCTION void toSample(Vec3 po, Vec3 doo, Vec3& p, Vec3& d) const {
        p = C + u*po.x + v*po.y + axis*(po.z - zFoc);
        d = u*doo.x + v*doo.y + axis*doo.z;
    }
};

KOKKOS_INLINE_FUNCTION Ray makeRay(Vec3 p, Vec3 d, double energy_keV) {
    Ray ray;
    ray.setStartCoordinates((float)p.x, (float)p.y, (float)p.z);
    ray.setEndCoordinates((float)d.x, (float)d.y, (float)d.z);
    ray.setSPol(0.f, 1.f, 0.f);          // PolyCap re-orthogonalises against the direction
    ray.setPPol(0.f, 0.f, 0.f);
    ray.setEnergyKeV((float)energy_keV);
    ray.setProb(1.f);
    ray.setIAFlag(false);
    return ray;
}

// ── deterministic per-ray RNG streams ────────────────────────────────────────
// Every (seed, index) pair gets its own xorshift64* state via splitmix64, so
// serial and parallel runs draw identical numbers for the same ray.

KOKKOS_INLINE_FUNCTION uint64_t splitmix(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
KOKKOS_INLINE_FUNCTION RNG makeRng(uint64_t seed, uint64_t idx) {
    return RNG(splitmix(seed ^ splitmix(idx)));
}

// Run body(i) for i in [0, n) — Kokkos::parallel_for in the Kokkos build
// (the functor is copied to the device), a plain loop in the host-only build.
template <class Body>
inline void forRange(const char* name, long n, const Body& body) {
#ifndef VOXTRACE_HOST_ONLY
    Kokkos::parallel_for(name, Kokkos::RangePolicy<>(0, n), body);
    Kokkos::fence();
#else
    (void)name;
    for (long i = 0; i < n; ++i) body(i);
#endif
}

// ── the sample data model: Sample grid + Voxel/Material/ChemElement arrays ────
// Inside kernels the pointers refer to device memory (DeviceBuffer::device()).
struct Grid {
    Sample sample;
    const Voxel*       voxels;
    const Material*    mats;
    const ChemElement* elems;
};

// Walk the ray from a sample-surface entry point through the voxel grid and
// sample the first interaction (optical depth ~ −ln U). Returns true and sets
// the interaction point @p P, its material index and its voxel index, or
// false if the ray escaped. @p dbg/@p dbgId drive the level-3 voxel-walk dump.
KOKKOS_INLINE_FUNCTION
bool sampleInteraction(const Grid& g, Vec3 entry, Vec3 dir, float E, double xi,
                       Vec3& P, int& matIdx, int& voxIdx,
                       const vtdbg::Ctx& dbg, long dbgId) {
    Ray r = makeRay(entry, dir, E);
    int vox = g.sample.getVoxelIdx((float)entry.x, (float)entry.y, (float)entry.z);
    double tau = -log(xi > 0 ? xi : 1e-30), acc = 0;
    while (vox >= 0) {
        const Voxel& v = g.voxels[vox];
        if (dbg.on(3, dbgId)) vtdbg::voxel(dbgId, "walk-in", vox, v);
        double len = v.intersect(r);                                   // path through voxel [cm]
        double mu  = g.mats[v.getMaterialIdx()].CS_Tot_Lin(E, g.elems); // 1/cm
        if (acc + mu*len >= tau) {
            double l = r.getTIn() + (tau - acc) / mu;                  // dist from entry
            P = entry + dir * l;
            matIdx = v.getMaterialIdx();
            voxIdx = vox;
            return true;
        }
        acc += mu * len;
        vox = v.getNN(r.getNextVoxel());
    }
    return false;                                                       // passed through
}

// Total optical depth from @p P along @p dir to the edge of the sample (the
// emitted photon's self-absorption on the way out), walking the voxel grid.
KOKKOS_INLINE_FUNCTION
double opticalDepthOut(const Grid& g, Vec3 P, Vec3 dir, float E) {
    Ray r = makeRay(P, dir, E);
    int vox = g.sample.getVoxelIdx((float)P.x, (float)P.y, (float)P.z);
    double total = 0;
    while (vox >= 0) {
        const Voxel& v = g.voxels[vox];
        double len = v.intersect(r);
        total += g.mats[v.getMaterialIdx()].CS_Tot_Lin(E, g.elems) * len;
        vox = v.getNN(r.getNextVoxel());
    }
    return total;
}

struct ExitRay { Vec3 pos, dir; double w; };     // beam ray leaving the source stage (optic frame)

// One detected photon: scan position, voxel it was emitted from, detector
// channel it was recorded in, and its Monte-Carlo weight.
struct Event { int pos; int vox; int ch; float w; };

// ── beam kernel: one source photon through the (optional) primary optic ──────
// Writes the exit ray — or a w = −1 sentinel — into its slot. Without the
// optic the source emits its bare parallel beam (unit weight, along the axis).
struct BeamKernel {
    PolyCap    primary;     // ignored when !hasOptic
    bool       hasOptic;
    double     srcR;
    double     energy;
    uint64_t   seed;
    long       base;        // chunk offset (slot i ↔ primary photon base+i)
    ExitRay*   slots;       // device
    vtdbg::Ctx dbg;

    KOKKOS_INLINE_FUNCTION void operator()(long i) const {
        slots[i].w = -1.0;
        RNG rng = makeRng(seed, (uint64_t)(base + i));
        double rr = sqrtf(rng.frand()) * srcR, ra = TWO_PI_D * rng.frand();
        Vec3 p0 = {rr*cos(ra), rr*sin(ra), 0};
        if (!hasOptic) {
            slots[i] = {p0, {0, 0, 1}, 1.0};
            return;
        }
        Ray p = makeRay(p0, {0,0,1}, energy);
        primary.trace(p);
        if (p.getIAFlag()) {
            slots[i] = {{p.getStartX(), p.getStartY(), p.getStartZ()},
                        {p.getDirX(),   p.getDirY(),   p.getDirZ()}, p.getProb()};
            if (dbg.on(2, base + i)) vtdbg::ray(base + i, "beam-exit", p);
        }
    }
};

// ── scan kernel: one (scan position, beam ray) sample of the full chain ──────
// surface entry → voxel walk → emission → self-absorption → [secondary optic
// or bare detector aperture] → detector. vr=true aims the emitted photon at
// the collection window and multiplies importance weights; vr=false emits
// isotropically and replaces every weight by a Bernoulli survival draw, so
// each detected count is one real photon. Writes the detected Event — or a
// ch = −1 sentinel — into its slot.
struct ScanKernel {
    Grid        grid;           // device pointers
    PolyCap     secondary;      // ignored when !hasSecondary
    bool        hasSecondary;
    Detector    detector;
    const ChemElement* detElems;   // device
    const ExitRay*     beam;       // device
    Event*             slots;      // device
    OpticFrame  primFrame, secFrame;
    Vec3        aimCtr, Epol, d_sec;   // collection window centre (optic or aperture)
    double      energy, rWin, eBin;    // rWin = window/aperture radius
    double      x0, y0, z0, LX, LY;
    int         nBin, di;
    bool        vr, usePol;
    uint64_t    scanSeed;
    long        beamN;
    vtdbg::Ctx  dbg;

    KOKKOS_INLINE_FUNCTION void operator()(long bi) const {
        slots[bi] = Event{0, 0, -1, 0.f};
        RNG rng = makeRng(scanSeed, (uint64_t)di*beamN + bi);
        const ExitRay& er = beam[bi];
        double wBeam = er.w;
        if (!vr) {                                            // analog: Bernoulli beam weight
            if (rng.frand() >= wBeam) return;
            wBeam = 1.0;
        }

        // (1) beam exit ray → sample frame → surface entry
        Vec3 ps, ds;
        primFrame.toSample(er.pos, er.dir, ps, ds);
        if (ds.z <= 0) return;
        double t = (z0 - ps.z) / ds.z;
        if (t < 0) return;
        Vec3 entry = ps + ds * t;
        if (entry.x <= x0 || entry.x >= x0 + LX ||
            entry.y <= y0 || entry.y >= y0 + LY) return;
        if (dbg.on(2, bi))
            VT_DBG(bi, "surface-entry", "pos %d: (%.5f %.5f) dir=(%+.4f %+.4f %+.4f)",
                   di, entry.x, entry.y, ds.x, ds.y, ds.z);

        // (2) walk the voxel grid to the first interaction point
        Vec3 P; int matIdx, voxIdx;
        if (!sampleInteraction(grid, entry, ds, (float)energy, rng.frand(),
                               P, matIdx, voxIdx, dbg, bi)) return;
        const Material& mat = grid.mats[matIdx];
        if (dbg.on(2, bi))
            VT_DBG(bi, "interaction", "pos %d: P=(%.5f %.5f %.5f) voxel %d",
                   di, P.x, P.y, P.z, voxIdx);

        // (3) emission direction: aimed at the collection window (importance
        //     sampling) or isotropic (brute force)
        Vec3 eDir;
        double wEmit = 1.0;
        if (vr) {
            double ar = sqrtf(rng.frand()) * rWin, aa = TWO_PI_D * rng.frand();
            Vec3   aim  = aimCtr + secFrame.u*(ar*cos(aa)) + secFrame.v*(ar*sin(aa));
            eDir = aim - P;
            double r2 = dot(eDir, eDir);
            eDir = norm(eDir);
            wEmit = (PI_D * rWin * rWin * fabs(dot(eDir, d_sec))) / (4.0*PI_D*r2);
        } else {
            double cth = 2.0*rng.frand() - 1.0, phi = TWO_PI_D * rng.frand();
            double sth = sqrt(fmax(0.0, 1.0 - cth*cth));
            eDir = {sth*cos(phi), sth*sin(phi), cth};
        }
        if (eDir.z >= 0) return;                              // must travel out (−z)

        // (4) the interaction: element, channel, emitted energy (scatter
        //     carries the polarization-dependent azimuthal weight if enabled)
        int ei   = mat.getInteractingElementIdx((float)energy, rng.frand(), grid.elems);
        const ChemElement& el = grid.elems[ei];
        int type = el.getInteractionType((float)energy, rng.frand());
        double cth = fmax(-1.0, fmin(1.0, dot(ds, eDir)));    // cos(scatter angle)
        double th  = acos(cth);
        double Ef, wPol = 1.0;
        if (type == 0) {                                      // photoelectric → fluorescence
            int shell = el.getExcitedShell((float)energy, rng.frand());
            if (rng.frand() >= el.Fluor_Y(shell)) return;     // Auger
            Ef = el.Line_Energy(el.getTransition(shell, rng.frand()));
        } else {                                              // scatter
            Ef = (type == 1) ? energy
                             : el.getComptEnergy((float)energy, (float)th);
            if (usePol) {
                Vec3   sperp = eDir - ds*cth;
                Vec3   eperp = Epol - ds*dot(Epol, ds);
                double sl = sqrt(dot(sperp,sperp)), el2 = sqrt(dot(eperp,eperp));
                double cphi = (sl > 1e-12 && el2 > 1e-12) ? dot(sperp,eperp)/(sl*el2) : 1.0;
                double phi  = acos(fmax(-1.0, fmin(1.0, cphi)));
                wPol = (type == 1) ? el.polFactorRayl((float)th, (float)phi)
                                   : el.polFactorCompt((float)energy, (float)th, (float)phi);
            }
        }
        if (Ef < 0.8) return;
        if (dbg.on(2, bi))
            VT_DBG(bi, "emission", "pos %d: type=%d Z=%d Ef=%.3f keV wPol=%.3f",
                   di, type, grid.elems[ei].Z(), Ef, wPol);

        // (5) self-absorption on the way out (weight or Bernoulli survival)
        double tauOut = opticalDepthOut(grid, P, eDir, (float)Ef);
        double wSelf = 1.0;
        if (vr) wSelf = exp(-tauOut);
        else if (rng.frand() >= exp(-tauOut)) return;

        // (6) collection: secondary polycap (confocal selection) or bare
        //     detector aperture (importance aim needs no further check; brute
        //     force must hit the aperture disk geometrically)
        double wSec = 1.0;
        if (hasSecondary) {
            Vec3 po, doo;
            secFrame.toOptic(P, eDir, po, doo);
            Ray sray = makeRay(po, doo, Ef);
            secondary.trace(sray);
            if (!sray.getIAFlag()) return;
            if (vr) wSec = sray.getProb();
            else if (rng.frand() >= sray.getProb()) return;
            if (dbg.on(2, bi)) vtdbg::ray(bi, "secondary-exit", sray);
        } else if (!vr) {
            double tn = dot(eDir, d_sec);
            if (tn <= 0) return;
            double tt = dot(aimCtr - P, d_sec) / tn;
            if (tt <= 0) return;
            Vec3 miss = P + eDir*tt - aimCtr;
            if (dot(miss, miss) > rWin*rWin) return;
        }

        // (7) Si(Li) detector response → measured channel + event record
        float wDet;
        float Emeas = detector.detect((float)Ef, rng, detElems, wDet);
        if (wDet <= 0.f) return;
        if (!vr) { if (rng.frand() >= wDet) return; wDet = 1.f; }

        double w = wBeam * wEmit * wPol * wSelf * wSec * wDet;
        int b = (int)(Emeas / eBin);
        if (b >= 0 && b < nBin) {
            slots[bi] = Event{di, voxIdx, b, (float)w};
            if (dbg.on(1, bi))
                VT_DBG(bi, "detected", "pos %d: E=%.3f keV ch=%d w=%.3e voxel %d",
                       di, (double)Emeas, b, w, voxIdx);
        }
    }
};

}  // namespace vt
