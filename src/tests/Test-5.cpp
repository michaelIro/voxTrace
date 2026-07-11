// Test-5: confocal micro-XRF voxel-weight reconstruction of NIST-1107 brass
//
//   source → primary polycap → SAMPLE (voxel grid) → secondary polycap →
//   Si(Li) detector → Spectrum → SpectrumLoss(χ² | weighted χ²) → ensmallen
//
// Test-3 runs the forward problem (spectra from a known sample); Test-5 runs
// the INVERSE problem on the same ray-trace chain. Every voxel of the sample
// carries an emission weight w_v (a local density / concentration scaling).
// A confocal scan is traced like Test-3, but each detected photon also
// records the voxel it was emitted from, which turns the trace into a sparse
// linear forward model (ResponseMatrix, one per scan position):
//
//     S_ch(w) = B_ch + Σ_v w_v R[v][ch]
//
// (weights scale emission only; attenuation stays at the nominal composition —
// the standard first-order linearisation of confocal-XRF reconstruction).
// The "measured" spectra come from a second, independently sampled trace of a
// GROUND-TRUTH weight field (the phantom in Setup.txt), scaled to an exposure
// tied to the Monte-Carlo statistics and Poisson-noised. A SpectrumLoss
// (plain χ² or ROI-weighted χ², evaluated by Spectrum::chiSquare*) compares
// model and measurement, and ensmallen's L-BFGS (via OptimizerAPI) minimises
// the total loss over the per-voxel weights with the exact analytic gradient.
// Voxels with too few events to constrain a parameter are frozen at w = 1 and
// contribute the fixed baseline B. The optimizer runs over x with w = x²
// (weights are physical, ≥ 0) plus a mild Tikhonov pull λ Σ m_v (w_v − 1)²
// that resolves laterally degenerate voxels.
//
// EVERYTHING about the experiment is read from a simulation directory
// (test-data/simulation/nist-1107-recon by default): Polycapillary.txt,
// Source.txt, Capillaries.txt, Sample.txt, Materials.txt, Simulation.txt and
// Setup.txt (statistics, loss/ROIs, optimizer, physics switches, phantom).
// Any Setup.txt key can be overridden on the command line:
//
//   ./build/src/Test5 [sim-dir] [key=value ...]
//   ./build/src/Test5 test-data/simulation/nist-1107-recon \
//        n_primary=2000000 loss=1 polarization=0 variance_reduction=0 debug=2
//
// Switches: polarization=0/1 toggles the scatter polarization weights;
// variance_reduction=1 aims each emitted photon at the secondary optic and
// carries importance weights, =0 is brute-force analog MC (isotropic
// emission, Bernoulli survival, unit-weight counts — every ray is calculated
// until it is thrown away; expect ~500× fewer detected events per primary).
// Diagnostics: debug=1..3 (+debug_ray=I) prints stage-tagged ray/voxel state
// (Debug.hpp); profile=1 prints phase timings, counters, CPU/memory usage
// (Profiler.hpp).
//
// GPU-READY DISPATCH (Phase 1): the two hot loops are explicit Kokkos functors
// (BeamKernel, ScanKernel) — not lambdas, so nvcc's extended-lambda limits
// never apply — that run the ENTIRE per-photon chain (primary optic, voxel
// walk, emission, self-absorption, secondary optic, detector) in device code:
// every helper on that path is KOKKOS_INLINE_FUNCTION with global-namespace
// math, all shared arrays travel through DeviceBuffer (Kokkos::View + host
// mirror; plain vector in host-only builds), functors capture trivially
// copyable state by value, and debug output uses device-safe printf via a
// value-captured vtdbg::Ctx. On the local OpenMP backend the deep copies are
// no-ops; under a CUDA/HIP Kokkos install the same source moves the buffers
// to the GPU and launches there (PolyCap keeps its double trace state —
// native on those GPUs). Per-ray RNG streams make results identical at any
// thread count. Only the fit (ensmallen) and file IO stay host-side.
//
// Build: make test5           (host-only, serial)
//        make test5-kokkos    (Kokkos build → build/src/Test5k)
// Plot:  python3 src/tests/plot_recon.py
//
// The per-photon chain (steps 1-7 below) is identical to Test-3.

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "Ray.hpp"
#include "PolyCap.hpp"
#include "ChemElement.hpp"
#include "Material.hpp"
#include "Voxel.hpp"
#include "Sample.hpp"
#include "Detector.hpp"
#include "Spectrum.hpp"
#include "SpectrumLoss.hpp"
#include "ResponseMatrix.hpp"
#include "Debug.hpp"
#include "DeviceBuffer.hpp"
#include "Profiler.hpp"
#include "io/SetupIO.hpp"
#include "../api/OptimizerAPI.hpp"

namespace {

constexpr double VT_PI_D = 3.14159265358979323846;
constexpr double VT_2PI_D = 6.28318530717958647692;

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
// optic-frame plane z = zFoc maps to the confocal point C in the sample frame.
// Constructed on the host, used (trivially copied) inside kernels.
struct OpticFrame {
    Vec3 C, axis, u, v;
    double zFoc;
    OpticFrame(Vec3 confocal, Vec3 ax, double z_focal) : C(confocal), axis(norm(ax)), zFoc(z_focal) {
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

// Deterministic per-ray RNG stream: every (seed, index) pair gets its own
// xorshift64* state via splitmix64, so serial and parallel runs — at any
// thread count, on any backend — draw identical numbers for the same ray.
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
void forRange(const char* name, long n, const Body& body) {
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

struct ExitRay { Vec3 pos, dir; double w; };     // primary-optic exit ray (optic frame)

// One detected photon: scan position, voxel it was emitted from, detector
// channel it was recorded in, and its Monte-Carlo weight.
struct Event { int pos; int vox; int ch; float w; };

// ── beam kernel: one source photon through the primary optic ─────────────────
// Writes the focused exit ray — or a w = −1 sentinel — into its slot.
struct BeamKernel {
    PolyCap    primary;
    double     srcR;
    double     energy;
    uint64_t   seed;
    long       base;        // chunk offset (slot i ↔ primary photon base+i)
    ExitRay*   slots;       // device
    vtdbg::Ctx dbg;

    KOKKOS_INLINE_FUNCTION void operator()(long i) const {
        slots[i].w = -1.0;
        RNG rng = makeRng(seed, (uint64_t)(base + i));
        double rr = sqrtf(rng.frand()) * srcR, ra = VT_2PI_D * rng.frand();
        Ray p = makeRay({rr*cos(ra), rr*sin(ra), 0}, {0,0,1}, energy);
        primary.trace(p);
        if (p.getIAFlag()) {
            slots[i] = {{p.getStartX(), p.getStartY(), p.getStartZ()},
                        {p.getDirX(),   p.getDirY(),   p.getDirZ()}, p.getProb()};
            if (dbg.on(2, base + i)) vtdbg::ray(base + i, "beam-exit", p);
        }
    }
};

// ── scan kernel: one (scan position, beam ray) sample of the full chain ──────
// Steps (1)-(7) are identical to Test-3, with two switchable behaviours:
// vr=true aims the emitted photon at the secondary window and multiplies
// importance weights; vr=false emits isotropically and replaces every weight
// by a Bernoulli survival draw, so each detected count is one real photon.
// Writes the detected Event — or a ch = −1 sentinel — into its slot.
struct ScanKernel {
    Grid        grid;           // device pointers
    PolyCap     secondary;
    Detector    detector;
    const ChemElement* detElems;   // device
    const ExitRay*     beam;       // device
    Event*             slots;      // device
    OpticFrame  primFrame, secFrame;
    Vec3        secWinCtr, Epol, d_sec;
    double      energy, rWin, eBin;
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

        // (1) primary exit ray → sample frame → surface entry
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

        // (3) emission direction: aimed at the secondary window (importance
        //     sampling) or isotropic (brute force)
        Vec3 eDir;
        double wEmit = 1.0;
        if (vr) {
            double ar = sqrtf(rng.frand()) * rWin, aa = VT_2PI_D * rng.frand();
            Vec3   aim  = secWinCtr + secFrame.u*(ar*cos(aa)) + secFrame.v*(ar*sin(aa));
            eDir = aim - P;
            double r2 = dot(eDir, eDir);
            eDir = norm(eDir);
            wEmit = (VT_PI_D * rWin * rWin * fabs(dot(eDir, d_sec))) / (4.0*VT_PI_D*r2);
        } else {
            double cth = 2.0*rng.frand() - 1.0, phi = VT_2PI_D * rng.frand();
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

        // (6) secondary polycap — only confocal-volume photons survive
        Vec3 po, doo;
        secFrame.toOptic(P, eDir, po, doo);
        Ray sray = makeRay(po, doo, Ef);
        secondary.trace(sray);
        if (!sray.getIAFlag()) return;
        double wSec = 1.0;
        if (vr) wSec = sray.getProb();
        else if (rng.frand() >= sray.getProb()) return;
        if (dbg.on(2, bi)) vtdbg::ray(bi, "secondary-exit", sray);

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

}  // namespace

int main(int argc, char* argv[]) {
#ifndef VOXTRACE_HOST_ONLY
    Kokkos::ScopeGuard kokkos(argc, argv);
#endif
    // ── configuration: simulation directory + key=value overrides ─────────────
    std::string dir = (argc > 1 && argv[1][0] != '-' && !std::strchr(argv[1], '='))
                          ? argv[1] : "test-data/simulation/nist-1107-recon";
    vtio::PolyCapDescr pc;
    vtio::SourceDescr  so;
    vtio::BeamDescr    bm;
    vtio::SampleDescr  sd;
    vtio::ScanDescr    sc;
    vtio::SetupDescr   cfg;
    try {
        VT_PROFILE("load-config");
        pc  = vtio::loadPolyCap(dir + "/Polycapillary.txt");
        so  = vtio::loadSource (dir + "/Source.txt");
        bm  = vtio::loadBeam   (dir + "/Capillaries.txt");
        sd  = vtio::loadSample (dir);
        sc  = vtio::loadScan   (dir + "/Simulation.txt");
        cfg = vtio::loadSetup  (dir + "/Setup.txt");
        for (int a = 1; a < argc; ++a)
            if (std::strchr(argv[a], '=')) vtio::applyOverride(cfg, argv[a]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    vtdbg::cfg.level = cfg.debug;
    vtdbg::cfg.only  = cfg.debugRay;
    std::filesystem::create_directories("test-data/out");

    const int    N_POS = (int)sc.points.size();
    const int    NBIN  = cfg.nBin;
    const double EBIN  = cfg.eBin;
    const bool   VR    = cfg.varianceReduction;

    // ── optics: Polycapillary.txt is the primary orientation; the secondary is
    //    the same optic mounted the other way round ──────────────────────────
    PolyCap primary   = pc.build();
    PolyCap secondary = pc.reversed().build();
    const double R_WIN = pc.rExtDown;               // secondary entrance = primary exit radius
    const double FOCAL = pc.focalDown;              // confocal working distance

    // ── sample: voxel grid + materials over the union of elements ────────────
    std::vector<int> uz;                             // union of element Z's
    for (const auto& zs : sd.matZ)
        for (int z : zs)
            if (std::find(uz.begin(), uz.end(), z) == uz.end()) uz.push_back(z);
    if ((int)uz.size() > MAX_ELEMENTS) {
        std::fprintf(stderr, "Materials use %zu elements; Material supports %d\n",
                     uz.size(), MAX_ELEMENTS);
        return 1;
    }
    std::vector<ChemElement> elems;
    for (int z : uz) elems.emplace_back(z);
    std::vector<Material> mats;
    for (size_t m = 0; m < sd.matZ.size(); ++m) {
        float w[MAX_ELEMENTS] = {};
        for (size_t i = 0; i < sd.matZ[m].size(); ++i)
            w[std::find(uz.begin(), uz.end(), sd.matZ[m][i]) - uz.begin()] = sd.matW[m][i];
        mats.emplace_back((int)uz.size(), w, elems.data());
    }

    const int xN = sd.xN, yN = sd.yN, zN = sd.zN;
    const double x0 = sd.x0, y0 = sd.y0, z0 = sd.z0;  // surface = plane z = z0
    const double VOX = sd.vz;
    std::vector<Voxel> voxels((size_t)xN*yN*zN);
    for (int i = 0; i < xN; ++i)
    for (int j = 0; j < yN; ++j)
    for (int k = 0; k < zN; ++k) {
        size_t idx = (size_t)i*yN*zN + j*zN + k;
        voxels[idx] = Voxel((float)(x0+i*sd.vx), (float)(y0+j*sd.vy), (float)(z0+k*sd.vz),
                            (float)sd.vx, (float)sd.vy, (float)sd.vz, sd.matOf(idx));
    }
    for (int i = 0; i < xN; ++i)
    for (int j = 0; j < yN; ++j)
    for (int k = 0; k < zN; ++k) {
        int nn[27], c = 0;                                                // 27-neighbour table
        for (int l = -1; l < 2; ++l) for (int m = -1; m < 2; ++m) for (int n = -1; n < 2; ++n) {
            int ni=i+n, nj=j+m, nk=k+l;
            nn[c++] = (ni<0||ni>=xN||nj<0||nj>=yN||nk<0||nk>=zN) ? -1 : ni*yN*zN + nj*zN + nk;
        }
        voxels[(size_t)i*yN*zN + j*zN + k].setNN(nn);
    }

    // ── Si(Li) detector: a Si crystal + Be window, reusing ChemElement physics ─
    std::vector<ChemElement> detElems{ ChemElement(14), ChemElement(4) };   // Si, Be
    Detector detector = Detector::make(/*siIdx*/0, /*beIdx*/1, /*thickness*/0.30f,
                                       /*beWin*/0.0025f, /*deadLayer*/1e-4f,
                                       /*fano*/0.114f, /*noiseFWHM*/0.080f);

    // ── shared arrays → device (no-op copies on host backends) ───────────────
    DeviceBuffer<Voxel>       voxBuf ("voxels",       voxels);
    DeviceBuffer<Material>    matBuf ("materials",    mats);
    DeviceBuffer<ChemElement> elemBuf("elements",     elems);
    DeviceBuffer<ChemElement> detBuf ("det-elements", detElems);
    Grid grid{ Sample((float)x0,(float)y0,(float)z0, (float)sd.LX,(float)sd.LY,(float)sd.LZ,
                      (float)sd.vx,(float)sd.vy,(float)sd.vz, xN,yN,zN),
               voxBuf.device(), matBuf.device(), elemBuf.device() };

    // decoded voxel-centre depth (grid layout: vox = i*yN*zN + j*zN + k)
    auto zOfVox = [&](int vox) { return z0 + ((vox % zN) + 0.5) * VOX; };

    // ── ground truth: the phantom layers from Setup.txt (later lines win) ────
    auto wTrueOfZ = [&](double z) {
        double w = 1.0;
        for (const auto& L : cfg.truth)
            if (z >= L.z0um*1e-4 && z < L.z1um*1e-4) w = L.w;
        return w;
    };

    // ── confocal geometry (Capillaries.txt: bench position + tilt angle) ─────
    const double A = bm.angleDeg * VT_PI_D / 180.0;
    const Vec3 d_prim = norm({ std::cos(A), 0,  std::sin(A)});  // into the sample (+z)
    const Vec3 d_sec  = norm({ std::cos(A), 0, -std::sin(A)});  // out of the sample (−z)
    const Vec3 C0     = {bm.posX, bm.posY, 0};                  // scan origin on the surface
    const Vec3 srcPol = {1, 0, 0};       // source E-field in the optic plane (u-axis)
    const bool usePol = cfg.polarization && so.polFactor > 0.0;

    std::printf("NIST-1107 brass — confocal voxel-weight reconstruction\n");
    std::printf("  config           = %s\n", dir.c_str());
    std::printf("  sample           = %dx%dx%d voxels (%zu materials, %zu elements)\n",
                xN, yN, zN, mats.size(), elems.size());
    std::printf("  scan positions   = %d (%.0f…%.0f µm below surface)   E = %.1f keV\n",
                N_POS, sc.points.front()[2]*1e4, sc.points.back()[2]*1e4, bm.energyKeV);
    std::printf("  switches         = polarization %s | %s\n",
                usePol ? "ON" : "OFF",
                VR ? "importance sampling (variance reduction)" : "BRUTE FORCE analog MC");

    // ── trace the primary beam through the primary optic ONCE; the focused beam
    //    is position-independent, so the stored exit rays are reused everywhere ─
    const double srcR = std::min(pc.rExtUp, so.radiusX);
    std::vector<ExitRay> beam;
    {
        VT_PROFILE("beam-trace");
        const long CHUNK = 2000000;
        DeviceBuffer<ExitRay> chunkBuf("beam-chunk", (size_t)std::min(CHUNK, (long)cfg.nPrimary));
        for (long base = 0; base < cfg.nPrimary; base += CHUNK) {
            long n = std::min(CHUNK, cfg.nPrimary - base);
            BeamKernel bk{
                .primary = primary, .srcR = srcR, .energy = bm.energyKeV,
                .seed = cfg.seed, .base = base, .slots = chunkBuf.device(),
                .dbg = vtdbg::cfg,
            };
            forRange("voxTrace::beam", n, bk);
            chunkBuf.toHost();
            const ExitRay* er = chunkBuf.host();
            for (long i = 0; i < n; ++i)
                if (er[i].w >= 0) beam.push_back(er[i]);
        }
    }
    const long beamN = (long)beam.size();
    vtprof::count("primaries", cfg.nPrimary);
    vtprof::count("beam-rays", beamN);
    std::printf("  primary photons  = %ld   transmitted = %ld (%.2f%%)\n",
                cfg.nPrimary, beamN, 100.0*beamN/cfg.nPrimary);
    if (beamN == 0) { std::printf("No beam — check the optic descriptor.\n"); return 1; }
    DeviceBuffer<ExitRay> beamBuf("beam", beam);

    // ── one full confocal scan → list of detected-photon events ───────────────
    // Per position, ScanKernel runs the whole chain for every beam ray in
    // parallel; the host compacts the slot array into the event list.
    DeviceBuffer<Event> slotsBuf("event-slots", (size_t)beamN);
    auto scanTrace = [&](uint64_t scanSeed, const char* phaseTag, const char* countTag) {
        VT_PROFILE(phaseTag);
        std::vector<Event> events;
        for (int di = 0; di < N_POS; ++di) {
            Vec3 C = C0 + Vec3{sc.points[di][0], sc.points[di][1], sc.points[di][2]};
            OpticFrame primFrame(C, d_prim,  pc.length + FOCAL);
            OpticFrame secFrame (C, d_sec,  -FOCAL);
            Vec3 secWinCtr = C + d_sec * FOCAL;                       // secondary entrance centre
            Vec3 Epol = primFrame.u*srcPol.x + primFrame.v*srcPol.y + primFrame.axis*srcPol.z;

            ScanKernel k{
                .grid = grid, .secondary = secondary, .detector = detector,
                .detElems = detBuf.device(), .beam = beamBuf.device(), .slots = slotsBuf.device(),
                .primFrame = primFrame, .secFrame = secFrame,
                .secWinCtr = secWinCtr, .Epol = Epol, .d_sec = d_sec,
                .energy = bm.energyKeV, .rWin = R_WIN, .eBin = EBIN,
                .x0 = x0, .y0 = y0, .z0 = z0, .LX = sd.LX, .LY = sd.LY,
                .nBin = NBIN, .di = di, .vr = VR, .usePol = usePol,
                .scanSeed = scanSeed, .beamN = beamN, .dbg = vtdbg::cfg,
            };
            forRange("voxTrace::scan", beamN, k);
            slotsBuf.toHost();
            const Event* se = slotsBuf.host();
            for (long bi = 0; bi < beamN; ++bi)
                if (se[bi].ch >= 0) events.push_back(se[bi]);
        }
        vtprof::count(countTag, (long)events.size());
        return events;
    };

    // ── two independently sampled scans: the "measurement" and the model ──────
    std::vector<Event> evMeas = scanTrace(cfg.seed ^ 0x6D656173ULL, "scan-measurement", "events-measurement");
    std::vector<Event> evFit  = scanTrace(cfg.seed ^ 0x7265636FULL, "scan-response",    "events-response");
    std::printf("  detected events  = %zu (measurement) / %zu (response)\n",
                evMeas.size(), evFit.size());
    if (evMeas.empty() || evFit.empty()) {
        std::printf("Too few detected events — increase n_primary.\n");
        return 1;
    }

    // ── measured spectra: phantom weight field, exposure scaling, Poisson noise ─
    vtprof::Phase phase("build-model");
    std::vector<std::vector<double>> raw(N_POS, std::vector<double>(NBIN, 0.0));
    double rawTotal = 0;
    for (const Event& e : evMeas) {
        double w = e.w * wTrueOfZ(zOfVox(e.vox));
        raw[e.pos][e.ch] += w;
        rawTotal += w;
    }
    const double expo = cfg.expoPerEvent * evMeas.size() / rawTotal;   // live-time / flux factor
    std::mt19937_64 noiseRng(cfg.seed ^ 0x6E6F697365ULL);
    std::vector<Spectrum> meas;
    meas.reserve(N_POS);
    long nChan = 0;                                                    // channels entering the χ²
    for (int d = 0; d < N_POS; ++d) {
        meas.emplace_back(NBIN, 0.f, (float)EBIN);
        for (int ch = 0; ch < NBIN; ++ch) {
            double mu = expo * raw[d][ch];
            if (mu <= 0) continue;
            long n = std::poisson_distribution<long>(mu)(noiseRng);
            if (n > 0) { meas[d].setCount(ch, (float)n); ++nChan; }
        }
    }

    // ── voxel parameters: only voxels with enough events are fitted ───────────
    std::unordered_map<int, int> evCount;
    for (const Event& e : evFit) ++evCount[e.vox];
    std::unordered_map<int, int> paramOf;
    std::vector<int> voxOfParam;
    for (const Event& e : evFit)
        if (evCount[e.vox] >= cfg.minEvents && !paramOf.count(e.vox)) {
            paramOf[e.vox] = (int)voxOfParam.size();
            voxOfParam.push_back(e.vox);
        }
    const int nP = (int)voxOfParam.size();
    if (nP == 0) {
        std::printf("No voxel reached %d events — increase n_primary.\n", cfg.minEvents);
        return 1;
    }
    vtprof::count("parameters", nP);

    // ── response matrices (per scan position), frozen voxels → baseline ───────
    std::vector<ResponseMatrix> R;
    R.reserve(N_POS);
    for (int d = 0; d < N_POS; ++d) R.emplace_back(nP, NBIN);
    double massFit = 0, massFrozen = 0;
    for (const Event& e : evFit) {
        auto it = paramOf.find(e.vox);
        if (it != paramOf.end()) { R[e.pos].add(it->second, e.ch, e.w); massFit    += e.w; }
        else                     { R[e.pos].addBaseline(e.ch, e.w);     massFrozen += e.w; }
    }
    size_t nnz = 0;
    for (int d = 0; d < N_POS; ++d) { R[d].finalize(); R[d].scale(expo); nnz += R[d].nnz(); }
    std::vector<double> mass(nP, 0.0);                                 // response mass per parameter
    for (int d = 0; d < N_POS; ++d) R[d].addParamMass(mass);
    std::printf("  measured counts  = %.0f over the scan (×%.1f per event)\n",
                expo * rawTotal, cfg.expoPerEvent);
    std::printf("  parameters       = %d voxels (≥%d events)   frozen voxels → baseline (%.1f%% of response)\n",
                nP, cfg.minEvents, 100.0*massFrozen/(massFit+massFrozen));
    std::printf("  response matrix  = %zu non-zeros over %d positions   exposure ×%.3e\n",
                nnz, N_POS, expo);

    // ── the loss: plain χ² or ROI-weighted χ² that emphasises chosen lines ────
    SpectrumLoss loss(cfg.lossMode == 1 ? SpectrumLoss::CHI2_WEIGHTED : SpectrumLoss::CHI2);
    if (cfg.lossMode == 1) {
        std::printf("  loss = weighted χ² with ROIs:");
        for (const auto& L : cfg.rois) {
            double half = std::fmax(0.10, 2.5 * detector.resolutionSigma((float)L.energyKeV));
            int c0 = std::max(0,        (int)((L.energyKeV - half) / EBIN));
            int c1 = std::min(NBIN - 1, (int)((L.energyKeV + half) / EBIN));
            loss.addROI(c0, c1, L.weight);
            std::printf("  %s ×%.0f [%.2f–%.2f keV]", L.name.c_str(), L.weight, c0*EBIN, (c1+1)*EBIN);
        }
        std::printf("\n");
    } else {
        std::printf("  loss = plain χ² (run with loss=1 for ROI-weighted)\n");
    }
    std::printf("  regularization   = λ %.3g × Σ m_v (w_v − 1)²\n", cfg.lambda);

    // ── objective for ensmallen: total loss + exact gradient (chain rule) ─────
    // The optimizer runs over x with w_v = x_v² — voxel weights are physical
    // emission scalings, so they must stay ≥ 0. Without this, laterally
    // degenerate voxels (indistinguishable in a confocal scan) can cancel each
    // other with huge ± weights; with it, the identifiable quantity — the
    // response-weighted layer average — stays meaningful.
    //
    // A mild Tikhonov term  λ Σ_v m_v (w_v − 1)²  (m_v = response mass) breaks
    // the remaining lateral degeneracy: the data fix each layer's response-
    // weighted SUM, the penalty distributes it evenly across the layer's
    // voxels, and weakly-observed voxels shrink towards the nominal w = 1
    // instead of chasing noise. λ = 0 turns the fit into the pure χ².
    const double lambda = cfg.lambda;
    Spectrum sim(NBIN, 0.f, (float)EBIN);
    std::vector<double> wbuf(nP), gw(nP);
    ObjGradFn objGrad = [&](const std::vector<double>& x, std::vector<double>& g) {
        for (int p = 0; p < nP; ++p) wbuf[p] = x[p]*x[p];
        std::fill(gw.begin(), gw.end(), 0.0);
        double L = 0;
        for (int d = 0; d < N_POS; ++d) {
            R[d].assemble(wbuf, sim);
            L += loss.value(sim, meas[d]);
            R[d].accumulateGrad(loss, sim, meas[d], gw);
        }
        for (int p = 0; p < nP; ++p) {
            L     += lambda * mass[p] * (wbuf[p] - 1.0) * (wbuf[p] - 1.0);
            gw[p] += 2.0 * lambda * mass[p] * (wbuf[p] - 1.0);
        }
        for (int p = 0; p < nP; ++p) g[p] = 2.0 * x[p] * gw[p];       // dL/dx = 2x·dL/dw
        return L;
    };
    std::vector<double> gscratch(nP);
    auto objValue = [&](const std::vector<double>& x) { return objGrad(x, gscratch); };

    // gradient sanity: analytic vs central finite difference on the 3 strongest voxels
    std::vector<double> x1(nP, 1.0), xT(nP), wT(nP);
    for (int p = 0; p < nP; ++p) {
        wT[p] = wTrueOfZ(zOfVox(voxOfParam[p]));
        xT[p] = std::sqrt(wT[p]);
    }
    {
        std::vector<int> order(nP);
        for (int p = 0; p < nP; ++p) order[p] = p;
        std::sort(order.begin(), order.end(), [&](int a, int b){ return mass[a] > mass[b]; });
        std::vector<double> g(nP);
        objGrad(x1, g);
        std::printf("\n  gradient check (analytic | finite-diff):");
        for (int t = 0; t < 3 && t < nP; ++t) {
            int p = order[t];
            const double h = 5e-3;
            std::vector<double> xp = x1, xm = x1;
            xp[p] += h; xm[p] -= h;
            double fd = (objValue(xp) - objValue(xm)) / (2*h);
            std::printf("  %.4e | %.4e", g[p], fd);
        }
        std::printf("\n");
    }

    const double L_init  = objValue(x1);
    const double L_truth = objValue(xT);
    std::printf("  loss  L(w=1) = %.1f   L(w=truth) = %.1f   (%ld channels, %d parameters)\n\n",
                L_init, L_truth, nChan, nP);

    // ── minimise over the voxel weights with ensmallen L-BFGS ─────────────────
    OptimizerConfig ocfg;
    ocfg.maxIter   = (size_t)cfg.maxIter;
    ocfg.tolerance = 1e-8;
    OptimizerResult res;
    {
        VT_PROFILE("optimize");
        res = OptimizerAPI::Minimize(Algorithm::LBFGS, objGrad, x1, ocfg);
    }
    std::vector<double> wFit(nP);
    for (int p = 0; p < nP; ++p) wFit[p] = res.x[p] * res.x[p];

    SpectrumLoss plain(SpectrumLoss::CHI2);                            // physical goodness of fit
    double chi2Fit = 0;
    for (int d = 0; d < N_POS; ++d) { R[d].assemble(wFit, sim); chi2Fit += plain.value(sim, meas[d]); }

    double e2i = 0, e2f = 0, mTot = 0;                                 // response-weighted truth error
    for (int p = 0; p < nP; ++p) {
        e2i += mass[p] * (1.0 - wT[p]) * (1.0 - wT[p]);
        e2f += mass[p] * (wFit[p] - wT[p]) * (wFit[p] - wT[p]);
        mTot += mass[p];
    }
    std::printf("%s: L = %.1f after ≤%zu iterations\n", res.algorithm.c_str(), res.fval, ocfg.maxIter);
    std::printf("  plain χ²/channel = %.2f   (starts at %.2f, truth gives %.2f; MC model noise\n"
                "  adds ≈%.0f×Poisson, so ≈%.1f — not 1 — is the consistent level)\n",
                chi2Fit/nChan, L_init/nChan, L_truth/nChan, cfg.expoPerEvent, 1.0+cfg.expoPerEvent);
    std::printf("  weight RMS error vs truth: %.3f → %.3f (response-weighted)\n\n",
                std::sqrt(e2i/mTot), std::sqrt(e2f/mTot));

    // ── depth profile: response-weighted layer averages of the fitted field ───
    phase.next("outputs");
    std::printf("  %-12s %-6s %-7s %-8s %-10s %s\n", "layer [µm]", "nvox", "mass%", "w_true", "<w_fit>", "sd");
    std::vector<double> lm(zN, 0), lw(zN, 0), lw2(zN, 0);
    std::vector<int>    ln(zN, 0);
    for (int p = 0; p < nP; ++p) {
        int k = voxOfParam[p] % zN;
        lm[k] += mass[p]; lw[k] += mass[p]*wFit[p]; lw2[k] += mass[p]*wFit[p]*wFit[p];
        ++ln[k];
    }
    std::ofstream fpz("test-data/out/recon_profile.csv");
    fpz << "z_um,n_vox,mass,w_true,w_fit_mean,w_fit_sd\n";
    for (int k = 0; k < zN; ++k) {
        if (ln[k] == 0) continue;
        double m = lw[k]/lm[k], sd2 = std::sqrt(std::fmax(0.0, lw2[k]/lm[k] - m*m));
        double wt = wTrueOfZ(z0 + (k+0.5)*VOX);
        std::printf("  %3.0f – %-6.0f %-6d %-7.2f %-8.2f %-10.3f %.3f\n",
                    k*VOX*1e4, (k+1)*VOX*1e4, ln[k], 100.0*lm[k]/mTot, wt, m, sd2);
        fpz << (k+0.5)*VOX*1e4 << "," << ln[k] << "," << lm[k] << ","
            << wt << "," << m << "," << sd2 << "\n";
    }

    // ── per-voxel weights + measured-vs-fitted spectra ─────────────────────────
    std::ofstream fw("test-data/out/recon_weights.csv");
    fw << "vox,i,j,k,x_um,y_um,z_um,events,mass,w_true,w_fit\n";
    for (int p = 0; p < nP; ++p) {
        int vox = voxOfParam[p];
        int i = vox/(yN*zN), j = (vox/zN)%yN, k = vox%zN;
        fw << vox << "," << i << "," << j << "," << k << ","
           << (x0+(i+0.5)*sd.vx)*1e4 << "," << (y0+(j+0.5)*sd.vy)*1e4 << "," << (z0+(k+0.5)*sd.vz)*1e4 << ","
           << evCount[vox] << "," << mass[p] << "," << wT[p] << "," << wFit[p] << "\n";
    }

    std::ofstream fs("test-data/out/recon_spectra.csv");
    fs << "energy_keV";
    for (int d = 0; d < N_POS; ++d) {
        int um = (int)std::lround(sc.points[d][2]*1e4);
        fs << ",meas_" << um << "um,fit_" << um << "um";
    }
    fs << "\n";
    std::vector<std::vector<float>> fitSpec(N_POS, std::vector<float>(NBIN));
    for (int d = 0; d < N_POS; ++d) {
        R[d].assemble(wFit, sim);
        for (int ch = 0; ch < NBIN; ++ch) fitSpec[d][ch] = sim.count(ch);
    }
    for (int ch = 0; ch < NBIN; ++ch) {
        fs << (ch + 0.5)*EBIN;
        for (int d = 0; d < N_POS; ++d)
            fs << "," << meas[d].count(ch) << "," << fitSpec[d][ch];
        fs << "\n";
    }

    std::printf("\nVoxel weights → test-data/out/recon_weights.csv\n");
    std::printf("Depth profile → test-data/out/recon_profile.csv\n");
    std::printf("Spectra       → test-data/out/recon_spectra.csv\n");
    phase.close();
    if (cfg.profile) vtprof::report();
    return 0;
}
