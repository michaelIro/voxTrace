// Test-3: confocal micro-XRF depth scan of NIST-1107 brass — full chain
//
//   source → primary polycap → SAMPLE (voxel grid) → secondary polycap →
//   Si(Li) detector → spectrum
//
// The sample is modelled with the voxTrace package data model: a Sample (voxel
// grid) of Voxels, each referencing a Material made of ChemElements. The primary
// beam is focused into the brass by a PC-236 polycap; the interaction point and
// all self-absorption are found by walking the voxel grid (Voxel::intersect +
// getNN, Material/ChemElement physics). Fluorescence/scatter is collected by a
// second PC-236 (the primary reversed) whose focus coincides with the primary's
// — so only the confocal volume is seen. Each collected photon is finally run
// through a Si(Li) Detector (efficiency, Si escape peaks, Compton continuum,
// finite resolution) so the recorded spectrum looks like a real EDXRF spectrum.
// The confocal point is stepped through the surface to produce a full depth scan.
//
// Build: make test3      Run: ./build/src/Test3 [n_primary] [seed]
// Plot:  python3 src/tests/plot_depthscan.py

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "Ray.hpp"
#include "PolyCap.hpp"
#include "ChemElement.hpp"
#include "Material.hpp"
#include "Voxel.hpp"
#include "Sample.hpp"
#include "Detector.hpp"

// ── PC-236 optic (Polycapillary.txt), cm ─────────────────────────────────────
static constexpr double OPT_LEN     = 4.03;
static constexpr double R_EXT_BIG   = 0.3175;     // large (collimated) window
static constexpr double R_EXT_SMALL = 0.095;      // small (focusing) window
static constexpr double R_CAP_BIG   = 0.000325;
static constexpr double R_CAP_SMALL = 0.0000975;
static constexpr double FOCAL_INF   = 1e8;        // collimated side
static constexpr double FOCAL       = 0.49;       // focusing side (cm from window)
static constexpr int    OPT_Z[2]    = {8, 14};    // SiO2
static constexpr float  OPT_W[2]    = {53.f, 47.f};
static constexpr double OPT_RHO     = 2.23;
static constexpr double OPT_ROUGH   = 5.0;
static constexpr int    OPT_NCAP    = 240000;

// ── NIST-1107 brass sample (Materials.txt) ────────────────────────────────────
static constexpr int   BR_N    = 6;
static constexpr int   BR_Z[6] = {26, 28, 29, 30, 50, 82};                       // Fe Ni Cu Zn Sn Pb
static constexpr float BR_W[6] = {0.0004f, 0.001f, 0.6119f, 0.3741f, 0.0107f, 0.0019f};

// ── sample voxel grid (Sample.txt: 300×300×150 µm, 5 µm voxels), cm ───────────
static constexpr double VOX        = 0.0005;      // 5 µm
static constexpr double SAMPLE_XY  = 0.030;       // 300 µm
static constexpr double SAMPLE_Z   = 0.015;       // 150 µm
// surface = plane z = 0; the bulk fills z ∈ [0, SAMPLE_Z] (+z = into the sample).

// ── confocal geometry & beam ──────────────────────────────────────────────────
static constexpr double COS45       = 0.70710678;
static constexpr double PRIM_ENERGY = 17.4;       // keV (Capillaries.txt)
static constexpr double SRC_RADIUS  = 0.37;       // collimated source radius (Source.txt)

// ── depth scan (Simulation.txt: 11 points over ±50 µm) ────────────────────────
static constexpr int    N_DEPTH = 11;
static constexpr double D_MIN   = -50e-4;         // −50 µm (confocal above surface)
static constexpr double D_STEP  =  10e-4;         //  10 µm step

namespace {

struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3 operator+(Vec3 o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(Vec3 o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
};
double dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
Vec3   cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
Vec3   norm(Vec3 a) { double n = std::sqrt(dot(a,a)); return {a.x/n, a.y/n, a.z/n}; }

// Maps between the sample frame and one optic's frame (optic axis = +z); the
// optic-frame plane z = zFoc maps to the confocal point C in the sample frame.
struct OpticFrame {
    Vec3 C, axis, u, v;
    double zFoc;
    OpticFrame(Vec3 confocal, Vec3 ax, double z_focal) : C(confocal), axis(norm(ax)), zFoc(z_focal) {
        v = {0, 1, 0};                 // both optics tilt in the x-z plane
        u = norm(cross(v, axis));
        v = cross(axis, u);
    }
    void toOptic(Vec3 p, Vec3 d, Vec3& po, Vec3& doo) const {
        Vec3 r = p - C;
        po = {dot(r,u), dot(r,v), zFoc + dot(r,axis)};
        doo = {dot(d,u), dot(d,v), dot(d,axis)};
    }
    void toSample(Vec3 po, Vec3 doo, Vec3& p, Vec3& d) const {
        p = C + u*po.x + v*po.y + axis*(po.z - zFoc);
        d = u*doo.x + v*doo.y + axis*doo.z;
    }
};

Ray makeRay(Vec3 p, Vec3 d, double energy_keV) {
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

// ── the sample data model: Sample grid + Voxel/Material/ChemElement arrays ────
struct Grid {
    Sample sample;
    const Voxel*       voxels;
    const Material*    mats;
    const ChemElement* elems;
};

// Walk the ray from a sample-surface entry point through the voxel grid and
// sample the first interaction (optical depth ~ −ln U). Returns true and sets
// the interaction point @p P and its material index, or false if it escaped.
bool sampleInteraction(const Grid& g, Vec3 entry, Vec3 dir, float E,
                       double xi, Vec3& P, int& matIdx) {
    Ray r = makeRay(entry, dir, E);
    int vox = g.sample.getVoxelIdx((float)entry.x, (float)entry.y, (float)entry.z);
    double tau = -std::log(xi), acc = 0;
    while (vox >= 0) {
        const Voxel& v = g.voxels[vox];
        double len = v.intersect(r);                                   // path through voxel [cm]
        double mu  = g.mats[v.getMaterialIdx()].CS_Tot_Lin(E, g.elems); // 1/cm
        if (acc + mu*len >= tau) {
            double l = r.getTIn() + (tau - acc) / mu;                  // dist from entry
            P = entry + dir * l;
            matIdx = v.getMaterialIdx();
            return true;
        }
        acc += mu * len;
        vox = v.getNN(r.getNextVoxel());
    }
    return false;                                                       // passed through
}

// Total optical depth from @p P along @p dir to the edge of the sample (the
// emitted photon's self-absorption on the way out), walking the voxel grid.
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

}  // namespace

int main(int argc, char* argv[]) {
    int      n_primary = (argc > 1) ? std::atoi(argv[1]) : 1000000;
    uint64_t seed      = (argc > 2) ? (uint64_t)std::atoll(argv[2]) : 1ULL;
    std::filesystem::create_directories("test-data/out");

    // ── optics: primary (large→small, focuses at exit), secondary = reversed ──
    int   optZ[2] = {OPT_Z[0], OPT_Z[1]};
    float optW[2] = {OPT_W[0], OPT_W[1]};
    PolyCap primary(0.f, OPT_LEN, R_EXT_BIG, R_EXT_SMALL, R_CAP_BIG, R_CAP_SMALL,
                    FOCAL_INF, FOCAL, PolyCap::ELLIPSOIDAL, 2, optZ, optW, OPT_RHO, OPT_ROUGH, OPT_NCAP);
    PolyCap secondary(0.f, OPT_LEN, R_EXT_SMALL, R_EXT_BIG, R_CAP_SMALL, R_CAP_BIG,
                      FOCAL, FOCAL_INF, PolyCap::ELLIPSOIDAL, 2, optZ, optW, OPT_RHO, OPT_ROUGH, OPT_NCAP);

    // ── sample: voxel grid of brass (one Material shared by every Voxel) ──────
    std::vector<ChemElement> elems;
    for (int i = 0; i < BR_N; ++i) elems.emplace_back(BR_Z[i]);
    float brW[6]; for (int i = 0; i < BR_N; ++i) brW[i] = BR_W[i];
    std::vector<Material> mats{ Material(BR_N, brW, elems.data()) };

    const int xN = (int)std::lround(SAMPLE_XY / VOX);
    const int yN = xN, zN = (int)std::lround(SAMPLE_Z / VOX);
    const double x0 = -SAMPLE_XY/2, y0 = -SAMPLE_XY/2, z0 = 0.0;          // surface at z=0
    std::vector<Voxel> voxels((size_t)xN*yN*zN);
    for (int i = 0; i < xN; ++i)
    for (int j = 0; j < yN; ++j)
    for (int k = 0; k < zN; ++k)
        voxels[(size_t)i*yN*zN + j*zN + k] =
            Voxel((float)(x0+i*VOX), (float)(y0+j*VOX), (float)(z0+k*VOX),
                  (float)VOX, (float)VOX, (float)VOX, 0);                  // all → material 0 (brass)
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
    Grid grid{ Sample((float)x0,(float)y0,(float)z0, (float)SAMPLE_XY,(float)SAMPLE_XY,(float)SAMPLE_Z,
                      (float)VOX,(float)VOX,(float)VOX, xN,yN,zN),
               voxels.data(), mats.data(), elems.data() };

    // ── Si(Li) detector: a Si crystal + Be window, reusing ChemElement physics ─
    std::vector<ChemElement> detElems{ ChemElement(14), ChemElement(4) };   // Si, Be
    Detector detector = Detector::make(/*siIdx*/0, /*beIdx*/1, /*thickness*/0.30f,
                                       /*beWin*/0.0025f, /*deadLayer*/1e-4f,
                                       /*fano*/0.114f, /*noiseFWHM*/0.080f);
    RNG detRng(seed ^ 0x9E3779B97F4A7C15ULL);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    // ── trace the primary beam through the primary optic ONCE; the focused beam
    //    is depth-independent, so the stored exit rays are reused at every depth ─
    std::vector<ExitRay> beam;
    for (int n = 0; n < n_primary; ++n) {
        double rr = std::sqrt(U(rng)) * std::min(R_EXT_BIG, SRC_RADIUS), ra = 2*M_PI*U(rng);
        Ray p = makeRay({rr*std::cos(ra), rr*std::sin(ra), 0}, {0,0,1}, PRIM_ENERGY);
        primary.trace(p);
        if (p.getIAFlag())
            beam.push_back({{p.getStartX(), p.getStartY(), p.getStartZ()},
                            {p.getDirX(),   p.getDirY(),   p.getDirZ()}, p.getProb()});
    }
    std::printf("NIST-1107 brass confocal depth scan\n");
    std::printf("  primary photons     = %d   transmitted = %zu (%.2f%%)\n\n",
                n_primary, beam.size(), 100.0*beam.size()/n_primary);

    // ── characteristic lines to track vs depth ────────────────────────────────
    struct Line { const char* name; double e; };
    const Line lines[] = {
        {"Fe-Ka",6.40},{"Ni-Ka",7.48},{"Cu-Ka",8.05},{"Zn-Ka",8.64},
        {"Cu-Kb",8.91},{"Zn-Kb",9.57},{"Pb-La",10.55},{"Elastic",17.4},
    };
    const int NL = sizeof(lines)/sizeof(lines[0]);

    constexpr int    NBIN = 1000;
    constexpr double EBIN = 0.02;                                          // keV/bin
    std::vector<std::vector<double>> spectra(N_DEPTH, std::vector<double>(NBIN, 0.0));
    std::vector<std::vector<double>> lineI(N_DEPTH, std::vector<double>(NL, 0.0));
    std::vector<int64_t> detected(N_DEPTH, 0);

    Vec3 d_prim = norm({ COS45, 0,  COS45});   // primary beam: down-right into sample (+z)
    Vec3 d_sec  = norm({ COS45, 0, -COS45});   // detection:    up-right out of sample (−z)

    std::printf("  %-7s %-10s %-11s %-11s %-11s\n", "depth", "detected", "Cu-Ka", "Zn-Ka", "Pb-La");
    for (int di = 0; di < N_DEPTH; ++di) {
        double depth = D_MIN + di*D_STEP;                                 // confocal depth below surface
        Vec3   C = {0, 0, depth};
        OpticFrame primFrame(C, d_prim,  OPT_LEN + FOCAL);
        OpticFrame secFrame (C, d_sec,  -FOCAL);
        Vec3 secWinCtr = C + d_sec * FOCAL;                               // secondary entrance centre

        for (const ExitRay& er : beam) {
            // (1) primary exit ray → sample frame → surface entry
            Vec3 ps, ds;
            primFrame.toSample(er.pos, er.dir, ps, ds);
            if (ds.z <= 0) continue;
            double t = (0 - ps.z) / ds.z;
            if (t < 0) continue;
            Vec3 entry = ps + ds * t;
            if (std::fabs(entry.x) >= SAMPLE_XY/2 || std::fabs(entry.y) >= SAMPLE_XY/2) continue;

            // (2) walk the voxel grid to the first interaction point
            Vec3 P; int matIdx;
            if (!sampleInteraction(grid, entry, ds, (float)PRIM_ENERGY, U(rng), P, matIdx)) continue;
            const Material& mat = grid.mats[matIdx];

            // (3) aim the emitted photon at the secondary window (importance sampling)
            double ar = std::sqrt(U(rng)) * R_EXT_SMALL, aa = 2*M_PI*U(rng);
            Vec3   aim  = secWinCtr + secFrame.u*(ar*std::cos(aa)) + secFrame.v*(ar*std::sin(aa));
            Vec3   eDir = aim - P;
            double r2   = dot(eDir, eDir);
            eDir = norm(eDir);
            if (eDir.z >= 0) continue;                                    // must travel out (−z)

            // (4) the interaction: element, channel, emitted energy
            int ei   = mat.getInteractingElementIdx((float)PRIM_ENERGY, (float)U(rng), grid.elems);
            const ChemElement& el = grid.elems[ei];
            int type = el.getInteractionType((float)PRIM_ENERGY, (float)U(rng));
            double Ef;
            if (type == 0) {                                              // photoelectric → fluorescence
                int shell = el.getExcitedShell((float)PRIM_ENERGY, (float)U(rng));
                if (U(rng) >= el.Fluor_Y(shell)) continue;                // Auger
                Ef = el.Line_Energy(el.getTransition(shell, (float)U(rng)));
            } else if (type == 1) {                                       // Rayleigh
                Ef = PRIM_ENERGY;
            } else {                                                      // Compton (angle = geometry)
                double th = std::acos(std::fmax(-1.0, std::fmin(1.0, dot(ds, eDir))));
                Ef = el.getComptEnergy((float)PRIM_ENERGY, (float)th);
            }
            if (Ef < 0.8) continue;

            // (5) emission weight + self-absorption out of the sample (voxel walk)
            double wEmit = (M_PI * R_EXT_SMALL * R_EXT_SMALL * std::fabs(dot(eDir, d_sec))) / (4.0*M_PI*r2);
            double wSelf = std::exp(-opticalDepthOut(grid, P, eDir, (float)Ef));

            // (6) secondary polycap — only confocal-volume photons survive
            Vec3 po, doo;
            secFrame.toOptic(P, eDir, po, doo);
            Ray sray = makeRay(po, doo, Ef);
            secondary.trace(sray);
            if (!sray.getIAFlag()) continue;

            // (7) Si(Li) detector response: efficiency, Si escape, Compton, resolution
            float wDet;
            float Emeas = detector.detect((float)Ef, detRng, detElems.data(), wDet);
            if (wDet <= 0.f) continue;

            double w = er.w * wEmit * wSelf * sray.getProb() * wDet;
            int b = (int)(Emeas / EBIN);
            if (b >= 0 && b < NBIN) { spectra[di][b] += w; ++detected[di]; }
        }

        // line intensities = measured spectrum integrated over each photopeak (±2.5σ)
        for (int L = 0; L < NL; ++L) {
            double half = std::fmax(0.10, 2.5 * detector.resolutionSigma((float)lines[L].e));
            int b0 = std::max(0,        (int)((lines[L].e - half) / EBIN));
            int b1 = std::min(NBIN - 1, (int)((lines[L].e + half) / EBIN));
            double s = 0.0;
            for (int b = b0; b <= b1; ++b) s += spectra[di][b];
            lineI[di][L] = s;
        }

        double cu = 0, zn = 0, pb = 0;
        for (int L = 0; L < NL; ++L) {
            if (std::string(lines[L].name) == "Cu-Ka") cu = lineI[di][L];
            if (std::string(lines[L].name) == "Zn-Ka") zn = lineI[di][L];
            if (std::string(lines[L].name) == "Pb-La") pb = lineI[di][L];
        }
        std::printf("  %+5.0fµm %-10lld %.4e  %.4e  %.4e\n",
                    depth*1e4, (long long)detected[di], cu, zn, pb);
    }

    // ── output: depth profile + per-depth spectra ─────────────────────────────
    std::ofstream fp("test-data/out/confocal_depthscan.csv");
    fp << "depth_um,detected";
    for (int L = 0; L < NL; ++L) fp << "," << lines[L].name;
    fp << "\n";
    for (int di = 0; di < N_DEPTH; ++di) {
        fp << (D_MIN + di*D_STEP)*1e4 << "," << detected[di];
        for (int L = 0; L < NL; ++L) fp << "," << lineI[di][L];
        fp << "\n";
    }
    std::ofstream fs("test-data/out/confocal_spectra.csv");
    fs << "energy_keV";
    for (int di = 0; di < N_DEPTH; ++di) fs << ",d" << (int)std::lround((D_MIN+di*D_STEP)*1e4);
    fs << "\n";
    for (int b = 0; b < NBIN; ++b) {
        fs << (b + 0.5)*EBIN;
        for (int di = 0; di < N_DEPTH; ++di) fs << "," << spectra[di][b];
        fs << "\n";
    }
    std::printf("\nDepth profile → test-data/out/confocal_depthscan.csv\n");
    std::printf("Spectra      → test-data/out/confocal_spectra.csv\n");
    return 0;
}
