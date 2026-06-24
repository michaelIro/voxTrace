// Test-3: confocal micro-XRF of NIST-1107 brass — full chain
//
//   source → primary polycap → sample → secondary polycap → detector spectrum
//
// Two PC-236 polycapillary optics (the secondary is the primary reversed) are
// placed 90° apart, each 45° to the sample surface, focusing to a common
// confocal volume just beneath the surface. A monochromatic primary beam is
// focused into the brass; fluorescence/scatter is emitted from the interaction
// point toward the secondary, which only transmits photons originating in its
// focus — so the recorded spectrum samples the confocal volume.
//
// Build: make test3      Run: ./build/src/Test3 [n_primary] [seed]
// Plot:  python3 src/tests/plot_spectrum.py

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

// ── NIST-1107 brass sample ────────────────────────────────────────────────────
static constexpr int   BR_N        = 6;
static constexpr int   BR_Z[6]     = {26, 28, 29, 30, 50, 82};               // Fe Ni Cu Zn Sn Pb
static constexpr float BR_W[6]     = {0.0004f, 0.001f, 0.6119f, 0.3741f, 0.0107f, 0.0019f};

// ── confocal geometry (sample frame, cm; surface = plane z=0, bulk z<0) ───────
static constexpr double COS45        = 0.70710678;
static constexpr double CONF_DEPTH   = 0.0015;    // confocal point 15 µm below surface
static constexpr double SAMPLE_THICK = 0.015;     // 150 µm
static constexpr double SAMPLE_HALF  = 0.015;     // 150 µm half-width in x,y
static constexpr double PRIM_ENERGY  = 17.4;      // keV (Capillaries.txt)
static constexpr double SRC_RADIUS   = 0.37;      // collimated source radius (Source.txt)

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

// Maps between the sample frame and one optic's frame (optic axis = +z). The
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

}  // namespace

int main(int argc, char* argv[]) {
    int      n_primary = (argc > 1) ? std::atoi(argv[1]) : 2000000;
    uint64_t seed      = (argc > 2) ? (uint64_t)std::atoll(argv[2]) : 1ULL;

    std::filesystem::create_directories("test-data/out");

    // ── optics: primary (large→small, focuses at exit), secondary = reversed ──
    int   optZ[2] = {OPT_Z[0], OPT_Z[1]};
    float optW[2] = {OPT_W[0], OPT_W[1]};
    PolyCap primary(0.f, OPT_LEN, R_EXT_BIG, R_EXT_SMALL, R_CAP_BIG, R_CAP_SMALL,
                    FOCAL_INF, FOCAL, PolyCap::ELLIPSOIDAL,
                    2, optZ, optW, OPT_RHO, OPT_ROUGH, OPT_NCAP);
    PolyCap secondary(0.f, OPT_LEN, R_EXT_SMALL, R_EXT_BIG, R_CAP_SMALL, R_CAP_BIG,
                      FOCAL, FOCAL_INF, PolyCap::ELLIPSOIDAL,
                      2, optZ, optW, OPT_RHO, OPT_ROUGH, OPT_NCAP);

    // ── brass material ────────────────────────────────────────────────────────
    std::vector<ChemElement> elems;
    for (int i = 0; i < BR_N; ++i) elems.emplace_back(BR_Z[i]);
    float brW[6]; for (int i = 0; i < BR_N; ++i) brW[i] = BR_W[i];
    Material brass(BR_N, brW, elems.data());

    // ── confocal placement: V-shape in the x-z plane, apex (focus) at C ───────
    Vec3 C        = {0, 0, -CONF_DEPTH};
    Vec3 d_prim   = norm({ COS45, 0, -COS45});   // primary beam: down-right into sample
    Vec3 d_sec    = norm({ COS45, 0,  COS45});   // detection:    up-right out of sample
    OpticFrame primFrame(C, d_prim,  OPT_LEN + FOCAL);  // C = exit-side focal point
    OpticFrame secFrame (C, d_sec,  -FOCAL);            // C = entrance-side focal point
    Vec3 secWinCtr = C + d_sec*FOCAL;                   // secondary entrance window centre

    // ── spectrum histogram ────────────────────────────────────────────────────
    constexpr int    NBIN = 1000;
    constexpr double EBIN = 0.02;                       // keV/bin, 0..20 keV
    std::vector<double> spec(NBIN, 0.0);
    auto addCount = [&](double e, double w) {
        int b = (int)(e / EBIN);
        if (b >= 0 && b < NBIN) spec[b] += w;
    };

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    int64_t transmitted_primary = 0, detected = 0;

    for (int n = 0; n < n_primary; ++n) {
        // (1) primary input: collimated beam (Source.txt divergence = 0) over the
        //     entrance aperture — the parallel illumination a focusing optic expects.
        double rr = std::sqrt(U(rng)) * std::min(R_EXT_BIG, SRC_RADIUS), ra = 2*M_PI*U(rng);
        Vec3 ent = {rr*std::cos(ra), rr*std::sin(ra), 0};
        Ray pray = makeRay(ent, {0, 0, 1}, PRIM_ENERGY);

        // (2) primary polycap
        primary.trace(pray);
        if (!pray.getIAFlag()) continue;
        ++transmitted_primary;
        double wPrim = pray.getProb();

        // (3) into the sample frame; propagate to the surface, then to interaction depth
        Vec3 ps, ds;
        primFrame.toSample({pray.getStartX(), pray.getStartY(), pray.getStartZ()},
                           {pray.getDirX(),   pray.getDirY(),   pray.getDirZ()}, ps, ds);
        if (ds.z >= 0) continue;
        Vec3 entry = ps + ds * ((0 - ps.z) / ds.z);
        if (std::fabs(entry.x) > SAMPLE_HALF || std::fabs(entry.y) > SAMPLE_HALF) continue;

        double muIn = brass.CS_Tot_Lin((float)PRIM_ENERGY, elems.data());   // 1/cm
        double s    = -std::log(U(rng)) / muIn;
        Vec3   P    = entry + ds * s;
        if (P.z > 0 || P.z < -SAMPLE_THICK ||
            std::fabs(P.x) > SAMPLE_HALF || std::fabs(P.y) > SAMPLE_HALF) continue;  // passed through

        // (4) aim the emitted photon at the secondary entrance window (importance sampling)
        double ar = std::sqrt(U(rng)) * R_EXT_SMALL, aa = 2*M_PI*U(rng);
        Vec3   aim = secWinCtr + secFrame.u*(ar*std::cos(aa)) + secFrame.v*(ar*std::sin(aa));
        Vec3   eDir = aim - P;
        double r2   = dot(eDir, eDir);
        eDir = norm(eDir);
        if (eDir.z <= 0) continue;                       // must travel up toward the surface

        // (5) the interaction: which element, which channel, emitted energy
        int   ei   = brass.getInteractingElementIdx((float)PRIM_ENERGY, (float)U(rng), elems.data());
        const ChemElement& el = elems[ei];
        int   type = el.getInteractionType((float)PRIM_ENERGY, (float)U(rng));
        double Ef;
        if (type == 0) {                                 // photoelectric → fluorescence
            int shell = el.getExcitedShell((float)PRIM_ENERGY, (float)U(rng));
            if (U(rng) >= el.Fluor_Y(shell)) continue;   // Auger: absorbed, no photon
            Ef = el.Line_Energy(el.getTransition(shell, (float)U(rng)));
        } else if (type == 1) {                          // Rayleigh (elastic)
            Ef = PRIM_ENERGY;
        } else {                                         // Compton (angle fixed by geometry)
            double theta = std::acos(std::fmax(-1.0, std::fmin(1.0, dot(ds, eDir))));
            Ef = el.getComptEnergy((float)PRIM_ENERGY, (float)theta);
        }
        if (Ef < 0.8) continue;

        // (6) emission weight: isotropic source sampled toward the window disk,
        //     × self-absorption of the emitted photon on its way out of the sample
        double cosD   = std::fabs(dot(eDir, d_sec));
        double wEmit  = (M_PI * R_EXT_SMALL * R_EXT_SMALL * cosD) / (4.0 * M_PI * r2);
        double Lout   = (0 - P.z) / eDir.z;                                  // path to surface
        double wSelf  = std::exp(-brass.CS_Tot_Lin((float)Ef, elems.data()) * Lout);

        // (7) secondary polycap — only confocal-volume photons survive
        Vec3 po, doo;
        secFrame.toOptic(P, eDir, po, doo);
        Ray sray = makeRay(po, doo, Ef);
        secondary.trace(sray);
        if (!sray.getIAFlag()) continue;

        addCount(Ef, wPrim * wEmit * wSelf * sray.getProb());
        ++detected;
    }

    // ── output ────────────────────────────────────────────────────────────────
    std::ofstream f("test-data/out/confocal_spectrum.csv");
    f << "energy_keV,weight\n";
    for (int b = 0; b < NBIN; ++b) f << (b + 0.5) * EBIN << "," << spec[b] << "\n";

    std::printf("NIST-1107 brass confocal point  (depth %.0f µm below surface)\n",
                CONF_DEPTH * 1e4);
    std::printf("  primary photons      = %d\n", n_primary);
    std::printf("  primary transmitted  = %lld\n", (long long)transmitted_primary);
    std::printf("  detected (confocal)  = %lld\n\n", (long long)detected);

    struct Line { const char* name; double e; };
    const Line lines[] = {
        {"Sn-La", 3.44}, {"Fe-Ka", 6.40}, {"Ni-Ka", 7.48}, {"Cu-Ka", 8.05},
        {"Zn-Ka", 8.64}, {"Cu-Kb", 8.91}, {"Zn-Kb", 9.57}, {"Pb-La", 10.55},
        {"Pb-Lb", 12.61}, {"Compton", 16.8}, {"Elastic", 17.4},
    };
    std::printf("  %-9s %-8s %s\n", "line", "E[keV]", "rel. intensity");
    for (const Line& L : lines) {
        double sum = 0;
        for (int b = 0; b < NBIN; ++b) {
            double e = (b + 0.5) * EBIN;
            if (std::fabs(e - L.e) < 0.12) sum += spec[b];
        }
        std::printf("  %-9s %-8.2f %.4e\n", L.name, L.e, sum);
    }
    std::printf("\nSpectrum → test-data/out/confocal_spectrum.csv\n");
    return 0;
}
