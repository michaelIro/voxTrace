// Test-2: PolyCap validation — original polycap-1.2 C library vs the native
// PolyCap.hpp (Ray-anchored, double-precision trace).
//
// The SAME set of sampled photons is traced through both implementations (the C
// library via polycap_photon_new + polycap_photon_launch, bypassing its internal
// RNG) so per-photon results are directly comparable. Reports per-energy
// transmission efficiency and writes exit-plane beam profiles for plotting.
//
// Build:  make test2
// Run:    ./build/src/Test2 [n_photons] [seed]
// Plot:   python3 src/tests/plot_beam_comparison.py
//
// Geometry: pc-236-descr.txt-style SiO2 polycapillary, L = 4.03 cm.

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ── original polycap C library ───────────────────────────────────────────────
extern "C" {
#include <polycap.h>
}

// ── repo-native implementation ───────────────────────────────────────────────
#include "Ray.hpp"
#include "PolyCap.hpp"

// ── pc-236 geometry ───────────────────────────────────────────────────────────
static constexpr double PC_LENGTH       = 4.03;
static constexpr double PC_REXT_IN      = 0.3175;
static constexpr double PC_REXT_OUT     = 0.095;
static constexpr double PC_RCAP_IN      = 3.25e-4;
static constexpr double PC_RCAP_OUT     = 9.75e-5;
static constexpr double PC_FOCAL_IN     = 1e8;
static constexpr double PC_FOCAL_OUT    = 0.49;
static constexpr int    PC_NELEM        = 2;
static constexpr int    PC_Z[2]         = {8, 14};
static constexpr double PC_WT[2]        = {53.0, 47.0};
static constexpr double PC_DENSITY      = 2.23;
static constexpr double PC_ROUGHNESS    = 5.0;
static constexpr int64_t PC_NCAP        = 240000;

// ── source geometry (point source, homogeneous illumination) ─────────────────
static constexpr double SRC_DIST   = 500.0;
static constexpr double SRC_X      = 0.01;
static constexpr double SRC_Y      = 0.01;
static constexpr double SRC_SHIFTX = 0.0;
static constexpr double SRC_SHIFTY = 0.0;
static constexpr double HOR_POL    = 0.9;

static const double ENERGIES[] = {4.0, 6.0, 8.0, 10.0, 12.0};
static constexpr int N_ENERGIES = 5;

static const double BEAM_PLANE_D[5] = {
    0.0,
    PC_FOCAL_OUT * 0.5,
    PC_FOCAL_OUT,
    PC_FOCAL_OUT * 1.5,
    PC_FOCAL_OUT * 2.0,
};
static constexpr int N_PLANES = 5;

namespace {

// ── shared photon: native polycap frame (x,y horizontal/vertical, z = optic axis) ──
struct SampledPhoton {
    double x = 0, y = 0, z = 0;
    double dx = 0, dy = 0, dz = 0;
    double ex = 0, ey = 0, ez = 0;  // electric vector
};

void normalize(double& x, double& y, double& z) {
    double n = std::sqrt(x*x + y*y + z*z);
    x /= n; y /= n; z /= n;
}

bool within_hex_boundary(double radius, double cx, double cy) {
    static constexpr double COSPI_6 = 0.86602540378443865;
    double d = std::sqrt(radius*radius - (radius/2.)*(radius/2.));
    if (std::fabs(cy) > d) return false;
    if (std::fabs(COSPI_6*cx + 0.5*cy) > d) return false;
    if (std::fabs(COSPI_6*cx - 0.5*cy) > d) return false;
    return true;
}

// Mirrors polycap_source_get_photon's uniform-illumination sampling so both
// implementations see identical photon statistics (own RNG → identical draws).
SampledPhoton sample_photon(std::mt19937_64& rng, double entrance_radius, double n_shells) {
    std::uniform_real_distribution<double> unit(0., 1.);
    SampledPhoton ph;

    double r = unit(rng);
    double phi = std::atan(SRC_Y / SRC_X * std::tan(2. * M_PI * r / 4.));
    r = unit(rng);
    if (r >= 0.25 && r < 0.50) phi = M_PI - phi;
    if (r >= 0.50 && r < 0.75) phi = M_PI + phi;
    if (r >= 0.75) phi = -phi;

    double max_radius = SRC_X * SRC_Y /
        std::sqrt(std::pow(SRC_Y * std::cos(phi), 2) + std::pow(SRC_X * std::sin(phi), 2));
    r = unit(rng);
    double src_x = std::sqrt(r) * max_radius * std::cos(phi) + SRC_SHIFTX;
    double src_y = std::sqrt(r) * max_radius * std::sin(phi) + SRC_SHIFTY;

    do {
        r = unit(rng);
        ph.x = (2.*r - 1.) * entrance_radius;
        r = unit(rng);
        ph.y = (2.*r - 1.) * entrance_radius;
    } while (n_shells > 0. && !within_hex_boundary(entrance_radius, ph.x, ph.y));
    ph.z = 0.;

    ph.dx = ph.x - src_x;
    ph.dy = ph.y - src_y;
    ph.dz = SRC_DIST;
    normalize(ph.dx, ph.dy, ph.dz);

    double horizontal_fraction = (1. + HOR_POL) / 2.;
    double ex = 1., ey = 0., ez = 0.;
    if (unit(rng) > horizontal_fraction) { ex = 0.; ey = 1.; }

    double dot = ex*ph.dx + ey*ph.dy + ez*ph.dz;   // orthogonalize against direction
    ph.ex = ex - ph.dx*dot;
    ph.ey = ey - ph.dy*dot;
    ph.ez = ez - ph.dz*dot;
    normalize(ph.ex, ph.ey, ph.ez);
    return ph;
}

// ── one transmitted photon at the exit plane ─────────────────────────────────
struct ExitRow {
    double x, y, z, dx, dy, dz, w;
};

// ═══════════════════════════ 1. original polycap C library ═══════════════════
std::vector<ExitRow> run_polycap_lib(polycap_description* desc,
                                     const std::vector<SampledPhoton>& photons,
                                     double energy_keV) {
    std::vector<ExitRow> rows;
    rows.reserve(photons.size());
    double energies[1] = {energy_keV};

    for (const auto& ph : photons) {
        polycap_vector3 start_coords    = {ph.x, ph.y, ph.z};
        polycap_vector3 start_direction = {ph.dx, ph.dy, ph.dz};
        polycap_vector3 start_elecv     = {ph.ex, ph.ey, ph.ez};

        polycap_error* error = nullptr;
        polycap_photon* photon = polycap_photon_new(desc, start_coords, start_direction,
                                                     start_elecv, &error);
        if (!photon) { polycap_error_free(error); continue; }

        double* weights = nullptr;
        int status = polycap_photon_launch(photon, 1, energies, &weights, false, &error);
        if (status == 1 && weights) {
            polycap_vector3 ec = polycap_photon_get_exit_coords(photon);
            polycap_vector3 ed = polycap_photon_get_exit_direction(photon);
            rows.push_back({ec.x, ec.y, ec.z, ed.x, ed.y, ed.z, weights[0]});
        }
        free(weights);
        polycap_photon_free(photon);
        polycap_error_free(error);
    }
    return rows;
}

// ═══════════════════════════ 2. PolyCap.hpp (native) ═════════════════════════
// PolyCap traces directly on the Ray's own (x,y,z); the optic axis is z, so the
// native frame maps 1:1.
std::vector<ExitRow> run_polycap(const PolyCap& optic,
                                 const std::vector<SampledPhoton>& photons,
                                 double energy_keV) {
    std::vector<ExitRow> rows;
    rows.reserve(photons.size());

    for (const auto& ph : photons) {
        Ray ray;
        ray.setStartCoordinates((float)ph.x, (float)ph.y, (float)ph.z);
        ray.setEndCoordinates((float)ph.dx, (float)ph.dy, (float)ph.dz);
        ray.setSPol((float)ph.ex, (float)ph.ey, (float)ph.ez);
        ray.setPPol(0.f, 0.f, 0.f);
        ray.setEnergyKeV((float)energy_keV);
        ray.setProb(1.f);
        ray.setIAFlag(false);

        optic.trace(ray);
        if (ray.getIAFlag()) {
            rows.push_back({ray.getStartX(), ray.getStartY(), ray.getStartZ(),
                            ray.getDirX(),   ray.getDirY(),   ray.getDirZ(),
                            ray.getProb()});
        }
    }
    return rows;
}

// ── propagate exit rows to downstream observation planes (native z axis) ─────
void write_plane_csv(const std::string& path, const std::vector<ExitRow>& rows) {
    std::ofstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return; }
    f << "plane_idx,x_cm,y_cm,weight\n";
    f << std::setprecision(10);
    for (const auto& r : rows) {
        for (int plane = 0; plane < N_PLANES; ++plane) {
            double dist = BEAM_PLANE_D[plane];
            double t = (std::fabs(r.dz) > 1.e-10) ? dist / r.dz : 0.0;
            f << plane << "," << (r.x + t*r.dx) << "," << (r.y + t*r.dy) << "," << r.w << "\n";
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    int      n_photons = (argc > 1) ? std::atoi(argv[1]) : 20000;
    uint64_t seed      = (argc > 2) ? (uint64_t)std::atoll(argv[2]) : 42ULL;

    std::filesystem::create_directories("test-data/out");

    std::cout << "PolyCap comparison: original polycap C library  vs  PolyCap.hpp\n"
              << "  n_photons = " << n_photons << "\n"
              << "  seed      = " << seed << "\n"
              << "  optic     = pc-236  L=" << PC_LENGTH << " cm  rExt=("
              << PC_REXT_IN << "->" << PC_REXT_OUT << ") cm\n\n";

    // ── Build original polycap C library description ─────────────────────────
    polycap_error* error = nullptr;
    polycap_profile* c_profile = polycap_profile_new(
        POLYCAP_PROFILE_ELLIPSOIDAL, PC_LENGTH,
        PC_REXT_IN, PC_REXT_OUT, PC_RCAP_IN, PC_RCAP_OUT,
        PC_FOCAL_IN, PC_FOCAL_OUT, &error);
    if (!c_profile) {
        std::cerr << "polycap_profile_new failed: " << (error ? error->message : "?") << "\n";
        return 1;
    }
    int iz_mut[2] = {PC_Z[0], PC_Z[1]};
    double wi_mut[2] = {PC_WT[0], PC_WT[1]};
    polycap_description* c_desc = polycap_description_new(
        c_profile, PC_ROUGHNESS, PC_NCAP, PC_NELEM, iz_mut, wi_mut, PC_DENSITY, &error);
    polycap_profile_free(c_profile);
    if (!c_desc) {
        std::cerr << "polycap_description_new failed: " << (error ? error->message : "?") << "\n";
        return 1;
    }

    // ── Build native PolyCap ──────────────────────────────────────────────────
    int   atomicZ[2] = {PC_Z[0], PC_Z[1]};
    float weights[2] = {(float)PC_WT[0], (float)PC_WT[1]};
    PolyCap optic(0.0f, (float)PC_LENGTH,
                  (float)PC_REXT_IN, (float)PC_REXT_OUT,
                  (float)PC_RCAP_IN, (float)PC_RCAP_OUT,
                  (float)PC_FOCAL_IN, (float)PC_FOCAL_OUT,
                  PolyCap::ELLIPSOIDAL,
                  PC_NELEM, atomicZ, weights,
                  (float)PC_DENSITY, (float)PC_ROUGHNESS, (int)PC_NCAP);

    double entrance_radius = PC_REXT_IN;
    double n_shells        = std::round(std::sqrt(12. * (double)PC_NCAP - 3.) / 6. - 0.5);

    std::cout << std::setw(10) << "E(keV)"
              << std::setw(16) << "polycap_eff"
              << std::setw(16) << "PolyCap_eff"
              << std::setw(14) << "ratio"
              << std::setw(14) << "lib_ms"
              << std::setw(14) << "PolyCap_ms" << "\n"
              << std::string(84, '-') << "\n";

    for (int ei = 0; ei < N_ENERGIES; ++ei) {
        double e = ENERGIES[ei];

        // Shared photon batch: identical for both implementations.
        std::mt19937_64 rng(seed + (uint64_t)ei * 7919ULL);
        std::vector<SampledPhoton> photons;
        photons.reserve(n_photons);
        for (int i = 0; i < n_photons; ++i)
            photons.push_back(sample_photon(rng, entrance_radius, n_shells));

        auto t0 = std::chrono::steady_clock::now();
        std::vector<ExitRow> rows_lib = run_polycap_lib(c_desc, photons, e);
        auto t1 = std::chrono::steady_clock::now();
        std::vector<ExitRow> rows_pc  = run_polycap(optic, photons, e);
        auto t2 = std::chrono::steady_clock::now();

        double sum_lib = 0., sum_pc = 0.;
        for (auto& r : rows_lib) sum_lib += r.w;
        for (auto& r : rows_pc)  sum_pc  += r.w;
        double eff_lib = sum_lib / (double)n_photons;
        double eff_pc  = sum_pc  / (double)n_photons;

        std::printf("%10.1f%16.6f%16.6f%14.3f%14.0f%14.0f\n",
                    e, eff_lib, eff_pc,
                    (eff_lib > 0.) ? eff_pc/eff_lib : 0.,
                    std::chrono::duration<double, std::milli>(t1-t0).count(),
                    std::chrono::duration<double, std::milli>(t2-t1).count());

        char fname[256];
        std::snprintf(fname, sizeof(fname), "test-data/out/beam_lib_E%.1f.csv", e);
        write_plane_csv(fname, rows_lib);
        std::snprintf(fname, sizeof(fname), "test-data/out/beam_polycap_E%.1f.csv", e);
        write_plane_csv(fname, rows_pc);
    }

    polycap_description_free(c_desc);
    polycap_error_free(error);
    return 0;
}
