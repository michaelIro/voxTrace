// Test-3way: three-way polycapillary comparison.
//   (1) the original polycap-1.2 C library (ground truth)
//   (2) PolyCap_new.hpp   — lean Kokkos/Metal-portable single-ray tracer
//   (3) PolyCap.hpp       — current header-only translation used by the rest of core
//
// All three trace the *same* set of sampled photons (identical start coords,
// direction, and electric vector per index) so per-photon results are directly
// comparable rather than only matching in aggregate statistics.
//
// Geometry: pc-236-descr.txt-style SiO2 polycapillary, L=4.03 cm (same optic as Test-2).
//
// Build:  make test3
// Run:    ./build/src/Test3 [n_photons] [seed]
// Then:   python3 src/tests/plot_beam_comparison_3way.py

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

// ── repo-native implementations ──────────────────────────────────────────────
#include "Ray.hpp"
#include "PolyCap_new.hpp"
#include "PolyCap_veryNew.hpp"
#include "PolyCap.hpp"

// ── pc-236 geometry (same as Test-2.cpp) ──────────────────────────────────────
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
static constexpr double SRC_SIGX   = -1.0;   // <0 -> uniform over PC entrance
static constexpr double SRC_SIGY   = -1.0;
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

double vec_norm(double x, double y, double z) { return std::sqrt(x*x + y*y + z*z); }

void normalize(double& x, double& y, double& z) {
    double n = vec_norm(x, y, z);
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

// Mirrors polycap_source_get_photon's sampling algorithm (uniform illumination
// branch, src_sigx/src_sigy < 0) so all three implementations see the same
// photon statistics. Uses our own RNG so we can feed identical draws to all 3.
SampledPhoton sample_photon(std::mt19937_64& rng, double entrance_radius,
                            double cap_radius, double n_shells) {
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

    // Uniform illumination over entrance window (src_sigx/src_sigy < 0 branch)
    if (n_shells == 0.) {
        do {
            r = unit(rng);
            ph.x = (2.*r - 1.) * cap_radius;
            r = unit(rng);
            ph.y = (2.*r - 1.) * cap_radius;
        } while (ph.x*ph.x + ph.y*ph.y > cap_radius*cap_radius);
    } else {
        do {
            r = unit(rng);
            ph.x = (2.*r - 1.) * entrance_radius;
            r = unit(rng);
            ph.y = (2.*r - 1.) * entrance_radius;
        } while (!within_hex_boundary(entrance_radius, ph.x, ph.y));
    }
    ph.z = 0.;

    ph.dx = ph.x - src_x;
    ph.dy = ph.y - src_y;
    ph.dz = SRC_DIST;
    normalize(ph.dx, ph.dy, ph.dz);

    double horizontal_fraction = (1. + HOR_POL) / 2.;
    double ex = 1., ey = 0., ez = 0.;
    if (unit(rng) > horizontal_fraction) { ex = 0.; ey = 1.; }

    // Orthogonalize electric vector against direction
    double dot = ex*ph.dx + ey*ph.dy + ez*ph.dz;
    ph.ex = ex - ph.dx*dot;
    ph.ey = ey - ph.dy*dot;
    ph.ez = ez - ph.dz*dot;
    double pn2 = ph.ex*ph.ex + ph.ey*ph.ey + ph.ez*ph.ez;
    if (pn2 < 1.e-12) {
        // direction nearly parallel to (1,0,0)/(0,1,0): fall back to a stable perpendicular
        if (std::fabs(ph.dx) < 0.9) { ph.ex = 1.; ph.ey = 0.; ph.ez = 0.; }
        else                        { ph.ex = 0.; ph.ey = 1.; ph.ez = 0.; }
        dot = ph.ex*ph.dx + ph.ey*ph.dy + ph.ez*ph.dz;
        ph.ex -= ph.dx*dot; ph.ey -= ph.dy*dot; ph.ez -= ph.dz*dot;
    }
    normalize(ph.ex, ph.ey, ph.ez);

    return ph;
}

// ── CSV row: one transmitted photon, exit coords/dir + per-energy weight ─────
struct ExitRow {
    double x, y, z, dx, dy, dz, w;
};

void write_csv(const std::string& path, const std::vector<ExitRow>& rows) {
    std::ofstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return; }
    f << "x,y,z,dx,dy,dz,w\n";
    f << std::setprecision(10);
    for (const auto& r : rows) {
        f << r.x << "," << r.y << "," << r.z << ","
          << r.dx << "," << r.dy << "," << r.dz << "," << r.w << "\n";
    }
}

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

// ═══════════════════════════ 2. PolyCap_new.hpp ═══════════════════════════════
// PolyCap_new traces directly on the Ray's own (x,y,z) — native frame maps 1:1
// since both use z as the optic axis.
std::vector<ExitRow> run_polycap_new(const PolyCap_new& optic,
                                     const std::vector<SampledPhoton>& photons,
                                     double energy_keV) {
    std::vector<ExitRow> rows;
    rows.reserve(photons.size());

    for (const auto& ph : photons) {
        Ray ray;
        ray.setStartCoordinates((float)ph.x, (float)ph.y, (float)ph.z);
        ray.setEndCoordinates((float)ph.dx, (float)ph.dy, (float)ph.dz);
        ray.setSPol((float)ph.ex, (float)ph.ey, (float)ph.ez);
        ray.setPPol(0.f, 0.f, 0.f);  // not used by PolyCap_new's reflect()
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

// ═══════════════════════════ 2b. PolyCap_veryNew.hpp ══════════════════════════
// Same native frame as PolyCap_new: optic axis = ray z, fed 1:1.
std::vector<ExitRow> run_polycap_verynew(const PolyCap_veryNew& optic,
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

// ═══════════════════════════ 3. PolyCap.hpp (current core) ═══════════════════
// PolyCap.hpp's facade expects rays in the repo convention (optic axis = Y);
// its internal repo_to_pc() maps repo (x,y,z) -> native (x,z,y). To feed it the
// same native-frame photon, place native.y into repo.z and native.z into repo.y.
std::vector<ExitRow> run_polycap_hpp(const PolyCap& optic,
                                     const std::vector<SampledPhoton>& photons,
                                     double energy_keV) {
    std::vector<ExitRow> rows;
    rows.reserve(photons.size());

    for (const auto& ph : photons) {
        Ray ray;
        ray.setStartCoordinates((float)ph.x, (float)ph.z, (float)ph.y);
        ray.setEndCoordinates((float)ph.dx, (float)ph.dz, (float)ph.dy);
        ray.setSPol((float)ph.ex, (float)ph.ez, (float)ph.ey);
        ray.setPPol(0.f, 0.f, 0.f);
        ray.setEnergyKeV((float)energy_keV);
        ray.setProb(1.f);
        ray.setIAFlag(false);

        PolyCapTraceResult result = optic.trace(ray);
        if (result.transmitted) {
            // result.ray is in repo convention; convert back to native frame for CSV.
            double rx = result.ray.getStartX();
            double ry = result.ray.getStartY();
            double rz = result.ray.getStartZ();
            double rdx = result.ray.getDirX();
            double rdy = result.ray.getDirY();
            double rdz = result.ray.getDirZ();
            rows.push_back({rx, rz, ry, rdx, rdz, rdy, result.primaryWeight()});
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

    std::cout << "Three-way polycapillary comparison\n"
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

    // ── Build PolyCap_new ──────────────────────────────────────────────────────
    int    nElements   = PC_NELEM;
    int    atomicZ[2]  = {PC_Z[0], PC_Z[1]};
    float  weights[2]  = {(float)PC_WT[0], (float)PC_WT[1]};
    PolyCap_new pc_new(0.0f, (float)PC_LENGTH,
                       (float)PC_REXT_IN, (float)PC_REXT_OUT,
                       (float)PC_RCAP_IN, (float)PC_RCAP_OUT,
                       (float)PC_FOCAL_IN, (float)PC_FOCAL_OUT,
                       PolyCap_new::ELLIPSOIDAL,
                       nElements, atomicZ, weights,
                       (float)PC_DENSITY, (float)PC_ROUGHNESS, (int)PC_NCAP);

    // ── Build PolyCap_veryNew ─────────────────────────────────────────────────
    PolyCap_veryNew pc_verynew(0.0f, (float)PC_LENGTH,
                               (float)PC_REXT_IN, (float)PC_REXT_OUT,
                               (float)PC_RCAP_IN, (float)PC_RCAP_OUT,
                               (float)PC_FOCAL_IN, (float)PC_FOCAL_OUT,
                               PolyCap_veryNew::ELLIPSOIDAL,
                               nElements, atomicZ, weights,
                               (float)PC_DENSITY, (float)PC_ROUGHNESS, (int)PC_NCAP);

    // ── Build PolyCap.hpp (current core) ──────────────────────────────────────
    std::vector<int> hpp_z(PC_Z, PC_Z + PC_NELEM);
    std::vector<double> hpp_wt(PC_WT, PC_WT + PC_NELEM);
    PolyCapProfile hpp_profile = PolyCapProfile::ellipsoidal(
        PC_LENGTH, PC_REXT_IN, PC_REXT_OUT, PC_RCAP_IN, PC_RCAP_OUT,
        PC_FOCAL_IN, PC_FOCAL_OUT);
    PolyCapWall hpp_wall(hpp_z, hpp_wt, PC_DENSITY, PC_ROUGHNESS);
    PolyCap pc_hpp(hpp_profile, hpp_wall, PC_NCAP);

    double entrance_radius = PC_REXT_IN;
    double cap_radius      = PC_RCAP_IN;
    double n_shells        = std::round(std::sqrt(12. * (double)PC_NCAP - 3.) / 6. - 0.5);

    std::cout << std::setw(8)  << "E(keV)"
              << std::setw(14) << "polycap_eff"
              << std::setw(14) << "new_eff"
              << std::setw(14) << "veryNew_eff"
              << std::setw(14) << "hpp_eff"
              << std::setw(12) << "new/lib"
              << std::setw(12) << "vNew/hpp"
              << std::setw(12) << "hpp/lib" << "\n"
              << std::string(100, '-') << "\n";

    for (int ei = 0; ei < N_ENERGIES; ++ei) {
        double e = ENERGIES[ei];

        // Shared photon batch: identical for all three implementations.
        std::mt19937_64 rng(seed + (uint64_t)ei * 7919ULL);
        std::vector<SampledPhoton> photons;
        photons.reserve(n_photons);
        for (int i = 0; i < n_photons; ++i)
            photons.push_back(sample_photon(rng, entrance_radius, cap_radius, n_shells));

        auto t0 = std::chrono::steady_clock::now();
        std::vector<ExitRow> rows_lib = run_polycap_lib(c_desc, photons, e);
        auto t1 = std::chrono::steady_clock::now();
        std::vector<ExitRow> rows_new = run_polycap_new(pc_new, photons, e);
        auto t2 = std::chrono::steady_clock::now();
        std::vector<ExitRow> rows_vnew = run_polycap_verynew(pc_verynew, photons, e);
        auto t3 = std::chrono::steady_clock::now();
        std::vector<ExitRow> rows_hpp = run_polycap_hpp(pc_hpp, photons, e);
        auto t4 = std::chrono::steady_clock::now();

        double sum_lib = 0., sum_new = 0., sum_vnew = 0., sum_hpp = 0.;
        for (auto& r : rows_lib)  sum_lib  += r.w;
        for (auto& r : rows_new)  sum_new  += r.w;
        for (auto& r : rows_vnew) sum_vnew += r.w;
        for (auto& r : rows_hpp)  sum_hpp  += r.w;
        double eff_lib  = sum_lib  / (double)n_photons;
        double eff_new  = sum_new  / (double)n_photons;
        double eff_vnew = sum_vnew / (double)n_photons;
        double eff_hpp  = sum_hpp  / (double)n_photons;

        std::printf("%8.1f%14.6f%14.6f%14.6f%14.6f%12.3f%12.3f%12.3f\n",
                    e, eff_lib, eff_new, eff_vnew, eff_hpp,
                    (eff_lib > 0.) ? eff_new/eff_lib  : 0.,
                    (eff_hpp > 0.) ? eff_vnew/eff_hpp : 0.,
                    (eff_lib > 0.) ? eff_hpp/eff_lib  : 0.);

        std::printf("  timing: lib=%.0fms new=%.0fms veryNew=%.0fms hpp=%.0fms\n",
                    std::chrono::duration<double, std::milli>(t1-t0).count(),
                    std::chrono::duration<double, std::milli>(t2-t1).count(),
                    std::chrono::duration<double, std::milli>(t3-t2).count(),
                    std::chrono::duration<double, std::milli>(t4-t3).count());

        char fname[256];
        std::snprintf(fname, sizeof(fname), "test-data/out/3way_lib_E%.1f.csv", e);
        write_plane_csv(fname, rows_lib);
        std::snprintf(fname, sizeof(fname), "test-data/out/3way_new_E%.1f.csv", e);
        write_plane_csv(fname, rows_new);
        std::snprintf(fname, sizeof(fname), "test-data/out/3way_verynew_E%.1f.csv", e);
        write_plane_csv(fname, rows_vnew);
        std::snprintf(fname, sizeof(fname), "test-data/out/3way_hpp_E%.1f.csv", e);
        write_plane_csv(fname, rows_hpp);
    }

    polycap_description_free(c_desc);
    polycap_error_free(error);
    return 0;
}
