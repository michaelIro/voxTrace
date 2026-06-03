// Test-2: PolyCap benchmark using the repo-native ray adapter and source API.
// Geometry: pc-236-descr.txt  —  SiO2 polycapillary, L=4.03 cm

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "Ray.hpp"
#include "PolyCap.hpp"

// ── pc-236 geometry (test-data/api/polycap/pc-236-descr.txt) ─────────────────
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

// ── source geometry ───────────────────────────────────────────────────────────
static constexpr double SRC_DIST        = 100.0;
static constexpr double SRC_CONE_ANGLE  = PC_REXT_IN / SRC_DIST;

// ── benchmark parameters ──────────────────────────────────────────────────────
static constexpr int N_TRACE_PHOTONS    = 50000;
static constexpr int N_SIM_PHOTONS      = 1000;
static constexpr double ENERGIES[]      = {4.0, 6.0, 8.0, 10.0, 12.0};
static constexpr int N_ENERGIES         = 5;

// ── beam profile output ───────────────────────────────────────────────────────
static constexpr int N_BEAM_TRACE       = 50000;
static constexpr int N_BEAM_SIM         = 1000;

static const double BEAM_PLANE_D[5] = {
    0.0,
    PC_FOCAL_OUT * 0.5,
    PC_FOCAL_OUT,
    PC_FOCAL_OUT * 1.5,
    PC_FOCAL_OUT * 2.0
};

namespace {

double halton(int index, int base) {
    double fraction = 1.0;
    double result = 0.0;
    while (index > 0) {
        fraction /= (double)base;
        result += fraction * (double)(index % base);
        index /= base;
    }
    return result;
}

PolyCap build_optic() {
    std::vector<int> atomic_numbers(PC_Z, PC_Z + PC_NELEM);
    std::vector<double> weights(PC_WT, PC_WT + PC_NELEM);
    PolyCapProfile profile = PolyCapProfile::ellipsoidal(
        PC_LENGTH,
        PC_REXT_IN,
        PC_REXT_OUT,
        PC_RCAP_IN,
        PC_RCAP_OUT,
        PC_FOCAL_IN,
        PC_FOCAL_OUT);
    PolyCapWall wall(atomic_numbers, weights, PC_DENSITY, PC_ROUGHNESS);
    return PolyCap(profile, wall, PC_NCAP);
}

Ray make_trace_ray(double energy_keV, int index) {
    double radius = std::sqrt(halton(index + 1, 2)) * PC_REXT_IN;
    double phi = 2.0 * M_PI * halton(index + 1, 3);
    double target_x = radius * std::cos(phi);
    double target_z = radius * std::sin(phi);

    double dir_x = target_x;
    double dir_y = SRC_DIST;
    double dir_z = target_z;
    double length = std::sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
    dir_x /= length;
    dir_y /= length;
    dir_z /= length;

    Ray ray;
    ray.setStartCoordinates(0.f, (float)-SRC_DIST, 0.f);
    ray.setEndCoordinates((float)dir_x, (float)dir_y, (float)dir_z);
    ray.setSPol(1.f, 0.f, 0.f);
    ray.setPPol(0.f, 0.f, 1.f);
    ray.setEnergyKeV((float)energy_keV);
    ray.setProb(1.f);
    ray.setIAFlag(false);
    ray.setIANum(index);
    return ray;
}

PolyCapSource make_source(double energy_keV) {
    PolyCapSource source;
    source.setSourceDistanceCm(SRC_DIST);
    source.setSourceHalfSizeCm(PC_REXT_IN, PC_REXT_IN);
    source.setAngularSpread(-1.0, -1.0);
    source.setSourceShiftCm(0.0, 0.0);
    source.setHorizontalPolarization(1.0);
    (void)energy_keV;
    return source;
}

void run_trace_benchmark(const PolyCap& optic,
                         double energy_keV,
                         double& efficiency,
                         double& ms) {
    double sum_weights = 0.0;
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < N_TRACE_PHOTONS; ++i) {
        PolyCapTraceResult traced = optic.trace(make_trace_ray(energy_keV, i));
        if (traced.transmitted) {
            sum_weights += traced.primaryWeight();
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    efficiency = sum_weights / (double)N_TRACE_PHOTONS;
}

void run_simulate_benchmark(const PolyCap& optic,
                            double energy_keV,
                            double& efficiency,
                            double& ms) {
    PolyCapSource source = make_source(energy_keV);

    auto t0 = std::chrono::steady_clock::now();
    PolyCapSimulationResult result = optic.simulate(source, {energy_keV}, N_SIM_PHOTONS, 12345ULL);
    auto t1 = std::chrono::steady_clock::now();

    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    efficiency = result.energies.empty() ? 0.0 : result.energies.front().efficiency;
}

void write_trace_planes(FILE* fp, const Ray& ray, double weight) {
    double ex = ray.getStartX();
    double ez = ray.getStartZ();
    double dx = ray.getDirX();
    double dy = ray.getDirY();
    double dz = ray.getDirZ();

    for (int plane = 0; plane < 5; ++plane) {
        double dist = BEAM_PLANE_D[plane];
        double t = (std::fabs(dy) > 1.e-10) ? dist / dy : 0.0;
        std::fprintf(fp, "%d,%.6f,%.6f,%.6f\n", plane, ex + t*dx, ez + t*dz, weight);
    }
}

void write_sim_planes(FILE* fp, const PolyCapExitPhoton& photon) {
    for (int plane = 0; plane < 5; ++plane) {
        double dist = BEAM_PLANE_D[plane];
        double t = (std::fabs(photon.dy) > 1.e-10) ? dist / photon.dy : 0.0;
        double weight = photon.weights.empty() ? 0.0 : photon.weights.front();
        std::fprintf(fp, "%d,%.6f,%.6f,%.6f\n",
                     plane,
                     photon.x_exit_cm + t*photon.dx,
                     photon.z_exit_cm + t*photon.dz,
                     weight);
    }
}

void save_trace_beam_profile(const PolyCap& optic, double energy_keV) {
    char fname[256];
    std::snprintf(fname, sizeof(fname), "test-data/out/beam_trace_E%.1f.csv", energy_keV);
    FILE* fp = std::fopen(fname, "w");
    if (!fp) {
        std::fprintf(stderr, "Cannot open %s\n", fname);
        return;
    }
    std::fprintf(fp, "plane_idx,x_cm,z_cm,weight\n");

    for (int i = 0; i < N_BEAM_TRACE; ++i) {
        PolyCapTraceResult traced = optic.trace(make_trace_ray(energy_keV, i));
        if (!traced.transmitted) {
            continue;
        }
        write_trace_planes(fp, traced.ray, traced.primaryWeight());
    }

    std::fclose(fp);
    std::printf("  trace     -> %s\n", fname);
}

void save_sim_beam_profile(const PolyCap& optic, double energy_keV) {
    char fname[256];
    std::snprintf(fname, sizeof(fname), "test-data/out/beam_sim_E%.1f.csv", energy_keV);
    FILE* fp = std::fopen(fname, "w");
    if (!fp) {
        std::fprintf(stderr, "Cannot open %s\n", fname);
        return;
    }
    std::fprintf(fp, "plane_idx,x_cm,z_cm,weight\n");

    PolyCapSource source = make_source(energy_keV);
    PolyCapSimulationResult result = optic.simulate(source, {energy_keV}, N_BEAM_SIM, 67890ULL);
    for (const PolyCapExitPhoton& photon : result.photons) {
        write_sim_planes(fp, photon);
    }

    std::fclose(fp);
    std::printf("  simulate  -> %s  (%lld exits / %lld launched)\n",
                fname,
                (long long)result.transmitted_count,
                (long long)result.launched_count);
}

}  // namespace

int main() {
    std::filesystem::create_directories("test-data/out");

    PolyCap optic = build_optic();

    std::printf("pc-236  L=%.2f cm  rExt=(%.4f->%.4f)cm  rho=%.2f g/cc\n",
                PC_LENGTH, PC_REXT_IN, PC_REXT_OUT, PC_DENSITY);
    std::printf("Source: point at %.0f cm, cone half-angle=%.2e rad (->entrance r=%.4f cm)\n",
                SRC_DIST, SRC_CONE_ANGLE, PC_REXT_IN);
    std::printf("Open area fraction: %.4f\n", optic.openArea());
    std::printf("Trace photons = %d, simulate transmitted target = %d\n\n",
                N_TRACE_PHOTONS, N_SIM_PHOTONS);

    std::printf("%-12s  %-14s  %-14s  %-12s  %-12s  %-8s\n",
                "E(keV)", "trace_eff", "simulate_eff", "trace_ms", "sim_ms", "ratio");
    std::printf("%s\n", std::string(82, '-').c_str());

    for (int ei = 0; ei < N_ENERGIES; ++ei) {
        double e = ENERGIES[ei];
        double trace_eff = 0.0;
        double trace_ms = 0.0;
        double sim_eff = 0.0;
        double sim_ms = 0.0;

        run_trace_benchmark(optic, e, trace_eff, trace_ms);
        run_simulate_benchmark(optic, e, sim_eff, sim_ms);

        std::printf("%-12.1f  %-14.6f  %-14.6f  %-12.1f  %-12.1f  %.2fx\n",
                    e,
                    trace_eff,
                    sim_eff,
                    trace_ms,
                    sim_ms,
                    (trace_ms > 0.0) ? sim_ms / trace_ms : 0.0);
    }

    std::printf("\nCollecting beam profiles (planes: exit, f/2, f, 3f/2, 2f)...\n");
    std::printf("%s\n", std::string(40, '-').c_str());
    for (int ei = 0; ei < N_ENERGIES; ++ei) {
        double e = ENERGIES[ei];
        std::printf("%.1f keV\n", e);
        save_trace_beam_profile(optic, e);
        save_sim_beam_profile(optic, e);
    }

    return 0;
}