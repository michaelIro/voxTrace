<<<<<<< HEAD
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
=======
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/PolyCap.hpp"
#include "core/Ray.hpp"

#include <polycap.h>

namespace {

struct PolycapConfig {
    double length = 0.0;
    double r_ext_in = 0.0;
    double r_ext_out = 0.0;
    double r_cap_in = 0.0;
    double r_cap_out = 0.0;
    double focal_in = 0.0;
    double focal_out = 0.0;
    int nelem = 0;
    std::vector<int> atomic_numbers;
    std::vector<double> weight_percent;
    double density = 0.0;
    double roughness = 0.0;
    int64_t n_capillaries = 0;
};

struct SeedRay {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    double dz = 1.0;
    double ex = 1.0;
    double ey = 0.0;
    double ez = 0.0;
};

struct BeamPoint {
    int plane = 0;
    double x = 0.0;
    double z = 0.0;
    double w = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
};

std::string trim(const std::string& s) {
    const auto b = std::find_if_not(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c) != 0; });
    if (b == s.end()) {
        return {};
    }
    const auto e = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) { return std::isspace(c) != 0; }).base();
    return std::string(b, e);
}

std::string stripComment(const std::string& s) {
    const std::size_t p = s.find("//");
    if (p == std::string::npos) {
        return s;
    }
    return s.substr(0, p);
}

double parseScalar(const std::string& s) {
    static const std::regex num_re(R"(([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))");
    std::smatch m;
    if (!std::regex_search(s, m, num_re)) {
        throw std::runtime_error("Could not parse numeric scalar from line: " + s);
    }
    return std::stod(m.str(1));
}

std::vector<double> parseArray(const std::string& s) {
    const std::size_t l = s.find('{');
    const std::size_t r = s.find('}', l == std::string::npos ? 0 : l + 1);
    if (l == std::string::npos || r == std::string::npos || r <= l) {
        throw std::runtime_error("Could not parse array from line: " + s);
    }

    std::vector<double> out;
    std::stringstream ss(s.substr(l + 1, r - l - 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
        const std::string t = trim(token);
        if (!t.empty()) {
            out.push_back(std::stod(t));
        }
    }
    return out;
}

PolycapConfig readConfig(const std::filesystem::path& file) {
    std::ifstream in(file);
    if (!in) {
        throw std::runtime_error("Failed to open polycap parameter file: " + file.string());
    }

    std::vector<std::string> lines;
    std::string raw;
    while (std::getline(in, raw)) {
        const std::string clean = trim(stripComment(raw));
        if (!clean.empty()) {
            lines.push_back(clean);
        }
    }

    if (lines.size() < 14) {
        throw std::runtime_error("Unexpected polycap parameter format in " + file.string());
    }

    PolycapConfig cfg;
    cfg.length = parseScalar(lines[1]);
    cfg.r_ext_in = parseScalar(lines[2]);
    cfg.r_ext_out = parseScalar(lines[3]);
    cfg.r_cap_in = parseScalar(lines[4]);
    cfg.r_cap_out = parseScalar(lines[5]);
    cfg.focal_in = parseScalar(lines[6]);
    cfg.focal_out = parseScalar(lines[7]);
    cfg.nelem = static_cast<int>(std::lround(parseScalar(lines[8])));

    const std::vector<double> iz = parseArray(lines[9]);
    const std::vector<double> wi = parseArray(lines[10]);
    if (static_cast<int>(iz.size()) != cfg.nelem || static_cast<int>(wi.size()) != cfg.nelem) {
        throw std::runtime_error("Element count does not match composition array sizes in " + file.string());
    }

    cfg.atomic_numbers.reserve(iz.size());
    for (double z : iz) {
        cfg.atomic_numbers.push_back(static_cast<int>(std::lround(z)));
    }
    cfg.weight_percent = wi;

    cfg.density = parseScalar(lines[11]);
    cfg.roughness = parseScalar(lines[12]);
    cfg.n_capillaries = static_cast<int64_t>(std::llround(parseScalar(lines[13])));
    return cfg;
}

std::vector<SeedRay> makeSeedRays(std::size_t n_rays, uint64_t seed,
                                  double source_radius, double divergence,
                                  double z_start, double focal_dist) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);

    std::vector<SeedRay> rays;
    rays.reserve(n_rays);

    for (std::size_t i = 0; i < n_rays; ++i) {
        SeedRay r;
        r.x = (g(rng) / 3.0) * source_radius;
        r.y = (g(rng) / 3.0) * source_radius;
        r.z = z_start;

        // Beam points toward +z; focal_dist maps to axial distance.
        double tx = (g(rng) / 3.0) * divergence - r.x;
        double ty = (g(rng) / 3.0) * divergence - r.y;
        double tz = focal_dist - z_start;

        const double n = std::sqrt(tx * tx + ty * ty + tz * tz);
        r.dx = tx / n;
        r.dy = ty / n;
        r.dz = tz / n;

        // Start with linear polarization along x (same default used elsewhere).
        r.ex = 1.0;
        r.ey = 0.0;
        r.ez = 0.0;

        rays.push_back(r);
    }

    return rays;
}

bool projectToPlane(double exit_x, double exit_y, double exit_z,
                    double dir_x, double dir_y, double dir_z,
                    double distance_after_exit,
                    double* out_x, double* out_y, double* out_z) {
    if (std::abs(dir_z) < 1e-12) {
        return false;
    }
    const double z_plane = exit_z + distance_after_exit;
    const double t = (z_plane - exit_z) / dir_z;
    if (t < 0.0) {
        return false;
    }

    *out_x = exit_x + t * dir_x;
    *out_y = exit_y + t * dir_y;
    *out_z = z_plane;
    return true;
}

std::vector<BeamPoint> traceVoxTrace(const PolycapConfig& cfg,
                                     const std::vector<SeedRay>& seeds,
                                     double energy_keV,
                                     const std::vector<double>& plane_dists) {
    std::vector<float> wt(cfg.weight_percent.begin(), cfg.weight_percent.end());

    PolyCap optic(
        0.0f,
        static_cast<float>(cfg.length),
        static_cast<float>(cfg.r_ext_in),
        static_cast<float>(cfg.r_ext_out),
        static_cast<float>(cfg.r_cap_in),
        static_cast<float>(cfg.r_cap_out),
        static_cast<float>(cfg.focal_in),
        static_cast<float>(cfg.focal_out),
        PolyCap::CONICAL,
        cfg.nelem,
        cfg.atomic_numbers.data(),
        wt.data(),
        static_cast<float>(cfg.density),
        static_cast<float>(cfg.roughness),
        static_cast<int>(cfg.n_capillaries));

    std::vector<BeamPoint> beam;
    beam.reserve(seeds.size() * plane_dists.size());

    for (const SeedRay& s : seeds) {
        Ray ray;
        ray.setStartCoordinates(static_cast<float>(s.x), static_cast<float>(s.y), static_cast<float>(s.z));
        ray.setEndCoordinates(static_cast<float>(s.dx), static_cast<float>(s.dy), static_cast<float>(s.dz));
        ray.setSPol(static_cast<float>(s.ex), static_cast<float>(s.ey), static_cast<float>(s.ez));
        ray.setPPol(0.0f, 1.0f, 0.0f);
        ray.setProb(1.0f);
        ray.setEnergyKeV(static_cast<float>(energy_keV));

        optic.trace(ray);
        if (!ray.getIAFlag() || ray.getProb() <= 0.0f) {
            continue;
        }

        const double ex = ray.getStartX();
        const double ey = ray.getStartY();
        const double ez = ray.getStartZ();
        const double dx = ray.getDirX();
        const double dy = ray.getDirY();
        const double dz = ray.getDirZ();
        const double w = ray.getProb();

        for (std::size_t ip = 0; ip < plane_dists.size(); ++ip) {
            double px = 0.0, py = 0.0, pz = 0.0;
            if (!projectToPlane(ex, ey, ez, dx, dy, dz, plane_dists[ip], &px, &py, &pz)) {
                continue;
            }
            beam.push_back(BeamPoint{static_cast<int>(ip), px, py, w, dx, dy, dz});
        }
    }

    return beam;
}

std::vector<BeamPoint> tracePolycapLib(const PolycapConfig& cfg,
                                       const std::vector<SeedRay>& seeds,
                                       double energy_keV,
                                       const std::vector<double>& plane_dists) {
    std::vector<BeamPoint> beam;
    beam.reserve(seeds.size() * plane_dists.size());

    polycap_error* err = nullptr;
    polycap_profile* profile = polycap_profile_new(
        POLYCAP_PROFILE_CONICAL,
        cfg.length,
        cfg.r_ext_in,
        cfg.r_ext_out,
        cfg.r_cap_in,
        cfg.r_cap_out,
        cfg.focal_in,
        cfg.focal_out,
        &err);
    if (err != nullptr || profile == nullptr) {
        const std::string msg = err != nullptr ? err->message : "unknown profile error";
        if (err != nullptr) {
            polycap_clear_error(&err);
        }
        throw std::runtime_error("polycap_profile_new failed: " + msg);
    }

    std::vector<int> iz = cfg.atomic_numbers;
    std::vector<double> wi = cfg.weight_percent;

    polycap_description* desc = polycap_description_new(
        profile,
        cfg.roughness,
        cfg.n_capillaries,
        static_cast<unsigned int>(cfg.nelem),
        iz.data(),
        wi.data(),
        cfg.density,
        &err);
    if (err != nullptr || desc == nullptr) {
        const std::string msg = err != nullptr ? err->message : "unknown description error";
        if (err != nullptr) {
            polycap_clear_error(&err);
        }
        polycap_profile_free(profile);
        throw std::runtime_error("polycap_description_new failed: " + msg);
    }

    for (const SeedRay& s : seeds) {
        // polycap C API requires positive z for both start coordinate and
        // propagation direction at launch.
        polycap_vector3 sc{s.x, s.y, std::abs(s.z)};
        polycap_vector3 sd{s.dx, s.dy, std::abs(s.dz)};
        polycap_vector3 se{s.ex, s.ey, s.ez};

        polycap_photon* photon = polycap_photon_new(desc, sc, sd, se, &err);
        if (err != nullptr || photon == nullptr) {
            const std::string msg = err != nullptr ? err->message : "unknown photon creation error";
            if (err != nullptr) {
                polycap_clear_error(&err);
            }
            polycap_description_free(desc);
            polycap_profile_free(profile);
            throw std::runtime_error("polycap_photon_new failed: " + msg);
        }

        double e_arr[1] = {energy_keV};
        double* w_arr = nullptr;
        const int status = polycap_photon_launch(photon, 1, e_arr, &w_arr, false, &err);

        if (err != nullptr) {
            const std::string msg = err->message;
            polycap_clear_error(&err);
            if (w_arr != nullptr) {
                polycap_free(w_arr);
            }
            polycap_photon_free(photon);
            polycap_description_free(desc);
            polycap_profile_free(profile);
            throw std::runtime_error("polycap_photon_launch failed: " + msg);
        }

        if (status == 1 && w_arr != nullptr && w_arr[0] > 0.0) {
            const polycap_vector3 ec = polycap_photon_get_exit_coords(photon);
            const polycap_vector3 ed = polycap_photon_get_exit_direction(photon);

            const double ex = ec.x;
            const double ey = ec.y;
            const double ez = ec.z;
            const double dx = ed.x;
            const double dy = ed.y;
            const double dz = ed.z;

            for (std::size_t ip = 0; ip < plane_dists.size(); ++ip) {
                double px = 0.0, py = 0.0, pz = 0.0;
                if (!projectToPlane(ex, ey, ez, dx, dy, dz, plane_dists[ip], &px, &py, &pz)) {
                    continue;
                }
                beam.push_back(BeamPoint{static_cast<int>(ip), px, py, w_arr[0], dx, dy, dz});
            }
        }

        if (w_arr != nullptr) {
            polycap_free(w_arr);
        }
        polycap_photon_free(photon);
    }

    polycap_description_free(desc);
    polycap_profile_free(profile);
    return beam;
}

void writeBeamCsv(const std::filesystem::path& file, const std::vector<BeamPoint>& beam) {
    std::ofstream out(file);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + file.string());
    }

    out << "plane,x_cm,z_cm,weight,dir_x,dir_y,dir_z\n";
    out << std::fixed << std::setprecision(9);
    for (const BeamPoint& p : beam) {
        out << p.plane << ','
            << p.x << ','
            << p.z << ','
            << p.w << ','
            << p.dx << ','
            << p.dy << ','
            << p.dz << '\n';
    }
}

} // namespace

int main() {
    try {
        const std::filesystem::path base = std::filesystem::current_path();
        const std::filesystem::path cfg_file = base / "test-data/simulation/nist-1107/Polycapillary.txt";
        const std::filesystem::path out_dir = base / "test-data/out";
        std::filesystem::create_directories(out_dir);

        const PolycapConfig cfg = readConfig(cfg_file);

        std::cout << "Loaded polycap config from " << cfg_file << "\n";
        std::cout << "L=" << cfg.length
                  << " rExtIn=" << cfg.r_ext_in
                  << " rExtOut=" << cfg.r_ext_out
                  << " focalIn=" << cfg.focal_in
                  << " focalOut=" << cfg.focal_out
                  << " nCap=" << cfg.n_capillaries << "\n";

        const std::vector<double> energies = {4.0, 6.0, 8.0, 10.0, 12.0};

        // Keep plane spacing consistent with plot_beam_comparison.py expectations.
        const double plane_focal = cfg.focal_in;
        const std::vector<double> plane_dists = {
            0.0,
            0.5 * plane_focal,
            1.0 * plane_focal,
            1.5 * plane_focal,
            2.0 * plane_focal
        };

        const std::size_t n_rays = 100000;
        const auto seeds = makeSeedRays(
            n_rays,
            42ULL,
            0.002,   // 20 um source radius envelope (3 sigma)
            0.003,   // 3 mrad divergence envelope (3 sigma)
            -0.50,   // start before optic entrance
            0.50);   // focal distance for initial direction generation

        for (double e : energies) {
            const auto vt_beam = traceVoxTrace(cfg, seeds, e, plane_dists);
            const auto pc_beam = tracePolycapLib(cfg, seeds, e, plane_dists);

            std::ostringstream vt_name;
            vt_name << "beam_vt_E" << std::fixed << std::setprecision(1) << e << ".csv";
            std::ostringstream pc_name;
            pc_name << "beam_pc_E" << std::fixed << std::setprecision(1) << e << ".csv";

            const std::filesystem::path vt_file = out_dir / vt_name.str();
            const std::filesystem::path pc_file = out_dir / pc_name.str();

            writeBeamCsv(vt_file, vt_beam);
            writeBeamCsv(pc_file, pc_beam);

            std::cout << "E=" << e << " keV: wrote "
                      << vt_beam.size() << " vt samples -> " << vt_file << " ; "
                      << pc_beam.size() << " polycap samples -> " << pc_file << "\n";
        }

        std::cout << "Done. Use src/tests/plot_beam_comparison.py to plot position comparison.\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Test-2 failed: " << ex.what() << "\n";
        return 1;
    }
}
>>>>>>> cdcd280 (123)
