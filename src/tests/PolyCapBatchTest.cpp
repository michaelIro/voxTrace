#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

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

static constexpr double SRC_DIST        = 100.0;
static constexpr int    N_BATCH_RAYS    = 2048;

namespace {

double halton(int index, int base) {
    double fraction = 1.0;
    double result = 0.0;
    while (index > 0) {
        fraction /= static_cast<double>(base);
        result += fraction * static_cast<double>(index % base);
        index /= base;
    }
    return result;
}

PolyCap buildOptic() {
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

Ray makeTraceRay(double energyKeV, int index) {
    double radius = std::sqrt(halton(index + 1, 2)) * PC_REXT_IN;
    double phi = 2.0 * static_cast<double>(VT_PI) * halton(index + 1, 3);
    double target_x = radius * std::cos(phi);
    double target_z = radius * std::sin(phi);

    double dir_x = target_x;
    double dir_y = SRC_DIST;
    double dir_z = target_z;
    double length = std::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
    dir_x /= length;
    dir_y /= length;
    dir_z /= length;

    Ray ray;
    ray.setStartCoordinates(0.f, static_cast<float>(-SRC_DIST), 0.f);
    ray.setEndCoordinates(static_cast<float>(dir_x),
                          static_cast<float>(dir_y),
                          static_cast<float>(dir_z));
    ray.setSPol(1.f, 0.f, 0.f);
    ray.setPPol(0.f, 0.f, 1.f);
    ray.setEnergyKeV(static_cast<float>(energyKeV));
    ray.setProb(1.f);
    ray.setIAFlag(false);
    ray.setIANum(index);
    return ray;
}

bool nearlyEqual(double lhs, double rhs, double tolerance = 1.e-6) {
    return std::fabs(lhs - rhs) <= tolerance;
}

bool sameRay(const Ray& lhs, const Ray& rhs, double tolerance = 1.e-6) {
    return nearlyEqual(lhs.getStartX(), rhs.getStartX(), tolerance)
        && nearlyEqual(lhs.getStartY(), rhs.getStartY(), tolerance)
        && nearlyEqual(lhs.getStartZ(), rhs.getStartZ(), tolerance)
        && nearlyEqual(lhs.getDirX(), rhs.getDirX(), tolerance)
        && nearlyEqual(lhs.getDirY(), rhs.getDirY(), tolerance)
        && nearlyEqual(lhs.getDirZ(), rhs.getDirZ(), tolerance)
        && nearlyEqual(lhs.getProb(), rhs.getProb(), tolerance)
        && lhs.getIAFlag() == rhs.getIAFlag()
        && lhs.getIANum() == rhs.getIANum();
}

int runBatchTraceCheck() {
    PolyCap optic = buildOptic();

    const double energies[] = {8.0, 12.0, 17.4};
    std::vector<Ray> rays;
    rays.reserve(N_BATCH_RAYS);
    for (int i = 0; i < N_BATCH_RAYS; ++i) {
        rays.push_back(makeTraceRay(energies[i % 3], i));
    }

    std::vector<PolyCapTraceResult> serial_results;
    serial_results.reserve(rays.size());
    int64_t serial_transmitted = 0;
    for (const Ray& ray : rays) {
        PolyCapTraceResult traced = optic.trace(ray);
        serial_transmitted += traced.transmitted ? 1 : 0;
        serial_results.push_back(std::move(traced));
    }

    PolyCapBatchTraceResult batch = optic.traceBatch(rays);
    if (batch.transmitted_count != serial_transmitted) {
        std::fprintf(stderr,
                     "PolyCap batch mismatch: transmitted_count=%lld serial=%lld\n",
                     static_cast<long long>(batch.transmitted_count),
                     static_cast<long long>(serial_transmitted));
        return 1;
    }

    for (std::size_t i = 0; i < rays.size(); ++i) {
        const PolyCapTraceResult& serial = serial_results[i];
        const PolyCapTraceResult& traced = batch.rays[i];
        if (serial.transmitted != traced.transmitted
            || serial.reflections != traced.reflections
            || !nearlyEqual(serial.travel_distance_cm, traced.travel_distance_cm)
            || !nearlyEqual(serial.primaryWeight(), traced.primaryWeight())
            || !sameRay(serial.ray, traced.ray)) {
            std::fprintf(stderr,
                         "PolyCap batch mismatch at ray %zu: transmitted=%d/%d reflections=%lld/%lld weight=%.8f/%.8f\n",
                         i,
                         serial.transmitted ? 1 : 0,
                         traced.transmitted ? 1 : 0,
                         static_cast<long long>(serial.reflections),
                         static_cast<long long>(traced.reflections),
                         serial.primaryWeight(),
                         traced.primaryWeight());
            return 1;
        }
    }

    std::printf("PolyCap batch trace verified: %d rays, transmitted=%lld, open area=%.4f\n",
                N_BATCH_RAYS,
                static_cast<long long>(batch.transmitted_count),
                optic.openArea());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int exit_code = 0;

#if !defined(VOXTRACE_HOST_ONLY) && !defined(VOXTRACE_METAL) && !defined(__METAL_VERSION__)
    Kokkos::initialize(argc, argv);
    {
        exit_code = runBatchTraceCheck();
    }
    Kokkos::finalize();
#else
    (void)argc;
    (void)argv;
    exit_code = runBatchTraceCheck();
#endif

    return exit_code;
}