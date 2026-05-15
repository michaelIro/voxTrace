// Test-2: Cone-beam transmission benchmark (point source at SRC_DIST)
// Compares voxTrace (Kokkos, full complex Fresnel + Debye-Waller) vs polycap.
// Geometry: pc-236-descr.txt  —  SiO2 polycapillary, L=4.03 cm

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>

// ── voxTrace physics ───────────────────────────────────────────────────────────
// Include Kokkos FIRST so its internals are processed before polycap.h
// defines R0 / N_AVOG / HC macros that conflict with Kokkos template params.
#include "Platform.hpp"
#include "Ray.hpp"
#include "PolyCap.hpp"
#include <Kokkos_Core.hpp>

// ── polycap C library ──────────────────────────────────────────────────────────
// Undef the clashing macros polycap.h emits before including it.
#ifdef R0
#  undef R0
#endif
#ifdef HC
#  undef HC
#endif
#ifdef N_AVOG
#  undef N_AVOG
#endif
#include <polycap.h>

// ── xraylib ─────────────────────────────────────────────────────────────────
#include <xraylib.h>

// ── pc-236 geometry (test-data/in/polycap/pc-236-descr.txt) ──────────────────
static constexpr double PC_LENGTH       = 4.03;       // cm
static constexpr double PC_REXT_IN      = 0.3175;     // cm entrance ext radius
static constexpr double PC_REXT_OUT     = 0.095;      // cm exit ext radius
static constexpr double PC_RCAP_IN      = 3.25e-4;    // cm capillary radius in
static constexpr double PC_RCAP_OUT     = 9.75e-5;    // cm capillary radius out
static constexpr double PC_FOCAL_IN     = 1e8;        // cm
static constexpr double PC_FOCAL_OUT    = 0.49;       // cm (effectively inf)
static constexpr int    PC_NELEM        = 2;
static constexpr int    PC_Z[2]         = {8, 14};    // O, Si
static constexpr double PC_WT[2]        = {53.0, 47.0};
static constexpr double PC_DENSITY      = 2.23;       // g/cm³
static constexpr double PC_ROUGHNESS    = 5.0;        // Ångström
static constexpr int64_t PC_NCAP        = 240000;

// ── source geometry ───────────────────────────────────────────────────────────
// Point source at (0, -SRC_DIST, 0); cone half-angle = PC_REXT_IN / SRC_DIST.
// Both voxTrace and polycap use this geometry for a fair comparison.
static constexpr double SRC_DIST        = 100.0;      // cm source-to-entrance
static constexpr double SRC_CONE_ANGLE  = PC_REXT_IN / SRC_DIST; // rad, ~9.5e-4

// ── benchmark parameters ──────────────────────────────────────────────────────
static constexpr int N_PHOTONS          = 100'000;
static constexpr double ENERGIES[]      = {4.0, 6.0, 8.0, 10.0, 12.0};
static constexpr int    N_ENERGIES      = 5;

// ─────────────────────────────────────────────────────────────────────────────
// Section A: voxTrace — Kokkos parallel_reduce
// Cone beam from point source (0, -SRC_DIST, 0) toward entrance disk.
// Rays directed toward a random point in the entrance aperture (radius rExtIn),
// so sinTheta = r/SRC_DIST — each photon gets a real grazing angle → reflections.
// Open-area stochastic rejection models capillary packing efficiency.
// mu_rho is the linear attenuation coefficient for the optic glass at this energy,
// used inside PolyCap::trace for the full complex Fresnel reflectivity.
// ─────────────────────────────────────────────────────────────────────────────
static void runVoxTrace(PolyCap optic,      // by value: setMuRho per energy
                        double energyKeV,
                        double& efficiency,
                        double& ms)
{
    const int N = N_PHOTONS;
    double sumW = 0.0;

    // ── Per-energy linear attenuation coefficient via xraylib ─────────────────
    // mu [cm²/g] = weighted sum of element cross-sections (photoelectric + scatter)
    // mu_rho = mu * density [cm⁻¹]
    float mu = 0.f;
    for (int i = 0; i < PC_NELEM; ++i)
        mu += (float)(PC_WT[i] / 100.0) * (float)CS_Total(PC_Z[i], energyKeV, nullptr);
    optic.setMuRho(mu * (float)PC_DENSITY);

    // Open area fraction: fraction of entrance face covered by capillary openings.
    // Photons landing in glass walls are rejected before tracing.
    const float openArea = (float)(PC_NCAP * PC_RCAP_IN * PC_RCAP_IN
                                   / (PC_REXT_IN * PC_REXT_IN));

    auto t0 = std::chrono::steady_clock::now();

    Kokkos::parallel_reduce(
        "vt_bench", N,
        KOKKOS_LAMBDA(int idx, double& lsum) {
            // Portable RNG seeded uniquely per photon
            RNG rng(775289ULL ^ (uint64_t)idx * 6364136223846793005ULL);

            // Halton sequence (base 2/3) for uniform disk sampling
            float h2 = 0.f, h3 = 0.f, f = 0.5f, g = 1.f/3.f;
            int n2 = idx + 1, n3 = idx + 1;
            while (n2) { h2 += f * (float)(n2 & 1); f *= 0.5f; n2 >>= 1; }
            while (n3) { int d = n3 % 3; h3 += g * (float)d; g /= 3.f; n3 /= 3; }

            // Target point on entrance disk (radius rExtIn)
            float r   = sqrtf(h2) * (float)PC_REXT_IN;
            float phi = 2.f * VT_PI * h3;
            float ex  = r * cosf(phi);          // entrance point x
            float ez  = r * sinf(phi);          // entrance point z

            // Reject photons that land in capillary wall (stochastic open area)
            if (rng.frand() >= openArea) return;

            // Direction from point source (0,-SRC_DIST,0) toward entrance (ex,0,ez)
            float ddx = ex;
            float ddy = (float)SRC_DIST;
            float ddz = ez;
            float dl  = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
            ddx /= dl; ddy /= dl; ddz /= dl;

            Ray ray(
                0.f, -(float)SRC_DIST, 0.f,  // start at point source
                ddx, ddy, ddz,                // direction toward entrance disk
                1.f, 0.f, 0.f,
                false, 0.f, idx,
                0.f, 0.f, 0.f,
                ddx, ddy, ddz,
                1.f
            );
            ray.setEnergyKeV((float)energyKeV);

            optic.trace(ray);

            if (ray.getIAFlag())
                lsum += (double)ray.getProb();
        },
        sumW
    );
    Kokkos::fence();

    auto t1 = std::chrono::steady_clock::now();
    ms         = std::chrono::duration<double, std::milli>(t1 - t0).count();
    efficiency = sumW / (double)N;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section B: polycap library — Pierre Tack's reference implementation
// ─────────────────────────────────────────────────────────────────────────────
static void runPolycap(double energyKeV,
                       double& efficiency,
                       double& ms)
{
    polycap_error* err = nullptr;

    int    iz[PC_NELEM]; double wi[PC_NELEM];
    for (int i = 0; i < PC_NELEM; ++i) { iz[i] = PC_Z[i]; wi[i] = PC_WT[i]; }

    polycap_profile* prof = polycap_profile_new(
        POLYCAP_PROFILE_ELLIPSOIDAL,
        PC_LENGTH,
        PC_REXT_IN,  PC_REXT_OUT,
        PC_RCAP_IN,  PC_RCAP_OUT,
        PC_FOCAL_IN, PC_FOCAL_OUT,
        &err);

    polycap_description* desc = polycap_description_new(
        prof,
        PC_ROUGHNESS, PC_NCAP,
        PC_NELEM, iz, wi,
        PC_DENSITY, &err);
    polycap_profile_free(prof);

    double energies[1] = { energyKeV };
    // Homogeneous illumination of the entrance disk: polycap samples all angles
    // within capillary acceptance (sigx=sigy=-1), src disk = optic entrance.
    // Efficiency returned = iexit / total_simulated; divide by openArea to get
    // per-capillary reflectivity comparable to voxTrace's cone-beam result.
    polycap_source* src = polycap_source_new(
        desc,
        SRC_DIST,
        PC_REXT_IN, PC_REXT_IN,     // source disk = entrance disk
        -1.0, -1.0,                  // homogeneous illumination
        0.0, 0.0,                    // no shift
        1.0,                         // polarisation
        1, energies, &err);
    polycap_description_free(desc);

    auto t0 = std::chrono::steady_clock::now();

    polycap_transmission_efficiencies* eff =
        polycap_source_get_transmission_efficiencies(
            src, -1, N_PHOTONS, false, nullptr, &err);

    auto t1 = std::chrono::steady_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    efficiency = 0.0;
    if (err) {
        fprintf(stderr, "polycap error: %s\n", err->message);
        polycap_error_free(err); err = nullptr;
    }
    if (eff) {
        double* effs = nullptr;
        polycap_transmission_efficiencies_get_data(eff, nullptr, nullptr, &effs, &err);
        if (err) { polycap_error_free(err); err = nullptr; }
        if (effs) { efficiency = effs[0]; polycap_free(effs); }
        polycap_transmission_efficiencies_free(eff);
    }

    polycap_source_free(src);
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        // Build voxTrace optic once (host + device trivially-copyable)
        int   izf[PC_NELEM]; float wtf[PC_NELEM];
        for (int i = 0; i < PC_NELEM; ++i) { izf[i] = PC_Z[i]; wtf[i] = (float)PC_WT[i]; }

        PolyCap optic(
            0.f,                     // posY (entrance at y=0)
            (float)PC_LENGTH,
            (float)PC_REXT_IN,  (float)PC_REXT_OUT,
            (float)PC_RCAP_IN,  (float)PC_RCAP_OUT,
            (float)PC_FOCAL_IN, (float)PC_FOCAL_OUT,
            PC_NELEM, izf, wtf,
            (float)PC_DENSITY, (float)PC_ROUGHNESS, (int)PC_NCAP);

        // Open area fraction (analytical): N_cap * r_cap_in^2 / r_ext_in^2
        double openArea = (double)PC_NCAP * PC_RCAP_IN * PC_RCAP_IN
                          / (PC_REXT_IN * PC_REXT_IN);

        printf("pc-236  L=%.2f cm  rExt=(%.4f->%.4f)cm  rho=%.2f g/cc\n",
               PC_LENGTH, PC_REXT_IN, PC_REXT_OUT, PC_DENSITY);
        printf("Source: point at %.0f cm, cone half-angle=%.2e rad (->entrance r=%.4f cm)\n",
               SRC_DIST, SRC_CONE_ANGLE, PC_REXT_IN);
        printf("Open area fraction (analytical): %.4f\n", openArea);
        printf("N_PHOTONS = %d\n\n", N_PHOTONS);

        // voxTrace: eff = transmitted/total  (incl. open-area rejection and
        //   per-bounce complex Fresnel + Debye-Waller reflectivity)
        // polycap:  eff = iexit/total_simulated  (polycap continues until
        //   iexit=N_PHOTONS; divide by openArea for per-capillary comparison)
        printf("%-12s  %-14s  %-14s  %-14s  %-12s  %-12s  %-8s\n",
               "E(keV)", "vt_eff", "pc_eff_abs", "pc_eff/OAF",
               "vt_ms", "pc_ms", "ratio");
        printf("%s\n", std::string(90, '-').c_str());

        for (int ei = 0; ei < N_ENERGIES; ++ei) {
            double e = ENERGIES[ei];
            double vtEff, vtMs, pcEff, pcMs;

            runVoxTrace(optic, e, vtEff, vtMs);
            runPolycap(e, pcEff, pcMs);

            printf("%-12.1f  %-14.6f  %-14.6f  %-14.6f  %-12.1f  %-12.1f  %.2fx\n",
                   e, vtEff, pcEff, pcEff / openArea, vtMs, pcMs,
                   (vtMs > 0) ? pcMs / vtMs : 0.0);
        }
    }
    Kokkos::finalize();
    return 0;
}
