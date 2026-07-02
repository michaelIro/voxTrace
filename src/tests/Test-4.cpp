// Test-4: X-ray reflectivity (XRR) of a layered stack
//
//   ambient (vacuum) → thin film(s) → substrate   →   R(θ) specular curve
//
// Unlike the Monte-Carlo ray-trace tests, XRR is a *coherent* (wave-optics)
// calculation: the specular reflectivity vs grazing angle comes from the
// interference of the Fresnel reflections at every interface. We solve it with
// the Abelès transfer-matrix method. The only material physics needed is the
// complex refractive index n = 1 − δ − iβ of each layer, and that is exactly
// what voxTrace already uses for the polycapillary walls (PolyCap::refractiveIndex):
//   δ = (r_e λ²/2π) Σ_i N_i (Z_i + f'_i)     N_i = ρ w_i N_A / A_i
//   β = λ μ / 4π                              μ  = ρ Σ_i w_i (μ/ρ)_i
// taken here from xraylib (A, Fi, CS_Tot) — so XRR reuses the package's optics.
//
// This same per-layer (δ,β) + transfer-matrix kernel is the foundation for
// GIXRF/TXRF: the matrix also yields the depth/angle field intensity |E(z,θ)|²
// that would drive the grazing-incidence fluorescence (see notes at the end).
//
// Build: make test4      Run: ./build/src/Test4 [energy_keV]
// Plot:  python3 src/tests/plot_xrr.py

#include <cstdio>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <vector>

#include "../api/XRayLibAPI.hpp"

using cd = std::complex<double>;

namespace {

constexpr double RE_CM  = 2.8179403262e-13;  // classical electron radius [cm]
constexpr double NAVOG  = 6.02214076e23;     // Avogadro [1/mol]
constexpr double KEV_A  = 12.39841984;       // λ[Å]·E[keV]

struct Element { int Z; double w; };          // atomic number, mass fraction

// A material layer. thickness/roughness in cm; thickness 0 ⇒ semi-infinite
// (ambient or substrate). An empty composition is vacuum (n = 1).
struct Layer {
    const char*          name;
    std::vector<Element> comp;
    double               rho;       // density [g/cm³]
    double               d;         // thickness [cm] (0 = semi-infinite)
    double               rough;     // r.m.s. roughness of its top interface [cm]
};

// Complex refractive index n = 1 − δ − iβ of a layer at energy E [keV].
cd refractiveIndex(const Layer& L, double E_keV) {
    if (L.comp.empty()) return cd(1.0, 0.0);                       // vacuum
    double lambda = KEV_A / E_keV * 1e-8;                          // [cm]
    double delta = 0.0, mu = 0.0;
    for (const Element& e : L.comp) {
        double A  = XRayLibAPI::A(e.Z);                            // [g/mol]
        double Ni = L.rho * e.w * NAVOG / A;                         // atoms/cm³
        delta += Ni * (e.Z + XRayLibAPI::Fi(e.Z, E_keV));         // (Z + f')
        mu    += L.rho * e.w * XRayLibAPI::CS_Tot(e.Z, E_keV);    // [1/cm]
    }
    delta *= RE_CM * lambda * lambda / (2.0 * M_PI);
    double beta = lambda * mu / (4.0 * M_PI);
    return cd(1.0 - delta, -beta);
}

// 2×2 complex matrix
struct Mat2 {
    cd a, b, c, d;
    Mat2 operator*(const Mat2& o) const {
        return {a*o.a + b*o.c, a*o.b + b*o.d,
                c*o.a + d*o.c, c*o.b + d*o.d};
    }
};

// Specular reflectivity R = |r|² of the stack at grazing angle θ [rad].
// layers[0] = ambient, layers.back() = substrate (both semi-infinite).
double reflectivity(const std::vector<Layer>& layers, double E_keV, double theta) {
    const int N = (int)layers.size();
    double lambda = KEV_A / E_keV * 1e-8;
    double k0 = 2.0 * M_PI / lambda;
    double c2 = std::cos(theta) * std::cos(theta);

    std::vector<cd> kz(N);                                         // ⊥ wavevector / layer
    for (int j = 0; j < N; ++j) {
        cd n = refractiveIndex(layers[j], E_keV);
        kz[j] = k0 * std::sqrt(n*n - c2);
    }

    // Abelès interface matrix between layers j and j+1 (Névot–Croce roughness).
    auto interface = [&](int j) {
        cd ka = kz[j], kb = kz[j+1];
        cd r  = (ka - kb) / (ka + kb) * std::exp(-2.0 * ka * kb * layers[j+1].rough * layers[j+1].rough);
        cd t  = 2.0 * ka / (ka + kb);
        return Mat2{ 1.0/t, r/t, r/t, 1.0/t };
    };
    // Propagation matrix through finite layer j.
    auto propagate = [&](int j) {
        cd phi = kz[j] * layers[j].d;
        return Mat2{ std::exp(cd(0,-1)*phi), 0, 0, std::exp(cd(0,1)*phi) };
    };

    Mat2 M = interface(0);                                         // ambient | layer 1
    for (int j = 1; j < N - 1; ++j) {                             // finite interior layers
        M = M * propagate(j);
        M = M * interface(j);
    }
    cd r = M.c / M.a;
    return std::norm(r);
}

double critAngleDeg(const Layer& L, double E_keV) {
    double delta = 1.0 - refractiveIndex(L, E_keV).real();
    return std::sqrt(2.0 * delta) * 180.0 / M_PI;
}

}  // namespace

int main(int argc, char* argv[]) {
    double E = (argc > 1) ? std::atof(argv[1]) : 8.04;            // Cu-Kα default
    std::filesystem::create_directories("test-data/out");

    // ── three test systems ────────────────────────────────────────────────────
    Layer vacuum  {"vacuum", {},                 0.0,   0.0, 0.0};
    Layer si_sub  {"Si",     {{14,1.0}},         2.33,  0.0, 3e-8};       // 3 Å rough
    Layer sio2_sub{"SiO2",   {{14,0.467},{8,0.533}}, 2.20, 0.0, 3e-8};

    std::vector<Layer> bareSi = { vacuum, si_sub };
    std::vector<Layer> niSi   = { vacuum,
                                  {"Ni", {{28,1.0}}, 8.90, 50e-7, 3e-8},  // 50 nm Ni
                                  si_sub };
    std::vector<Layer> ptSiO2 = { vacuum,
                                  {"Pt", {{78,1.0}}, 21.45, 30e-7, 3e-8}, // 30 nm Pt
                                  sio2_sub };

    std::printf("X-ray reflectivity  (E = %.3f keV, λ = %.4f Å)\n", E, KEV_A/E);
    std::printf("  critical angles:  Si %.4f°   Ni %.4f°   Pt %.4f°\n",
                critAngleDeg(si_sub, E), critAngleDeg(niSi[1], E), critAngleDeg(ptSiO2[1], E));
    double lambdaA = KEV_A / E;
    std::printf("  Kiessig spacing (≈λ/2d):  Ni 50nm %.4f°   Pt 30nm %.4f°\n",
                lambdaA*1e-1/(2*50.0)*180.0/M_PI,    // λ[nm]/(2 d[nm]) → rad → deg
                lambdaA*1e-1/(2*30.0)*180.0/M_PI);

    // ── sweep grazing angle and write the curves ──────────────────────────────
    std::ofstream f("test-data/out/xrr.csv");
    f << "theta_deg,Si,Ni50nm_Si,Pt30nm_SiO2\n";
    const double th0 = 0.01, th1 = 3.0, dth = 0.004;             // degrees
    for (double tdeg = th0; tdeg <= th1; tdeg += dth) {
        double th = tdeg * M_PI / 180.0;
        f << tdeg << ","
          << reflectivity(bareSi, E, th) << ","
          << reflectivity(niSi,   E, th) << ","
          << reflectivity(ptSiO2, E, th) << "\n";
    }
    std::printf("\nReflectivity curves → test-data/out/xrr.csv\n");

    // ── sanity check: R just below θc(Si) should be ≈1 (total reflection) ──────
    double thc = critAngleDeg(si_sub, E);
    std::printf("  check: R_Si(0.5·θc) = %.4f   R_Si(2·θc) = %.2e\n",
                reflectivity(bareSi, E, 0.5*thc*M_PI/180.0),
                reflectivity(bareSi, E, 2.0*thc*M_PI/180.0));
    return 0;
}
