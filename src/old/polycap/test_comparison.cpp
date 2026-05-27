// test_comparison.cpp
// Traces the polycapillary defined in Polycapillary.txt using both:
//   (1) The original polycap-1.2 C library
//   (2) The header-only PolyCap.hpp translation
// Writes exit-plane photon data to CSV files for subsequent plotting.
//
// Build (adjust include/library paths as needed):
//   g++ -std=c++17 -O2 -I./polycap-1.2/include \
//       test_comparison.cpp -o test_comparison \
//       -lpolycap -lxraylib -lm
//
// Usage:  ./test_comparison [n_photons] [seed]
//   n_photons: number of transmitted photons to collect (default 5000)
//   seed:      RNG seed; 0 = random (default 42 for reproducibility)

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>

// ── original polycap C library ───────────────────────────────────────────────
#ifdef __cplusplus
extern "C" {
#endif
#include <polycap.h>
#include <polycap-source.h>
#include <polycap-profile.h>
#include <polycap-description.h>
#include <polycap-transmission-efficiencies.h>
#include <polycap-error.h>
#include <polycap-rng.h>
#ifdef __cplusplus
}
#endif

// ── header-only translation ──────────────────────────────────────────────────
#include "PolyCap.hpp"

// ═══════════════════════════ Optic parameters ════════════════════════════════
// From Polycapillary.txt:
static const double LENGTH           = 4.03;        // cm
static const double RAD_EXT_UPSTREAM = 0.3175;      // cm
static const double RAD_EXT_DSTREAM  = 0.095;       // cm
static const double RAD_INT_UPSTREAM = 0.000325;    // cm
static const double RAD_INT_DSTREAM  = 0.0000975;   // cm
static const double FOCAL_DIST_UP    = 100000000.0; // cm (≈ infinity)
static const double FOCAL_DIST_DOWN  = 0.49;        // cm
static const int64_t N_CAP           = 240000;
static const double DENSITY          = 2.23;        // g/cm³
static const double SIG_ROUGH        = 5.0;         // Å
// Elements: SiO₂ glass  O(Z=8) 53 wt%, Si(Z=14) 47 wt%
static const int    IZ[2]            = {8, 14};
static const double WI[2]            = {0.53, 0.47};

// Source parameters (representative divergent source far from optic)
static const double D_SOURCE  = 500.0;    // cm
static const double SRC_X     = 0.01;     // cm
static const double SRC_Y     = 0.01;     // cm
static const double SRC_SIGX  = -1.0;     // <0 → uniform over PC entrance
static const double SRC_SIGY  = -1.0;
static const double SRC_SHFTX = 0.0;
static const double SRC_SHFTY = 0.0;
static const double HOR_POL   = 0.9;

// Energies: 1, 2, … , 25 keV
static std::vector<double> make_energies() {
    std::vector<double> e;
    for (int i = 1; i <= 25; ++i) e.push_back((double)i);
    return e;
}

// ─────────────────────────── CSV helpers ─────────────────────────────────────
static void write_csv(const std::string& path,
                       const std::vector<std::vector<double>>& cols,
                       const std::vector<std::string>& headers) {
    std::ofstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return; }
    for (int i = 0; i < (int)headers.size(); ++i)
        f << headers[i] << (i+1 < (int)headers.size() ? "," : "\n");
    int nrows = (int)cols[0].size();
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < (int)cols.size(); ++c)
            f << std::setprecision(12) << cols[c][r]
              << (c+1 < (int)cols.size() ? "," : "\n");
    }
}

// ═══════════════════════════ 1. polycap library ═══════════════════════════════
static int run_polycap(int n_photons, uint64_t seed,
                        const std::vector<double>& energies) {
    std::cout << "\n── Running polycap library (" << n_photons
              << " photons) ──────────────────\n";
    auto t0 = std::chrono::steady_clock::now();

    polycap_error* error = nullptr;

    // Profile
    polycap_profile* profile = polycap_profile_new(
        POLYCAP_PROFILE_ELLIPSOIDAL,
        LENGTH,
        RAD_EXT_UPSTREAM, RAD_EXT_DSTREAM,
        RAD_INT_UPSTREAM, RAD_INT_DSTREAM,
        FOCAL_DIST_UP,    FOCAL_DIST_DOWN,
        &error);
    if (!profile) {
        std::cerr << "polycap_profile_new failed: "
                  << (error ? error->message : "?") << "\n";
        polycap_error_free(error); return -1;
    }

    // Description
    int iz_mut[2] = {IZ[0], IZ[1]};
    double wi_mut[2] = {WI[0], WI[1]};
    polycap_description* desc = polycap_description_new(
        profile,
        SIG_ROUGH, N_CAP,
        2, iz_mut, wi_mut,   // n_elements, Z array, weight fractions
        DENSITY,
        &error);
    polycap_profile_free(profile);
    if (!desc) {
        std::cerr << "polycap_description_new failed: "
                  << (error ? error->message : "?") << "\n";
        polycap_error_free(error); return -1;
    }

    // Source
    std::vector<double> e_copy = energies;  // non-const copy for C API
    polycap_source* src = polycap_source_new(
        desc,
        D_SOURCE, SRC_X, SRC_Y, SRC_SIGX, SRC_SIGY,
        SRC_SHFTX, SRC_SHFTY, HOR_POL,
        (size_t)e_copy.size(), e_copy.data(),
        &error);
    polycap_description_free(desc);
    if (!src) {
        std::cerr << "polycap_source_new failed: "
                  << (error ? error->message : "?") << "\n";
        polycap_error_free(error); return -1;
    }

    // Simulate (progress_monitor = NULL → no callback)
    polycap_transmission_efficiencies* effs =
        polycap_source_get_transmission_efficiencies(src, -1, n_photons, false, nullptr, &error);
    polycap_source_free(src);
    if (!effs) {
        std::cerr << "polycap_source_get_transmission_efficiencies failed: "
                  << (error ? error->message : "?") << "\n";
        polycap_error_free(error); return -1;
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1-t0).count();
    std::cout << "  Elapsed: " << std::fixed << std::setprecision(2) << elapsed << " s\n";

    // Extract exit data
    int64_t         n_exit   = 0;
    polycap_vector3 *ex_coord = nullptr, *ex_dir = nullptr, *ex_elecv = nullptr;
    int64_t         *n_refl  = nullptr;
    double          *d_travel = nullptr;
    size_t           n_energies_out = 0;
    double          **ex_weights = nullptr;

    bool ok = polycap_transmission_efficiencies_get_exit_data(
        effs, &n_exit,
        &ex_coord, &ex_dir, &ex_elecv,
        &n_refl, &d_travel,
        &n_energies_out, &ex_weights,
        &error);
    if (!ok) {
        std::cerr << "get_exit_data failed: "
                  << (error ? error->message : "?") << "\n";
        polycap_error_free(error);
        polycap_transmission_efficiencies_free(effs);
        return -1;
    }

    // Efficiencies via polycap_transmission_efficiencies_get_data
    size_t n_en = 0;
    double* eff_vals = nullptr;
    double* en_vals  = nullptr;
    polycap_transmission_efficiencies_get_data(effs, &n_en, &en_vals, &eff_vals, &error);
    free(en_vals);
    std::cout << "  Exit photons: " << n_exit << "\n";
    std::cout << "  Efficiencies (polycap): ";
    for (size_t e = 0; e < n_en && e < 5; ++e)
        std::cout << std::setprecision(4) << eff_vals[e] << " ";
    if (n_en > 5) std::cout << "...";
    std::cout << "\n";
    free(eff_vals);

    // Build CSV columns: x,y,z,dx,dy,dz,w1,w2,...,wN
    std::vector<std::string> headers;
    headers.push_back("x");  headers.push_back("y");  headers.push_back("z");
    headers.push_back("dx"); headers.push_back("dy"); headers.push_back("dz");
    for (int e = 0; e < (int)n_en; ++e) headers.push_back("w" + std::to_string(e+1));

    std::vector<std::vector<double>> cols(6 + n_en, std::vector<double>(n_exit));
    for (int64_t i = 0; i < n_exit; ++i) {
        cols[0][i] = ex_coord[i].x;
        cols[1][i] = ex_coord[i].y;
        cols[2][i] = ex_coord[i].z;
        cols[3][i] = ex_dir[i].x;
        cols[4][i] = ex_dir[i].y;
        cols[5][i] = ex_dir[i].z;
        for (size_t e = 0; e < n_en; ++e)
            cols[6+e][i] = ex_weights[i][e];
    }
    write_csv("polycap_exit.csv", cols, headers);
    std::cout << "  Written: polycap_exit.csv\n";

    // Cleanup
    free(ex_coord); free(ex_dir); free(ex_elecv);
    free(n_refl);   free(d_travel);
    for (int64_t i = 0; i < n_exit; ++i) free(ex_weights[i]);
    free(ex_weights);
    polycap_transmission_efficiencies_free(effs);
    polycap_error_free(error);
    return 0;
}

// ═══════════════════════════ 2. PolyCap.hpp ══════════════════════════════════
static int run_polycap_hpp(int n_photons, uint64_t seed,
                            const std::vector<double>& energies) {
    std::cout << "\n── Running PolyCap.hpp (" << n_photons
              << " photons) ────────────────────\n";
    auto t0 = std::chrono::steady_clock::now();

    // Build optic description
    PC::Profile prof = PC::Profile::ellipsoidal(
        LENGTH,
        RAD_EXT_UPSTREAM, RAD_EXT_DSTREAM,
        RAD_INT_UPSTREAM, RAD_INT_DSTREAM,
        FOCAL_DIST_UP, FOCAL_DIST_DOWN);

    PC::Description desc(
        std::move(prof),
        SIG_ROUGH, N_CAP,
        {IZ[0], IZ[1]},
        {WI[0], WI[1]},
        DENSITY);

    // Source parameters
    PC::SourceParams sp;
    sp.d_source   = D_SOURCE;
    sp.src_x      = SRC_X;
    sp.src_y      = SRC_Y;
    sp.src_sigx   = SRC_SIGX;
    sp.src_sigy   = SRC_SIGY;
    sp.src_shiftx = SRC_SHFTX;
    sp.src_shifty = SRC_SHFTY;
    sp.hor_pol    = HOR_POL;
    sp.energies   = energies;

    PC::SimResult result = PC::simulate(sp, desc, n_photons, seed);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1-t0).count();
    std::cout << "  Elapsed: " << std::fixed << std::setprecision(2) << elapsed << " s\n";
    std::cout << "  Transmitted: " << result.n_transmitted
              << "  (launched: " << result.n_launched << ")\n";
    std::cout << "  Efficiencies (PolyCap.hpp): ";
    for (int e = 0; e < std::min((int)result.efficiencies.size(), 5); ++e)
        std::cout << std::setprecision(4) << result.efficiencies[e] << " ";
    if (result.efficiencies.size() > 5) std::cout << "...";
    std::cout << "\n";

    int ne = (int)energies.size();
    int np = (int)result.photons.size();
    std::vector<std::string> headers;
    headers.push_back("x");  headers.push_back("y");  headers.push_back("z");
    headers.push_back("dx"); headers.push_back("dy"); headers.push_back("dz");
    for (int e = 0; e < ne; ++e) headers.push_back("w" + std::to_string(e+1));

    std::vector<std::vector<double>> cols(6 + ne, std::vector<double>(np));
    for (int i = 0; i < np; ++i) {
        const auto& p = result.photons[i];
        cols[0][i] = p.x_exit;
        cols[1][i] = p.y_exit;
        cols[2][i] = p.z_exit;
        cols[3][i] = p.dx;
        cols[4][i] = p.dy;
        cols[5][i] = p.dz;
        for (int e = 0; e < ne; ++e) cols[6+e][i] = p.weights[e];
    }
    write_csv("polycap_hpp_exit.csv", cols, headers);
    std::cout << "  Written: polycap_hpp_exit.csv\n";

    return 0;
}

// ═══════════════════════════ Efficiency comparison table ═════════════════════
static void print_efficiency_table(const std::vector<double>& energies) {
    // Re-read both CSVs and compare summed weights
    auto read_eff = [&](const std::string& path) {
        std::ifstream f(path);
        if (!f) return std::vector<double>(energies.size(), -1.);
        std::string line;
        std::getline(f, line); // header
        int ne = (int)energies.size();
        std::vector<double> sums(ne, 0.);
        int count = 0;
        while (std::getline(f, line)) {
            std::vector<double> vals;
            std::string tok;
            for (char c : line + ',') {
                if (c == ',') {
                    try { vals.push_back(std::stod(tok)); }
                    catch (...) { vals.push_back(0.0); }
                    tok.clear();
                } else tok += c;
            }
            // columns: x(0),y(1),z(2),dx(3),dy(4),dz(5),w1..wN
            for (int e = 0; e < ne && 6+e < (int)vals.size(); ++e)
                sums[e] += vals[6+e];
            count++;
        }
        for (auto& s : sums) if (count > 0) s /= count;
        return sums;
    };

    std::vector<double> eff_pc  = read_eff("polycap_exit.csv");
    std::vector<double> eff_hpp = read_eff("polycap_hpp_exit.csv");

    std::cout << "\n── Efficiency comparison ────────────────────────────────────\n"
              << std::setw(10) << "E [keV]"
              << std::setw(14) << "polycap"
              << std::setw(14) << "PolyCap.hpp"
              << std::setw(12) << "ratio\n"
              << std::string(50, '-') << "\n";
    for (int e = 0; e < (int)energies.size(); ++e) {
        double ratio = (eff_pc[e] > 0.) ? eff_hpp[e] / eff_pc[e] : 0.;
        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(10) << energies[e]
                  << std::scientific << std::setprecision(4)
                  << std::setw(14) << eff_pc[e]
                  << std::setw(14) << eff_hpp[e]
                  << std::fixed    << std::setprecision(3)
                  << std::setw(12) << ratio << "\n";
    }
}

// ═══════════════════════════════ main ════════════════════════════════════════
int main(int argc, char* argv[]) {
    int      n_photons = (argc > 1) ? std::atoi(argv[1]) : 5000;
    uint64_t seed      = (argc > 2) ? (uint64_t)std::atoll(argv[2]) : 42ULL;

    std::cout << "Polycapillary comparison test\n"
              << "  n_photons = " << n_photons << "\n"
              << "  seed      = " << seed      << "\n";

    auto energies = make_energies();

    int r1 = run_polycap    (n_photons, seed, energies);
    int r2 = run_polycap_hpp(n_photons, seed, energies);

    if (r1 == 0 && r2 == 0)
        print_efficiency_table(energies);

    return (r1 | r2);
}
