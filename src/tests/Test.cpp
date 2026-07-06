#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include "../api/XRayLibAPI.hpp"
#include "../api/OptimizerAPI.hpp"
#include "PolyCap.hpp"
#include "Ray.hpp"
#include "RNG.hpp"
#include "Source.hpp"
#include "io/SetupIO.hpp"

namespace {

Ray makeCenteredRay(float energy_keV, int index) {
    Ray ray;
    // PolyCap's optic axis is the ray's +z; aim a centred ray straight down it.
    ray.setStartCoordinates(0.f, 0.f, -100.f);
    ray.setEndCoordinates(0.f, 0.f, 1.f);
    ray.setSPol(1.f, 0.f, 0.f);
    ray.setPPol(0.f, 1.f, 0.f);
    ray.setEnergyKeV(energy_keV);
    ray.setProb(1.f);
    ray.setIAFlag(false);
    ray.setIANum(index);
    return ray;
}

}  // namespace

int main() {

    // ── XRayLib test ──────────────────────────────────────────────────────────
    std::cout << "Atomic number of Cu: " << XRayLibAPI::SymToZ("Cu") << "\n";

    // ── Optimizer test ────────────────────────────────────────────────────────
    auto result = OptimizerAPI::Minimize(
        Algorithm::NelderMead,
        [](const std::vector<double>& x){ return x[0]*x[0] + x[1]*x[1]; },
        {3.0, 3.0}
    );
    std::cout << "Nelder-Mead min: f=" << result.fval
              << "  x=(" << result.x[0] << ", " << result.x[1] << ")\n";

    // ── Source test ───────────────────────────────────────────────────────────
    Source src = Source::fromCapGeom(0.002f, 0.5f, 0.003f, 8.0f);
    src.print();

    RNG rng(42ULL);
    for (int i = 0; i < 3; ++i) {
        Ray ray = src.generate(i, rng);
        ray.print();
    }

    // ── PolyCap ray tracing test (descriptor loaded via SetupIO) ─────────────
    PolyCap optic = vtio::loadPolyCap("test-data/api/polycap/pc-236-descr.txt").build();

    std::vector<float> energies_keV = {8.0f, 12.0f, 17.4f};
    int transmitted = 0;
    for (int i = 0; i < (int)energies_keV.size(); ++i) {
        Ray ray = makeCenteredRay(energies_keV[i], i);
        optic.trace(ray);
        bool ok = ray.getIAFlag();
        transmitted += ok ? 1 : 0;
        std::cout << "PolyCap trace E=" << energies_keV[i] << " keV"
                  << " transmitted=" << ok
                  << " prob=" << ray.getProb()
                  << " reflections=" << ray.getIANum() << "\n";
    }

    if (transmitted == 0) {
        std::cerr << "PolyCap ray tracing smoke test failed: no rays transmitted\n";
        return 1;
    }

    return 0;
}
