#include <iostream>
#include <cmath>
#include "../api/XRayLibAPI.hpp"
#include "../api/OptimizerAPI.hpp"
#include "../core/Ray.hpp"
#include "../core/RNG.hpp"
#include "../core/PolyCap.hpp"
#include "../core/Source.hpp"

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

    // ── PolyCap setup ─────────────────────────────────────────────────────────
    int   Z[]  = {8, 14};
    float wt[] = {53.0f, 47.0f};

    PolyCap optic(
        /*posY=*/0.0f,       /*length=*/4.03f,
        /*rExtIn=*/0.095f,   /*rExtOut=*/0.3175f,
        /*rCapIn=*/9.75e-5f, /*rCapOut=*/3.25e-4f,
        /*focalIn=*/0.49f,   /*focalOut=*/1e8f,
        /*nElem=*/2, Z, wt,
        /*density=*/2.23f,   /*roughness=*/5.0f, /*nCap=*/240000
    );
    optic.print();

    // ── Trace rays through optic (host) ──────────────────────────────────────
    float angles[] = { 0.000f, 0.002f, 0.005f, 0.010f, 0.020f };
    int   nRays    = 5;

    std::cout << "\nRay tracing through PolyCap (host):\n";
    std::cout << "  idx  angle[rad]  entering  iters\n";

    for (int i = 0; i < nRays; ++i) {
        float ang  = angles[i];
        float dirY = cosf(ang);
        float dirX = sinf(ang);

        Ray ray(
            0.0f, -0.5f, 0.0f,
            dirX, dirY,  0.0f,
            1.0f, 0.0f,  0.0f,
            true, 0.0f, i,
            0.0f, 0.0f, 0.0f,
            0.0f, 1.0f,  0.0f,
            1.0f
        );
        ray.setEnergyKeV(8.0f);

        bool entered = optic.isEntering(ray);
        optic.trace(ray);

        std::cout << "  [" << i << "]  "
                  << ang << "       " << entered
                  << "    " << ray.getIANum() << "\n";
    }

    // ── Source test ───────────────────────────────────────────────────────────
    Source src = Source::fromCapGeom(0.002f, 0.5f, 0.003f, 8.0f);
    src.print();

    RNG rng(42ULL);
    for (int i = 0; i < 3; ++i) {
        Ray ray = src.generate(i, rng);
        ray.print();
    }

    return 0;
}
