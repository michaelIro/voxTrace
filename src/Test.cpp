#include <iostream>
#include "api/XRayLibAPI.hpp"
#include "api/OptimizerAPI.hpp"
#include "cuda/RayGPU.cu"
#include "cuda/PolyCap.cu"
#include "cuda/Source.cu"

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

    // ── Create rays and trace ─────────────────────────────────────────────────
    // Rays start at y=-0.5cm (before entrance), directed along +Y with small spread
    float angles[] = { 0.000f, 0.002f, 0.005f, 0.010f, 0.020f };
    int   nRays    = 5;

    std::cout << "\nRay tracing through PolyCap (host):\n";
    std::cout << "  idx  angle[rad]  entering  prob_out  iters\n";

    for (int i = 0; i < nRays; ++i) {
        float ang  = angles[i];
        float dirY = cosf(ang);
        float dirX = sinf(ang);

        // RayGPU(startX,Y,Z, dirX,Y,Z, asX,Y,Z, flag, k, q, opd, fS, fP, apX,Y,Z, prob)
        RayGPU ray(
            0.0f, -0.5f, 0.0f,   // start: on-axis, before entrance
            dirX, dirY,  0.0f,   // direction: slight angle in XY plane
            1.0f,  0.0f, 0.0f,   // s-polarization
            true, 0.0f, i,       // flag, wavenumber (set via energy), index
            0.0f, 0.0f, 0.0f,    // opd, phases
            0.0f,  1.0f, 0.0f,   // p-polarization
            1.0f                 // initial probability
        );
        ray.setEnergyKeV(8.0f);  // Cu K-alpha

        bool entered = optic.isEntering(ray);
        optic.trace(ray);

        std::cout << "  [" << i << "]  "
                  << ang      << "       "
                  << entered  << "         "
                  << ray.getProb()    << "     "
                  << ray.getIANum()   << "\n";
    }


// Gaussian source at y=-0.5cm, aimed along +Y
Source src(
    0.0f, -0.5f, 0.0f,    // position
    0.0f,  1.0f, 0.0f,    // direction
    8.0f,                  // Cu K-alpha
    0.002f,                // source sigma = 20 µm
    0.003f,                // divergence sigma = 3 mrad
    SourceType::Gaussian
);
src.print();

for (int i = 0; i < 5; ++i) {
    RayGPU ray = src.generateHost(i, /*seed=*/42);
    ray.print();
    optic.trace(ray);
    std::cout << "ray " << i << "  prob=" << ray.getProb() << "\n";
}

    return 0;
}
