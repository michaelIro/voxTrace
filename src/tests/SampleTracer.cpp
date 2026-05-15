#include <iostream>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include <Kokkos_Core.hpp>
#include "../core/Tracer.hpp"
#include "../io/SimulationParameter.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: SampleTracer <simulation-directory>\n";
        return 1;
    }

    Kokkos::initialize(argc, argv);
    {
        std::cout << "START: SampleTracer\n";
        SimulationParameter sim_param(argv[1]);
        Tracer::callTraceNewBeam(sim_param);
        std::cout << "END: SampleTracer\n";
    }
    Kokkos::finalize();
    return 0;
}
