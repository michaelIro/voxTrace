#include <iostream>
#include <cuda_runtime.h>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "cuda/RayGPU.cu"
#include "cuda/ChemElementGPU.cu"
#include "cuda/MaterialGPU.cu"
#include "cuda/VoxelGPU.cu"
#include "cuda/SampleGPU.cu"
#include "cuda/TracerGPU.cuh"

#include "io/SimulationParameter.hpp"

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "No Input Folder given!" << std::endl;
        return 1;
    }
    
    std::cout << "START: SampleTracer" << std::endl;
//---------------------------------------------------------------------------------------------	

    std::string simulation_dir = argv[1];
    SimulationParameter sim_param(simulation_dir);
    TracerGPU::callTraceNewBeam(sim_param);
    
//---------------------------------------------------------------------------------------------
    std::cout << "END: SampleTracer" << std::endl;

    return 0;
}
