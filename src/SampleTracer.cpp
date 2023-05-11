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

	std::cout << "START: SampleTracer" << std::endl;
//---------------------------------------------------------------------------------------------	

    SimulationParameter sim_param("/media/miro/Data-1TB/simulation/triple-cross");
    TracerGPU::callTraceNewBeam(sim_param);
    
//---------------------------------------------------------------------------------------------
    std::cout << "END: SampleTracer" << std::endl;

    return 0;
}