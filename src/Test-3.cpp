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

int main(int argc, const char* argv[]) {

	std::cout << "START: Test-3" << std::endl;
//---------------------------------------------------------------------------------------------	

	//TracerGPU::callTracePreBeam(); 
    TracerGPU::callTraceNewBeam();
    
//---------------------------------------------------------------------------------------------
    std::cout << "END: Test-3" << std::endl;

    return 0;
}