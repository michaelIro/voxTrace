#include <iostream>
#include <cuda_runtime.h>

#include "cuda/RayGPU.cu"
#include "cuda/ChemElementGPU.cu"
#include "cuda/MaterialGPU.cu"
#include "cuda/VoxelGPU.cu"
#include "cuda/SampleGPU.cu"
#include "cuda/TracerGPU.cuh"

#include <armadillo>


int main() {

	std::cout << "START: Test-3" << std::endl;
//---------------------------------------------------------------------------------------------	

	TracerGPU::callTrace(); 

//---------------------------------------------------------------------------------------------
    std::cout << "END: Test-3" << std::endl;

    return 0;
}