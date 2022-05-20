#include <iostream>
#include <cuda_runtime.h>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include "api/PolyCapAPI.hpp"
//#include <polycap-private.h>
//#define __STDC_FORMAT_MACROS
//#include <inttypes.h>
#include <omp.h>

//#include <boost/random/mersenne_twister.hpp>
//#include <boost/random/uniform_01.hpp>
//#include <thread> 

#include "cuda/RayGPU.cu"
#include "cuda/ChemElementGPU.cu"
#include "cuda/MaterialGPU.cu"
#include "cuda/VoxelGPU.cu"
#include "cuda/SampleGPU.cu"
#include "cuda/TracerGPU.cuh"

int main() {

	std::cout << "START: Test-3" << std::endl;
//---------------------------------------------------------------------------------------------	

	TracerGPU::callTrace(); 
    
    PolyCapAPI mySecondaryPolycap((char*) "../test-data/in/polycap/pc-236-descr.txt");

    //XRBeam myDetectorBeam(mySecondaryPolycap.traceFast(fluorescence_.getMatrix()));
	//std::cout << "Detector size:" << myDetectorBeam.getRays().size() << std::endl;
	
    //TracerGPU::callTest();

//---------------------------------------------------------------------------------------------
    std::cout << "END: Test-3" << std::endl;

    return 0;
}