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

    float offset[3] = {0.0f, 0.0f, 0.0f}; 
    int n_rays = 100000;

    int n_el = 6;
    int els[n_el] = {26,28,29,30,50,82};
    float wgt[n_el] = {0.0004,0.0010,0.6119,0.3741,0.0107,0.0019}; 

    float prim_trans_param[5] = {300000.0, 300000.0,0.0, 5100.0, 45.0}; 
    float sec_trans_param[6] = {300000.0, 300000.0, 0.0, 4900.0, 45.0, 950.0}; 
    float prim_cap_geom[4] = {0.1075f,0.5100f, 0.00165f, 17.4f};

    for(int i = -5; i < 1; i++){
        offset[2] = ((float) i) * 10.0f;
        std::string appendix = std::to_string(offset[2]);
        std::string path_out = "/media/miro/Data/nist-1107-pos-(" + appendix + ").h5";
        TracerGPU::callTraceNewBeam(offset, n_rays, n_el, els, wgt, prim_trans_param, sec_trans_param, prim_cap_geom, path_out);
    }

    
//---------------------------------------------------------------------------------------------
    std::cout << "END: Test-3" << std::endl;

    return 0;
}