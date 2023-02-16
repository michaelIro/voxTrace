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
    int n_rays = 1000000;

    int n_el = 6;
    int els[n_el] = {26,28,29,30,50,82};
    float wgt[n_el] = {0.0004,0.0010,0.6119,0.3741,0.0107,0.0019}; 

    float prim_cap_geom[4] = {1075.0f, 5100.0f, 16.5f, 17.4f};                  // {0.1075f,0.5100f, 0.00165f, 17.4f};
    float prim_trans_param[5] = {150.0f, 150.0f, 0.0f, 5100.0f, 45.0f};         // Everything in um and °
    float sec_trans_param[6] = {150.0f, 150.0f, 0.0f, 4900.0f, 45.0f, 950.0f}; 

    float sample_start[3] = {0.0f, 0.0f, 0.0f};
    float sample_length[3] = {300.0f, 300.0f, 300.0f};
    float sample_voxel_length[3] = {4.0f, 4.0f, 4.0f};

    for(int i =-20; i < 21; i++){
        offset[2] = ((float) i) * 2.0f;
        std::string appendix = std::to_string(offset[2]);
        std::string path_out = "/media/miro/Data-1TB/nist-1107-simulation/nist-1107-pos-(" + appendix + ").h5";
        TracerGPU::callTraceNewBeam(offset, n_rays, n_el, els, wgt, prim_trans_param, sec_trans_param, prim_cap_geom, path_out,sample_start, sample_length, sample_voxel_length);
    }

    
//---------------------------------------------------------------------------------------------
    std::cout << "END: Test-3" << std::endl;

    return 0;
}