
#ifndef TRACER_GPU_H
#define TRACER_GPU_H

#define ARMA_ALLOW_FAKE_GCC 

#include <iostream>
#include <math.h>
#include <armadillo>
#include <filesystem>

#include "../cuda/RayGPU.cu"
#include "../cuda/ChemElementGPU.cu"
#include "../cuda/MaterialGPU.cu"
#include "../cuda/VoxelGPU.cu"
#include "../cuda/SampleGPU.cu"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

class TracerGPU{	

	public:

		static void callTracePreBeam(); 
		static void callTraceNewBeam(); 

		__device__ static void traceForward(RayGPU* ray, VoxelGPU* currentVoxel, curandState_t *localState);
		__device__ static void generateRayGPU(curandState_t *localState, float r_out, float f, float r_f, float energy_keV, RayGPU* ray);
};

#endif


