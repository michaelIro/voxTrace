
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

		static void callTrace(); 	//arma::Mat<double>& beam_
		static void callAdd();
		static void callTest();

		__device__ static RayGPU* traceForward(RayGPU* ray, VoxelGPU* currentVoxel, int* nextVoxel, SampleGPU *sample, curandState_t *localState);

		//static void callAdd(int N, int M, float *x, float *y, RayGPU* rays_1, RayGPU* rays_2, ChemElementGPU* elements, MaterialGPU* materials,ChemElementGPU* myElements,float* myElements_weights); 

};

#endif


