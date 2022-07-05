#ifndef Material_GPU_H
#define Material_GPU_H

/** Material-Object for GPU */

#include "../cuda/ChemElementGPU.cu"
#include <device_launch_parameters.h>

class MaterialGPU {
	private:
		int num_elements_;
		ChemElementGPU* elements_;
		float* weights_;
		float rho_;

	public:

  		__host__ __device__ MaterialGPU(){};

	 	__host__ __device__ MaterialGPU(int num_elements, ChemElementGPU* elements, float* weights, float density){ 
			num_elements_ = num_elements;
			elements_ = elements;
			weights_ = weights;
			rho_ = density; 
		};

		__host__ __device__ MaterialGPU(int num_elements, ChemElementGPU* elements, float* weights){ 
			num_elements_ = num_elements;
			elements_ = elements;
			weights_ = weights;
			rho_ = 0.;
			for(int i = 0; i < num_elements_; i++)
        		rho_ += elements_[i].Rho()*weights_[i];
		};

		__host__ __device__ float Rho() const { return rho_;};

		__device__ float CS_Tot(float energy) const{
			float cs_tot_= 0.;
			for(int i = 0; i < num_elements_; i++)
        		cs_tot_ += weights_[i]*elements_[i].CS_Tot(energy);

			return cs_tot_;
		}
		
		__device__ float CS_Tot_Lin(float energy) const { return (CS_Tot(energy) * rho_); };

		__device__ ChemElementGPU* getInteractingElement(float energy, float randomN) const {

			float muMassTot = CS_Tot(energy);
			float sum = 0.;

			for(int i = 0; i < num_elements_; i++){
				sum += elements_[i].CS_Tot(energy)*weights_[i]/muMassTot;

				if(sum >= randomN){
					return &(elements_[i]);	
				}	
			}

			return &(elements_[num_elements_-1]);
		};
};

#endif