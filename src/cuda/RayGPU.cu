#ifndef Ray_GPU_H
#define Ray_GPU_H

// X-Ray-Object for GPU
// Memory-Estimation: 1 instance of Ray approx. 133 byte => 10⁶ Rays approx. 133 MB 

#include <iostream>
#include <math.h> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


class RayGPU {
	// These Parameters are taken 1:1 from Shadow3
	float x0_, y0_, z0_;			// Start coordinates of X-Ray-photon
	float dirX_, dirY_, dirZ_;		// Coordinates of direction Vector of X-Ray-Photon
	float asX_, asY_, asZ_;			// sPolarization
	float apX_, apY_, apZ_;			// pPolarization
	bool flag_;						// Flag
	float k_;						// wave number
	int q_;							// ray index/number
	float opd_;						// optical path length 
	float fS_, fP_;					// Phases

	// Further Parameters needed for voxTrace
	int iaNum_;						// Number of interactions of this ray
	bool iaFlag_;					// Interaction Flag
	float prob_;					// Probability

	// int sum of sizes of all member variables
	size_t memory_size_= sizeof(float)* 17 + sizeof(int)*2 + sizeof(bool)*2 + sizeof(size_t);
		
  public:

	__host__ __device__ RayGPU() {};

  	__host__ __device__ RayGPU(float startX, float startY, float startZ, float dirX, float dirY, float dirZ, 
	  float asX, float asY, float asZ, bool flag, float k, int q, float opd, float fS, 
	  float fP, float apX, float apY, float apZ, float prob){
		x0_ = startX;
		y0_ = startY;
		z0_ = startZ;
	
		dirX_ = dirX; 	
		dirY_ = dirY;
		dirZ_ = dirZ;		

		asX_= asX;
		asY_= asY;
		asZ_= asZ;
		apX_= apX;
		apY_= apY;
		apZ_= apZ;

		flag_=flag;
		k_=k;
		q_=q;
		opd_=opd;
		fS_=fS;
		fP_=fP;
	
		iaNum_=0;
		iaFlag_=false;
		prob_= prob;
	};
	

	__host__ __device__ void setFlag(bool flag){flag_=flag;};
    __host__ __device__ void setStartCoordinates (float x, float y, float z) { x0_ = x; y0_ = y; z0_ = z;};
    __host__ __device__ void setEndCoordinates (float x, float y, float z) { dirX_ = x; dirY_ = y; dirZ_ = z;};
    __host__ __device__ void setSPol (float x, float y, float z) { asX_ = x; asY_ = y; asZ_ = z;};
    __host__ __device__ void setPPol (float x, float y, float z) { apX_ = x; apY_ = y; apZ_ = z;};
    __host__ __device__ void setEnergyKeV(float keV) {k_=keV*50677300.0;};
	__host__ __device__ void setIAFlag(bool iaFlag) {iaFlag_=iaFlag;};
	__host__ __device__ void setIANum(int iaNum) {iaNum_=iaNum;};

	__host__ __device__ void rotate(float phi, float theta){
			float diX = cos(theta)*cos(phi)*dirX_ - sin(phi)*dirY_ + sin(theta)*cos(phi)*dirZ_;
			float diY = cos(theta)*sin(phi)*dirX_ + cos(phi)*dirY_ + sin(theta)*sin(phi)*dirZ_;
			float diZ = -sin(theta)*dirX_+cos(theta)*dirZ_;
			dirX_=diX;
			dirY_=diY;
			dirZ_=diZ;
	}

	__host__ __device__ float getStartX() const {return x0_;};
	__host__ __device__ float getStartY() const {return y0_;};
	__host__ __device__ float getStartZ() const {return z0_;};
	__host__ __device__ float getDirX() const {return dirX_;};		
	__host__ __device__ float getDirY() const {return dirY_;};		
	__host__ __device__ float getDirZ() const {return dirZ_;};

	__host__ __device__ float getSPolX() const {return asX_;};
	__host__ __device__ float getSPolY() const {return asY_;};
	__host__ __device__ float getSPolZ() const {return asZ_;};
	__host__ __device__ float getPPolX() const {return apX_;};
	__host__ __device__ float getPPolY() const {return apY_;};
	__host__ __device__ float getPPolZ() const {return apZ_;};
	__host__ __device__ float getSPhase() const {return fS_;};
	__host__ __device__ float getPPhase() const {return fP_;};
		
	__host__ __device__ int getIndex() const {return q_;};
    __host__ __device__ bool getFlag() const {return flag_;};
    __host__ __device__ float getWaveNumber() const {return k_;};
    __host__ __device__ float getOpticalPath() const {return opd_;};
	__host__ __device__ float getEnergyEV() const {return (k_ / 50677300.0);};
    __host__ __device__ float getEnergyKeV() const {return (k_ / 50677300.0);};

	__host__ __device__ bool getIAFlag() const {return iaFlag_;};
	__host__ __device__ int getIANum() const {return iaNum_;};
	__host__ __device__ float getProb() const {return prob_;};

	__host__ void print() const { 
		std::cout << "Ray " << q_ << "\t Energy: \t" << getEnergyKeV() << " keV \t Memory-Size: " << memory_size_ << " Byte"<<std::endl;
		//std::cout << "Start: \t" << x0_ << "\t" << y0_ << "\t" << z0_ << std::endl;
		///std::cout << "Direction: \t" << dirX_ << "\t" << dirY_ << "\t" << dirZ_ << std::endl;
		//std::cout << "SPol: \t" << asX_ << "\t" << asY_ << "\t" << asZ_ << std::endl;
		//std::cout << "PPol: \t" << apX_ << "\t" << apY_ << "\t" << apZ_ << std::endl;
		//std::cout << "SPhase: \t" << fS_ << std::endl;
		//std::cout << "PPhase: \t" << fP_ << std::endl;
		//std::cout << "Flag: \t" << flag_ << std::endl;
		//std::cout << "WaveNumber: \t" << k_ << std::endl;
		//std::cout << "OpticalPath: \t" << opd_ << std::endl;
		//std::cout << "IAFlag: \t" << iaFlag_ << std::endl;
		//std::cout << "IANum: \t" << iaNum_ << std::endl;
		//std::cout << "Prob: \t" << prob_ << std::endl;
	};
};


#endif