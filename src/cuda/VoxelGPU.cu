#ifndef Voxel_GPU_H
#define Voxel_GPU_H

/** Voxel-Object for GPU */

//#include <iostream>
//#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include "../cuda/RayGPU.cu"
#include "../cuda/MaterialGPU.cu"

class VoxelGPU{
	private:
		float x0_,y0_,z0_;
		float x1_,y1_,z1_;
		MaterialGPU* mat_;
		VoxelGPU *nn_[27];

  	public:
		__host__ __device__ VoxelGPU(){ };
		__host__ __device__ VoxelGPU(float x0, float y0, float z0, float x1, float y1, float z1, MaterialGPU* mat){
			x0_ = x0;
			y0_ = y0;
			z0_ = z0;
			x1_ = x1;
			y1_ = y1;
			z1_ = z1;
			mat_ = mat;
		};

		__host__ __device__ void setNN(VoxelGPU *v[27]){
			for (int i = 0; i < 27; i++) 
				nn_[i] = v[i]; 
		};

		__device__ float getX0() const {return x0_;};
		__device__ float getY0() const {return y0_;};
		__device__ float getZ0() const {return z0_;};
		__device__ float getX1() const {return (x0_ + x1_);};
		__device__ float getY1() const {return (y0_ + y1_);};
		__device__ float getZ1() const {return (z0_ + z1_);};

		__device__ MaterialGPU* getMaterial() {return mat_;};
		__device__ VoxelGPU* getNN(int i) const{return nn_[i];};

		__device__ float intersect(RayGPU* ray, int* nextVoxelGPU, float *tIn){	
	
			float t0x, t1x;
			float t0y, t1y;
			float t0z, t1z;
			float t0_max, t1_min;
			bool xDir=true,yDir=true,zDir=true;

			/*if Ray is x-Parallel no intersection with x-Planes possible*/
			if((*ray).getDirX() != 0.){
				t0x = ( getX0()-(*ray).getStartX() ) / (*ray).getDirX(); 
				t1x = ( getX1()-(*ray).getStartX() ) / (*ray).getDirX();
				if(t0x>t1x){
					float temp = t0x;
					t0x=t1x;
					t1x=temp;
					xDir=false;
				}
			}
			else{
				t0x=FLT_MIN;
				t1x=FLT_MAX;
			}
			if((*ray).getDirY() != 0.){
				t0y = ( getY0()-(*ray).getStartY() ) / (*ray).getDirY(); 
				t1y = ( getY1()-(*ray).getStartY() ) / (*ray).getDirY();  
				if(t0y>t1y){
					float temp = t0y;
					t0y=t1y;
					t1y=temp;
					yDir=false;
				}
			}
			else{
				t0y=FLT_MIN;
				t1y=FLT_MAX;
			}
			if((*ray).getDirZ() != 0.){
				t0z = ( getZ0()-(*ray).getStartZ() ) / (*ray).getDirZ(); 
				t1z = ( getZ1()-(*ray).getStartZ() ) / (*ray).getDirZ();  
				if(t0z>t1z){
					float temp = t0z;
					t0z=t1z;
					t1z=temp;
					zDir=false;
				}
			}
			else{
				t0z=FLT_MIN;
				t1z=FLT_MAX;
			}

			//cout<<"t0x "<<t0x<<"t1x "<<t1x<<endl;
			//cout<<"t0y "<<t0y<<"t1x "<<t1y<<endl;
			//cout<<"t0z "<<t0z<<"t1x "<<t1z<<endl;
			//cout<<"IANUM: "<<(*ray).getIANum()<<endl;

			if((*ray).getIAFlag() ){
				t0_max = 0.;
				(*ray).setIAFlag(false);
			}
			else t0_max = max(max(t0x,t0y),t0z);
			t1_min = min(min(t1x,t1y),t1z);

			//cout<<"t1min"<<t1_min<<endl;

			if((t1_min == t1z) && (t1_min != t1y) && (t1_min != t1x)) { 
				if(zDir)	(*nextVoxelGPU)= 22; 	
				else		(*nextVoxelGPU)= 4; 	
			}
			else if	((t1_min != t1z) && (t1_min == t1y) && (t1_min != t1x))	{
				if(yDir) 		(*nextVoxelGPU)= 16; 	
				else 			(*nextVoxelGPU)= 10; 	
			}
			else if	((t1_min != t1z) && (t1_min != t1y) && (t1_min == t1x))	{
				if(xDir) 		(*nextVoxelGPU)= 14; 	
				else 			(*nextVoxelGPU)= 12; 	
			}

	
			/*The following options are very unlikely*/
			else if	((t1_min == t1z) && (t1_min == t1y) && (t1_min != t1x))	{
				if(zDir){
					if(yDir) 	(*nextVoxelGPU)= 25; 	
					else 		(*nextVoxelGPU)= 19; 	
				}
				else{
					if(yDir) 	(*nextVoxelGPU)= 1; 	
					else 		(*nextVoxelGPU)= 7; 	
				}
			}
			else if	((t1_min == t1z) && (t1_min != t1y) && (t1_min == t1x))	{ 
				if(zDir){
					if(xDir) 	(*nextVoxelGPU)= 23; 	
					else 		(*nextVoxelGPU)= 21; 	
				}
				else{
					if(xDir) 	(*nextVoxelGPU)= 3; 	
					else 		(*nextVoxelGPU)= 5; 	
				}
			}	
			else if	((t1_min != t1z) && (t1_min == t1y) && (t1_min == t1x))	{ 
				if(yDir){
					if(xDir) 	(*nextVoxelGPU)= 17; 	
					else 		(*nextVoxelGPU)= 15; 	
				}
				else{
					if(xDir) 	(*nextVoxelGPU)= 11; 	
					else 		(*nextVoxelGPU)= 9; 	
				}
			}	
	

			else if	((t1_min == t1z) && (t1_min == t1y) && (t1_min == t1x))	{ 
				if(zDir){
					if(yDir) {
						if(xDir) 	(*nextVoxelGPU)= 26; 	
						else 		(*nextVoxelGPU)= 24; 	
					}
					else{
						if(xDir) 	(*nextVoxelGPU)= 20; 	
						else 		(*nextVoxelGPU)= 18; 	
					}
				}
				else{
					if(yDir) {
						if(xDir) 	(*nextVoxelGPU)= 8; 	
						else 		(*nextVoxelGPU)= 6; 	
					}
					else{
						if(xDir) 	(*nextVoxelGPU)= 2; 	
						else 		(*nextVoxelGPU)= 0; 	
					}
				}
			}
	
			*tIn = t0_max;
			return (t1_min-t0_max); // TODO: direction  
		};

		__device__ float min(float x, float y){
			if(x<y) return x;
			else return y;
		};

		__device__ float max(float x, float y){
			if(x>y) return x;
			else return y;
		};
};

#endif

