#ifndef Sample_GPU_H
#define Sample_GPU_H

/** Sample-Object for GPU */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../cuda/ChemElementGPU.cu"
#include "../cuda/MaterialGPU.cu"
#include "../cuda/VoxelGPU.cu"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


class SampleGPU{

	private:
		float x_, y_,z_;
		float xL_,yL_,zL_;
		float xLV_,yLV_,zLV_;
		int xN_,yN_,zN_,voxN_;

		VoxelGPU* voxels_;
		VoxelGPU* oobVoxel_;

		//VoxelGPU* zeroVoxel_;
		//thrust::device_vector<ChemElementGPU> chemical_elements_;
		//vector<vector<vector<Voxel>>> voxels_;
        //size_t memory_size_= sizeof(float)*9;

	public:
		__host__ __device__ SampleGPU(){ };

		__host__ __device__ SampleGPU(float x, float y, float z, float xL, float yL, float zL, float xLV, float yLV, float zLV, int xN, int yN, int zN, VoxelGPU* voxels, VoxelGPU *oobVoxel){
	
    		x_ = x;
			y_ = y;
			z_ = z;

			xL_ = xL;
			yL_ = yL;
			zL_ = zL;

			xN_ = (int)(xL_/xLV)+1;
			yN_ = (int)(yL_/yLV)+1;
			zN_ = (int)(zL_/zLV)+1;

			voxN_= xN_*yN_*zN_;
    
			xLV_ = xL_/((float)(xN_-1));
			yLV_ = yL_/((float)(yN_-1));
			zLV_ = zL_/((float)(zN_-1));

			voxels_ = voxels;

			//for(int i=0; i<xN_; i++){
			//	for(int j=0; j<yN_; j++){
			//		for(int k=0; k<zN_; k++){
			//			voxels_[i][j][k] = VoxelGPU(x_+i*xLV_, y_+j*yLV_, z_+k*zLV_, xLV_, yLV_, zLV_, &(materials_[i][j][k]));
			//		}
			//	}
			//}

			/*Define Zero-Voxel and Out-of-Bounds-Voxel*/
			//zeroVoxel = &voxels_[0][0][0];
			oobVoxel_ = oobVoxel;
			//oobVoxel_ =  new VoxelGPU(-1.,-1.,-1.,-1.,-1.,-1.,&(materials_[0][0][0]));

			/*Define Nearest Neighbours for each Voxel*/
			for(int i=0; i<xN_;i++)
			{
				for(int j=0; j<yN_;j++)
				{
					for(int k=0; k<zN_;k++)
					{
						VoxelGPU *nn[27];
						int count=0;
							for(int l=-1; l<2;l++)
							{
								for(int m=-1; m<2;m++)
								{
									for(int n=-1; n<2;n++)
									{
										if((i+n < 0) || (i+n>=xN_) || (j+m < 0) || (j+m>=yN_) ||(k+l< 0) || (k+l>=xN_))
											nn[count++]=oobVoxel_;	
										//else
										//	nn[count++]=&voxels_[i+n][j+m][k+l];	
								}
							}
						}
						//(voxels_[i][j][k]).setNN(nn);
					}
				}
			}
		};


		__device__ float getXPos() const {return x_;};
		__device__ float getYPos() const {return y_;};
		__device__ float getZPos() const {return z_;};
		__device__ float getXLen() const {return xL_;};
		__device__ float getYLen() const {return yL_;};
		__device__ float getZLen() const {return zL_;};
		__device__ float getXLenVox() const {return xLV_;};
		__device__ float getYLenVox() const {return yLV_;};
		__device__ float getZLenVox() const {return zLV_;};
		__device__ int getXN() const {return xN_;};
		__device__ int getYN() const {return yN_;};
		__device__ int getZN() const {return zN_;};
		__device__ int getVoxN() const {return voxN_;};
		
		__device__  VoxelGPU* 	getVoxel(float x, float y, float z){
			VoxelGPU *temp = oobVoxel_;
		
			int xSteps = (int) floor(x/xLV_);
			int ySteps = (int) floor(y/yLV_);
			int zSteps = (int) floor(z/zLV_);
			if(((xSteps < xN_)&&(zSteps < zN_)&&(zSteps < zN_)) && ((xSteps >= 0)&&(zSteps >= 0)&&(zSteps >= 0)))
				temp = &voxels_[xSteps*yN_*zN_+ySteps*zN_+zSteps];

			return temp;
		};

		__device__  VoxelGPU* 	getVoxel(int x, int y, int z){
			//VoxelGPU *temp = oobVoxel_;
			//temp = &(voxels_[x][y][z]);
			//			return temp;

			return &voxels_[x*yN_*zN_+y*zN_+z];
		};

		__device__ bool isOOB(VoxelGPU* vox) const { return (vox == oobVoxel_);};
		__device__ VoxelGPU* getOOBVoxel() const {return oobVoxel_;};

		__device__ VoxelGPU* findStartVoxel(RayGPU *ray){
  
  			//printf("Jambalaya!\n");

			float x_in = ray->getStartX();
			float y_in = ray->getStartY();
			float z_in = ray->getStartZ();

						  //printf("%f\n",x_in);			  
						  //printf("%f\n",y_in);
						  //printf("%f\n",z_in);

			// Check if the ray is a primary ray -> If so, calculate coordinates of Voxel which is touched first by the ray from the top. 
			if(ray->getStartZ() < getZPos()){
				float t = ( getZPos()-ray->getStartZ() ) / ray->getDirZ();
				x_in += t*ray->getDirX();
				y_in += t*ray->getDirY();
				z_in += t*ray->getDirZ();
			}

			// Check if ray hits the sample from the top direction -> If not, calculate coordinates of Voxel which is touched first by the ray from the side. 
			if( (x_in<0.) || (y_in<0.) || (z_in<0.) ){
				//printf("Jambalaya! The wrongness is before.\n");
				float t = ( getYPos()-ray->getStartY() ) / ray->getDirY();
				x_in = ray->getStartX() + t*ray->getDirX();
				y_in = ray->getStartY() + t*ray->getDirY();
				z_in = ray->getStartZ() + t*ray->getDirZ();
			}
				
			return getVoxel(x_in,y_in,z_in);
		};
};

#endif
