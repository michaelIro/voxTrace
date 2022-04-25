/** Sample-Object for GPU */

#include "SampleGPU.cuh"

Sample::Sample(float x, float y, float z, float xL, float yL, float zL, float xLV, float yLV, float zLV, vector<vector<vector<Material>>> materials_){
	
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


	for(int i=0; i<xN_; i++){
		vector<vector<Voxel>> a_;
		for(int j=0; j<yN_; j++){
			vector<Voxel> b_;
			for(int k=0; k<zN_; k++){
				Voxel c_(x_+i*xLV_,y_+j*yLV_,z_+k*zLV_,xLV_,yLV_,zLV_,materials_[i][j][k]);
				b_.push_back(c_);
			}
			a_.push_back(b_);
		}
		voxels_.push_back(a_);
	}

	/*Define Zero-Voxel and Out-of-Bounds-Voxel*/
	zeroVoxel = &voxels_[0][0][0];
	oobVoxel =  new Voxel(-1.,-1.,-1.,-1.,-1.,-1.,materials_[0][0][0]);

	/*Define Nearest Neighbours for each Voxel*/
	for(int i=0; i<xN_;i++)
	{
		for(int j=0; j<yN_;j++)
		{
			for(int k=0; k<zN_;k++)
			{
				Voxel *nn[27];
				int count=0;
					for(int l=-1; l<2;l++)
					{
						for(int m=-1; m<2;m++)
						{
							for(int n=-1; n<2;n++)
							{
								if((i+n < 0) || (i+n>=xN_) || (j+m < 0) || (j+m>=yN_) ||(k+l< 0) || (k+l>=xN_))
									nn[count++]=oobVoxel;	
								else
									nn[count++]=&voxels_[i+n][j+m][k+l];	
							}
						}
					}
				(voxels_[i][j][k]).setNN(nn);
			}
		}
	}
}

float Sample::getXPos() const {return x_;}
float Sample::getYPos() const {return y_;}
float Sample::getZPos() const {return z_;}
float Sample::getXLen() const {return xL_;}
float Sample::getYLen() const {return yL_;}
float Sample::getZLen() const {return zL_;}
float Sample::getXLenVox() const {return xLV_;}
float Sample::getYLenVox() const {return yLV_;}
float Sample::getZLenVox() const {return zLV_;}
int Sample::getXN() const {return xN_;}
int Sample::getYN() const {return yN_;}
int Sample::getZN() const {return zN_;}
int Sample::getVoxN() const {return voxN_;}
//vector<ChemElement> Sample::getElements() const{ return elements_;}

Voxel* Sample::getVoxel(float x, float y, float z){
	Voxel *temp = oobVoxel;
		
	int xSteps = (int) floor(x/xLV_);
	int ySteps = (int) floor(y/yLV_);
	int zSteps = (int) floor(z/zLV_);
	if(((xSteps < xN_)&&(zSteps < zN_)&&(zSteps < zN_)) && ((xSteps >= 0)&&(zSteps >= 0)&&(zSteps >= 0)))
		temp = &voxels_[xSteps][ySteps][zSteps];

	return temp;
}

Voxel* Sample::getVoxel(int x, int y, int z) {
	Voxel *temp = oobVoxel;
	temp = &(voxels_[x][y][z]);
	return temp;
}

bool Sample::isOOB(Voxel* vox) const { return (vox == oobVoxel); }

/*
float* Sample::sample2problem(){
	int problemSize=0;
	for(int i=0; i<xN_;i++)
		for(int j=0; j<yN_;j++)
			for(int k=0; k<zN_;k++)
				problemSize += (*voxPtrArray[i][j][k]).getMaterial().getMassFractions().size() + 1;
	
	float *problem { new float[static_cast<std::size_t>(problemSize)]{} };

	int varParam =0;
	for(int i=0; i<xN_;i++)
		for(int j=0; j<yN_;j++)
			for(int k=0; k<zN_;k++){
				problem[varParam++]=(*voxPtrArray[i][j][k]).getMaterial().Rho();
				for(auto const& it: (*voxPtrArray[i][j][k]).getMaterial().getMassFractions())
					problem[varParam++]=it.second;
			}		
	return problem;
}
*/


void Sample::print(){
	cout.precision(1);
	cout<<scientific;
	cout<<"Print Sample "<<xN_<<" "<<yN_<<" "<<zN_<<endl; 
	for(int i=0; i< xN_; i++)
		for(int j=0; j< yN_; j++)
			for(int k=0; k< zN_; k++){
				//cout<<(*voxPtrArray[i][j][k]).getSpectrum()<<" ";
				cout << i << " " << j << " " << k << " ";
				//cout<<(*voxPtrArray[i][j][k]).getXPos0()<<" "<<(*voxPtrArray[i][j][k]).getYPos0()<<" "<<(*voxPtrArray[i][j][k]).getZPos0()<<" ";
				//cout<<endl;
				voxels_[i][j][k].getMaterial().print();
			}
}

/** Find the Voxel which is touched first by the ray.
* @param ray  
* @return Voxel* Pointer to Voxel of first Interaction
*/
Voxel* Sample::findStartVoxel(Ray *ray){

	float x_in = (*ray).getStartX();
	float y_in = (*ray).getStartY();
	float z_in = (*ray).getStartZ();

	// Check if the ray is a primary ray -> If so, calculate coordinates of Voxel which is touched first by the ray from the top. 
	if((*ray).getStartZ() < getZPos()){
		float t = ( getZPos()-(*ray).getStartZ() ) / (*ray).getDirZ();
		x_in += t*(*ray).getDirX();
		y_in += t*(*ray).getDirY();
		z_in += t*(*ray).getDirZ();
	}

	// Check if ray hits the sample from the top direction -> If not, calculate coordinates of Voxel which is touched first by the ray from the side. 
	if( (x_in<0.) || (y_in<0.) || (z_in<0.) ){
		float t = ( getYPos()-(*ray).getStartY() ) / (*ray).getDirY();
		x_in = (*ray).getStartX() + t*(*ray).getDirX();
		y_in = (*ray).getStartY() + t*(*ray).getDirY();
		z_in = (*ray).getStartZ() + t*(*ray).getDirZ();
	}
				
	return getVoxel(x_in,y_in,z_in);
}