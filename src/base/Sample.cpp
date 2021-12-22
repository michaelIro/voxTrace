/*Sample*/

#include "Sample.hpp"

using namespace std;

//Sample::Sample(){ }

// TODO: Look @ alternative Constructors from Old Code 

Sample::Sample(double x, double y, double z, double xL, double yL, double zL, double xLV, double yLV, double zLV, vector<vector<vector<Material>>> materials_){
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
	//cout<<"xN:"<<xN_<<" "<<yN_<<" "<<zN_<<endl;
	//cout<<"xN:"<<materials_.size()<<" "<<materials_[0].size()<<" "<<materials_[0][0].size()<<endl;


	xLV_ = xL_/((double)(xN_-1));
	yLV_ = yL_/((double)(yN_-1));
	zLV_ = zL_/((double)(zN_-1));

	//cout<<"xlV:"<<xLV_<<" "<<yLV_<<" "<<zLV_<<endl;

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

double Sample::getXPos() const {return x_;}
double Sample::getYPos() const {return y_;}
double Sample::getZPos() const {return z_;}
double Sample::getXLen() const {return xL_;}
double Sample::getYLen() const {return yL_;}
double Sample::getZLen() const {return zL_;}
double Sample::getXLenVox() const {return xLV_;}
double Sample::getYLenVox() const {return yLV_;}
double Sample::getZLenVox() const {return zLV_;}
int Sample::getXN() const {return xN_;}
int Sample::getYN() const {return yN_;}
int Sample::getZN() const {return zN_;}
int Sample::getVoxN() const {return voxN_;}
//vector<ChemElement> Sample::getElements() const{ return elements_;}

Voxel* Sample::getVoxel(double x, double y, double z){
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
double* Sample::sample2problem(){
	int problemSize=0;
	for(int i=0; i<xN_;i++)
		for(int j=0; j<yN_;j++)
			for(int k=0; k<zN_;k++)
				problemSize += (*voxPtrArray[i][j][k]).getMaterial().getMassFractions().size() + 1;
	
	double *problem { new double[static_cast<std::size_t>(problemSize)]{} };

	int varParam =0;
	for(int i=0; i<xN_;i++)
		for(int j=0; j<yN_;j++)
			for(int k=0; k<zN_;k++){
				problem[varParam++]=(*voxPtrArray[i][j][k]).getMaterial().getRho();
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

	double x_in = (*ray).getStartX();
	double y_in = (*ray).getStartY();
	double z_in = (*ray).getStartZ();

	// Check if the ray is a primary ray -> If so, calculate coordinates of Voxel which is touched first by the ray from the top. 
	if((*ray).getStartZ() < getZPos()){
		double t = ( getZPos()-(*ray).getStartZ() ) / (*ray).getDirZ();
		x_in += t*(*ray).getDirX();
		y_in += t*(*ray).getDirY();
		z_in += t*(*ray).getDirZ();
	}

	// Check if ray hits the sample from the top direction -> If not, calculate coordinates of Voxel which is touched first by the ray from the side. 
	if( (x_in<0.) || (y_in<0.) || (z_in<0.) ){
		double t = ( getYPos()-(*ray).getStartY() ) / (*ray).getDirY();
		x_in = (*ray).getStartX() + t*(*ray).getDirX();
		y_in = (*ray).getStartY() + t*(*ray).getDirY();
		z_in = (*ray).getStartZ() + t*(*ray).getDirZ();
	}
				
	return getVoxel(x_in,y_in,z_in);
}