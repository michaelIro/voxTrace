/*Sample*/

#include "Sample.hpp"

using namespace std;



//Sample::Sample(){ }

/*Sample::Sample(const Sample &sample, double *problem){
	x_ = sample.x_;
	y_ = sample.y_;
	z_ = sample.z_;
	xL_ = sample.xL_;
	yL_ = sample.yL_;
	zL_ = sample.zL_;
	xN_ = sample.xN_;
	yN_ = sample.yN_;
	zN_ = sample.zN_;
	voxN_= sample.voxN_;
	xLV_ = sample.xLV_;
	yLV_ = sample.yLV_;
	zLV_ = sample.zLV_;

	//Allocate voxPtrArray
	voxPtrArray = new Voxel***[xN_];
	for(int i=0; i<xN_;i++){
		voxPtrArray[i]= new Voxel**[yN_];
		for(int j=0; j<yN_;j++)
			voxPtrArray[i][j] = new Voxel*[zN_];
	}

	int varParam =0;
	//Initialize voxPtrArray
	for(int i=0; i<xN_;i++)
		for(int j=0; j<yN_;j++)
			for(int k=0; k<zN_;k++){

				double rho_ = problem[varParam++];
				map<ChemElement,double> concentrations_= (*(sample.voxPtrArray[i][j][k])).getMaterial().getMasses();
				for(auto & it: concentrations_)
					it.second = problem[varParam++];
				Material mat_(concentrations_,rho_);

				voxPtrArray[i][j][k] = (new Voxel(x_+i*xLV_,y_+j*yLV_,z_+k*zLV_,xLV_,yLV_,zLV_,mat_));
			}
			
	//Define Zero-Voxel and Out-of-Bounds-Voxel
	*zeroVoxel = *sample.zeroVoxel;
	*oobVoxel =  *sample.oobVoxel;

	//Define Nearest Neighbours for each Voxel
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
									nn[count++]=voxPtrArray[i+n][j+m][k+l];	
							}
						}
					}
				(*voxPtrArray[i][j][k]).setNN(nn);
			}
		}
	}
}*/

//Sample::Sample(Scan scan){ Sample(0.,0.,0.,scan.getLengths()[0],scan.getLengths()[1],scan.getLengths()[2],15.,15.,15.,scan.getMaterials(),scan.getElements()); }

Sample::Sample(double x, double y, double z, double xL, double yL, double zL, double xLV, double yLV, double zLV, vector<vector<vector<Material>>> materials_, vector<ChemElement> elements){
	x_ = x;
	y_ = y;
	z_ = z;
	xL_ = xL;
	yL_ = yL;
	zL_ = zL;

	elements_ = elements;
	
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
vector<ChemElement> Sample::getElements() const{ return elements_;}

Voxel* Sample::getVoxel(double x, double y, double z){
	Voxel *temp = oobVoxel;
		
	int xSteps = (int) floor(x/xLV_);
	int ySteps = (int) floor(y/yLV_);
	int zSteps = (int) floor(z/zLV_);
	if((xSteps <= xN_)&&(zSteps <= zN_)&&(zSteps <= zN_))
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

Voxel* Sample::findStartVoxel(Ray *ray){

	double x_in = (*ray).getStartX();
	double y_in = (*ray).getStartY();
	double z_in = (*ray).getStartZ();

	//cout<<"Start:"<<x_in<<y_in<<z_in<<endl;
	//cout<<"Start-Sample:"<<getZPos()<<endl;

	/*Check if its primary ray*/
	if((*ray).getStartX() < getXPos()){
		double t = ( getXPos()-(*ray).getStartX() ) / (*ray).getDirX();
		
		//cout<<t <<endl;
		x_in += t*(*ray).getDirX();
		y_in += t*(*ray).getDirY();
		z_in += t*(*ray).getDirZ();
	}
	//cout<<"1Ray-Start: "<<x_in<<" "<<y_in<<" "<<z_in <<endl;
	if( (x_in<0.) || (y_in<0.) || (z_in<0.) ){
		double t = ( getYPos()-(*ray).getStartY() ) / (*ray).getDirY();

		x_in = (*ray).getStartX() + t*(*ray).getDirX();
		y_in = (*ray).getStartY() + t*(*ray).getDirY();
		z_in = (*ray).getStartZ() + t*(*ray).getDirZ();
	}

	cout<<"2Ray-Start: "<<x_in<<" "<<y_in<<" "<<z_in <<endl;
				
	return getVoxel(x_in,y_in,z_in);
}
/*********************************/

