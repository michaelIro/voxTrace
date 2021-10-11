/*Voxel*/

#include "Voxel.h"

using namespace std;

Voxel::Voxel(){}

Voxel::Voxel(double x0, double y0, double z0, double x1, double y1, double z1, Material mat){
	x0_ = x0;
	y0_ = y0;
	z0_ = z0;
	x1_ = x1;
	y1_ = y1;
	z1_ = z1;
	mat_ = mat;
	//spe_ = spe;
}

double Voxel::getX0() const {return x0_;}
double Voxel::getY0() const {return y0_;}
double Voxel::getZ0() const {return z0_;}
//double Voxel::getXLen() const {return xL_;}
//double Voxel::getYLen() const {return yL_;}
//double Voxel::getZLen() const {return zL_;}
double Voxel::getX1() const {return (x0_ + x1_);}
double Voxel::getY1() const {return (y0_ + y1_);}
double Voxel::getZ1() const {return (z0_ + z1_);}

Material Voxel::getMaterial() const {return mat_;}
//Spectrum Voxel::getSpectrum() const {return spe_;}

/*Nearest-Neighbour-Array getter*/
Voxel* Voxel::getNN(int i) const {return nn_[i];}

/*Nearest-Neighbour-Array setter*/
void Voxel::setNN(Voxel *v[27]){
	for (int i = 0; i < 27; i++) 
		nn_[i] = v[i]; 
}

void Voxel::print(){
	cout<<x0_<<" "<<y0_<<" "<<z0_<<" ";
	mat_.print();
}

/*********************************/
double Voxel::intersect(Ray* ray, int* nextVoxel, double *tIn){	
	
	double t0x, t1x;
	double t0y, t1y;
	double t0z, t1z;
	double t0_max, t1_min;
	bool xDir=true,yDir=true,zDir=true;

	/*if Ray is x-Parallel no intersection with x-Planes possible*/
	if((*ray).getDirX() != 0.){
		t0x = ( getX0()-(*ray).getStartX() ) / (*ray).getDirX(); 
		t1x = ( getX1()-(*ray).getStartX() ) / (*ray).getDirX();
		if(t0x>t1x){
			double temp = t0x;
			t0x=t1x;
			t1x=temp;
			xDir=false;
		}
	}
	else{
		t0x=std::numeric_limits<double>::min();
		t1x=std::numeric_limits<double>::max();
	}
	if((*ray).getDirY() != 0.){
		t0y = ( getY0()-(*ray).getStartY() ) / (*ray).getDirY(); 
		t1y = ( getY1()-(*ray).getStartY() ) / (*ray).getDirY();  
		if(t0y>t1y){
			double temp = t0y;
			t0y=t1y;
			t1y=temp;
			yDir=false;
		}
	}
	else{
		t0y=std::numeric_limits<double>::min();
		t1y=std::numeric_limits<double>::max();
	}
	if((*ray).getDirZ() != 0.){
		t0z = ( getZ0()-(*ray).getStartZ() ) / (*ray).getDirZ(); 
		t1z = ( getZ1()-(*ray).getStartZ() ) / (*ray).getDirZ();  
		if(t0z>t1z){
			double temp = t0z;
			t0z=t1z;
			t1z=temp;
			zDir=false;
		}
	}
	else{
		t0z=std::numeric_limits<double>::min();
		t1z=std::numeric_limits<double>::max();
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
		if(zDir)	(*nextVoxel)= 22; 	
		else		(*nextVoxel)= 4; 	
	}
	else if	((t1_min != t1z) && (t1_min == t1y) && (t1_min != t1x))	{
		if(yDir) 		(*nextVoxel)= 16; 	
		else 			(*nextVoxel)= 10; 	
	}
	else if	((t1_min != t1z) && (t1_min != t1y) && (t1_min == t1x))	{
		if(xDir) 		(*nextVoxel)= 14; 	
		else 			(*nextVoxel)= 12; 	
	}

	
	/*The following options are very unlikely*/
	else if	((t1_min == t1z) && (t1_min == t1y) && (t1_min != t1x))	{
		if(zDir){
			if(yDir) 	(*nextVoxel)= 25; 	
			else 		(*nextVoxel)= 19; 	
		}
		else{
			if(yDir) 	(*nextVoxel)= 1; 	
			else 		(*nextVoxel)= 7; 	
		}
	}
	else if	((t1_min == t1z) && (t1_min != t1y) && (t1_min == t1x))	{ 
		if(zDir){
			if(xDir) 	(*nextVoxel)= 23; 	
			else 		(*nextVoxel)= 21; 	
		}
		else{
			if(xDir) 	(*nextVoxel)= 3; 	
			else 		(*nextVoxel)= 5; 	
		}
	}	
	else if	((t1_min != t1z) && (t1_min == t1y) && (t1_min == t1x))	{ 
		if(yDir){
			if(xDir) 	(*nextVoxel)= 17; 	
			else 		(*nextVoxel)= 15; 	
		}
		else{
			if(xDir) 	(*nextVoxel)= 11; 	
			else 		(*nextVoxel)= 9; 	
		}
	}	
	

	else if	((t1_min == t1z) && (t1_min == t1y) && (t1_min == t1x))	{ 
		if(zDir){
			if(yDir) {
				if(xDir) 	(*nextVoxel)= 26; 	
				else 		(*nextVoxel)= 24; 	
			}
			else{
				if(xDir) 	(*nextVoxel)= 20; 	
				else 		(*nextVoxel)= 18; 	
			}
		}
		else{
			if(yDir) {
				if(xDir) 	(*nextVoxel)= 8; 	
				else 		(*nextVoxel)= 6; 	
			}
			else{
				if(xDir) 	(*nextVoxel)= 2; 	
				else 		(*nextVoxel)= 0; 	
			}
		}
	}
	
	*tIn = t0_max;
	return (t1_min-t0_max);
}
/*********************************/

