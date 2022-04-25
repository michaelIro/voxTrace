/** Voxel-Object for GPU */

#include "VoxelGPU.cuh"

//Material VoxelGPU::getMaterial() const {return mat_;}

/*Nearest-Neighbour-Array getter*/
//VoxelGPU* VoxelGPU::getNN(int i) const {return nn_[i];}

/*Nearest-Neighbour-Array setter*/
/*void VoxelGPU::setNN(VoxelGPU *v[27]){
	for (int i = 0; i < 27; i++) 
		nn_[i] = v[i]; 
}*/

void VoxelGPU::print(){
	std::cout << x0_ << "\t" << y0_ << "\t" << z0_ << "\n";
	//mat_.print();
}

/*********************************/

double VoxelGPU::intersect(Ray* ray, int* nextVoxelGPU, double *tIn){	
	
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
}
/*********************************/

