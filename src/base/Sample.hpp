#ifndef Sample_H
#define Sample_H

/**Sample*/

#include <iostream>
#include <vector>

#include "./ChemElement.hpp"
#include "./Voxel.hpp"
//#include "../io/Scan.hpp"


using namespace std;

class Sample{

	private:
		double x_, y_,z_;
		double xL_,yL_,zL_;
		double xLV_,yLV_,zLV_;

		int xN_,yN_,zN_,voxN_;

		vector<vector<vector<Voxel>>> voxels_;
		Voxel *zeroVoxel;
		Voxel *oobVoxel;

	public:
		Sample();
		Sample(double x, double y, double z, double xL, double yL, double zL, double xLV, double yLV, double zLV, vector<vector<vector<Material>>> materials_);
		//Sample(Scan scan_): Sample(0.,0.,0.,scan_.getLengths()[0],scan_.getLengths()[1],scan_.getLengths()[2],15.,15.,15.,scan_.getMaterials()){};
		//Sample(const Sample &sample, double *problem);

		double getXPos() const;
		double getYPos() const;
		double getZPos() const;
		double getXLen() const;
		double getYLen() const;
		double getZLen() const;
		double getXLenVox() const;
		double getYLenVox() const;
		double getZLenVox() const;
		int getXN() const;
		int getYN() const;
		int getZN() const;
		int getVoxN() const;
		
		Voxel* 	getVoxel(double x, double y, double z);
		Voxel* 	getVoxel(int x, int y, int z);
		bool	isOOB(Voxel* vox) const;

		Voxel* findStartVoxel(Ray *ray);
	
		void print();
};

#endif

