#ifndef Voxel_H
#define Voxel_H

/**Voxel*/

#include <iostream>
#include "./Material.h"
#include "./Spectrum.h"

using namespace std;

class Voxel{
	private:
		double x0_,y0_,z0_;
		double x1_,y1_,z1_;
		//Spectrum spe_;
		Material mat_;
		Voxel *nn_[27];

  public:
		Voxel();
		Voxel(double x0, double y0, double z0, double x1, double y1, double z1, Material mat);

		double getX0() const;
		double getY0() const;
		double getZ0() const;
		double getX1() const;
		double getY1() const;
		double getZ1() const;

		Material getMaterial() const;
		Voxel* getNN(int i) const;
		//Spectrum getSpectrum() const;		

		double intersect(Ray* ray, int* nextVoxel, double *tIn);

		void setMaterial(Material mat);
		void setNN(Voxel *v[27]);

		void print();
};

#endif