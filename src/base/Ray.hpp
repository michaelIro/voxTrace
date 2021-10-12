#ifndef Ray_H
#define Ray_H

// X-Ray
// Memory-Estimation: 1 instance of Ray approx. 133 byte => 10⁶ Rays approx. 133 MB 

#include <iostream>
#include <math.h> 

using namespace std;
class Ray {
	
	double startX_, startY_, startZ_;	// Start coordinates of X-Ray-photon
	double dirX_, dirY_, dirZ_;			// End Coordinates of direction Vector of X-Ray-Photon
	double asX_, asY_, asZ_;			// sPolarization
	double apX_, apY_, apZ_;			// pPolarization
	bool flag_;							// Flag
	double k_;							// wave number
	int q_;								// ray index/number
	double opd_;						// optical path length 
	double fS_, fP_;					// Phases

	int iaNum_;							// Number of interactions of this ray
	bool iaFlag_;						// Interaction Flag
		
  public:
  	Ray();
	Ray(const Ray& ray);
	Ray(const Ray& ray, double x, double y, double z, double phi, double theta);
  	Ray(double startX, double startY, double startZ, double dirX, double dirY, double dirZ, 
	  double asX, double asY, double asZ, bool flag, double k, int q, double opd, double fS, 
	  double fP, double apX, double apY, double apZ);
	
	double 	getStartX() const;
	double 	getStartY() const;
	double 	getStartZ() const;
	double 	getDirX() const;		
	double 	getDirY() const;		
	double 	getDirZ() const;

	double getSPolX() const;
	double getSPolY() const;
	double getSPolZ() const;
	double getPPolX() const;
	double getPPolY() const;
	double getPPolZ() const;
	double getSPhase() const;
	double getPPhase() const;
		
	int 	getIndex() const;
    bool 	getFlag() const;
    double 	getK() const;
    double 	getOpticalPath() const;
	double 	getEnergyEV() const;
	double 	getEnergyKeV() const;

	bool 	getIAFlag() const;
	int 	getIANum() const;

    
	void setFlag(bool flag);
    void setStartCoordinates (double x, double y, double z);
    void setEndCoordinates (double x, double y, double z);
	void rotate(double phi, double theta);
	void translate(double xDist, double yDist, double zDist);
    void setSPol (double x, double y, double z);
    void setPPol (double x, double y, double z);
	void setEnergy(double keV);

	void setIAFlag(bool iaFlag);
	void setIANum(int iaNum);

	void fluorescence(double newStartX,double newStartY,double newStartZ, double theta, double phi, double Energy);
	void rayleigh(double newStartX,double newStartY,double newStartZ, double theta, double phi);
	

	double* getShadowRay() const;
	void print(int i) const;  
};

#endif

