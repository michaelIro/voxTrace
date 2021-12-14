#ifndef Ray_H
#define Ray_H

// X-Ray
// Memory-Estimation: 1 instance of Ray approx. 133 byte => 10⁶ Rays approx. 133 MB 

#include <iostream>
#include <math.h> 

using namespace std;
class Ray {
	// These Parameters are taken 1:1 from Shadow3
	double x0_, y0_, z0_;	// Start coordinates of X-Ray-photon
	double dirX_, dirY_, dirZ_;			// Coordinates of direction Vector of X-Ray-Photon
	double asX_, asY_, asZ_;			// sPolarization
	double apX_, apY_, apZ_;			// pPolarization
	bool flag_;							// Flag
	double k_;							// wave number
	int q_;								// ray index/number
	double opd_;						// optical path length 
	double fS_, fP_;					// Phases

	// Further Parameters needed for voxTrace
	int iaNum_;							// Number of interactions of this ray
	bool iaFlag_;						// Interaction Flag
	double prob_;						// Probability
		
  public:
  	Ray();
	Ray(const Ray& ray);
	Ray(const Ray& ray, double x, double y, double z, double phi, double theta);
  	Ray(double startX, double startY, double startZ, double dirX, double dirY, double dirZ, 
	  double asX, double asY, double asZ, bool flag, double k, int q, double opd, double fS, 
	  double fP, double apX, double apY, double apZ, double prob);
	
	void rotate(double phi, double theta);
	void translate(double xDist, double yDist, double zDist);

	double* getShadowRay() const;
	void print(int i) const;  

	// Basic Setter and Getter Functions
	void setFlag(bool flag);
    void setStartCoordinates (double x, double y, double z);
    void setEndCoordinates (double x, double y, double z);
    void setSPol (double x, double y, double z);
    void setPPol (double x, double y, double z);
	void setEnergy(double keV);
	void setIAFlag(bool iaFlag);
	void setIANum(int iaNum);

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
    double 	getWaveNumber() const;
    double 	getOpticalPath() const;
	double 	getEnergyEV() const;
	double 	getEnergyKeV() const;

	bool 	getIAFlag() const;
	int 	getIANum() const;
	double 	getProb() const;
};

#endif

