/**Source*/

#include "XRSource.hpp"

using namespace std;

XRSource::XRSource(){

	Shadow3API myShadowSource((char*) "../test-data/shadow3");
	arma::Mat<double> myShadowBeam = myShadowSource.getBeamFromSource(10); 
}

XRSource::XRSource(Shadow3API shadowSource){

	Shadow3API myShadowSource((char*) "../test-data/shadow3");
	arma::Mat<double> myShadowBeam = myShadowSource.getBeamFromSource(10); 
}
/*
XRSource::XRSource(string path, double rayNum, double workingDistance, double spotSize){

	RayGenerator rayGenerator_(rayNum,path);

	double x0 = workingDistance / sqrt(2.) - spotSize / 2.;
	double y0 = workingDistance / sqrt(2.) - spotSize / 2.;
	double z0 = -spotSize / 2.;

	int i=0;
	for (Ray ray : rayGenerator_.getRayList()) {
		ray.print(-i);
		rayList_.push_back(*(new Ray(ray,x0,y0,z0,-M_PI/4.,0.)));
		//rayList_.back().print(i++);
	}
}

XRSource::XRSource(XRSource zeroSource, double x, double y, double z){
	for(auto ray: zeroSource.getRayList()){
		rayList_.push_back(*(new Ray(ray,x,y,z,0.,0.)));
	}
}
*/
list<Ray> XRSource::getRayList() const{
	return rayList_;
}

void XRSource::print() const{
	int i =0;
	for(auto ray: rayList_)
		ray.print(i++);
}

