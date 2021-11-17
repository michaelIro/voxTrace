/**Source*/

#include "XRSource.hpp"

using namespace std;

/* Empty constructor */
XRSource::XRSource(){}

/* Constructor */
XRSource::XRSource(vector<Ray> beam, double position, double workingDistance, double spotSize){

	rayList_= beam;
	position_=position;

	double x0 = workingDistance / sqrt(2.) - spotSize / 2.;
	double y0 = workingDistance / sqrt(2.) - spotSize / 2.;
	double z0 = -spotSize / 2.;

	int i=0;
	for (Ray ray : rayList_) {
		ray.print(-i);
		//rayList_.push_back(*(new Ray(ray,x0,y0,z0,-M_PI/4.,0.)));
		//rayList_.back().print(i++);
	}
}


XRSource::XRSource(XRSource zeroSource, double x, double y, double z){
	for(auto ray: zeroSource.getRayList()){
		rayList_.push_back(*(new Ray(ray,x,y,z,0.,0.)));
	}
}


/** Getter*/
vector<Ray> XRSource::getRayList() const{
	return rayList_;
}

void XRSource::print() const{
	int i =0;
	for(auto ray: rayList_)
		ray.print(i++);
}

