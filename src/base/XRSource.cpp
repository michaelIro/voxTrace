/**Source*/

#include "XRSource.hpp"

using namespace std;

/* Empty constructor */
XRSource::XRSource(){}

/* Constructor */
XRSource::XRSource(vector<Ray> beam, double x0, double y0, double d, double alpha){

	//rayList_= beam;
	//position_=position;
	vector<Ray> rList_;

	alpha = alpha / 180 * M_PI;

	int i=0;
	for (Ray ray : beam) {

		double x0_ = x0 + ray.getStartX();
		double y0_ = y0 - d * cos(alpha) + cos(alpha)*ray.getStartY()-sin(alpha)*ray.getStartZ();;
		double z0_ = -d * sin(alpha) + sin(alpha)*ray.getStartY()+cos(alpha)*ray.getStartZ();

		double xd_ = ray.getDirX(); 
		double yd_ = cos(alpha)*ray.getDirY()-sin(alpha)*ray.getDirZ();
		double zd_ = sin(alpha)*ray.getDirY()+cos(alpha)*ray.getDirZ();

		rList_.push_back(*(new Ray(x0_,y0_,z0_,xd_,yd_,zd_, 0.,0.,0., false, 17.4,i,0.,0.,0.,0.,0.,0.)));
		//ray.print(-i);
		//rList_.back().print(i++);
	}

	rayList_=rList_;
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

