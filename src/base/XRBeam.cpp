/**Source*/

#include "XRBeam.hpp"

using namespace std;

/** Empty constructor */
XRBeam::XRBeam(){}

/** Empty constructor */
XRBeam::XRBeam(vector<Ray> beam){
	rayList_ = beam;
}


/* Constructor */
XRBeam::XRBeam(vector<Ray> beam, double x0, double y0, double z0, double d, double alpha){

	rayList_={};

	alpha = alpha / 180 * M_PI;

	int i=0;
	for (Ray ray : beam) {

		double x0_ = x0 + ray.getStartX();
		double y0_ = y0 - d * cos(alpha) + cos(alpha)*ray.getStartY()-sin(alpha)*ray.getStartZ();
		double z0_ = z0 - d * sin(alpha) + sin(alpha)*ray.getStartY()+cos(alpha)*ray.getStartZ();

		double xd_ = ray.getDirX(); 
		double yd_ = cos(alpha)*ray.getDirY()-sin(alpha)*ray.getDirZ();
		double zd_ = sin(alpha)*ray.getDirY()+cos(alpha)*ray.getDirZ();

		rayList_.push_back(*(new Ray(x0_,y0_,z0_,xd_,yd_,zd_, 0.,0.,0., false, 17.4,i++,0.,0.,0.,0.,0.,0.)));
	}
}

void XRBeam::secondaryTransform(double x0, double y0, double z0, double d, double alpha){

	
	vector<Ray> rays_;
	alpha = alpha / 180 * M_PI;

	int i=0;
	for (Ray ray : rayList_) {

		double x0_ = x0 + ray.getStartX();
		double y0_ = y0 - d * cos(alpha) + cos(alpha)*ray.getStartY()-sin(alpha)*ray.getStartZ();;
		double z0_ = z0 + d * sin(alpha) + sin(alpha)*ray.getStartY()+cos(alpha)*ray.getStartZ();

		double xd_ = ray.getDirX(); 
		double yd_ = cos(alpha)*ray.getDirY()-sin(alpha)*ray.getDirZ();
		double zd_ = sin(alpha)*ray.getDirY()+cos(alpha)*ray.getDirZ();

		rays_.push_back(*(new Ray(x0_,y0_,z0_,xd_,yd_,zd_, 0.,0.,0., false, 17.4,i++,0.,0.,0.,0.,0.,0.)));
	}
	rayList_=rays_;

}


XRBeam::XRBeam(XRBeam beam, double xShift, double yShift, double zShift){

	rayList_ = {};
	for(auto ray: beam.getRays()){
		rayList_.push_back(*(new Ray(ray,ray.getStartX()+xShift,ray.getStartY()+yShift,ray.getStartZ()+zShift,0.,0.)));
	}

}


/** Getter*/
vector<Ray> XRBeam::getRays() const{
	return rayList_;
}

/** Getter*/
void XRBeam::print() const{
	int i =0;
	for(auto ray: rayList_)
		ray.print(i++);
}