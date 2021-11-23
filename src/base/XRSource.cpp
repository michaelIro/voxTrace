/**Source*/

#include "XRSource.hpp"

using namespace std;

/* Empty constructor */
XRSource::XRSource(){}

/* Constructor */
XRSource::XRSource(vector<Ray> beam, double x0, double y0, double d, double alpha, double beta){

	//rayList_= beam;
	//position_=position;
	vector<Ray> rList_;

	alpha = alpha / 180 * M_PI;
	beta = beta / 180 * M_PI;
	
	double x0_ = x0 - d * cos(alpha) * sin(beta);
	double y0_ = y0 - d * cos(alpha) * cos(beta);
	double z0_ = -d * sin(alpha);

	
	//double x0_ = 0.;
	//double y0_ = -1.;
	//double z0_ = -1.;

	int i=0;
	for (Ray ray : beam) {

		double xd_ = 0.;
		double yd_ = 1.;
		double zd_ = 0.;

		xd_ = cos(beta)*ray.getDirX() - sin(beta) * (cos(alpha)*ray.getDirY()-sin(alpha)*ray.getDirZ());
		yd_ = sin(beta)*ray.getDirX() - cos(beta) * (cos(alpha)*ray.getDirY()-sin(alpha)*ray.getDirZ());
		zd_ = sin(alpha)*ray.getDirY() + cos(alpha)*ray.getDirZ();





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

