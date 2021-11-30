/** Source */

#include "XRBeam.hpp"

/** Empty Constructor */
XRBeam::XRBeam(){}

/** Standard Constructor 
 * @param rays A vector of Ray
 * @return XRBeam
 */
XRBeam::XRBeam(vector<Ray> rays){
	rayList_ = rays;
}

/** Transform Beam from 
 * @param xShift description
 * @param yShift description
 * @param zShift description
 */
void XRBeam::shift(double xShift, double yShift, double zShift){

	vector<Ray> rays_ = {};
	for(auto ray: rayList_){
		rays_.push_back(*(new Ray(ray,ray.getStartX()+xShift,ray.getStartY()+yShift,ray.getStartZ()+zShift,0.,0.)));
	}

	rayList_=rays_;
}

/** Transform Primary Beam from 
 * @param x0 description
 * @param y0 description
 * @param z0 description
 * @param d description
 * @param alpha description
 */
void XRBeam::primaryTransform(double x0, double y0, double z0, double d, double alpha){

	vector<Ray> rays_={};

	alpha = alpha / 180 * M_PI;

	int i=0;
	for (Ray ray : rayList_) {

		double x0_ = x0 + ray.getStartX();
		double y0_ = y0 - d * cos(alpha) + cos(alpha)*ray.getStartY()-sin(alpha)*ray.getStartZ();
		double z0_ = z0 - d * sin(alpha) + sin(alpha)*ray.getStartY()+cos(alpha)*ray.getStartZ();

		double xd_ = ray.getDirX(); 
		double yd_ = cos(alpha)*ray.getDirY()-sin(alpha)*ray.getDirZ();
		double zd_ = sin(alpha)*ray.getDirY()+cos(alpha)*ray.getDirZ();

		rays_.push_back(*(new Ray(
			x0_, y0_, z0_,
			xd_, yd_, zd_,
			ray.getSPolX(),ray.getSPolY(),ray.getSPolZ(), 
			ray.getFlag(), ray.getEnergyKeV(),ray.getIndex(),
			ray.getOpticalPath(),ray.getSPhase(),ray.getPPhase(),
			ray.getPPolX(),ray.getPPolY(),ray.getPPolZ(), 
			ray.getProb()
		)));
	}
	rayList_=rays_;
}

/** Transform Secondary Beam from 
 * @param x0 description
 * @param y0 description
 * @param z0 description
 * @param d description
 * @param beta description
 */
void XRBeam::secondaryTransform(double x0, double y0, double z0, double d, double beta){
	vector<Ray> rays_;
	beta = beta / 180 * M_PI;
	int i =0;
	for(Ray ray: rayList_){
		if((ray.getStartZ()>=0) && (ray.getDirZ()<0) ){

			double x0_=ray.getStartX()-x0;
			double y0_=cos(beta)*(ray.getStartY()-y0)-sin(beta)*(ray.getStartZ()-z0);
			double z0_=sin(beta)*(ray.getStartY()-y0)+cos(beta)*(ray.getStartZ()-z0);

			double xd_ = ray.getDirX(); 
			double yd_ = cos(beta)*ray.getDirY()-sin(beta)*ray.getDirZ();
			double zd_ = sin(beta)*ray.getDirY()+cos(beta)*ray.getDirZ();

			rays_.push_back(*(new Ray(
				x0_, y0_, z0_,
				xd_, yd_, zd_,
				ray.getSPolX(),ray.getSPolY(),ray.getSPolZ(), 
				ray.getFlag(), ray.getEnergyKeV(),ray.getIndex(),
				ray.getOpticalPath(),ray.getSPhase(),ray.getPPhase(),
				ray.getPPolX(),ray.getPPolY(),ray.getPPolZ(), 
				ray.getProb()
			)));
		}
	}
	rayList_=rays_;
}


/** Getter*/
vector<Ray> XRBeam::getRays() const{
	return rayList_;
}

/** Getter*/
arma::Mat<double> XRBeam::getMatrix() const{
	arma::Mat<double> rays = arma::ones(rayList_.size(), 6);
	int i =0;
	for(auto ray: rayList_)
		rays.row(i++)={ray.getStartX(),ray.getStartY(),ray.getStartZ(),ray.getDirX(),ray.getDirY(),ray.getDirZ()};

	return rays;
}

/** Print all Rays */
void XRBeam::print() const{
	int i =0;
	for(auto ray: rayList_)
		ray.print(i++);
}