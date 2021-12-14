/** XRBeam */

#include "XRBeam.hpp"

/** Empty Constructor */
XRBeam::XRBeam(){}

/** Standard Constructor 
 * @param rays A vector of Ray
 * @return XRBeam-Object containing vec<Ray> and Member-Functions
 */
XRBeam::XRBeam(vector<Ray> rays){
	rays_ = rays;
}

/** Alternative Constructor 
 * @param rays An arma::Mat<double> of Ray
 * @return XRBeam-Object containing vec<Ray> and Member-Functions
 */
XRBeam::XRBeam(arma::Mat<double> rays){
	for(int i = 0; i < rays.n_rows; i++){
		rays_.push_back(
			Ray(rays(i,0),rays(i,1),rays(i,2),rays(i,3),
			rays(i,4),rays(i,5),rays(i,6),
			rays(i,7),rays(i,8),rays(i,9),
			rays(i,10)*50677.3*1000.,rays(i,11),rays(i,12), //TODO: Conversion here very bad!
			rays(i,13),rays(i,14),rays(i,15),
			rays(i,16),rays(i,17),rays(i,18)));
	}


}

/** Merges multiple XRBeams to one
 * @param beams A vector of XRBeams 
 * @return One XRBeam-Object containing vec<Ray> and Member-Functions
 */
XRBeam XRBeam::probabilty(XRBeam beam){
	vector<Ray> rays__;
	srand(time(NULL)); 

	for(auto beam_: beam.getRays())
		if(beam_.getProb() > ((double) rand()) / ((double) RAND_MAX))
			rays__.push_back(beam_);

	return XRBeam(rays__);
}

/** Merges multiple XRBeams to one
 * @param beams A vector of XRBeams 
 * @return One XRBeam-Object containing vec<Ray> and Member-Functions
 */
XRBeam XRBeam::merge(vector<XRBeam> beams){
	vector<Ray> rays__;

	for(XRBeam beam_: beams)
		for(Ray ray_: beam_.getRays())
			rays__.push_back(ray_);

	return XRBeam(rays__);
}

/** Transform Beam Coordinate System -> Shift in x-/y-/z-direction
 * @param xShift shift in x-direction in ...
 * @param yShift shift in y-direction in ...
 * @param zShift shift in z-direction in ...
 */
void XRBeam::shift(double xShift, double yShift, double zShift){

	vector<Ray> rays__ = {};
	for(auto ray: rays_){
		rays__.push_back(*(new Ray(ray,ray.getStartX()+xShift,ray.getStartY()+yShift,ray.getStartZ()+zShift,0.,0.)));
	}
	rays_=rays__;
}

/** Transform Primary Beam from 
 * @param x0 description
 * @param y0 description
 * @param z0 description
 * @param d description
 * @param alpha description
 */
void XRBeam::primaryTransform(double x0, double y0, double z0, double d, double alpha){

	vector<Ray> rays__={};

	alpha = alpha / 180 * M_PI;

	int i=0;
	for (Ray ray : rays_) {

		double x0_ = x0 + ray.getStartX();
		double y0_ = y0 - d * cos(alpha) + cos(alpha)*ray.getStartY()-sin(alpha)*ray.getStartZ();
		double z0_ = z0 - d * sin(alpha) + sin(alpha)*ray.getStartY()+cos(alpha)*ray.getStartZ();

		double xd_ = ray.getDirX(); 
		double yd_ = cos(alpha)*ray.getDirY()-sin(alpha)*ray.getDirZ();
		double zd_ = sin(alpha)*ray.getDirY()+cos(alpha)*ray.getDirZ();

		rays__.push_back(*(new Ray(
			x0_, y0_, z0_,
			xd_, yd_, zd_,
			ray.getSPolX(),ray.getSPolY(),ray.getSPolZ(), 
			ray.getFlag(), ray.getWaveNumber(),ray.getIndex(),
			ray.getOpticalPath(),ray.getSPhase(),ray.getPPhase(),
			ray.getPPolX(),ray.getPPolY(),ray.getPPolZ(), 
			ray.getProb()
		)));
	}
	rays_=rays__;
}

/** Transform Secondary Beam from 
 * @param x0 description
 * @param y0 description
 * @param z0 description
 * @param d description
 * @param beta description
 */
void XRBeam::secondaryTransform(double x0, double y0, double z0, double d, double beta){
	vector<Ray> rays__;
	beta = beta / 180 * M_PI;
	int i =0;
	for(Ray ray: rays_){
		if((ray.getStartZ()>=0.0) && (ray.getDirZ()<0.0)){

			double x0_=ray.getStartX()-x0;
			double y0_=cos(beta)*(ray.getStartY()-y0)-sin(beta)*(ray.getStartZ()-z0);
			double z0_=sin(beta)*(ray.getStartY()-y0)+cos(beta)*(ray.getStartZ()-z0);

			double xd_ = ray.getDirX(); 
			double yd_ = cos(beta)*ray.getDirY()-sin(beta)*ray.getDirZ();
			double zd_ = sin(beta)*ray.getDirY()+cos(beta)*ray.getDirZ();


			double dfac_= 0.49 / yd_;
			double rin_= 0.1; //actually 0.095
			double r_spot_ = sqrt( (xd_*dfac_)*(xd_*dfac_) + (zd_*dfac_)*(zd_*dfac_));
			//if(r_spot_ < rin_){
				rays__.push_back(*(new Ray(
					x0_, y0_, z0_,
					xd_, yd_, zd_,
					ray.getSPolX(),ray.getSPolY(),ray.getSPolZ(), 
					ray.getFlag(), ray.getWaveNumber(),ray.getIndex(),
					ray.getOpticalPath(),ray.getSPhase(),ray.getPPhase(),
					ray.getPPolX(),ray.getPPolY(),ray.getPPolZ(), 
					ray.getProb()
				)));
			//}
		}
	}
	rays_=rays__;
}


/** Getter*/
vector<Ray> XRBeam::getRays() const{
	return rays_;
}

/** Getter*///x0,y0,z0,   xd,yd,zd,   asx,asy,asz,    flag,k,index,   opd,fs,fp,  apx,apy,apz
arma::Mat<double> XRBeam::getMatrix() const{
	arma::Mat<double> rays__ = arma::ones(rays_.size(), 19);
	int i =0;
	for(auto ray: rays_){
		//arma::rowvec row_ = 
		rays__.row(i++)={
			ray.getStartX(),ray.getStartY(),ray.getStartZ(),
			ray.getDirX(),ray.getDirY(),ray.getDirZ(),
			ray.getSPolX(),ray.getSPolY(),ray.getSPolZ(),
			(double) ray.getFlag(), ray.getEnergyKeV(),(double) ray.getIndex(),
			ray.getOpticalPath(),ray.getSPhase(),ray.getPPhase(),
			ray.getPPolX(),ray.getPPolY(),ray.getPPolZ(),
			ray.getProb()};
	}
		
	return rays__;
}

/** Print all Rays */
void XRBeam::print() const{
	int i =0;
	for(auto ray: rays_)
		ray.print(i++);
}