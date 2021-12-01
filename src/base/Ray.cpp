/**X-Ray*/

#include "Ray.hpp"

using namespace std;

/** Empty constructor */
Ray::Ray(){}

/** Copy constructor */
Ray::Ray(const Ray& ray){
	startX_ = ray.getStartX();
	startY_ = ray.getStartY();
	startZ_ = ray.getStartZ();
	dirX_ = ray.getDirX();
	dirY_ = ray.getDirY();
	dirZ_ = ray.getDirZ();
	asX_= ray.getSPolX();
	asY_= ray.getSPolY();
	asZ_= ray.getSPolZ();
	apX_= ray.getPPolX();
	apY_= ray.getPPolY();
	apZ_= ray.getPPolZ();
	flag_=ray.getFlag();
	k_=ray.getWaveNumber();
	q_=ray.getIndex();
	opd_=ray.getOpticalPath();
	fS_=ray.getSPhase();
	fP_=ray.getPPhase();

	iaNum_=ray.getIANum();
	iaFlag_ =ray.getIAFlag();
	prob_= ray.getProb();
}

/** Coordinate Transofrmation */
Ray::Ray(const Ray& ray, double x, double y, double z, double phi, double theta){

	startX_ = ray.getStartX() - x;
	startY_ = ray.getStartY() - y;
	startZ_ = ray.getStartZ() - z;

	dirX_ = ray.getDirX();
	dirY_ = ray.getDirY();
	dirZ_ = ray.getDirZ();

	double diX = cos(theta)*cos(phi)*dirX_ - sin(phi)*dirY_ + sin(theta)*cos(phi)*dirZ_;
	double diY = cos(theta)*sin(phi)*dirX_ + cos(phi)*dirY_ + sin(theta)*sin(phi)*dirZ_;
	double diZ = -sin(theta)*dirX_+cos(theta)*dirZ_;

	dirX_=diX;
	dirY_=diY;
	dirZ_=diZ;

	asX_= ray.getSPolX();
	asY_= ray.getSPolY();
	asZ_= ray.getSPolZ();

	double asX = cos(theta)*cos(phi)*asX_ - sin(phi)*asY_ + sin(theta)*cos(phi)*asZ_;
	double asY = cos(theta)*sin(phi)*asX_ + cos(phi)*asY_ + sin(theta)*sin(phi)*asZ_;
	double asZ = -sin(theta)*asX_+cos(theta)*asZ_;

	asX_= asX;
	asY_= asY;
	asZ_= asY;

	apX_= ray.getPPolX();
	apY_= ray.getPPolY();
	apZ_= ray.getPPolZ();

	double apX = cos(theta)*cos(phi)*apX_ - sin(phi)*apY_ + sin(theta)*cos(phi)*apZ_;
	double apY = cos(theta)*sin(phi)*apX_ + cos(phi)*apY_ + sin(theta)*sin(phi)*apZ_;
	double apZ = -sin(theta)*apX_+cos(theta)*apZ_;

	apX_= apX;
	apY_= apY;
	apZ_= apZ;

	flag_=ray.getFlag();
	k_=ray.getWaveNumber();
	q_=ray.getIndex();
	opd_=ray.getOpticalPath();
	fS_=ray.getSPhase();
	fP_=ray.getPPhase();
	
	iaNum_=ray.getIANum();
	iaFlag_=ray.getIAFlag();
}

Ray::Ray(double startX, double startY, double startZ, double dirX, double dirY, double dirZ, 
		double asX, double asY, double asZ, bool flag, double k, int q, double opd, double fS, 
		double fP, double apX, double apY, double apZ, double prob){
	startX_ = startX;
	startY_ = startY;
	startZ_ = startZ;
	

	dirX_ = dirX; 	
	dirY_ = dirY;
	dirZ_ = dirZ;		

	asX_= asX;
	asY_= asY;
	asZ_= asZ;
	apX_= apX;
	apY_= apY;
	apZ_= apZ;

	flag_=flag;
	k_=k;
	q_=q;
	opd_=opd;
	fS_=fS;
	fP_=fP;
	
	iaNum_=0;
	iaFlag_=false;
	prob_= prob;
}

/*Member-Getter*/
double Ray::getStartX() const {return startX_;}
double Ray::getStartY() const {return startY_;}
double Ray::getStartZ() const {return startZ_;}
double Ray::getDirX() const {return dirX_;}
double Ray::getDirY() const {return dirY_;}
double Ray::getDirZ() const {return dirZ_;}
double Ray::getSPolX() const {return asX_;}
double Ray::getSPolY() const {return asY_;}
double Ray::getSPolZ() const {return asZ_;}
double Ray::getPPolX() const {return apX_;}
double Ray::getPPolY() const {return apY_;}
double Ray::getPPolZ() const {return apZ_;}
double Ray::getSPhase() const {return fS_;}
double Ray::getPPhase() const {return fP_;}
double Ray::getWaveNumber() const {return k_;}
int Ray::getIndex() const {return q_;}
bool Ray::getFlag() const {return flag_;}
double Ray::getOpticalPath() const {return opd_;}
double Ray::getProb() const {return prob_;}

int Ray::getIANum() const {return iaNum_;}
bool Ray::getIAFlag() const {return iaFlag_;}


double Ray::getEnergyKeV() const {
	return (k_)/*phy::hBar*phy::c/phy::e)/10.*/;
} // in keV

void Ray::setFlag(bool flag){flag_=flag;}
void Ray::setIANum(int iaNum){iaNum_=iaNum;}
void Ray::setIAFlag(bool iaFlag){iaFlag_=iaFlag;}

void Ray::rotate(double phi, double theta){
	double diX = cos(theta)*cos(phi)*dirX_ - sin(phi)*dirY_ + sin(theta)*cos(phi)*dirZ_;
	double diY = cos(theta)*sin(phi)*dirX_ + cos(phi)*dirY_ + sin(theta)*sin(phi)*dirZ_;
	double diZ = -sin(theta)*dirX_+cos(theta)*dirZ_;
	dirX_=diX;
	dirY_=diY;
	dirZ_=diZ;
}

void Ray::translate(double xDist, double yDist, double zDist){
	startX_ -= xDist;
	startY_ -= yDist;
	startZ_ -= zDist;
}

void Ray::setStartCoordinates (double x, double y, double z){
	startX_ = x;
	startY_ = y;
	startZ_ = z;
}

void Ray::setEnergy(double keV){
	k_=keV/*10*phy::e/(phy::hBar*phy::c)*/;
}

double* Ray::getShadowRay() const{
	double *ray;
	return ray;	
}

void Ray::print(int i)const{
	cout << endl;
	cout.precision(19);
	cout << "Ray " << i << "\t Start: \t" << startX_ << "\t " << startY_ << "\t " << startZ_ << endl;
	cout << "Ray " << i << "\t Direct: \t" << dirX_ << "\t " << dirY_ << "\t " << dirZ_ << endl;
	//cout << "Ray " << i << "\t S-Pol: \t" << asX_ << "\t " << asY_ << "\t " << asZ_ << endl;
	//cout << "Ray " << i << "\t P-Pol: \t" << apX_ << "\t " << apY_ << "\t " << apZ_ << endl;
	//cout << "Ray " << i << "\t Phases: \t" << fS_ << "\t " << fP_ << endl;
	
	cout << "Ray " << i << "\t K & E: \t" << k_ << "\t " << getEnergyKeV() << endl;
	//cout << "Ray " << i << "\t Index & Flag & OPD: \t" << q_ << "\t " << flag_ << "\t " << opd_ << endl;
	//cout << "Ray " << i << "\t #IA: \t" << iaNum_<< "\t IA-Flag: \t" << iaFlag_<< endl;
	cout << endl;
	
	//long double dir_xL= (long double) dirX_;
	//long double dir_yL= (long double) dirY_;
	//long double dir_zL= (long double) dirZ_;
	//long double lengthLong = dir_xL*dir_xL+dir_yL*dir_yL+dir_zL*dir_zL;
	//cout<<"Long Dir-Length "<<dir_xL<<" "<<dir_yL<<" "<<" "<<dir_zL<<" "<<lengthLong<<endl;
	//cout<<"Long Dir-Length² "<<dir_xL*dir_xL<<" "<<dir_yL*dir_yL<<" "<<" "<<dir_zL*dir_zL<<" "<<lengthLong<<endl;
	cout << endl;
}

