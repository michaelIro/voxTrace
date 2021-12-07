/** PrimaryBeam */

#include "PrimaryBeam.hpp"

/** Empty Constructor */
PrimaryBeam::PrimaryBeam(){}

/** Standard Constructor 
 * @param rays A vector of Ray
 * @return PrimaryBeam
 */
PrimaryBeam::PrimaryBeam(char* shadowPath, char* polycapPath){

	shadowSource_ = Shadow3API(shadowPath);
	shadowSource_.trace(INT_MAX/100);
    arma::Mat<double> myShadowBeam = shadowSource_.getBeam();


    //PolyCapAPI polyCap(polycapPath);
	//vector<Ray> myPrimaryCapBeam = myPrimaryPolycap.traceSource(myShadowBeam,100000);
	//XRBeam myPrimaryBeam(myPrimaryCapBeam);
	//myPrimaryBeam.getMatrix().save("../test-data/beam/primaryBeam.csv", arma::csv_ascii);
	//myPrimaryBeam.primaryTransform(70.0, 70.0,0.0, 0.51, 45.0);
	//myPrimaryBeam.print();
}

//Shadow3API PrimaryBeam::getShadow3API(){return shadowSource_;}