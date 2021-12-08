/** PrimaryBeam */

#include "PrimaryBeam.hpp"

/** Empty Constructor */
PrimaryBeam::PrimaryBeam(){}

/** Standard Constructor */
PrimaryBeam::PrimaryBeam(Shadow3API* shadowSource, PolyCapAPI* polyCap){

	srand(time(NULL)); 
		double randomN = ((double) rand()) / ((double) RAND_MAX);
	
	shadowSource_ = shadowSource;
	polyCap_ = polyCap;

	
	#pragma omp parallel for
    for(int i = 0; i < 16; i++){
		//int mySeed = rand();
		Shadow3API sourceCopy = (*shadowSource);
		sourceCopy.trace(100000000,rand());
		std::cout << i << std::endl;
	}

	//(*shadowSource_).trace(100000000);
    //arma::Mat<double> myShadowBeam = (*shadowSource_).getBeam();
	//myShadowBeam.save("../test-data/beam/shadowBeam.csv", arma::csv_ascii);

	//XRBeam myDetectorBeam((*polyCap_).traceSource(myShadowBeam,1000000));
    //myDetectorBeam.getMatrix().save("../test-data/beam/detectorBeam.csv", arma::csv_ascii);

	//vector<Ray> myPrimaryCapBeam = myPrimaryPolycap.traceSource(myShadowBeam,100000);

	//myPrimaryBeam.primaryTransform(70.0, 70.0,0.0, 0.51, 45.0);
	//myPrimaryBeam.print();
}

//Shadow3API PrimaryBeam::getShadow3API(){return shadowSource_;}