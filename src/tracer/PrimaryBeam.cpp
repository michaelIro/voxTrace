/** PrimaryBeam */

#include "PrimaryBeam.hpp"

/** Empty Constructor */
PrimaryBeam::PrimaryBeam(){}

/** Standard Constructor */
PrimaryBeam::PrimaryBeam(Shadow3API* shadowSource, PolyCapAPI* polyCap){

	srand(time(NULL)); 
	double randomN = ((double) rand()) / ((double) RAND_MAX);

	//shadowSource_ = shadowSource;
	//polyCap_ = polyCap;

	int threadNum = 32;
	int raysPerThread = 1000000;
	vector<int> randomNumbers;
	vector<XRBeam> beams_(32);
	//vector<arma::Mat<double>> beam_(threadNum);

	PolyCapAPI polyCapCopy = *polyCap;
	//PolyCapAPI polyCapCopy = (*polyCap);

	std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();
	#pragma omp parallel
	for(int i = 0; i < threadNum; i++){

		//(*shadowSource).trace(raysPerThread,rand());

    	Shadow3API myShadowSource = (*shadowSource);
		myShadowSource.trace(raysPerThread,rand());

    	PolyCapAPI myPrimaryPolycap((char*) "../test-data/polycap/pc-246-descr.txt");	
		//PolyCapAPI polyCapCopy_ = polyCapCopy;
		XRBeam myDetectorBeam(
			myPrimaryPolycap.trace(myShadowSource.getBeamMatrix(),12500,"../test-data/beam/beam.hdf5")
			);
		beams_[i] = XRBeam::probabilty(myDetectorBeam);
	}

	std::chrono::steady_clock::time_point t1_ = std::chrono::steady_clock::now();
	XRBeam finalBeam = XRBeam::merge(beams_);
	finalBeam.getMatrix().save(arma::hdf5_name("/media/miro/Data/Shadow-Beam/PrimaryBeam.h5", "my_data"));
	std::chrono::steady_clock::time_point t2_ = std::chrono::steady_clock::now();

	std::cout << "t1 - t0 = " << std::chrono::duration_cast<std::chrono::microseconds>(t1_ - t0_).count() << "[µs]"  << std::endl;
	std::cout << "t2 - t1 = " << std::chrono::duration_cast<std::chrono::microseconds>(t2_ - t1_).count() << "[µs]" << std::endl;


	/*	#pragma omp parallel for
	for(int i = 0; i < threadNum; i++){
		Shadow3API shadowCopy_ = (*shadowSource);
		shadowCopy_.trace(raysPerThread/threadNum,randomNumbers[i+1]);
		//beam_.push_back((*shadowSource).getBeamMatrix());
		//std::string name = "../test-data/beam/Beam-" + std::to_string(i) + ".h5";
		//(*shadowSource).getBeamMatrix().save(arma::hdf5_name(name, "my_data"));
	} 
	
	arma::Mat<double> temp;
	for(int i = 0; i <24; i++){
		std::string name = "/media/miro/Data/Shadow-Beam/Beam-" + std::to_string(i) + ".h5";
		temp.load(arma::hdf5_name(name, "my_data"));
		temp.print();
		(*polyCap).trace(temp, 10);
	}
	*/

	//myPrimaryBeam.primaryTransform(70.0, 70.0,0.0, 0.51, 45.0);
}