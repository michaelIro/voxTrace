/** PrimaryBeam */

#include "PrimaryBeam.hpp"

/** Standard Constructor 
 * @param shadowSource
 * @param polyCap
*/
PrimaryBeam::PrimaryBeam(Shadow3API& shadowSource, PolyCapAPI& polyCap){

	srand(time(NULL)); 

	for(int i = 0 ; i < 1; i ++){

		std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();

		vector<XRBeam> beams_;
		int	counter_ = 0;
		bool fresh_ = false;
		int max_iter_ = 25;
		std::mutex mu_;

		auto shadow_lambda_ = [&shadowSource, &fresh_, &counter_, max_iter_]() {
			while(counter_ < max_iter_){
				if(fresh_ == true)
					std::this_thread::sleep_for(std::chrono::milliseconds(1)); 
				else{
					//Shadow3API shadow_source_copy_(&shadowSource);
					shadowSource.trace(10000000,rand());
					fresh_ = true;
				}
			}
    	};

		auto polycap_lambda_ = [&shadowSource, &fresh_, &counter_, max_iter_, &beams_,&mu_]() {
			while(counter_ < max_iter_){
				if(fresh_ == true){
					arma::Mat<double> shadow_beam_ = shadowSource.getBeamMatrix();
					fresh_=false;

					PolyCapAPI myPrimaryPolycap((char*) "../test-data/polycap/pc-246-descr.txt");
					XRBeam tracedBeam(myPrimaryPolycap.trace(shadow_beam_ ,100000,(char *)"../test-data/beam/beam.hdf5"));	
					
					mu_.lock();
					beams_.push_back(XRBeam::probabilty(tracedBeam));
					mu_.unlock();

					counter_++;
				}
			}
   	 	};

		thread shadow_thread(shadow_lambda_);
		thread polycap_thread(polycap_lambda_);

		shadow_thread.join();
		polycap_thread.join();

		std::chrono::steady_clock::time_point t1_ = std::chrono::steady_clock::now();

		XRBeam beam_ = XRBeam::merge(beams_);
		beam_.getMatrix().save(arma::hdf5_name("/media/miro/Data/Shadow-Beam/Transmission/PrimaryBeam-1keV-"+std::to_string(i)+".h5","my_data"));
		
		std::cout << "Beam size: " << beam_.getRays().size() << std::endl;

		std::chrono::steady_clock::time_point t2_ = std::chrono::steady_clock::now();
		std::cout << "Iteration #" << i << " from " << 11 << std::endl; 
		std::cout << "t1 - t0 = " << std::chrono::duration_cast<std::chrono::microseconds>(t1_ - t0_).count() << "[µs]"  << std::endl;
		std::cout << "t2 - t1 = " << std::chrono::duration_cast<std::chrono::microseconds>(t2_ - t1_).count() << "[µs]" << std::endl;
	}

	//double randomN = ((double) rand()) / ((double) RAND_MAX);
	/*
	int threadNum = 4;
	int raysPerThread = 8000000;
	vector<int> randomNumbers;
	vector<XRBeam> beams_(threadNum);

	#pragma omp parallel
	{
		//arma::Mat<double> temp_beam_;

		#pragma omp for
		for(int i = 0; i < threadNum; i++){
    		//Shadow3API myShadowSource((char*) "../test-data/shadow3");
			Shadow3API myShadowSource(shadowSource);
			myShadowSource.trace(raysPerThread,rand());

    		PolyCapAPI myPrimaryPolycap((char*) "../test-data/polycap/pc-246-descr.txt");	

			XRBeam myDetectorBeam(
				myPrimaryPolycap.trace(myShadowSource.getBeamMatrix(),100000,(char *)"../test-data/beam/beam.hdf5")
			);

			beams_[i] = XRBeam::probabilty(myDetectorBeam);
		}
	}

	XRBeam finalBeam = XRBeam::merge(beams_);
	finalBeam.getMatrix().save(arma::hdf5_name("/media/miro/Data/Shadow-Beam/PrimaryBeam.h5", "my_data"));


	arma::Mat<double> temp_;
    //temp_.load(arma::hdf5_name("/media/miro/Data/Shadow-Beam/PrimaryBeam.h5", "my_data"));
	*/
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