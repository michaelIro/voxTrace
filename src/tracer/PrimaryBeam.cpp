/** PrimaryBeam */

#include "PrimaryBeam.hpp"

/** Standard Constructor 
 * @param shadowSource
 * @param polyCap
*/
PrimaryBeam::PrimaryBeam(Shadow3API& shadowSource, PolyCapAPI& polyCap, int job_id, int rand_seed, int n_sh_rays, int n_iter, int n_files){

	//srand(time(NULL)); 
	srand(rand_seed);
	std::cout << "Job ID: " << job_id << "\tRandom Seed: " << rand_seed;
	std::cout << "\t#Shadow-Rays: " << n_sh_rays << "\t#ITER: " << n_iter;
	std::cout << "\t#Files " << n_files << std::endl;

	for(int i = 0; i < n_files; i ++){
		
		std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();

		vector<XRBeam> beams_;
		int	counter_ = 0;
		bool fresh_ = false;
		int max_iter_ = n_iter;
		std::mutex mu_;

		auto shadow_lambda_ = [&shadowSource, &fresh_, &counter_, max_iter_,n_sh_rays]() {
			while(counter_ < max_iter_){
				if(fresh_ == true)
					std::this_thread::sleep_for(std::chrono::milliseconds(1)); 
				else{
					Shadow3API shadow_source_copy_(shadowSource);
					shadowSource.trace(n_sh_rays,rand());
					fresh_ = true;
				}
			}
    	};

		auto polycap_lambda_ = [&shadowSource, &fresh_, &counter_, max_iter_, &beams_,&mu_,&polyCap]() {
			while(counter_ < max_iter_){
				if(fresh_ == true){
					arma::Mat<double> shadow_beam_ = shadowSource.getBeamMatrix();
					fresh_=false;

					//PolyCapAPI myPrimaryPolycap((char*) "../test-data/in/polycap/pc-246-descr.txt");
					PolyCapAPI myPrimaryPolycap(polyCap);
					XRBeam tracedBeam(
						//myPrimaryPolycap.trace(shadow_beam_ ,100000,"../test-data/out/beam/beam-"+std::to_string(counter_)+".hdf5",false)
						myPrimaryPolycap.traceFast(shadow_beam_)
					);	
					
					mu_.lock();
					//beams_.push_back(XRBeam::probabilty(tracedBeam));
					beams_.push_back(tracedBeam);
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
		//std::cout << "Here we are!" << std::endl;
		std::cout << ("./out/PrimaryBeam-"+std::to_string(job_id)+"-"+std::to_string(i)+".h5") << std::endl;
		beam_.getMatrix().save(arma::hdf5_name("./out/PrimaryBeam-"+std::to_string(job_id)+"-"+std::to_string(i)+".h5","my_data"));
		
		std::cout << "Beam size: " << beam_.getRays().size() << std::endl;

		std::chrono::steady_clock::time_point t2_ = std::chrono::steady_clock::now();
		std::cout << "Iteration #" << i << " from " << 11 << std::endl; 
		std::cout << "t1 - t0 = " << std::chrono::duration_cast<std::chrono::microseconds>(t1_ - t0_).count() << "[µs]"  << std::endl;
		std::cout << "t2 - t1 = " << std::chrono::duration_cast<std::chrono::microseconds>(t2_ - t1_).count() << "[µs]" << std::endl;
		
	}
	
}