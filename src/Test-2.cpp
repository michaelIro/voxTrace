#include <iostream>
#include <boost/random/mersenne_twister.hpp>

#include "api/PolyCapAPI.hpp"
#include "api/Shadow3API.hpp"

#include "base/XRBeam.hpp"

#include "tracer/PrimaryBeam.hpp"


int main() {
	std::cout << "START: Test-2" << std::endl;
    
    Shadow3API shadow_((char*) "../test-data/in/shadow3");
    PolyCapAPI pc1_((char*) "../test-data/in/polycap/pc-246-descr.txt");

    unsigned int seed_ = chrono::steady_clock::now().time_since_epoch().count();
    boost::mt19937 rand_gen_(seed_);

	std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();

    shadow_.trace(80000000,rand_gen_());
    arma::Mat<double> shadowBeam_;
    shadowBeam_ = shadow_.getBeamMatrix();
    //shadowBeam_.save(arma::hdf5_name("../test-data/out/beam/shadow_beam.h5", "my_data"));
    //shadowBeam_.load(arma::hdf5_name("../test-data/out/beam/shadow_beam.h5", "my_data")); 

	std::chrono::steady_clock::time_point t1_ = std::chrono::steady_clock::now();

    //XRBeam beam0_( pc1_.traceFastThread(shadowBeam_ ) );
    //beam0_.getMatrix().save(arma::hdf5_name("../test-data/out/beam/beam-00.h5", "my_data"));

    std::chrono::steady_clock::time_point t2_ = std::chrono::steady_clock::now();

    //XRBeam beam1_( pc1_.traceFast(shadowBeam_ ) );
    //beam1_.getMatrix().save(arma::hdf5_name("../test-data/out/beam/beam-01.h5", "my_data"));

    std::chrono::steady_clock::time_point t3_ = std::chrono::steady_clock::now();
    
    XRBeam beam2_( pc1_.trace(shadowBeam_ ,1000000,"../test-data/out/beam/beam-02.hdf5",true) );
    //XRBeam beam3_ = XRBeam::probabilty(beam2_);
    //beam2_.getMatrix().save(arma::hdf5_name("../test-data/out/beam/beam-02.h5", "my_data"));
    //beam3_.getMatrix().save(arma::hdf5_name("../test-data/out/beam/beam-03.h5", "my_data"));

    std::chrono::steady_clock::time_point t4_ = std::chrono::steady_clock::now();


    //std::cout << std::endl << std::endl; //
    //std::cout << "Successfully traced Rays: " << beam0_.getRays().size() << " vs. " <<  beam1_.getRays().size() << " vs. " <<  beam3_.getRays().size() << std::endl << std::endl;
	//std::cout << "t1 - t0 = " << std::chrono::duration_cast<std::chrono::microseconds>(t1_ - t0_).count() << "[µs]"  << std::endl;
	//std::cout << "t2 - t1 = " << std::chrono::duration_cast<std::chrono::microseconds>(t2_ - t1_).count() << "[µs]" << std::endl;
    //std::cout << "t3 - t2 = " << std::chrono::duration_cast<std::chrono::microseconds>(t3_ - t2_).count() << "[µs]" << std::endl;
    //std::cout << "t4 - t3 = " << std::chrono::duration_cast<std::chrono::microseconds>(t4_ - t3_).count() << "[µs]" << std::endl;
    //std::cout << std::endl << std::endl;

	std::cout << "END: Test-2" << std::endl << std::endl;
    return 0;
}


