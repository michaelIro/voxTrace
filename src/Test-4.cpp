#include <iostream>
#include <boost/random/mersenne_twister.hpp>

#include "api/PolyCapAPI.hpp"
#include "api/Shadow3API.hpp"

#include "base/XRBeam.hpp"

#include "tracer/PrimaryBeam.hpp"


int main() {
	std::cout << "START: Test-4" << std::endl;
    
    Shadow3API shadow_((char*) "../test-data/in/shadow3");
    PolyCapAPI pc1_((char*) "../test-data/in/polycap/pc-246-descr.txt");

    unsigned int seed_ = chrono::steady_clock::now().time_since_epoch().count();
    boost::mt19937 rand_gen_(seed_);

	std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();

    shadow_.trace(800000,rand_gen_());
    arma::Mat<double> shadowBeam_ = shadow_.getBeamMatrix();
    
    for(int i = 0; i < 20; i++){
        arma::Col<double> eneCol(shadowBeam_.n_rows, arma::fill::value(i*50677300.0));
        shadowBeam_.col(10) = eneCol; 

        XRBeam beam1_( pc1_.traceFast(shadowBeam_) );
        beam1_.getMatrix().save(arma::hdf5_name("../test-data/out/beam/beam-"+std::to_string(i)+"keV.h5", "my_data"));
    }


	//std::chrono::steady_clock::time_point t1_ = std::chrono::steady_clock::now();
    //std::chrono::steady_clock::time_point t2_ = std::chrono::steady_clock::now();
    //std::chrono::steady_clock::time_point t3_ = std::chrono::steady_clock::now();
    //std::chrono::steady_clock::time_point t4_ = std::chrono::steady_clock::now();

    //std::cout << std::endl << std::endl; //
    //std::cout << "Successfully traced Rays: " <<  beam1_.getRays().size() << std::endl << std::endl;
	//std::cout << "t1 - t0 = " << std::chrono::duration_cast<std::chrono::microseconds>(t1_ - t0_).count() << "[µs]"  << std::endl;
	//std::cout << "t2 - t1 = " << std::chrono::duration_cast<std::chrono::microseconds>(t2_ - t1_).count() << "[µs]" << std::endl;
    //std::cout << "t3 - t2 = " << std::chrono::duration_cast<std::chrono::microseconds>(t3_ - t2_).count() << "[µs]" << std::endl;
    //std::cout << "t4 - t3 = " << std::chrono::duration_cast<std::chrono::microseconds>(t4_ - t3_).count() << "[µs]" << std::endl;
    //std::cout << std::endl << std::endl;

	std::cout << "END: Test-4" << std::endl << std::endl;
    return 0;
}


