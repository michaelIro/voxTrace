#include <iostream>
#include <boost/random/mersenne_twister.hpp>

#include "api/PolyCapAPI.hpp"
#include "api/Shadow3API.hpp"

#include "base/XRBeam.hpp"

#include "tracer/PrimaryBeam.hpp"


int main() {
	std::cout << "START: Test-4" << std::endl;
    
    Shadow3API shadow_((char*) "../test-data/in/shadow3");
    PolyCapAPI pc1_((char*) "../test-data/in/polycap/pc-236-descr-turned.txt");
    //PolyCapAPI pc1_((char*) "../test-data/in/polycap/pc-246-descr.txt");

    unsigned int seed_ = chrono::steady_clock::now().time_since_epoch().count();
    boost::mt19937 rand_gen_(seed_);

    shadow_.trace(500000,rand_gen_());
    arma::Mat<double> shadowBeam_ = shadow_.getBeamMatrix();

    
   pc1_.trace(shadowBeam_, 10000,"/media/miro/Data/Shadow-Beam/Transmission/PC-236/no-leak.h5", true);
/*
    for(int i = 6; i < 101; i++){
        //double i = 6.5;
        arma::Col<double> eneCol(shadowBeam_.n_rows, arma::fill::value(i*0.2*50677300.0));
        shadowBeam_.col(10) = eneCol; 

        XRBeam beam_( pc1_.traceFast(shadowBeam_) );

        std::string myID;
        if(i*0.2<10)
            myID = "0"+std::to_string(i*0.2);
        else
            myID = std::to_string(i*0.2);

        beam_.getMatrix().save(arma::hdf5_name("/media/miro/Data/Shadow-Beam/Transmission/PC-246/beam-246-"+myID+"keV.h5", "my_data"));
    }*/

	std::cout << "END: Test-4" << std::endl << std::endl;
    return 0;
}


