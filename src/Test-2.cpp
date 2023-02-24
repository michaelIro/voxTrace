#include <iostream>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "api/PolyCapAPI.hpp"
#include "base/XRBeam.hpp"

int main() {

	std::cout << "START: Test-2" << std::endl;
    
	// create string with path to file
	
	for (int i = -3; i < 10; i++){

		//std::string path = "/media/miro/Data-1TB/nist-1107-simulation/nist-1107-pos-(0.000000).h5";
		std::string appendix = std::to_string(i*15.0);
		std::string path_in = "/media/miro/Data-1TB/nist-1107-simulation/nist-1107-pos-(" + appendix + ").h5";
		std::string path_out = path_in.substr(0, path_in.size()-3) + "-detector.h5";

    	arma::Mat<double> sec_beam_mat;
		sec_beam_mat.load(arma::hdf5_name(path_in,"my_data"));
	
		PolyCapAPI mySecondaryPolycap((char*) "/home/miro/Software/1st-party/voxTrace/test-data/in/polycap/pc-236-descr.txt");	
		XRBeam myDetectorBeam(mySecondaryPolycap.traceFast(sec_beam_mat));

		std::cout << "Detector size:" << myDetectorBeam.getRays().size() << std::endl;
    	myDetectorBeam.getMatrix().save(arma::hdf5_name(path_out,"my_data"));
	}


	std::cout << "END: Test-2" << std::endl << std::endl;
    return 0;
}