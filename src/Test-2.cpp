#include <iostream>

#include "api/PolyCapAPI.hpp"
#include "base/XRBeam.hpp"

int main() {
	std::cout << "START: Test-2" << std::endl;
    
    arma::Mat<double> sec_beam_mat;
	sec_beam_mat.load(arma::hdf5_name("/media/miro/Data/pos-4.h5","my_data"));

	//sec_beam_mat.col(3) = sec_beam_mat.col(3)*-1.;
	//sec_beam_mat.col(4) = sec_beam_mat.col(4)*-1.;
	//sec_beam_mat.col(5) = sec_beam_mat.col(5)*10000.;
	//std::cout << sec_beam_mat.col(4) << std::endl;
	//XRBeam sec_beam(sec_beam_mat);

	PolyCapAPI mySecondaryPolycap((char*) "../test-data/in/polycap/pc-236-descr.txt");	
	XRBeam myDetectorBeam(mySecondaryPolycap.traceFast(sec_beam_mat));
	std::cout << "Detector size:" << myDetectorBeam.getRays().size() << std::endl;
    myDetectorBeam.getMatrix().save(arma::hdf5_name("/media/miro/Data/pos-4-detector.h5","my_data"));

	std::cout << "END: Test-2" << std::endl << std::endl;
    return 0;
}