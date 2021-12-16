#include <iostream>
#include <armadillo>

#include "api/PolyCapAPI.hpp"
#include "api/Shadow3API.hpp"
#include "api/XRayLibAPI.hpp"

#include "base/ChemElement.hpp"
#include "base/Material.hpp"
#include "base/XRBeam.hpp"
#include "base/Sample.hpp"

#include "tracer/Tracer.hpp"
#include "tracer/PrimaryBeam.hpp"

int main() {
	
	std::cout << "START: Test-1" << std::endl;

//---------------------------------------------------------------------------------------------

	Shadow3API shadow_((char*) "../test-data/shadow3");
	PolyCapAPI pc1_((char*) "../test-data/polycap/pc-246-descr.txt");	

	PrimaryBeam primary_(shadow_, pc1_);

	//std::vector<XRBeam> beams_;
	//arma::Mat<double> temp_;
	//for(int i=0; i<50; i++){
	//	temp_.load(arma::hdf5_name("/media/miro/Data/Shadow-Beam/PrimaryBeam-"+std::to_string(i)+".h5","my_data"));
	//	beams_.push_back(XRBeam(temp_));
	//}
		
	//XRBeam temp1_ = XRBeam::merge(beams_);
	//temp1_.getMatrix().save(arma::hdf5_name("/media/miro/Data/Shadow-Beam/PrimaryBeam.h5","my_data"));
	//temp_.load(arma::hdf5_name("/media/miro/Data/Shadow-Beam/PrimaryBeam.h5","my_data"));

	//XRBeam prim_(temp_);
	//prim_.getRays()[1].print(1);
	//prim_.primaryTransform(70.0, 70.0,0.0, 0.51, 45.0);
	//prim_.getRays()[1].print(1);

	std::cout << "END: Test-1" << std::endl;
    return 0;

}