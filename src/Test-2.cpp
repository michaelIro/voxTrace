#include <iostream>

#include "api/PolyCapAPI.hpp"
#include "api/Shadow3API.hpp"

#include "base/XRBeam.hpp"

#include "tracer/PrimaryBeam.hpp"

int main() {
	std::cout << "START: Test-2" << std::endl;
    
    Shadow3API shadow_((char*) "../test-data/in/shadow3");
    PolyCapAPI pc1_((char*) "../test-data/in/polycap/pc-246-descr.txt");	

    shadow_.trace(8000000,rand());

    XRBeam beam_(
			pc1_.trace(shadow_.getBeamMatrix(),100000,(char *)"../test-data/out/beam/beam.hdf5",true)
	);

	beam_.getMatrix().save(arma::hdf5_name("../test-data/out/beam/beam.h5", "my_data"));
    
	std::cout << "END: Test-2" << std::endl;
    return 0;
}


