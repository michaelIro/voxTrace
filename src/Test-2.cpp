#include <iostream>

//namespace Shadow3{
// #include <shadow_bind_cpp.hpp>
//}
/**
//#include "api/OptimizerAPI.hpp"
#include "api/PlotAPI.hpp"
#include "api/PolyCapAPI.hpp"
#include "api/Shadow3API.hpp"
#include "api/XRayLibAPI.hpp"

#include "base/ChemElement.hpp"
#include "base/Material.hpp"
#include "base/XRBeam.hpp"

#include "tracer/PrimaryBeam.hpp"**/

int main() {
	std::cout << "START: Test-2" << std::endl;


    
    // Shadow3::Source mysrc;
    /**
    Shadow3API shadow_((char*) "../test-data/shadow3");
    PolyCapAPI pc1_((char*) "../test-data/polycap/pc-246-descr.txt");	

    shadow_.trace(8000000,rand());

    XRBeam beam_(
			pc1_.trace(shadow_.getBeamMatrix(),100000,(char *)"../test-data/beam/beam.hdf5")
	);

	beam_.getMatrix().save(arma::hdf5_name("../test-data/beam/beam.h5", "my_data"));
    **/
	std::cout << "END: Test-2" << std::endl;
    return 0;
}


