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

	Shadow3API shadow_((char*) "../test-data/in/shadow3");
	PolyCapAPI pc1_((char*) "../test-data/in/polycap/pc-246-descr.txt");	

	PrimaryBeam primary_(shadow_, pc1_);

	std::cout << "END: Test-1" << std::endl;
    return 0;

}