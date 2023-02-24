#include <iostream>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include "api/XRayLibAPI.hpp"
#include "api/Shadow3API.hpp"

int main() {
	std::cout << "START: Test-4" << std::endl;
	std::cout << XRayLibAPI::ZToSym(29) << std::endl;
	Shadow3API shadow_((char*) "/home/miro/Software/1st-party/voxTrace/test-data/in/shadow3");
	shadow_.trace(5);
	shadow_.getBeamMatrix().print();
	std::cout << "END: Test-4" << std::endl;
    return 0;
}


