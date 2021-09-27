//!	Main function
#include <iostream>
#include "api/XRayLibAPI.hpp"
#include "api/Shadow3API.hpp"
#include "api/PolyCapAPI.hpp"

int main() {

	//arma::Mat<double> myBeam = Shadow3API::getBeam(7);
	//myBeam.print();

	//PolyCapAPI myPolycap;
	arma::mat A;
	A.load("../test-data/polycap/Primary.txt", arma::arma_ascii);
	A.print();

    return 0;
}