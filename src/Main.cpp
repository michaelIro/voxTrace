//!	Main function
#include <iostream>

#include "api/OptimizerAPI.hpp"
#include "api/PlotAPI.hpp"
#include "api/PolyCapAPI.hpp"
#include "api/Shadow3API.hpp"
#include "api/XRayLibAPI.hpp"

int main() {

	arma::Mat<double> myBeam = Shadow3API::getBeam(7);
	myBeam.print();

	PolyCapAPI myPolycap;
	myPolycap.defineSource();
	//myPolycap.traceSource();
	myPolycap.traceSinglePhoton();

	arma::Mat<double> A;
	A.load("../test-data/polycap/Primary.txt", arma::auto_detect);
	A.print();

	XRayLibAPI::A(22);

    return 0;
}