#include <iostream>
#include "cuda/MyDummy.cuh"
//#include "api/PlotAPI.hpp"
#include "base/XRBeam.hpp"
#include "api/PolyCapAPI.hpp"

int main() {
	std::cout << "START: Test-3" << std::endl;

    //int N = 1<<20;
    //float *x, *y;

    //MyDummy::callAdd(N, x, y);

    //arma::Mat<double> dummy;
    //PlotAPI::scatter_plot((char*) "../test-data/out/plots/example-sine-functions.pdf",true,true, dummy);
    //---------------------------------------------------------------------------------------------

	//some test comment
	//OptimizerAPI myOptimizer;

//---------------------------------------------------------------------------------------------


    arma::Mat<double> temp_;
    temp_.load(arma::hdf5_name("../test-data/out/beam/fluorescenceBeam.h5","my_data"));
	XRBeam fluorescence_(temp_);
    std::cout << "Detector size:" << fluorescence_.getRays().size() << std::endl;

//---------------------------------------------------------------------------------------------

	PolyCapAPI mySecondaryPolycap((char*) "../test-data/in/polycap/pc-236-descr.txt");	
	//XRBeam myDetectorBeam(mySecondaryPolycap.trace(fluorescence_.getMatrix(),2,(char*) "../test-data/out/beam/detectorBeam.hdf5",false));
	XRBeam myDetectorBeam(mySecondaryPolycap.traceFast(fluorescence_.getMatrix()));
	std::cout << "Detector size:" << myDetectorBeam.getRays().size() << std::endl;

    std::cout << "END: Test-3" << std::endl;
    return 0;
}


