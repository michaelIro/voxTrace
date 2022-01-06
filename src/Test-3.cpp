#include <iostream>
#include "cuda/MyDummy.cuh"
//#include "api/PlotAPI.hpp"
#include "base/XRBeam.hpp"

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

    vector<XRBeam> beams_;
    for(int i = 1; i < 15; i++){
        arma::Mat<double> beam_;
        beam_.load(arma::hdf5_name("/media/miro/Data/Shadow-Beam/Fast/PC-236/PrimaryBeam-"+std::to_string(i)+".h5","my_data")); 
        XRBeam temp_(beam_);
        beams_.push_back(temp_);
    }
    XRBeam total_ = XRBeam::merge(beams_);
    total_.getMatrix().save(arma::hdf5_name("/media/miro/Data/Shadow-Beam/Fast/PC-236/PrimaryBeam-Total-15.h5","my_data")); 

    std::cout << "END: Test-3" << std::endl;
    return 0;
}


