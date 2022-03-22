#include <iostream>
#include "cuda/MyDummy.cuh"
//#include "api/PlotAPI.hpp"
#include "base/XRBeam.hpp"
#include "api/PolyCapAPI.hpp"

int main() {
	std::cout << "START: Test-3" << std::endl;

    int N = 1<<20;
    float *x, *y;

    MyDummy::callAdd(N, x, y);

    //arma::Mat<double> dummy;
    //PlotAPI::scatter_plot((char*) "../test-data/out/plots/example-sine-functions.pdf",true,true, dummy);




    std::cout << "END: Test-3" << std::endl;
    return 0;
}


