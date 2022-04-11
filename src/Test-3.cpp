#include <iostream>
#include "cuda/GPUTracer.cuh"
//#include "api/PlotAPI.hpp"
#include "base/XRBeam.hpp"
#include "api/PolyCapAPI.hpp"

int main() {
	std::cout << "START: Test-3" << std::endl;
//---------------------------------------------------------------------------------------------
    ChemElement cu(29);
    ChemElement sn(50);
	ChemElement pb(82);
	map<ChemElement* const,double> bronzeMap{{&cu,0.7},{&sn,0.2},{&pb,0.1}};

	//arma::field<Material> myMaterials(11,11,11); TODO: change vec<vec<vec>> to field
	vector<vector<vector<Material>>> myMat;	
	for(int i = 0; i < 11; i++){
		vector<vector<Material>> myMat1;
		for(int j = 0; j < 11; j++){
			vector<Material> myMat2;
			for(int k = 0; k < 11; k++){
				myMat2.push_back(Material(bronzeMap,8.96));
				//myMaterials(i,j,k) = Material(cuMatMap,8.96);
			}
			myMat1.push_back(myMat2);
		}
		myMat.push_back(myMat1);
	} 

//---------------------------------------------------------------------------------------------

	Sample sample_ (0.,0.,0.,150.,150.,5.,15.,15.,0.5,myMat);
	//sample_.print();
//---------------------------------------------------------------------------------------------

	arma::Mat<double> myPrimaryCapBeam;
    myPrimaryCapBeam.load(arma::hdf5_name("/media/miro/Data/Documents/TU Wien/VSC-BEAM/PrimaryBeam-1-0.h5","my_data"));
	XRBeam myPrimaryBeam(myPrimaryCapBeam);
//---------------------------------------------------------------------------------------------
    (myPrimaryBeam.getRays())[0].print(9);

    int N = 1<<20;
    float *x, *y;

    GPUTracer::callAdd(N, x, y,&sample_, &myPrimaryBeam);

    //arma::Mat<double> dummy;
    //PlotAPI::scatter_plot((char*) "../test-data/out/plots/example-sine-functions.pdf",true,true, dummy);




    std::cout << "END: Test-3" << std::endl;
    return 0;
}


