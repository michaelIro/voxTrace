#include <iostream>
#include <armadillo>

//#include "api/OptimizerAPI.hpp"
#include "api/PlotAPI.hpp"
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

//---------------------------------------------------------------------------------------------

	Shadow3API shadow_((char*) "../test-data/shadow3");
	PolyCapAPI pc1_((char*) "../test-data/polycap/pc-246-descr.txt");	

	PrimaryBeam primary_(&shadow_, &pc1_);
	//PolyCapAPI pc2_((char*) "../test-data/polycap/pc-236-descr.txt");	

	arma::Mat<double> temp_;
	temp_.load(arma::hdf5_name("/media/miro/Data/Shadow-Beam/PrimaryBeam.h5","my_data"));

	XRBeam prim_(temp_);
	//prim_.getRays()[1].print(1);
	prim_.primaryTransform(70.0, 70.0,0.0, 0.51, 45.0);
	//prim_.getRays()[1].print(1);
//---------------------------------------------------------------------------------------------

	vector<vector<vector<Material>>> myMat;
	ChemElement cu(29);
	ChemElement sn(50);
	ChemElement pb(82);
	map<ChemElement* const,double> bronzeMap{{&cu,0.7},{&sn,0.2},{&pb,0.1}};
		
	for(int i = 0; i < 11; i++){
		vector<vector<Material>> myMat1;
		for(int j = 0; j < 11; j++){
			vector<Material> myMat2;
			for(int k = 0; k < 11; k++){
				myMat2.push_back(Material(bronzeMap,8.96));
			}
			myMat1.push_back(myMat2);
		}
		myMat.push_back(myMat1);
	} 

//---------------------------------------------------------------------------------------------

	Sample sample_ (0.,0.,0.,150.,150.,5.,15.,15.,0.5,myMat);
	//sample_.print();

//---------------------------------------------------------------------------------------------

	Tracer tracer_(prim_, sample_);
	tracer_.start();

//---------------------------------------------------------------------------------------------

	XRBeam fluorescence_(tracer_.getBeam());
	fluorescence_.secondaryTransform(70.0, 70.0,0.0, 0.49, 45.0);
	fluorescence_.getMatrix().save(arma::hdf5_name("/media/miro/Data/Shadow-Beam/SecondaryBeam.h5","my_data"));

//---------------------------------------------------------------------------------------------

	//PolyCapAPI mySecondaryPolycap((char*) "../test-data/polycap/pc-236-descr.txt");	
	//XRBeam myDetectorBeam(mySecondaryPolycap.trace(fluorescence_.getMatrix(),4,(char*) "../test-data/polycap/pc-236.hdf5"));

//---------------------------------------------------------------------------------------------
	std::cout << "END: Test-1" << std::endl;
    return 0;

}