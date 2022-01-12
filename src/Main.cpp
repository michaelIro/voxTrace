//!	Main function
#include <iostream>

#include "api/OptimizerAPI.hpp"
#include "api/PlotAPI.hpp"
#include "api/PolyCapAPI.hpp"
#include "api/Shadow3API.hpp"
#include "api/XRayLibAPI.hpp"

#include "base/ChemElement.hpp"
#include "base/Material.hpp"
#include "base/Sample.hpp"
#include "base/XRBeam.hpp"
#include "tracer/Tracer.hpp"

int main() {
//---------------------------------------------------------------------------------------------

	vector<vector<vector<Material>>> myMat;
    ChemElement cu(29);
    ChemElement sn(50);
	ChemElement pb(82);
	map<ChemElement* const,double> bronzeMap{{&cu,0.7},{&sn,0.2},{&pb,0.1}};

	//arma::field<Material> myMaterials(11,11,11); TODO: change vec<vec<vec>> to field
		
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
	myPrimaryCapBeam.load(arma::hdf5_name("/media/miro/Data/Shadow-Beam/Fast/PC-246/PrimaryBeam-1.h5","my_data"));

	XRBeam myPrimaryBeam(myPrimaryCapBeam);

	//std::vector<XRBeam> beamVec = {myPrimaryBeam,myPrimaryBeam1};
	//myPrimaryBeam = XRBeam::merge(beamVec);
	//myPrimaryBeam.getMatrix().save("../test-data/out/beam/primaryBeam.csv", arma::csv_ascii);
	
	myPrimaryBeam.primaryTransform(70.0, 70.0,0.0, 0.51, 45.0);

//---------------------------------------------------------------------------------------------


	Tracer tracer_(myPrimaryBeam, sample_);
	//vector<XRBeam> tracedBeams;
	//for(int i= 0; i<1; i++){
	tracer_.start();
		//tracedBeams.push_back(tracer_.getBeam());
		//std::cout<< i<< std::endl;
	//}

	//vector<Ray> superBeam;
	//for(XRBeam xb: tracedBeams)
	//	for(Ray ray: xb.getRays())
	//		superBeam.push_back(ray);


//---------------------------------------------------------------------------------------------

	XRBeam fluorescence_= tracer_.getSecondaryBeam();
	fluorescence_.secondaryTransform(70.0, 70.0,0.0, 0.49, 45.0);

	//if(outermosti==0)
		//oneBeamToRuleThemAll = fluorescence_.getMatrix();
	//else	
	//	oneBeamToRuleThemAll = arma::join_cols(oneBeamToRuleThemAll,fluorescence_.getMatrix());
	
	//std::cout << std::endl << std::endl << "Iteration #-" << outermosti << std::endl << std::endl;
//}


	//fluorescence_.getMatrix().save("../test-data/out/beam/fluorescenceBeam.csv", arma::csv_ascii);
	//fluorescence_.print();
	//arma::Mat<double> temp_;
	//temp_.load("../test-data/out/beam/fluorescenceBeam.csv", arma::csv_ascii);
	//XRBeam fluorescence_(temp_);

//---------------------------------------------------------------------------------------------
	//oneBeamToRuleThemAll.load("../test-data/out/beam/fluorescenceBeam.csv", arma::csv_ascii);
	PolyCapAPI mySecondaryPolycap((char*) "../test-data/in/polycap/pc-236-descr.txt");	
	//XRBeam myDetectorBeam1(mySecondaryPolycap.trace(fluorescence_.getMatrix(),2,(char*) "../test-data/in/polycap/pc-236.hdf5",true));
	XRBeam myDetectorBeam(mySecondaryPolycap.traceFast(fluorescence_.getMatrix()));
	int finalBeamsN = myDetectorBeam.getRays().size();
	//int finalBeamsN1 = myDetectorBeam1.getRays().size();
/***********************************************************************************/
    return 0;
}