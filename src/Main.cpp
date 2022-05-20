//!	Main function
#include <iostream>
#include <string>
#include <filesystem>

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
	//std::cout << "JKLLKJÖL" << std::endl;
    std::string path = "/media/miro/Data/Documents/TU Wien/VSC-BEAM/";

	std::vector<XRBeam> beams_;
    for (const auto & file : std::filesystem::directory_iterator(path)){
        //cout << file.path() << endl;
	//for(int i= 1; i < 3; i++){
	std::string pathname = file.path();

	arma::Mat<double> myPrimaryCapBeam;
	//myPrimaryCapBeam.load(arma::hdf5_name("/media/miro/Data/Documents/TU Wien/Shadow-Beam/Fast/PC-246/PrimaryBeam-"+std::to_string(i)+".h5","my_data"));
	//myPrimaryCapBeam.load(arma::hdf5_name("/media/miro/Data/Documents/TU Wien/VSC-BEAM/PrimaryBeam-"+std::to_string(i)+"-0.h5","my_data"));
	myPrimaryCapBeam.load(arma::hdf5_name(file.path(),"my_data"));
	XRBeam myPrimaryBeam(myPrimaryCapBeam);
	//myPrimaryBeam.print();

	//std::vector<XRBeam> beamVec = {myPrimaryBeam,myPrimaryBeam1};
	//myPrimaryBeam = XRBeam::merge(beamVec);
	//myPrimaryBeam.getMatrix().save("../test-data/out/beam/primaryBeam.csv", arma::csv_ascii);
	
	myPrimaryBeam.primaryTransform(70.0, 70.0,0.0, 0.51, 45.0);
	std::cout << "Primary size:" << myPrimaryBeam.getRays().size() << std::endl;

//---------------------------------------------------------------------------------------------

	Tracer tracer_(myPrimaryBeam, sample_);
	tracer_.start();

//---------------------------------------------------------------------------------------------

	XRBeam fluorescence_= tracer_.getSecondaryBeam();
	fluorescence_.secondaryTransform(70.0, 70.0,0.0, 0.49, 45.0);
	std::cout << "Size at 2nd-Polycap-Entry:" << fluorescence_.getRays().size() << std::endl;
	beams_.push_back(fluorescence_);
}
	XRBeam fluorescence_ = XRBeam::merge(beams_);
	std::cout << "Secondary size:" << fluorescence_.getRays().size() << std::endl;
	//fluorescence_.getMatrix().save("../test-data/out/beam/fluorescenceBeam.csv", arma::csv_ascii);
	fluorescence_.getMatrix().save(arma::hdf5_name("../test-data/out/beam/fluorescenceBeam.h5","my_data"));

//---------------------------------------------------------------------------------------------

	PolyCapAPI mySecondaryPolycap((char*) "../test-data/in/polycap/pc-236-descr.txt");	
	//XRBeam myDetectorBeam(mySecondaryPolycap.trace(fluorescence_.getMatrix(),2,(char*) "../test-data/out/beam/detectorBeam.hdf5",false));
	XRBeam myDetectorBeam(mySecondaryPolycap.traceFast(fluorescence_.getMatrix()));
	std::cout << "Detector size:" << myDetectorBeam.getRays().size() << std::endl;
	

/***********************************************************************************/
    return 0;
}