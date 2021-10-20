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



int main() {

	//arma::Mat<double> myBeam = Shadow3API::getBeamFromSource(7,(char*) "../test-data/shadow3/start.00");

	PolyCapAPI myPolycap;
	myPolycap.defineSource();
	myPolycap.traceSource();
	//myPolycap.traceSinglePhoton(myBeam);

	//arma::Mat<double> A;
	//A.load("../test-data/polycap/Primary.txt", arma::auto_detect);
	//A.print();

	int a = XRayLibAPI::A(22);

	vector<vector<vector<Material>>> myMat;

	map<int,double> cuMatMap{{29,1.0}};

	//arma::field<Material> myMaterials(11,11,11); TODO: change vec<vec<vec>> to field
		
	for(int i = 0; i < 11; i++){
		vector<vector<Material>> myMat1;
		for(int j = 0; j < 11; j++){
			vector<Material> myMat2;
			for(int k = 0; k < 11; k++){
				myMat2.push_back(Material(cuMatMap,8.96));
				//myMaterials(i,j,k) = Material(cuMatMap,8.96);
			}
			myMat1.push_back(myMat2);
		}
		myMat.push_back(myMat1);
	}

	vector<ChemElement> myElements;
	myElements.push_back(ChemElement(29));

	Sample sample_ (0.,0.,0.,150.,150.,150.,15.,15.,15.,myMat,myElements);

	//Tracer tracer_(source_, sample_);
	//tracer_.start();

    return 0;
}