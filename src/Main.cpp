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
/***********************************************************************************/

	Shadow3API myShadowSource((char*) "../test-data/shadow3");
	arma::Mat<double> myShadowBeam = myShadowSource.getBeam(100000); //15000000	

	myShadowBeam.save("../test-data/shadowBeam.csv", arma::csv_ascii);
	//std::cout << "Shadow-Beam: " << std::endl;
	//myShadowBeam.print();

/***********************************************************************************/

	PolyCapAPI myPolycap;
	//myPolycap.compareBeams(myShadowBeam);
	vector<Ray> myPolyCapBeam = myPolycap.traceSource(myShadowBeam,1000);

	int counter=0;

	//for (auto it: myPolyCapBeam)
    //	it.print(counter++);

/***********************************************************************************/

	//some test comment
	//OptimizerAPI myOptimizer;

	//arma::Mat<double> A;
	//A.load("../test-data/polycap/Primary.txt", arma::auto_detect);
	//A.print();

	//int a = XRayLibAPI::A(22);

/***********************************************************************************/

	vector<vector<vector<Material>>> myMat;

	map<int,double> bronze{{29,0.7},{50,0.2},{82,0.1}};

	//arma::field<Material> myMaterials(11,11,11); TODO: change vec<vec<vec>> to field
		
	for(int i = 0; i < 11; i++){
		vector<vector<Material>> myMat1;
		for(int j = 0; j < 11; j++){
			vector<Material> myMat2;
			for(int k = 0; k < 11; k++){
				myMat2.push_back(Material(bronze,8.96));
				//myMaterials(i,j,k) = Material(cuMatMap,8.96);
			}
			myMat1.push_back(myMat2);
		}
		myMat.push_back(myMat1);
	}

	vector<ChemElement> myElements;
	myElements.push_back(ChemElement(29));
	myElements.push_back(ChemElement(50));
	myElements.push_back(ChemElement(82));
/***********************************************************************************/

	Sample sample_ (0.,0.,0.,150.,150.,150.,15.,15.,15.,myMat,myElements);
	//sample_.print();

	XRBeam source_(myPolyCapBeam, 75.0, 75.0, 0.51, 45.0);
	source_.print();

	Tracer tracer_(source_, sample_);
	tracer_.start();

	XRBeam fluorescence_(tracer_.getBeam(), 75.0, 75.0, 0.51, 45.0);

    return 0;
}