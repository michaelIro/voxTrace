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
	myShadowSource.trace(1000000);

	arma::Mat<double> myShadowBeam = myShadowSource.getBeam();

	//myShadowBeam.save("../test-data/beam/shadowBeam.csv", arma::csv_ascii);
	//std::cout << "Shadow-Beam: " << std::endl;
	//myShadowBeam.print();

/***********************************************************************************/

	PolyCapAPI myPrimaryPolycap((char*) "../test-data/polycap/pc-246-descr.txt");
	vector<Ray> myPrimaryCapBeam = myPrimaryPolycap.traceSource(myShadowBeam,10000);

	XRBeam myPrimaryBeam(myPrimaryCapBeam);
	//myPrimaryBeam.getMatrix().save("../test-data/beam/primaryBeam.csv", arma::csv_ascii);
	myPrimaryBeam.primaryTransform(70.0, 70.0,0.0, 0.51, 45.0);

	//myPrimaryBeam.print();

/***********************************************************************************/

	//some test comment
	//OptimizerAPI myOptimizer;

	//arma::Mat<double> A;
	//A.load("../test-data/polycap/Primary.txt", arma::auto_detect);
	//A.print();

	//int a = XRayLibAPI::A(22);

/***********************************************************************************/

	vector<vector<vector<Material>>> myMat;
    ChemElement cu(29);
    ChemElement sn(50);
	ChemElement pb(82);
	map<ChemElement* const,double> bronzeMap{{&cu,0.7},{&sn,0.2},{&pb,0.1}};

	//map<int,double> bronze{{29,0.7},{50,0.2},{82,0.1}};

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

/***********************************************************************************/

	Sample sample_ (0.,0.,0.,150.,150.,5.,15.,15.,0.5,myMat);
	//sample_.print();

/***********************************************************************************/

	Tracer tracer_(myPrimaryBeam, sample_);
	vector<XRBeam> tracedBeams;
	for(int i= 0; i<200; i++){
		tracer_.start();
		tracedBeams.push_back(tracer_.getBeam());
		std::cout<< i<< std::endl;
	}

	vector<Ray> superBeam;
	for(XRBeam xb: tracedBeams)
		for(Ray ray: xb.getRays())
			superBeam.push_back(ray);


/***********************************************************************************/

	XRBeam fluorescence_(superBeam);
	fluorescence_.secondaryTransform(70.0, 70.0,0.0, 0.49, 45.0);
	fluorescence_.getMatrix().save("../test-data/beam/fluorescenceBeam.csv", arma::csv_ascii);
	//fluorescence_.print();

/***********************************************************************************/

	PolyCapAPI mySecondaryPolycap((char*) "../test-data/polycap/pc-236-descr.txt");	
	XRBeam myDetectorBeam(mySecondaryPolycap.traceSource(fluorescence_.getMatrix(),1));

/***********************************************************************************/
    return 0;
}