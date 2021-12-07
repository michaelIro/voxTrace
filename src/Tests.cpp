// -*- lsst-c++ -*-
//!  Perform Tests for all external libraries
/*!
  A more elaborate class description.
*/
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

#include "tracer/PrimaryBeam.hpp"

int main() {
    /*
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    std::cout << "Hello World!" << std::endl;
    /////////////////////////////////////////////////////////////////////////////////////////////////////// Armadillo Tests
    // Initialize the random generator
    arma::arma_rng::set_seed_random();
  
    // Create a 4x4 random matrix and print it on the screen
    arma::Mat<double> A = arma::randu(4,4);
    std::cout << "A:\n" << A << "\n";
 
    // Multiply A with his transpose:
    std::cout << "A * A.t() =\n";
    std::cout << A * A.t() << "\n";
    // Access/Modify rows and columns from the array:
    A.row(0) = A.row(1) + A.row(3);
    A.col(3).zeros();
    std::cout << "add rows 1 and 3, store result in row 0, also fill 4th column with zeros:\n";
    std::cout << "A:\n" << A << "\n";
 
    // Create a new diagonal matrix using the main diagonal of A:
    arma::Mat<double>B = arma::diagmat(A);
    std::cout << "B:\n" << B << "\n";
 
    // Save matrices A and B:
    A.save("../test-data/armadillo/Armadillo-Matrix-A.txt", arma::arma_ascii);
    B.save("../test-data/armadillo/Armadillo-Matrix-B.txt", arma::arma_ascii);
    /////////////////////////////////////////////////////////////////////////////////////////////////////// PlotAPI Tests
    PlotAPI::test();
    /////////////////////////////////////////////////////////////////////////////////////////////////////// PolyCapAPI Tests
    /////////////////////////////////////////////////////////////////////////////////////////////////////// Shadow3API Tests
    /////////////////////////////////////////////////////////////////////////////////////////////////////// XrayLibAPI Tests
    XRayLibAPI::test();
    /////////////////////////////////////////////////////////////////////////////////////////////////////// OptimizerAPI Tests
    ChemElement cu(29);
    ChemElement zn(30);
	  map<ChemElement* const,double> brassMap{{&cu,0.7},{&zn,0.3}};
    Material brass(brassMap,8.96);

    brass.getMuLin(17.4);*/

    PrimaryBeam myPrimaryBeam((char*) "../test-data/shadow3",(char*) "../test-data/polycap/pc-246-descr.txt");
    //myPrimaryBeam.getShadow3API();

	  //Shadow3API myShadowSource((char*) "../test-data/shadow3");
	  //myShadowSource.trace(10000);

	  //arma::Mat<double> myShadowBeam = myShadowSource.getBeam();
    //myShadowBeam.save("../test-data/beam/shadowBeam.csv", arma::csv_ascii);

    //PolyCapAPI mySecondaryPolycap((char*) "../test-data/polycap/pc-236-descr.txt");	
	  //XRBeam myDetectorBeam(mySecondaryPolycap.traceSource(myShadowBeam,100));
    //myDetectorBeam.getMatrix().save("../test-data/beam/detectorBeam.csv", arma::csv_ascii);

    return 0;
}