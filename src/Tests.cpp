// -*- lsst-c++ -*-
//!  A test class. 
/*!
  A more elaborate class description.
*/
#include <iostream>
#include <armadillo>

#include <ensmallen.hpp>
#include <gsl/gsl_sf_bessel.h>

#include <sciplot/sciplot.hpp>
#include <polycap.h>
#include "xraylib.h"

#include <shadow_bind_cpp.hpp>

#define OPTIM_ENABLE_ARMA_WRAPPERS
#include "optim.hpp"

//using namespace std;
//using namespace sciplot;

// Ackley function
double ackley_fn(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    const double x = vals_inp(0);
    const double y = vals_inp(1);
    const double pi = arma::datum::pi;

    double obj_val = -20*std::exp( -0.2*std::sqrt(0.5*(x*x + y*y)) ) - std::exp( 0.5*(std::cos(2*pi*x) + std::cos(2*pi*y)) ) + 22.718282L;

    return obj_val;
}

int main() {
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    std::cout << "Hello World!" << std::endl;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
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
    A.save("../test-data/Armadillo-Matrix-A.txt", arma::arma_ascii);
    B.save("../test-data/Armadillo-Matrix-B.txt", arma::arma_ascii);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    double x0 = 5.0;
    double y0 = gsl_sf_bessel_J0 (x0);
    printf ("J0(%g) = %.18e\n", x0, y0);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create a vector with values from 0 to pi divived into 200 uniform intervals for the x-axis
    sciplot::Vec x = sciplot::linspace(0.0, PI, 200);

    // Create a Plot object
    sciplot::Plot plot;

    // Set the x and y labels
    plot.xlabel("x");
    plot.ylabel("y");

    // Set the x and y ranges
    plot.xrange(0.0, PI);
    plot.yrange(0.0, 1.0);

    // Set the legend to be on the bottom along the horizontal
    plot.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);

    // Plot sin(i*x) from i = 1 to i = 6
    plot.drawCurve(x, std::sin(1.0 * x)).label("sin(x)");
    plot.drawCurve(x, std::sin(2.0 * x)).label("sin(2x)");
    plot.drawCurve(x, std::sin(3.0 * x)).label("sin(3x)");
    plot.drawCurve(x, std::sin(4.0 * x)).label("sin(4x)");
    plot.drawCurve(x, std::sin(5.0 * x)).label("sin(5x)");
    plot.drawCurve(x, std::sin(6.0 * x)).label("sin(6x)");

    // Show the plot in a pop-up window
    //plot.show();

    // Save the plot to a PDF file
    plot.save("../test-data/plots/example-sine-functions.pdf");
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    polycap_error *error = NULL;
	polycap_profile *profile;
	polycap_description *description;
	polycap_source *source;
	polycap_transmission_efficiencies *efficiencies;
	int i;

	// Optic parameters
	double optic_length = 3.94;					//optic length in cm
	double rad_ext_upstream = 0.74;				//external radius upstream, at entrance window, in cm
	double rad_ext_downstream = 0.215;			//external radius downstream, at exit window, in cm
	double rad_int_upstream = 0.000805; 		//single capillary radius, at optic entrance, in cm
	double rad_int_downstream = 0.000234; 		//single capillary radius, at optic exit, in cm
	double focal_dist_upstream = 10000000000000.0; 	//focal distance on entrance window side, in cm
	double focal_dist_downstream = 0.51; 		//focal distance on exit window side, in cm
	int n_elem = 2;								//amount of elements in optic material
	int iz[2]={8,14};							//polycapillary optic material composition: atomic numbers and corresponding weight percentages
	double wi[2]={53.0,47.0};					//SiO2
	double density = 2.23;						//optic material density, in g/cm^3 
	double surface_rough = 5.;					//surface roughness in Angstrom
	double n_capillaries = 200000.;				//number of capillaries in the optic

	// Photon source parameters
	double source_dist = 6.0;					//distance between optic entrance and source along z-axis
	double source_rad_x = 1.0;					//source radius in x, in cm
	double source_rad_y = 1.0;					//source radius in y, in cm
	double source_div_x = 0.000471239;			//source divergence in x, in rad
	double source_div_y = 0.01;					//source divergence in y, in rad
	double source_shift_x = 0.;					//source shift in x compared to optic central axis, in cm
	double source_shift_y = 0.;					//source shift in y compared to optic central axis, in cm
	double source_polar = 0.5;					//source polarisation factor
	int n_energies = 7;							//number of discrete photon energies
	double energies[7]={1,5,10,15,20,25,30};	//energies for which transmission efficiency should be calculated, in keV

	// Simulation parameters
	int n_threads = -1;			//amount of threads to use; -1 means use all available
	int n_photons = 1000;		//simulate 30000 succesfully transmitted photons (excluding leak events)
	bool leak_calc = false;		//choose to perform leak photon calculations or not. Leak calculations take significantly more time


	//define optic profile shape
	profile = polycap_profile_new(POLYCAP_PROFILE_ELLIPSOIDAL, optic_length, rad_ext_upstream, rad_ext_downstream, rad_int_upstream, rad_int_downstream, focal_dist_upstream, focal_dist_downstream, &error);

	//define optic description
	description = polycap_description_new(profile, surface_rough, n_capillaries, n_elem, iz, wi, density, &error);
	polycap_profile_free(profile); //We can free the profile structure, as it is now contained in description

	//define photon source, including optic description
	source = polycap_source_new(description, source_dist, source_rad_x, source_rad_y, source_div_x, source_div_y, source_shift_x, source_shift_y, source_polar, n_energies, energies, &error);
	polycap_description_free(description); //We can free the description structure, as now it is contained in source

	//calculate transmission efficiency curve
	efficiencies = polycap_source_get_transmission_efficiencies(source, n_threads, n_photons, leak_calc, NULL, &error);

	polycap_transmission_efficiencies_write_hdf5(efficiencies,"../test-data/pc-246.hdf5",NULL);

	double *efficiencies_arr = NULL;
	polycap_transmission_efficiencies_get_data(efficiencies, NULL, NULL, &efficiencies_arr, NULL);

	//print out efficiencies:
	for(i = 0 ; i < n_energies ; i++){
		printf("Energy: %lf keV, Transmission Efficiency: %lf percent.\n", energies[i], efficiencies_arr[i]*100.);
	}

	polycap_source_free(source);
	polycap_transmission_efficiencies_free(efficiencies);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // initial values:
    
    arma::vec xn = arma::ones(2,1) + 1.0;

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

    bool success = optim::de(xn,ackley_fn,nullptr);

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    if (success) {
        std::cout << "de: Ackley test completed successfully.\n"
                  << "elapsed time: " << elapsed_seconds.count() << "s\n";
    } else {
        std::cout << "de: Ackley test completed unsuccessfully." << std::endl;
    }

    arma::cout << "\nde: solution to Ackley test:\n" << xn << arma::endl;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    Source src;
    OE     oe1;
    Beam   ray;
    
    // load variables from start.00
    src.load( (char*) "../test-data/start.00");

    std::cout << " Number of rays: " << src.NPOINT << std::endl;
    src.NPOINT=100000;
    std::cout << " Number of rays (modified): " << src.NPOINT << std::endl;

    // calculate source
    ray.genSource(&src);
    ray.write( (char*) "../test-data/begin.dat");
    
    // load start.01 into oe1
    //oe1.load( (char*) "start.01");
    
    // traces OE1
    //ray.traceOE(&oe1,1);

    // write file star.01
    //ray.write( (char*) "star.01");
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    cout	<<	"Cu - sigmai-K "	<<	CS_Photo_Partial(29,K_SHELL,17.4,NULL)	<<	endl;
	cout	<<	"Zn - sigmai-K "	<<	CS_Photo_Partial(30,K_SHELL,17.4,NULL)	<<	endl;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
 
    return 0;
}