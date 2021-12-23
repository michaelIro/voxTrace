/*!
API to PolyCap, a C library to calculate X-ray transmission through polycapillaries, published by Pieter Tack, Tom Schoonjans and Laszlo Vincze.
  
  Documentation is available @  https://pietertack.github.io/polycap
  
  Repositories can be found @ https://github.com/PieterTack/polycap

  For more general usage this Interface also implements an Interface to Rays generated/traced with the Shadow3 code.
*/

/*  
Add this to polycap-photon.h then reinstall polycap and copy polycap-private.h and config.h to install dir  
  
POLYCAP_EXTERN
double polycap_scalar(polycap_vector3 vect1, polycap_vector3 vect2);

POLYCAP_EXTERN
void polycap_norm(polycap_vector3 *vect);

POLYCAP_EXTERN
polycap_leak* polycap_leak_new(polycap_vector3 leak_coords, polycap_vector3 leak_dir, polycap_vector3 leak_elecv, int64_t n_refl, size_t n_energies, double *weights, polycap_error **error);

POLYCAP_EXTERN
int polycap_photon_within_pc_boundary(double polycap_radius, polycap_vector3 photon_coord, polycap_error **error);
*/

#ifndef PolyCapAPI_H
#define PolyCapAPI_H

#include <armadillo>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

#include <polycap-private.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <omp.h>

#include "../base/Ray.hpp"

class PolyCapAPI{
  private: 
  	polycap_profile *profile;
    polycap_error *error;
    polycap_description *description;
    polycap_source *source;
	polycap_transmission_efficiencies *efficiencies;

	  // Polycapillary parameters
	  double optic_length;					// optic length in cm
	  double rad_ext_upstream;			// external radius upstream, at entrance window, in cm
	  double rad_ext_downstream;		// external radius downstream, at exit window, in cm
	  double rad_int_upstream; 		  // single capillary radius, at optic entrance, in cm
	  double rad_int_downstream; 		// single capillary radius, at optic exit, in cm
	  double focal_dist_upstream; 	// focal distance on entrance window side, in cm
	  double focal_dist_downstream; // focal distance on exit window side, in cm
	  int n_elem;								    // amount of elements in optic material
	  int* iz;							        // polycapillary optic material composition: atomic numbers and corresponding weight percentages
	  double* wi;					          // SiO2
	  double density;						    // optic material density, in g/cm^3 
	  double surface_rough;					// surface roughness in Angstrom
	  double n_capillaries;				  // number of capillaries in the optic

    // Photon source parameters -> FIXME: Please load even if you use Shadow-Source
	  double source_dist;				    // distance between optic entrance and source along z-axis
	  double source_rad_x;					// source radius in x, in cm
	  double source_rad_y;					// source radius in y, in cm
	  double source_div_x;					// source divergence in x, in rad
	  double source_div_y;					// source divergence in y, in rad
	  double source_shift_x;				// source shift in x compared to optic central axis, in cm
	  double source_shift_y;				// source shift in y compared to optic central axis, in cm
	  double source_polar;					// source polarisation factor
	  int n_energies;								// number of discrete photon energies
	  double* energies;					    // energies for which transmission efficiency should be calculated, in keV

    void load_source_param(char* path);
    void load_cap_param(char* path);
    void compareBeams(arma::Mat<double> shadowBeam);
    void overwritePhoton(arma::rowvec shadowRay, polycap_photon *photon);

  public:
    PolyCapAPI() = delete;
    PolyCapAPI(char* path);
    //PolyCapAPI(const PolyCapAPI& polyCapAPI);
    
	//vector<Ray> traceFast(arma::Mat<double> shadowBeam);
    vector<Ray> trace(arma::Mat<double> shadowBeam, int nPhotons, std::filesystem::path savePath, bool save);
    //vector<Ray> trace(arma::Mat<double> shadowBeam, int nPhotons): PolyCapAPI::trace(shadowBeam, nPhotons, " ", false);

    polycap_transmission_efficiencies* polycap_shadow_source_get_transmission_efficiencies(polycap_source *source, int max_threads, int n_photons, bool leak_calc, polycap_progress_monitor *progress_monitor, polycap_error **error, arma::Mat<double> shadowBeam);

    void print();
};

#endif