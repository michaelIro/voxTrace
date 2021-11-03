/*!
API to PolyCap, a C library to calculate X-ray transmission through polycapillaries, published by Pieter Tack, Tom Schoonjans and Laszlo Vincze.
  
  Documentation is available @  https://pietertack.github.io/polycap
  
  Repositories can be found @ https://github.com/PieterTack/polycap

  For more general usage this Interface also implements an Interface to Rays generated/traced with the Shadow3 code.
*/
/*  Add this to polycap-photon.h the reinstall polycap and copy polycap-private.h and config.h to install dir  
  
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

//#include <polycap.h>
#include <polycap-private.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <omp.h>

#include <chrono>
#include <filesystem>

#include "./Shadow3API.hpp"

class PolyCapAPI{
  private: 
    polycap_error *error;
    polycap_description *description;
    polycap_source *source;
	  polycap_transmission_efficiencies *efficiencies;

    void defineSource();

  public:
    PolyCapAPI();
    
    void traceSource();

    void traceSinglePhoton(arma::Mat<double> shadowBeam);

    polycap_transmission_efficiencies* polycap_shadow_source_get_transmission_efficiencies(polycap_source *source, int max_threads, int n_photons, bool leak_calc, polycap_progress_monitor *progress_monitor, polycap_error **error);
};

#endif