
#include "PolyCapAPI.hpp"

/* Empty constructor */
PolyCapAPI::PolyCapAPI(){

	error = NULL;

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

	//define optic profile shape
	polycap_profile *profile;
	profile = polycap_profile_new(POLYCAP_PROFILE_ELLIPSOIDAL, optic_length, rad_ext_upstream, rad_ext_downstream, rad_int_upstream, rad_int_downstream, focal_dist_upstream, focal_dist_downstream, &error);

	//define optic description
	description = polycap_description_new(profile, surface_rough, n_capillaries, n_elem, iz, wi, density, &error);
	polycap_profile_free(profile); //We can free the profile structure, as it is now contained in description
}

/* Blah */
void PolyCapAPI::defineSource(){
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

	//define photon source, including optic description
	source = polycap_source_new(description, source_dist, source_rad_x, source_rad_y, source_div_x, source_div_y, source_shift_x, source_shift_y, source_polar, n_energies, energies, &error);
	//polycap_description_free(description); //We can free the description structure, as now it is contained in source
}

/* Blah */
void PolyCapAPI::traceSource(){
	int i;
    double** weights;


	// Simulation parameters
	int n_threads = -1;			//amount of threads to use; -1 means use all available
	int n_photons = 1000000;		//simulate 30000 succesfully transmitted photons (excluding leak events)
	bool leak_calc = false;		//choose to perform leak photon calculations or not. Leak calculations take significantly more time


	//calculate transmission efficiency curve
	efficiencies = polycap_source_get_transmission_efficiencies(source, n_threads, n_photons, leak_calc, NULL, &error);

	polycap_transmission_efficiencies_write_hdf5(efficiencies,"../test-data/polycap/pc-246.hdf5",NULL);

	double *efficiencies_arr = NULL;
	polycap_transmission_efficiencies_get_data(efficiencies, NULL, NULL, &efficiencies_arr, NULL);

	polycap_source_free(source);
	polycap_transmission_efficiencies_free(efficiencies);
}

/* Blah */
void PolyCapAPI::traceSinglePhoton(){

	polycap_vector3 start_coords;	
	start_coords.x = 1.0;	
	start_coords.y = 1.0;	
	start_coords.z = 1.0;

	polycap_vector3 start_direction;	
	start_direction.x = 1.0;	
	start_direction.y = 1.0;	
	start_direction.z = 1.0;

	polycap_vector3 start_electric_vector;	
	start_electric_vector.x = 1.0;	
	start_electric_vector.y = 1.0;	
	start_electric_vector.z = 1.0;

	polycap_photon* myPhoton;
	myPhoton = polycap_photon_new(description, start_coords, start_direction, start_electric_vector, &error);

	int n_energies = 7;							//number of discrete photon energies
	double energies[7]={1,5,10,15,20,25,30};	//energies for which transmission efficiency should be calculated, in keV
	//double** weights;

	//int success = polycap_photon_launch(myPhoton, n_energies, energies, weights, true, &error);

	polycap_vector3 exit_coords = polycap_photon_get_exit_coords(myPhoton);
	polycap_vector3 exit_dir = polycap_photon_get_exit_direction(myPhoton);
	polycap_vector3 exit_electric_vector = polycap_photon_get_exit_electric_vector(myPhoton);

	polycap_photon_free(myPhoton);	


	int ao=0;	

	//polycap_vector3 testvec;
	 
	//polycap_photon* myShadowPhoton = polycap_photon_new(description, )
	//polycap_rng *rng = polycap_rng_new();
    //polycap_photon *a = polycap_source_get_photon(source, rng, &error);
    //polycap_photon_launch (a, n_energies, energies, weights, false, &error);
	//polycap_vector3 exitcoords = polycap_photon_get_exit_coords (a);
	//std::cout << exitcoords.x << std::endl;
    //////////////////////////////////////////////////////
}