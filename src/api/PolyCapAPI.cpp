/** PolyCapAPI */
#include "PolyCapAPI.hpp"

/* Constructor */
PolyCapAPI::PolyCapAPI(char* path){


	load(path);
	error = NULL;

	//define optic profile shape
	polycap_profile *profile;
	profile = polycap_profile_new(POLYCAP_PROFILE_ELLIPSOIDAL, optic_length, rad_ext_upstream, rad_ext_downstream, rad_int_upstream, rad_int_downstream, focal_dist_upstream, focal_dist_downstream, &error);

	//define optic description
	description = polycap_description_new(profile, surface_rough, n_capillaries, n_elem, iz, wi, density, &error);
	polycap_profile_free(profile); 				//We can free the profile structure, as it is now contained in description

	defineSource();


}

/** Load policapillary parameters */
void PolyCapAPI::load(char* path){
	std::ifstream inFile;
	std::string x;
	inFile.open(path);

	std::getline(inFile, x);
	//std::cout << x << std::endl<< std::endl;

	std::getline(inFile, x);
	optic_length = std::stod(x.substr(0,x.find(';'))); 

	std::getline(inFile, x);
	rad_ext_upstream = std::stod(x.substr(0,x.find(';'))); 

	std::getline(inFile, x);
	rad_ext_downstream = std::stod(x.substr(0,x.find(';'))); 

	std::getline(inFile, x);
	rad_int_upstream = std::stod(x.substr(0,x.find(';'))); 

	std::getline(inFile, x);
	rad_int_downstream = std::stod(x.substr(0,x.find(';'))); 

	std::getline(inFile, x);
	focal_dist_upstream = std::stod(x.substr(0,x.find(';'))); 

	std::getline(inFile, x);
	focal_dist_downstream = std::stod(x.substr(0,x.find(';'))); 

	std::getline(inFile, x);
	n_elem= std::stoi(x.substr(0,x.find(';'))); 

	std::getline(inFile, x);
	iz = new int[n_elem];
	int i = 0;
	std::stringstream ss0(x.substr(x.find('{')+1, x.find('}')-x.find('{')-1));
	std::string s;
    while (std::getline(ss0, s, ',')) 
		iz[i++] = stoi(s); 

	std::getline(inFile, x);
	wi = new double[n_elem];
	i = 0;
	std::stringstream ss1(x.substr(x.find('{')+1, x.find('}')-x.find('{')-1));
    while (std::getline(ss1, s, ',')) 
		wi[i++] = stod(s); 

	std::getline(inFile, x);
	density = std::stod(x.substr(0,x.find(';'))); 

	std::getline(inFile, x);
	surface_rough = std::stod(x.substr(0,x.find(';'))); 

	std::getline(inFile, x);
	n_capillaries = std::stod(x.substr(0,x.find(';'))); 
}

/** Print policapillary parameters */
void PolyCapAPI::print(){
	std::cout << "Polycapillary Parameters:" << std::endl<< std::endl;
	std::cout << "optic_length: " << optic_length << std::endl<< std::endl;
	std::cout << "rad_ext_upstream: " << rad_ext_upstream << std::endl<< std::endl;
	std::cout << "rad_ext_downstream: " << rad_ext_downstream << std::endl<< std::endl;
	std::cout << "rad_int_upstream: " << rad_int_upstream << std::endl<< std::endl;
	std::cout << "rad_int_downstream: " << rad_int_downstream << std::endl<< std::endl;
	std::cout << "focal_dist_upstream: " << focal_dist_upstream << std::endl<< std::endl;
	std::cout << "focal_dist_downstream: " << focal_dist_downstream << std::endl<< std::endl;
	std::cout << "n_elem : " << n_elem << "\t" << iz[0] << "\t" << iz[1] << std::endl<< std::endl;
	std::cout << "n_elem-Z : " << "\t" << iz[0] << "\t" << iz[1] << std::endl<< std::endl;
	std::cout << "n_elem-W: " << "\t" << wi[0] << "\t" << wi[1] << std::endl<< std::endl;
	std::cout << "density: " << density << std::endl<< std::endl;
	std::cout << "surface_rough " << surface_rough << std::endl<< std::endl;
	std::cout << "n_capillaries : " << n_capillaries << std::endl<< std::endl;
}

/** Define PolyCap X-Ray-Source */
void PolyCapAPI::defineSource(){

	// Photon source parameters TODO: IO for clean looking .txt File for these parameters (should be adaptable without recompiling)
	double source_dist = 10.0;						//distance between optic entrance and source along z-axis
	double source_rad_x = 0.37;						//source radius in x, in cm
	double source_rad_y = 0.37;						//source radius in y, in cm
	double source_div_x = 0.0;						//source divergence in x, in rad
	double source_div_y = 0.0;						//source divergence in y, in rad
	double source_shift_x = 0.;						//source shift in x compared to optic central axis, in cm
	double source_shift_y = 0.;						//source shift in y compared to optic central axis, in cm
	double source_polar = 1.0;						//source polarisation factor
	int n_energies = 1;								//number of discrete photon energies
	double energies[1]={1.0};						//energies for which transmission efficiency should be calculated, in keV

	//define photon source, including optic description
	source = polycap_source_new(description, source_dist, source_rad_x, source_rad_y, source_div_x, source_div_y, source_shift_x, source_shift_y, source_polar, n_energies, energies, &error);
}

/** Tracer */
vector<Ray> PolyCapAPI::traceSource(arma::Mat<double> shadowBeam, int nPhotons){

	int i;
    double** weights;

	// Simulation parameters
	int n_threads = -1;				// amount of threads to use; -1 means use all available
	int n_photons = nPhotons;		// simulate nPhotons succesfully transmitted photons (excluding leak events)
	bool leak_calc = false;			// choose to perform leak photon calculations or not. Leak calculations take significantly more time

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	//calculate transmission efficiency curve
	//std::cout << "Original" << std::endl;
	//efficiencies = polycap_source_get_transmission_efficiencies(source, n_threads, n_photons, leak_calc, NULL, &error);
	//std::cout << std::endl;

	std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();

	//std::cout << "Modified" << std::endl;
	efficiencies = polycap_shadow_source_get_transmission_efficiencies(source, n_threads, n_photons, leak_calc, NULL, &error, shadowBeam);
	//std::cout << std::endl;

	std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();

	//std::cout << "Original Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin).count() << "[µs]"  << std::endl;
	//std::cout << "Modified Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - end1).count() << "[µs]" << std::endl;

	vector<Ray> polycapBeam(nPhotons);
	
	int rayCounter = 0;
	for( int i = 0; i < efficiencies->images->i_exit; i++ ){
		if( efficiencies->images->exit_coord_weights[i] >0){
			Ray ray_(
				efficiencies->images->pc_exit_coords[0][i],
				0.,
				efficiencies->images->pc_exit_coords[1][i],
				efficiencies->images->pc_exit_dir[0][i],
				efficiencies->images->pc_exit_dir[2][i],
				efficiencies->images->pc_exit_dir[1][i],
				0.,
				0.,
				0., 
				false, 
				17.4, 
				rayCounter++, 
				3.94, 
				0.0, 
				0.0,  
				0., 
				0., 
				0.,
				efficiencies->images->exit_coord_weights[i]
				);
			polycapBeam.push_back(ray_);
/*
			std::cout << efficiencies->images->pc_exit_coords[0][i] << " ";
			std::cout << efficiencies->images->pc_exit_coords[1][i] << " ";
			std::cout << efficiencies->images->pc_exit_coords[2][i] << std::endl;

			std::cout << efficiencies->images->pc_exit_dir[0][i] << " ";
			std::cout << efficiencies->images->pc_exit_dir[1][i] << " ";
			std::cout << efficiencies->images->pc_exit_dir[2][i] << std::endl << std::endl; */
		}
	}


	polycap_transmission_efficiencies_write_hdf5(efficiencies,"../test-data/polycap/pc-246.hdf5",NULL);

	double *efficiencies_arr = NULL;
	polycap_transmission_efficiencies_get_data(efficiencies, NULL, NULL, &efficiencies_arr, NULL);

	polycap_source_free(source);
	polycap_transmission_efficiencies_free(efficiencies);

	return polycapBeam;
}

void PolyCapAPI::overwritePhoton(arma::rowvec shadowRay, polycap_photon *photon){
	photon->start_coords.x = shadowRay(0); 
	photon->start_coords.y = shadowRay(2); 							// Switch Coordinate System:  y-shadow = z-polycap  
	photon->start_coords.z = shadowRay(1); 

	photon->start_direction.x = shadowRay(3); 
	photon->start_direction.y = shadowRay(5); 
	photon->start_direction.z = shadowRay(4); 

	photon->start_electric_vector.x = shadowRay(6); 
	photon->start_electric_vector.y = shadowRay(8); 
	photon->start_electric_vector.z = shadowRay(7); 
}

/* Blah 
void PolyCapAPI::traceSinglePhoton(arma::Mat<double> shadowBeam){

	int succesfully_traced = 0;
	int absorbed = 0;
	int no_in = 0;
	int wall_hit =0;
	int err =0;

	//Test photon
	polycap_rng *rng;
	rng = polycap_rng_new();
	int n_energies = 1;							//number of discrete photon energies


	//std::cout << "Shadow 3 - Beam: " << std::endl;
	//shadowBeam.print();


	//arma::Mat<double> polycapBeamAfter = arma::ones(shadowBeam.n_rows,9);
	arma::Mat<double> polycapBeamTraced = arma::ones(shadowBeam.n_rows,10);
	
	#pragma omp parallel for
    for(int i = 0; i < shadowBeam.n_rows; i++){
		polycap_photon  *polyCapPhot;
		polyCapPhot = polycap_source_get_photon(source, rng, NULL);

		overwritePhoton(shadowBeam.row(i), polyCapPhot);

		double energies[1]={shadowBeam(i,10)*HC/(2*M_PI)};	//energies for which transmission efficiency should be calculated, in keV
		double *weights_temp;

		polycap_error *myError = NULL;

		int success = polycap_photon_launch(polyCapPhot, n_energies, energies, &weights_temp, true, &myError);


		polycapBeamTraced(i,0) = polyCapPhot->exit_coords.x; 
		polycapBeamTraced(i,1) = polyCapPhot->exit_coords.y; 
		polycapBeamTraced(i,2) = polyCapPhot->exit_coords.z; 

		polycapBeamTraced(i,3) = polyCapPhot->exit_direction.x; 
		polycapBeamTraced(i,4) = polyCapPhot->exit_direction.y; 
		polycapBeamTraced(i,5) = polyCapPhot->exit_direction.z; 

		polycapBeamTraced(i,6) = polyCapPhot->exit_electric_vector.x; 
		polycapBeamTraced(i,7) = polyCapPhot->exit_electric_vector.y; 
		polycapBeamTraced(i,8) = polyCapPhot->exit_electric_vector.z; 
		polycapBeamTraced(i,9) = (double) success; 

		polycap_vector3 exit_coords = polycap_photon_get_exit_coords(polyCapPhot);
		polycap_vector3 exit_dir = polycap_photon_get_exit_direction(polyCapPhot);
		polycap_vector3 exit_electric_vector = polycap_photon_get_exit_electric_vector(polyCapPhot);

		if(success == 0)
			absorbed++;
		if(success == 1)
			succesfully_traced++;
		if(success == 2)
			wall_hit++;
		if(success == -1)
			err++;
		if(success == -2)
			no_in++;

			//if iesc == 0 here a new photon should be simulated/started as the photon was absorbed within it.
			//if iesc == 1 check whether photon is in PC exit window as photon reached end of PC
			//if iesc == 2 a new photon should be simulated/started as the photon hit the walls -> can still leak
			//if iesc == -2 a new photon should be simulated/started as the photon missed the optic entrance window
			//if iesc == -1 some error occured

		polycap_photon_free(polyCapPhot);	
		free(weights_temp);

	}
 
	//std::cout << "Polycap - Beam Before: " << std::endl;
	//polycapBeamBefore.print();

	//std::cout << "Polycap - Beam After: " << std::endl;
	//polycapBeamAfter.print();

	//std::cout << "Polycap - Beam Traced: " << std::endl;
	//polycapBeamTraced.print();

	std::cout << std::endl << std::endl;
	std::cout << "Succesfully Traced Photons: " << succesfully_traced << std::endl;
	std::cout << "Absorbed Photons: " << absorbed << std::endl;
	std::cout << "Photons that hit Wall: " << wall_hit << std::endl;
	std::cout << "Photons that miss entry: " << no_in << std::endl;
	std::cout << "Error: " << err << std::endl;
}*/


/* A helper function to compare coordinates from Shadow with coordinates from Polycap. FIXME: Remove this function from final code!*/
void PolyCapAPI::compareBeams(arma::Mat<double> shadowBeam){
	
	polycap_rng *rng;
	rng = polycap_rng_new();

	std::cout << "Shadow 3 - Beam: " << std::endl;
	shadowBeam.print();

	arma::Mat<double> polycapBeamBefore = arma::ones(shadowBeam.n_rows,9);
	arma::Mat<double> polycapBeamAfter = arma::ones(shadowBeam.n_rows,9);
	
	#pragma omp parallel for
    for(int i = 0; i < shadowBeam.n_rows; i++){
		polycap_photon  *polyCapPhot;
		polyCapPhot = polycap_source_get_photon(source, rng, NULL);
		
		//overwritePhoton(shadowBeam.row(i), polyCapPhot);
		//std::cout << shadowBeam.row(i)(0) << std::endl;

		polycapBeamBefore(i,0) = polyCapPhot->start_coords.x; 
		polycapBeamBefore(i,1) = polyCapPhot->start_coords.y;			
		polycapBeamBefore(i,2) = polyCapPhot->start_coords.z; 

		polycapBeamBefore(i,3) = polyCapPhot->start_direction.x; 
		polycapBeamBefore(i,4) = polyCapPhot->start_direction.y; 		 
		polycapBeamBefore(i,5) = polyCapPhot->start_direction.z; 

		polycapBeamBefore(i,6) = polyCapPhot->start_electric_vector.x; 
		polycapBeamBefore(i,7) = polyCapPhot->start_electric_vector.y; 		
		polycapBeamBefore(i,8) = polyCapPhot->start_electric_vector.z;

		polyCapPhot->start_coords.x = shadowBeam(i,0); 
		polyCapPhot->start_coords.y = shadowBeam(i,2); 							// Switch Coordinate System:  y-shadow = z-polycap  
		polyCapPhot->start_coords.z = shadowBeam(i,1); 

		polyCapPhot->start_direction.x = shadowBeam(i,3); 
		polyCapPhot->start_direction.y = shadowBeam(i,5); 
		polyCapPhot->start_direction.z = shadowBeam(i,4); 

		polyCapPhot->start_electric_vector.x = shadowBeam(i,6); 
		polyCapPhot->start_electric_vector.y = shadowBeam(i,8); 
		polyCapPhot->start_electric_vector.z = shadowBeam(i,7); 

		polycapBeamAfter(i,0) = polyCapPhot->start_coords.x; 
		polycapBeamAfter(i,1) = polyCapPhot->start_coords.y; 
		polycapBeamAfter(i,2) = polyCapPhot->start_coords.z; 

		polycapBeamAfter(i,3) = polyCapPhot->start_direction.x; 
		polycapBeamAfter(i,4) = polyCapPhot->start_direction.y; 
		polycapBeamAfter(i,5) = polyCapPhot->start_direction.z; 
		
		polycapBeamAfter(i,6) = polyCapPhot->start_electric_vector.x; 
		polycapBeamAfter(i,7) = polyCapPhot->start_electric_vector.y; 
		polycapBeamAfter(i,8) = polyCapPhot->start_electric_vector.z; 
	}
 
	std::cout << "Polycap - Beam Before: " << std::endl;
	polycapBeamBefore.print();

	std::cout << "Polycap - Beam After: " << std::endl;
	polycapBeamAfter.print();
}

/* 	This is a modified copy of the function polycap_source_get_transmission_efficiencies from polycap code. 
	"For a given array of energies, and a full polycap_description, get the transmission efficiencies."
	Can be adapted for further information about inner-capillary processes. TODO: Adapt this further */
polycap_transmission_efficiencies* PolyCapAPI::polycap_shadow_source_get_transmission_efficiencies(polycap_source *source, int max_threads, int n_photons, bool leak_calc, polycap_progress_monitor *progress_monitor, polycap_error **error, arma::Mat<double> shadowBeam){
	int i, row=0;
	int64_t sum_iexit=0, sum_irefl=0, sum_not_entered=0, sum_not_transmitted=0;
	int64_t *iexit_temp, *not_entered_temp, *not_transmitted_temp;
	int64_t leak_counter, intleak_counter;
	double *sum_weights;
	polycap_transmission_efficiencies *efficiencies;

	// check max_threads
	if (max_threads < 1 || max_threads > omp_get_max_threads())
		max_threads = omp_get_max_threads();

	// Prepare arrays to save results
	sum_weights = (double *) malloc(sizeof(double)*source->n_energies);

	if(sum_weights == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for sum_weights -> %s", strerror(errno));
		return NULL;
	}
	for(i=0; i < source->n_energies; i++)
		sum_weights[i] = 0.;

	// Thread specific started photon counter
	iexit_temp = (int64_t *) malloc(sizeof(int64_t)*max_threads);
	if(iexit_temp == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for iexit_temp -> %s", strerror(errno));
		return NULL;
	}
	not_entered_temp = (int64_t *) malloc(sizeof(int64_t)*max_threads);
	if(not_entered_temp == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for not_entered_temp -> %s", strerror(errno));
		return NULL;
	}
	not_transmitted_temp = (int64_t *) malloc(sizeof(int64_t)*max_threads);
	if(not_transmitted_temp == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for not_transmitted_temp -> %s", strerror(errno));
		return NULL;
	}
	for(i=0; i < max_threads; i++){
		iexit_temp[i] = 0;
		not_entered_temp[i] = 0;
		not_transmitted_temp[i] = 0;
	}

	// Assign polycap_transmission_efficiencies memory
	efficiencies = (polycap_transmission_efficiencies *) calloc(1, sizeof(polycap_transmission_efficiencies));
	if(efficiencies == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies -> %s", strerror(errno));
		free(sum_weights);
		return NULL;
	}
	efficiencies->energies = (double *) malloc(sizeof(double)*source->n_energies);
	if(efficiencies->energies == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->energies -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->efficiencies = (double *) malloc(sizeof(double)*source->n_energies);
	if(efficiencies->efficiencies == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->efficiencies -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}

	//Assign image coordinate array (initial) memory
	efficiencies->images = (_polycap_images *) calloc(1, sizeof(struct _polycap_images));
	if(efficiencies->images == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_start_coords[0] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->pc_start_coords[0] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_start_coords[0] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_start_coords[1] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->pc_start_coords[1] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_start_coords[1] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->src_start_coords[0] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->src_start_coords[0] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->src_start_coords[0] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->src_start_coords[1] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->src_start_coords[1] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->src_start_coords[1] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_start_dir[0] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->pc_start_dir[0] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_start_dir[0] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_start_dir[1] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->pc_start_dir[1] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_start_dir[1] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_start_elecv[0] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->pc_start_elecv[0] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_start_elecv[0] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_start_elecv[1] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->pc_start_elecv[1] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_start_elecv[1] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_exit_coords[0] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->pc_exit_coords[0] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_exit_coords[0] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_exit_coords[1] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->pc_exit_coords[1] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_exit_coords[1] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_exit_coords[2] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->pc_exit_coords[2] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_exit_coords[2] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_exit_dir[0] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->pc_exit_dir[0] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_exit_dir[0] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_exit_dir[1] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->pc_exit_dir[1] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_exit_dir[1] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_exit_elecv[0] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->pc_exit_elecv[0] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_exit_elecv[0] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_exit_elecv[1] = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->pc_exit_elecv[1] == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_exit_elecv[1] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_exit_nrefl = (int64_t *) malloc(sizeof(int64_t)*n_photons);
	if(efficiencies->images->pc_exit_nrefl == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_exit_nrefl -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->pc_exit_dtravel = (double *) malloc(sizeof(double)*n_photons);
	if(efficiencies->images->pc_exit_dtravel == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_exit_dtravel -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->images->exit_coord_weights = (double *) malloc(sizeof(double)*n_photons*source->n_energies);
	if(efficiencies->images->exit_coord_weights == NULL){
		polycap_set_error(error, POLYCAP_ERROR_MEMORY, "polycap_source_get_transmission_efficiencies: could not allocate memory for efficiencies->images->pc_exit_dir[1] -> %s", strerror(errno));
		polycap_transmission_efficiencies_free(efficiencies);
		free(sum_weights);
		return NULL;
	}
	efficiencies->source = source;

//	// use cancelled as global variable to indicate that the OpenMP loop was aborted due to an error
//	bool cancelled = false;	

//	if (!omp_get_cancellation()) {
//		polycap_set_error_literal(error, POLYCAP_ERROR_OPENMP, "polycap_transmission_efficiencies: OpenMP cancellation support is not available");
//		polycap_transmission_efficiencies_free(efficiencies);
//		free(sum_weights);
//		return NULL;
//	}


//OpenMP loop
#pragma omp parallel \
	default(shared) \
	shared(row)\
	private(i) \
	num_threads(max_threads)
{
	int thread_id = omp_get_thread_num();
	int j = 0;
	polycap_rng *rng;
	polycap_photon *photon;
	int iesc=0, k, l;
	double *weights;
	double *weights_temp;
	//polycap_error *local_error = NULL; // to be used when we are going to call methods that take a polycap_error as argument
	polycap_leak **extleak = NULL; // define extleak structure for each thread
	int64_t n_extleak=0;
	polycap_leak **intleak = NULL; // define intleak structure for each thread
	int64_t n_intleak=0;
	int64_t leak_mem_size=0, intleak_mem_size=0; //memory size indicator for leak and intleak structure arrays
	polycap_vector3 temp_vect; //temporary vector to store electric_vectors during projection onto photon direction
	double cosalpha, alpha; //angle between initial electric vector and photon direction
	double c_ae, c_be;
	polycap_leak **extleak_temp = NULL; // define extleak_temp structure for each thread
	int64_t n_extleak_temp = 0;
	polycap_leak **intleak_temp = NULL; // define intleak_temp structure for each thread
	int64_t n_intleak_temp = 0;
	int64_t leak_mem_size_temp=0, intleak_mem_size_temp=0; //memory size indicator for leak and intleak temp structure arrays

	weights = (double *) malloc(sizeof(double)*source->n_energies);

	for(k=0; k<source->n_energies; k++)
		weights[k] = 0.;

	// Create new rng
	rng = polycap_rng_new();

	i=0; //counter to monitor calculation proceeding
	#pragma omp for
	for(j=0; j < n_photons; j++){
		do{
			// Create photon structure
			photon = polycap_source_get_photon(source, rng, NULL);

			#pragma omp critical
			{
				//std::cout << "Photo #" << row++ <<std::endl;
				overwritePhoton(shadowBeam.row(row++), photon); // FIXME: Ugly solution
				//	arma::rowvec myRayOrTheHighRay = shadow_source.getSingleRay();
			}

			// Launch photon
			iesc = polycap_photon_launch(photon, source->n_energies, source->energies, &weights_temp, leak_calc, NULL);
			//if iesc == 0 here a new photon should be simulated/started as the photon was absorbed within it.
			//if iesc == 1 check whether photon is in PC exit window as photon reached end of PC
			//if iesc == 2 a new photon should be simulated/started as the photon hit the walls -> can still leak
			//if iesc == -2 a new photon should be simulated/started as the photon missed the optic entrance window
			//if iesc == -1 some error occured

			if(iesc == 0)
				not_transmitted_temp[thread_id]++; //photon did not reach end of PC
			if(iesc == 2)
				not_entered_temp[thread_id]++; //photon never entered PC (hit capillary wall instead of opening)
			if(iesc == 1) {
				//check whether photon is within optic exit window
					//different check for monocapillary case...
				temp_vect.x = photon->exit_coords.x + photon->exit_direction.x * (description->profile->z[description->profile->nmax] - photon->exit_coords.z)/photon->exit_direction.z;
				temp_vect.y = photon->exit_coords.y + photon->exit_direction.y * (description->profile->z[description->profile->nmax] - photon->exit_coords.z)/photon->exit_direction.z;
				temp_vect.z = description->profile->z[description->profile->nmax];
				if(round(sqrt(12. * photon->description->n_cap - 3.)/6.-0.5) == 0.){ //monocapillary case
					if(sqrt((temp_vect.x)*(temp_vect.x) + (temp_vect.y)*(temp_vect.y)) > description->profile->ext[description->profile->nmax]){ 
						iesc = 0;
					} else {
						iesc = 1;
					}
				} else { //polycapillary case
					iesc = polycap_photon_within_pc_boundary(description->profile->ext[description->profile->nmax],temp_vect, NULL);
				}
			}
			//Register succesfully transmitted photon, as well as save start coordinates and direction
			if(iesc == 1){
				iexit_temp[thread_id]++;
				efficiencies->images->src_start_coords[0][j] = photon->src_start_coords.x;
				efficiencies->images->src_start_coords[1][j] = photon->src_start_coords.y;
				efficiencies->images->pc_start_coords[0][j] = photon->start_coords.x;
				efficiencies->images->pc_start_coords[1][j] = photon->start_coords.y;
				efficiencies->images->pc_start_dir[0][j] = photon->start_direction.x;
				efficiencies->images->pc_start_dir[1][j] = photon->start_direction.y;
				//the start_electric_vector here is along polycapillary axis, better to project this to photon direction axis (i.e. result should be 1 0 or 0 1)
				cosalpha = polycap_scalar(photon->start_electric_vector, photon->start_direction);
				alpha = acos(cosalpha);
				c_ae = 1./sin(alpha);
				c_be = -1.*c_ae*cosalpha;
				temp_vect.x = photon->start_electric_vector.x * c_ae + photon->start_direction.x * c_be;
				temp_vect.y = photon->start_electric_vector.y * c_ae + photon->start_direction.y * c_be;
				temp_vect.z = photon->start_electric_vector.z * c_ae + photon->start_direction.z * c_be;
				polycap_norm(&temp_vect);
				efficiencies->images->pc_start_elecv[0][j] = round(temp_vect.x);
				efficiencies->images->pc_start_elecv[1][j] = round(temp_vect.y);
			}
			if(leak_calc) { //store potential leak and intleak events for photons that did not reach optic exit window
				if(iesc == 0 || iesc == 2){ 
					// this photon did not reach end of PC or this photon hit capilary wall at optic entrance
					//	but could contain leak info to pass on to future photons,
					if(photon->n_extleak > 0){
						n_extleak_temp += photon->n_extleak;
						if(n_extleak_temp > leak_mem_size_temp){
							if (leak_mem_size_temp == 0){
								leak_mem_size_temp = n_extleak_temp;
							} else {
								leak_mem_size_temp *= 2;
								if (leak_mem_size_temp < n_extleak_temp) leak_mem_size_temp = n_extleak_temp; //not doing this could be dangerous at low values
							}
							extleak_temp = (polycap_leak **) realloc(extleak_temp, sizeof(struct _polycap_leak*) * leak_mem_size_temp);
						}
						for(k = 0; k < photon->n_extleak; k++){
							polycap_leak *new_leak = polycap_leak_new(photon->extleak[k]->coords, photon->extleak[k]->direction, photon->extleak[k]->elecv, photon->extleak[k]->n_refl, source->n_energies, photon->extleak[k]->weight, NULL);
							extleak_temp[n_extleak_temp-photon->n_extleak+k] = new_leak;
						}
					}
					if(photon->n_intleak > 0){
						n_intleak_temp += photon->n_intleak;
						if(n_intleak_temp > intleak_mem_size_temp){
							if (intleak_mem_size_temp == 0){
								intleak_mem_size_temp = n_intleak_temp;
							} else {
								intleak_mem_size_temp *= 2;
								if (intleak_mem_size_temp < n_intleak_temp) intleak_mem_size_temp = n_intleak_temp; //not doing this could be dangerous at low values
							}
							intleak_temp = (polycap_leak **) realloc(intleak_temp, sizeof(struct _polycap_leak*) * intleak_mem_size_temp);
						}
						for(k = 0; k < photon->n_intleak; k++){
							polycap_leak *new_leak = polycap_leak_new(photon->intleak[k]->coords, photon->intleak[k]->direction, photon->intleak[k]->elecv, photon->intleak[k]->n_refl, source->n_energies, photon->intleak[k]->weight, NULL);
							intleak_temp[n_intleak_temp-photon->n_intleak+k] = new_leak;
						}
					}	
				}
				if(iesc == 1){ //this photon reached optic exit window,
					// so pass on all previously acquired leak info (leak_temp, intleak_temp) to this photon
					if(n_extleak_temp > 0){
						photon->n_extleak += n_extleak_temp;
						photon->extleak = (polycap_leak **) realloc(photon->extleak, sizeof(struct _polycap_leak*) * photon->n_extleak);
						for(k = 0; k < n_extleak_temp; k++){
							polycap_leak *new_leak = polycap_leak_new(extleak_temp[k]->coords, extleak_temp[k]->direction, extleak_temp[k]->elecv, extleak_temp[k]->n_refl, source->n_energies, extleak_temp[k]->weight, NULL);
							photon->extleak[photon->n_extleak-n_extleak_temp+k] = new_leak;
						}	

						//free the temp intleak and leak structs
						if(extleak_temp){
							for(k = 0; k < n_extleak_temp; k++){
								polycap_leak_free(extleak_temp[k]);
							}
							free(extleak_temp);
							extleak_temp = NULL;
						}
						//and set their memory counters to 0
						leak_mem_size_temp = 0;
						n_extleak_temp = 0;
					}
					if(n_intleak_temp > 0){
						photon->n_intleak += n_intleak_temp;
						photon->intleak = (polycap_leak **) realloc(photon->intleak, sizeof(struct _polycap_leak*) * photon->n_intleak);
						for(k = 0; k < n_intleak_temp; k++){
							polycap_leak *new_leak = polycap_leak_new(intleak_temp[k]->coords, intleak_temp[k]->direction, intleak_temp[k]->elecv, intleak_temp[k]->n_refl, source->n_energies, intleak_temp[k]->weight, NULL);
							photon->intleak[photon->n_intleak-n_intleak_temp+k] = new_leak;
						}	

						//free the temp intleak and leak structs
						if(intleak_temp){
							for(k = 0; k < n_intleak_temp; k++){
								polycap_leak_free(intleak_temp[k]);
							}
							free(intleak_temp);
							intleak_temp = NULL;
						}
						//and set their memory counters to 0
						intleak_mem_size_temp = 0;
						n_intleak_temp = 0;
					}
				}
			} // if(leak_calc)
			if(iesc != 1) {
				polycap_photon_free(photon); //Free photon here as a new one will be simulated 
				free(weights_temp);
			}
		} while(iesc == 0 || iesc == 2 || iesc == -2 || iesc == -1); //TODO: make this function exit if polycap_photon_launch returned -1... Currently, if returned -1 due to memory shortage technically one would end up in infinite loop

		if(thread_id == 0 && (double)i/((double)n_photons/(double)max_threads/10.) >= 1.){
			printf("%d%% Complete\t%" PRId64 " reflections\tLast reflection at z=%f, d_travel=%f\n",((j*100)/(n_photons/max_threads)),photon->i_refl,photon->exit_coords.z, photon->d_travel);
			i=0;
		}
		i++;//counter just to follow % completed

		//save photon->weight in thread unique array
		for(k=0; k<source->n_energies; k++){
			weights[k] += weights_temp[k];
			efficiencies->images->exit_coord_weights[k+j*source->n_energies] = weights_temp[k];
		}
		//save photon exit coordinates and propagation vector
		//Make sure to calculate exit_coord at capillary exit (Z = capillary length); currently the exit_coord is the coordinate of the last photon-wall interaction
//printf("** coords: %lf, %lf, %lf; length: %lf\n", photon->exit_coords.x, photon->exit_coords.y, photon->exit_coords.z, );
		efficiencies->images->pc_exit_coords[0][j] = photon->exit_coords.x + photon->exit_direction.x*
			(description->profile->z[description->profile->nmax] - photon->exit_coords.z)/photon->exit_direction.z;
		efficiencies->images->pc_exit_coords[1][j] = photon->exit_coords.y + photon->exit_direction.y*
			(description->profile->z[description->profile->nmax] - photon->exit_coords.z)/photon->exit_direction.z;
		efficiencies->images->pc_exit_coords[2][j] = photon->exit_coords.z + photon->exit_direction.z*
			(description->profile->z[description->profile->nmax] - photon->exit_coords.z)/photon->exit_direction.z;
		efficiencies->images->pc_exit_dir[0][j] = photon->exit_direction.x;
		efficiencies->images->pc_exit_dir[1][j] = photon->exit_direction.y;
		// the electric_vector here is along polycapillary axis, better to project this to photon direction axis (i.e. result should be 1 0 or 0 1)
		cosalpha = polycap_scalar(photon->start_electric_vector, photon->start_direction);
		alpha = acos(cosalpha);
		c_ae = 1./sin(alpha);
		c_be = -1.*c_ae*cosalpha;
		temp_vect.x = photon->exit_electric_vector.x * c_ae + photon->exit_direction.x * c_be;
		temp_vect.y = photon->exit_electric_vector.y * c_ae + photon->exit_direction.y * c_be;
		temp_vect.z = photon->exit_electric_vector.z * c_ae + photon->exit_direction.z * c_be;
		polycap_norm(&temp_vect);
		efficiencies->images->pc_exit_elecv[0][j] = round(temp_vect.x);
		efficiencies->images->pc_exit_elecv[1][j] = round(temp_vect.y);
		efficiencies->images->pc_exit_nrefl[j] = photon->i_refl;
		efficiencies->images->pc_exit_dtravel[j] = photon->d_travel + 
			sqrt( (efficiencies->images->pc_exit_coords[0][j] - photon->exit_coords.x)*(efficiencies->images->pc_exit_coords[0][j] - photon->exit_coords.x) + 
			(efficiencies->images->pc_exit_coords[1][j] - photon->exit_coords.y)*(efficiencies->images->pc_exit_coords[1][j] - photon->exit_coords.y) + 
			(description->profile->z[description->profile->nmax] - photon->exit_coords.z)*(description->profile->z[description->profile->nmax] - photon->exit_coords.z));

		//Assign memory to arrays holding leak photon information (and fill them)
		if(leak_calc){
			n_extleak += photon->n_extleak;
			if(n_extleak > leak_mem_size){
				if (leak_mem_size == 0){
					leak_mem_size = n_extleak;
				} else {
					leak_mem_size *= 2;
					if (leak_mem_size < n_extleak) leak_mem_size = n_extleak; //not doing this could be dangerous at low values
				}
				extleak = (polycap_leak **) realloc(extleak, sizeof(struct _polycap_leak*) * leak_mem_size);
			}
			n_intleak += photon->n_intleak;
			if(n_intleak > intleak_mem_size){
				if (intleak_mem_size == 0){
					intleak_mem_size = n_intleak;
				} else {
					intleak_mem_size *= 2;
					if (intleak_mem_size < n_intleak) intleak_mem_size = n_intleak; //not doing this could be dangerous at low values
				}
				intleak = (polycap_leak **) realloc(intleak, sizeof(struct _polycap_leak*) * intleak_mem_size);
			}

			//Write leak photon data.
			if(photon->n_extleak > 0){
				for(k=0; k<photon->n_extleak; k++){
					polycap_leak *new_leak = polycap_leak_new(photon->extleak[k]->coords, photon->extleak[k]->direction, photon->extleak[k]->elecv, photon->extleak[k]->n_refl, source->n_energies, photon->extleak[k]->weight, NULL);
					extleak[n_extleak-photon->n_extleak+k] = new_leak;
				}
			}
			if(photon->n_intleak > 0){
				for(k=0; k<photon->n_intleak; k++){
					polycap_leak *new_leak = polycap_leak_new(photon->intleak[k]->coords, photon->intleak[k]->direction, photon->intleak[k]->elecv, photon->intleak[k]->n_refl, source->n_energies, photon->intleak[k]->weight, NULL);
					intleak[n_intleak-photon->n_intleak+k] = new_leak;
				}
			}
		}

		#pragma omp critical
		{
		sum_irefl += photon->i_refl;
		}

		//free photon structure (new one created for each for loop instance)
		polycap_photon_free(photon);
		free(weights_temp);
	} //for(j=0; j < n_photons; j++)

	#pragma omp critical
	{
	for(i=0; i<source->n_energies; i++) sum_weights[i] += weights[i];
	if(leak_calc){
		efficiencies->images->i_extleak += n_extleak;
		efficiencies->images->i_intleak += n_intleak;
	}
	}

	if(leak_calc){
		#pragma omp barrier //All threads must reach here before we continue.
		#pragma omp single //Only one thread should allocate following memory. There is an automatic barrier at the end of this block.
		{
		efficiencies->images->extleak_coords[0] = (double *) realloc(efficiencies->images->extleak_coords[0], sizeof(double)* efficiencies->images->i_extleak);
		efficiencies->images->extleak_coords[1] = (double *) realloc(efficiencies->images->extleak_coords[1], sizeof(double)* efficiencies->images->i_extleak);
		efficiencies->images->extleak_coords[2] = (double *) realloc(efficiencies->images->extleak_coords[2], sizeof(double)* efficiencies->images->i_extleak);
		efficiencies->images->extleak_dir[0] = (double *) realloc(efficiencies->images->extleak_dir[0], sizeof(double)* efficiencies->images->i_extleak);
		efficiencies->images->extleak_dir[1] = (double *) realloc(efficiencies->images->extleak_dir[1], sizeof(double)* efficiencies->images->i_extleak);
		efficiencies->images->extleak_n_refl = (int64_t *) realloc(efficiencies->images->extleak_n_refl, sizeof(int64_t)* efficiencies->images->i_extleak);
		efficiencies->images->extleak_coord_weights = (double *) realloc(efficiencies->images->extleak_coord_weights, sizeof(double)*source->n_energies* efficiencies->images->i_extleak);
		efficiencies->images->intleak_coords[0] = (double *) realloc(efficiencies->images->intleak_coords[0], sizeof(double)* efficiencies->images->i_intleak);
		efficiencies->images->intleak_coords[1] = (double *) realloc(efficiencies->images->intleak_coords[1], sizeof(double)* efficiencies->images->i_intleak);
		efficiencies->images->intleak_coords[2] = (double *) realloc(efficiencies->images->intleak_coords[2], sizeof(double)* efficiencies->images->i_intleak);
		efficiencies->images->intleak_dir[0] = (double *) realloc(efficiencies->images->intleak_dir[0], sizeof(double)* efficiencies->images->i_intleak);
		efficiencies->images->intleak_dir[1] = (double *) realloc(efficiencies->images->intleak_dir[1], sizeof(double)* efficiencies->images->i_intleak);
		efficiencies->images->intleak_elecv[0] = (double *) realloc(efficiencies->images->intleak_elecv[0], sizeof(double)* efficiencies->images->i_intleak);
		efficiencies->images->intleak_elecv[1] = (double *) realloc(efficiencies->images->intleak_elecv[1], sizeof(double)* efficiencies->images->i_intleak);
		efficiencies->images->intleak_n_refl = (int64_t *) realloc(efficiencies->images->intleak_n_refl, sizeof(int64_t)* efficiencies->images->i_intleak);
		efficiencies->images->intleak_coord_weights = (double *) realloc(efficiencies->images->intleak_coord_weights, sizeof(double)*source->n_energies* efficiencies->images->i_intleak);
		leak_counter = 0;
		intleak_counter = 0;
		}//#pragma omp single
		#pragma omp critical //continue with all threads, but one at a time...
		{
		for(k=0; k < n_extleak; k++){
			efficiencies->images->extleak_coords[0][leak_counter] = extleak[k]->coords.x;
			efficiencies->images->extleak_coords[1][leak_counter] = extleak[k]->coords.y;
			efficiencies->images->extleak_coords[2][leak_counter] = extleak[k]->coords.z;
			efficiencies->images->extleak_dir[0][leak_counter] = extleak[k]->direction.x;
			efficiencies->images->extleak_dir[1][leak_counter] = extleak[k]->direction.y;
			efficiencies->images->extleak_n_refl[leak_counter] = extleak[k]->n_refl;
			for(l=0; l < source->n_energies; l++)
				efficiencies->images->extleak_coord_weights[leak_counter*source->n_energies+l] = extleak[k]->weight[l];
			leak_counter++;
		}
		for(k=0; k < n_intleak; k++){
			efficiencies->images->intleak_coords[0][intleak_counter] = intleak[k]->coords.x;
			efficiencies->images->intleak_coords[1][intleak_counter] = intleak[k]->coords.y;
			efficiencies->images->intleak_coords[2][intleak_counter] = intleak[k]->coords.z;
			efficiencies->images->intleak_dir[0][intleak_counter] = intleak[k]->direction.x;
			efficiencies->images->intleak_dir[1][intleak_counter] = intleak[k]->direction.y;
			efficiencies->images->intleak_elecv[0][intleak_counter] = intleak[k]->elecv.x;
			efficiencies->images->intleak_elecv[1][intleak_counter] = intleak[k]->elecv.y;
			efficiencies->images->intleak_n_refl[intleak_counter] = intleak[k]->n_refl;
			for(l=0; l < source->n_energies; l++)
				efficiencies->images->intleak_coord_weights[intleak_counter*source->n_energies+l] = intleak[k]->weight[l];
			intleak_counter++;
		}
		}//#pragma omp critical
	}
	if(extleak){
		for(k = 0; k < n_extleak; k++){
			polycap_leak_free(extleak[k]);
		}
		free(extleak);
		extleak = NULL;
	}
	if(intleak){
		for(k = 0; k < n_intleak; k++){
			polycap_leak_free(intleak[k]);
		}
		free(intleak);
		intleak = NULL;
	}
	polycap_rng_free(rng);
	free(weights);
} //#pragma omp parallel

//	if (cancelled)
//		return NULL;

	//add all started photons together
	for(i=0; i < max_threads; i++){
		sum_iexit += iexit_temp[i];
		sum_not_entered += not_entered_temp[i];
		sum_not_transmitted += not_transmitted_temp[i];
	}
	
	printf("Average number of reflections: %lf, Simulated photons: %" PRId64 "\n",(double)sum_irefl/n_photons,sum_iexit+sum_not_entered+sum_not_transmitted);
	printf("Open area Calculated: %lf, Simulated: %lf\n",((round(sqrt(12. * description->n_cap - 3.)/6.-0.5)+0.5)*6.)*((round(sqrt(12. * description->n_cap - 3.)/6.-0.5)+0.5)*6.)/12.*(description->profile->cap[0]*description->profile->cap[0]*M_PI)/(3.*sin(M_PI/3)*description->profile->ext[0]*description->profile->ext[0]), (double)(sum_iexit+sum_not_transmitted)/(sum_iexit+sum_not_entered+sum_not_transmitted));
	printf("iexit: %" PRId64 ", no enter: %" PRId64 ", no trans: %" PRId64 "\n",sum_iexit,sum_not_entered,sum_not_transmitted);

	//Continue working with simulated open area, as this should be a more honoust comparisson?
	description->open_area = (double)(sum_iexit+sum_not_transmitted)/(sum_iexit+sum_not_entered+sum_not_transmitted);

	// Complete output structure
	efficiencies->n_energies = source->n_energies;
	efficiencies->images->i_start = sum_iexit+sum_not_entered+sum_not_transmitted;
	efficiencies->images->i_exit = sum_iexit;
//printf("//////\n");
	for(i=0; i<source->n_energies; i++){
		efficiencies->energies[i] = source->energies[i];
		efficiencies->efficiencies[i] = (sum_weights[i] / ((double)sum_iexit+(double)sum_not_transmitted)) * description->open_area;
//printf("	Energy: %lf keV, Weight: %lf \n", efficiencies->energies[i], sum_weights[i]);
	}
//printf("//////\n");


	//free alloc'ed memory
	free(sum_weights);
	free(iexit_temp);
	free(not_entered_temp);
	free(not_transmitted_temp);
	return efficiencies;
}