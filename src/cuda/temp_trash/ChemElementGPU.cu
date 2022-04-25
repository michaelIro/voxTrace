/** Chemical-Element-Object for GPU */

#include "ChemElementGPU.cuh"

/** Constructor using Atomic number Z 
 * @param z Atomic number
 * @return ChemElementGPU-Object containing all information about chemical element.
 */
ChemElementGPU::ChemElementGPU(const int& z){
	z_ =	z;
	sym_ =	XRayLibAPI::ZToSym(z_);
	a_ =	XRayLibAPI::A(z_);
	rho_ = 	XRayLibAPI::Rho(z_);

	discretize();
	getMemorySize();
}

/** Constructor using element symbol used in periodic table 
 * @param symbol Element symbol used in periodic table 
 * @return ChemElementGPU-Object containing all information about chemical element.
 */
ChemElementGPU::ChemElementGPU(const std::string symbol){
	z_=			XRayLibAPI::SymToZ(symbol.c_str());
	sym_ =		symbol;
	a_ =		XRayLibAPI::A(z_);
	rho_ = 		XRayLibAPI::Rho(z_);

	discretize();
	getMemorySize();
}

/** Discretize in energy and angle grid
 * @return void
 */
void ChemElementGPU::discretize(){
	for(int i=1; i<=energy_entries; i++){
		float e = i*energy_resolution;

		//cs_tot.push_back(XRayLibAPI::CS_Tot(z_,e));
		//cs_phot_prob.push_back( XRayLibAPI::CS_Phot(z_,e) / XRayLibAPI::CS_Tot(z_,e) );
		//cs_ray_prob.push_back( XRayLibAPI::CS_Ray(z_,e) / XRayLibAPI::CS_Tot(z_,e) );
		//cs_compt_prob.push_back( XRayLibAPI::CS_Compt(z_,e) / XRayLibAPI::CS_Tot(z_,e) );

		cs_tot[i] = XRayLibAPI::CS_Tot(z_,e);
		cs_phot[i] = (XRayLibAPI::CS_Phot(z_,e) / XRayLibAPI::CS_Tot(z_,e));
		cs_ray_prob[i] = XRayLibAPI::CS_Ray(z_,e) / XRayLibAPI::CS_Tot(z_,e);
		cs_compt_prob[i] = XRayLibAPI::CS_Compt(z_,e) / XRayLibAPI::CS_Tot(z_,e);

		//thrust::host_vector<float> rayl;
		//thrust::host_vector<float> compt;
		for(int j = 0; j < angle_entries; j++){
			float a = j*angle_resolution;
			//rayl.push_back(XRayLibAPI::DCS_Rayl(z_,e,a));
			//compt.push_back(XRayLibAPI::DCS_Compt(z_,e,a));

			dcs_rayl[i][j] = XRayLibAPI::DCS_Rayl(z_,e,a);
			//dcs_compt_alt[i][j] = 
			dcs_comp[i][j]=XRayLibAPI::DCS_Compt(z_,e,a);
			//std::cout << XRayLibAPI::DCS_Compt(z_,e,a) << std::endl;
		}
		//dcs_rayl.push_back(rayl);
		//dcs_compt.push_back(compt);
	}

	for(int i = 0; i < 31; i++){
		//fluor_yield.push_back( XRayLibAPI::FluorY(z_,i) );
		//auger_yield.push_back( XRayLibAPI::AugY(z_,i) );

		fluor_yield[i] = XRayLibAPI::FluorY(z_,i);
		auger_yield[i] = XRayLibAPI::AugY(z_,i);

		//thrust::host_vector<float> fluor_l;
		//thrust::host_vector<float> phot_part;
		for(int j = 0; j < energy_entries; j++){
			float e = j*energy_resolution;
			//fluor_l.push_back(XRayLibAPI::CS_FluorL(z_,j,e));
			//phot_part.push_back( XRayLibAPI::CS_Phot_Part(z_,j,e) / XRayLibAPI::CS_Tot(z_,e) );
			cs_fluor_l[i][j] = XRayLibAPI::CS_FluorL(z_,j,e);
			cs_phot_part[i][j] = XRayLibAPI::CS_Phot_Part(z_,j,e) / XRayLibAPI::CS_Tot(z_,e);
		}
		//cs_fluor_l.push_back(fluor_l);
		//cs_phot_part.push_back(phot_part);
	}

	for(int i = 0; i < 382; i++){
		line_energies[i] = XRayLibAPI::LineE(z_,i);
		rad_rate[i] = XRayLibAPI::RadRate(z_,i*-1-1);
		//line_energies.push_back( XRayLibAPI::LineE(z_,i*-1-1) );
		//rad_rate.push_back( XRayLibAPI::RadRate(z_,i*-1-1) );
	}
} 

		/** Getter-Functions for member variables
__device__ float ChemElementGPU::A() const {return a_;}
__device__ int ChemElementGPU::Z() const {return z_;}
__device__ float ChemElementGPU::Rho() const {return rho_;}
__host__ std::string ChemElementGPU::Sym() const {return sym_;}*/
/** Calculate interaction type based on comparison with random number
 * @param energy Energy of interacting Ray in keV
 * @param randomN Random number between 0 and 1
 * @return type of Interaction -> 0 = Photo-Effect / 1 = Rayleigh-Scattering / 2 = Compton-Scattering
*/
__device__ int ChemElementGPU::getInteractionType(float energy, float randomN) const{
	
	float tot = CS_Tot(energy);
	float phot = CS_Phot(energy) / tot;
	float photRayleigh = (CS_Phot(energy) + CS_Rayl(energy)) / tot;

	if(randomN <= phot) return 0;
	else if(randomN <= photRayleigh) return 1;
	else return 2;
}

/** Get excited shell based on comparison with random number
 * @param energy Energy of interacting Ray in keV
 * @param randomN Random number between 0 and 1
 * @return Excited shell as int 
*/
__device__ int ChemElementGPU::getExcitedShell(float energy, float randomN) const{

	int shell_;
	float temp_= 0.;
	float sum_ = 0.;
	float cs_tot = CS_Phot(energy);

	if(cs_tot!=0.0)
	for(shell_ = 0; shell_ < max_shell; shell_++){
		temp_= CS_Phot_Part(shell_, energy) / cs_tot;
		sum_ += temp_;
		if(sum_ > randomN) break;
	}

	return shell_;
}

/** Get transition for excited shell based on comparison with random number
 * @param shell Excited shell
 * @param randomN Random number between 0 and 1
 * @return Transition as int
*/
int ChemElementGPU::getTransition(int shell, float randomN) const{
	thrust::host_vector<int> line_ratios_line;
	thrust::host_vector<float> line_ratios_ratio;	

	for(int myLine =shell_lines[shell][0]; myLine <= shell_lines[shell][1]; myLine++){
		int lineInput = myLine*-1-1;
		//int lineInput = myLine;
		float ratio = Rad_Rate(lineInput);
		if(ratio!=0){
			line_ratios_line.push_back(myLine);
			line_ratios_ratio.push_back(ratio);	
				//lineRatios.insert({myLine,ratio});
		}
	}

	float mySum =0.;
	int myLine;
					
	for(int i = 0; i < line_ratios_line.size(); i++){
		//cout<<line.first<<" "<<line.second<<endl;		
		mySum += line_ratios_ratio[i];
		myLine = line_ratios_line[i];
		if(mySum>randomN) break; 	
	}
	return myLine;
}

/*TODO:  Use new discretisation for this Integral
float ChemElementGPU::getThetaCompt(float energy, float randomN){
	int stepsize = 200;
	int arraysize = (int) (M_PI/(1./((float)stepsize)))+1;

	float probSum=0.;
	float prob[arraysize];
	float theta[arraysize];

	float photLambda = 1. / (energy/12.39841930);


	float x =0.;
	for(int i=0; i<arraysize; i++){
		x = ((float)i)/((float)(stepsize));
		theta[i]= 2*asin(x*photLambda);
		prob[i] = XRayLibAPI::DCS_Compt(z_,energy,theta[i]);

		if((isnan(prob[i]))) break;
		if(i!=0) probSum += prob[i-1]*(cos(theta[i]) - cos(theta[i-1]));	
	}

  	float integral = abs(probSum*2*M_PI);
	//float expected = CS_Rayl(z_,energy,NULL);

	probSum =0.;
	int j;
	for(j=0; j<arraysize; j++){
		probSum += prob[j]/integral;
		if(probSum >randomN)break;
	}
	return theta[j];
}

float ChemElementGPU::getThetaRayl(float energy, float randomN){
	int stepsize = 200;
	int arraysize = (int) (M_PI/(1./((float)stepsize)))+1;

	float probSum=0.;
	float prob[arraysize];
	float theta[arraysize];

	float photLambda = 1. / (energy/12.39841930);


	float x =0.;
	for(int i=0; i<arraysize; i++){
		x = ((float)i)/((float)(stepsize));
		theta[i]= 2*asin(x*photLambda);
		prob[i] = XRayLibAPI::DCS_Rayl(z_,energy,theta[i]);

		if((isnan(prob[i]))) break;
		if(i!=0) probSum += prob[i-1]*(cos(theta[i]) - cos(theta[i-1]));	
	}

  	float integral = abs(probSum*2*M_PI);
	//float expected = CS_Rayl(z_,energy,NULL);

	probSum =0.;
	int j;
	for(j=0; j<arraysize; j++){
		probSum += prob[j]/integral;
		if(probSum >randomN)break;
	}
	return theta[j];
}
*/

/** Returns an estimate for the memory size of the ChemElementGPU Object in MB
 * @returns an estimate for the size of the object in MB
 * @warning This is not a reliable method to get the exact size of an object.
 */
size_t ChemElementGPU::getMemorySize() const{

	size_t z_size = sizeof(int);
	size_t sym_size = sizeof(std::string);
	size_t a_size = sizeof(float);
	size_t rho_size = sizeof(float);

	size_t cs_tot_size = sizeof(float)*energy_entries;
	size_t cs_phot_prob_size = sizeof(float)*energy_entries;
	size_t cs_ray_prob_size = sizeof(float)*energy_entries;
	size_t cs_compt_prob_size = sizeof(float)*energy_entries;

	size_t cs_fluor_l_size = sizeof(float)*energy_entries*31;
	size_t cs_phot_part_size = sizeof(float)*energy_entries*31;

	size_t dcs_rayl_size = sizeof(float)*energy_entries*angle_entries;
	size_t dcs_compt_size = sizeof(float)*energy_entries*angle_entries;

	size_t fluor_yield_size = sizeof(float)*31;
	size_t aug_yield_size = sizeof(float)*31;

	size_t line_energies_size = sizeof(float)*382;
	size_t rad_rate_size = sizeof(float)*382;

	size_t tot_size = z_size + sym_size + a_size + rho_size + cs_tot_size + cs_phot_prob_size + cs_ray_prob_size + cs_compt_prob_size + cs_fluor_l_size + cs_phot_part_size + fluor_yield_size + aug_yield_size + line_energies_size + rad_rate_size + dcs_rayl_size + dcs_compt_size;
	tot_size = tot_size/1024/1024;

	//print all the sizes
	/*
	std::cout	<<	"z_size: \t\t"			<<	z_size 						<<	"\t Byte" << std::endl;
	std::cout	<<	"sym_size: \t\t"		<<	sym_size					<<	"\t Byte" << std::endl;
	std::cout	<<	"a_size: \t\t"			<<	a_size						<<	"\t Byte" << std::endl;
	std::cout	<<	"rho_size: \t\t"		<<	rho_size					<<	"\t Byte" << std::endl;
	std::cout	<<	"cs_tot_size: \t\t"		<<	cs_tot_size/1024			<<	"\t KB" << std::endl;
	std::cout	<<	"cs_phot_prob_size: \t"	<<	cs_phot_prob_size/1024		<<	"\t KB" << std::endl;
	std::cout	<<	"cs_ray_prob_size: \t"	<<	cs_ray_prob_size/1024		<<	"\t KB" << std::endl;
	std::cout	<<	"cs_compt_prob_size: \t"<<	cs_compt_prob_size/1024		<<	"\t KB" << std::endl;
	std::cout	<<	"cs_fluorL_size: \t"	<<	cs_fluor_l_size/1024		<<	"\t KB" << std::endl;
	std::cout	<<	"cs_phot_part_size: \t"	<<	cs_phot_part_size/1024		<<	"\t KB" << std::endl;
	std::cout	<<	"fluor_yield_size: \t"	<<	fluor_yield_size/1024		<<	"\t KB" << std::endl;
	std::cout	<<	"aug_yield_size: \t"	<<	aug_yield_size/1024			<<	"\t KB" <<std::endl;
	std::cout	<<	"line_energies_size: \t"<<	line_energies_size/1024		<<	"\t KB" <<std::endl;
	std::cout	<<	"rad_rate_size: \t\t"	<<	rad_rate_size/1024			<<	"\t KB" <<std::endl;
	std::cout	<<	"dcs_rayl_size: \t\t"	<<	dcs_rayl_size/1024			<<	"\t KB" <<std::endl;
	std::cout	<<	"dcs_compt_size: \t"	<<	dcs_compt_size/1024			<<	"\t KB" <<std::endl;*/
	std::cout	<<	"Total size: \t\t"		<<	tot_size					<<"\t MB"<<std::endl;


	// return total size
	return tot_size;
}

/** Interpolates any value between discretized energy / angle values 
 * @param arg the energy / angle value to interpolate
 * @param stepsize the stepsize for the given vector
 * @param vec the vector to interpolate
 * @returns the interpolated value
 */
/*float ChemElementGPU::interpolate(float arg, float stepsize, thrust::host_vector<float> vec) const {

	float x = arg/stepsize;
	int i = ceilf(x); //std::ceil(x);

	float x1 = (i-1)*stepsize;
	float x2 = i*stepsize;

	float y1 = vec[i-1];
	float y2 = vec[i];

	return y1 + (y2-y1)/(x2-x1)*(arg-x1);
}	*/

/*__device__ __device__ float ChemElementGPU::interpolate(float arg, float stepsize, const float* vec) const {

	float x = arg/stepsize;
	int i = ceilf(x); //std::ceil(x);

	float x1 = (i-1)*stepsize;
	float x2 = i*stepsize;

	float y1 = vec[i-1];
	float y2 = vec[i];

	return y1 + (y2-y1)/(x2-x1)*(arg-x1);
}*/
