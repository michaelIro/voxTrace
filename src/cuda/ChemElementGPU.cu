#ifndef ChemElementGPU_H
#define ChemElementGPU_H

/** Chemical-Element-Object for GPU */

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <math.h>
#include "../api/XRayLibAPI.hpp"

class ChemElementGPU {
	private:
		std::string sym_;															// Chemical Symbol
		int z_;																		// Atomic number
		float a_;																	// Atomic Mass number [g/mol]
		float rho_;																	// Density at room temperature [g/cm³]

		static constexpr float energy_resolution = 0.1;								// Resolution of the energy grid [keV] -> 0.01 keV
		static constexpr float max_energy = 20.0;									// Maximum energy of the energy grid [keV] -> 40 keV
		static constexpr int energy_entries = (int)(max_energy/energy_resolution);	// Number of energy grid entries

		static constexpr float angle_resolution = 0.1;								// Resolution of the angle grid [rad] -> 0.005 rad
		static constexpr float max_angle = 2*M_PI;									// Maximum angle of the angle grid [rad] -> 2*pi
		static constexpr int angle_entries = (int)(max_angle/angle_resolution);		// Number of angle grid entries

		static constexpr int line_entries = 382;									// Number of lines in the line grid
		static constexpr int shell_entries = 25;									// Number of shells in the shell grid
		static constexpr int max_shell = shell_entries;								// Maximum shell number

		float cs_tot[energy_entries];
		float cs_phot[energy_entries];
		float cs_ray_prob[energy_entries];
		float cs_compt_prob[energy_entries];

		float cs_fluor_l[energy_entries][shell_entries];
		float cs_phot_part[energy_entries][shell_entries];

		float dcs_rayl[energy_entries][angle_entries];
		float dcs_comp[energy_entries][angle_entries];

		float line_energies[line_entries];
		float rad_rate[line_entries];

		float fluor_yield[shell_entries];
		float auger_yield[shell_entries];

		int shell_lines[shell_entries][2]= {
			{0,28},																		// K lines
			{29,57},{85,112},{113,135},													// L lines	
			{136,157},{158,179},{180,199},{200,218},									// M lines
			{219,236},{237,253},{254,269},{270,284},{285,298},{299,311},{312,323},		// N lines
			{321,334},{335,344},{345,353},{354,361},{362,368},{369,371},{372,373},		// O lines
			{374,377},{378,380},{381,382}												// P lines
		};

		__host__ void discretize(){
			for(int i=1; i<=energy_entries; i++){
				float e = i*energy_resolution;

				cs_tot[i] = XRayLibAPI::CS_Tot(z_,e);
				cs_phot[i] = XRayLibAPI::CS_Phot(z_,e) / XRayLibAPI::CS_Tot(z_,e);
				cs_ray_prob[i] = XRayLibAPI::CS_Ray(z_,e) / XRayLibAPI::CS_Tot(z_,e);
				cs_compt_prob[i] = XRayLibAPI::CS_Compt(z_,e) / XRayLibAPI::CS_Tot(z_,e);

				for(int j = 0; j < angle_entries; j++){
					float a = j*angle_resolution;
					dcs_rayl[i][j] = XRayLibAPI::DCS_Rayl(z_,e,a);
					dcs_comp[i][j] = XRayLibAPI::DCS_Compt(z_,e,a);
				}
			}

			for(int i = 0; i < 31; i++){
				fluor_yield[i] = XRayLibAPI::FluorY(z_,i);
				auger_yield[i] = XRayLibAPI::AugY(z_,i);

				for(int j = 0; j < energy_entries; j++){
					float e = j*energy_resolution;

					cs_fluor_l[i][j] = XRayLibAPI::CS_FluorL(z_,j,e);
					cs_phot_part[i][j] = XRayLibAPI::CS_Phot_Part(z_,j,e) / XRayLibAPI::CS_Tot(z_,e);
				}
			}

			for(int i = 0; i < 382; i++){
				line_energies[i] = XRayLibAPI::LineE(z_,i);
				rad_rate[i] = XRayLibAPI::RadRate(z_,i*-1-1);
			}
		};

  	public:

  		__host__ ChemElementGPU() {};

  		__host__ ChemElementGPU(const std::string symbol){
			z_=			XRayLibAPI::SymToZ(symbol.c_str());
			sym_ =		symbol;
			a_ =		XRayLibAPI::A(z_);
			rho_ = 		XRayLibAPI::Rho(z_);

			discretize();
			getMemorySize();
		};

		__host__ ChemElementGPU(const int& z){
			z_ =	z;
			sym_ =	XRayLibAPI::ZToSym(z_);
			a_ =	XRayLibAPI::A(z_);
			rho_ = 	XRayLibAPI::Rho(z_);

			discretize();
			getMemorySize();
		};
		
		// Member-Getter
		__host__ __device__ float A() const {return a_;};
		__host__ __device__ int Z() const {return z_;};
		__host__ __device__ float Rho() const {return rho_;};	
		__host__ std::string Sym() const {return sym_;};

		// Simple DB-Access
		__device__ float Fluor_Y(int shell) const {return fluor_yield[shell];};
		__device__ float Aug_Y(int shell) const {return auger_yield[shell];};
		__device__ float Rad_Rate(int line) const {return rad_rate[line];};
		__device__ float Line_Energy(int line) const {return line_energies[line];};
		__device__ float CS_Tot(float energy) const { return interpolate(energy, energy_resolution, cs_tot);};
		__device__ float CS_Phot(float energy) const { return interpolate(energy, energy_resolution, cs_phot);};
		__device__ float CS_Rayl(float energy) const { return interpolate(energy, energy_resolution, cs_ray_prob);};
		__device__ float CS_Compt(float energy) const { return interpolate(energy, energy_resolution, cs_compt_prob);};
		__device__ float CS_Fluor_Line(int shell, float energy) const {return interpolate(energy,energy_resolution, cs_fluor_l[shell]);}; 
		__device__ float CS_Phot_Part(int shell, float energy) const {return interpolate(energy,energy_resolution, cs_phot_part[shell]);};

		//__device__ float DCS_Compt(float energy, float angle) const {return interpolate(energy, angle_resolution, dcs_comp);};

		// "Decisions" with random number
		__device__ int getInteractionType(float energy, float randomN) const{
	
			float tot = CS_Tot(energy);
			float phot = CS_Phot(energy) / tot;
			float photRayleigh = (CS_Phot(energy) + CS_Rayl(energy)) / tot;

			if(randomN <= phot) return 0;
			else if(randomN <= photRayleigh) return 1;
			else return 2;
		};

		__device__ int getExcitedShell(float energy, float randomN) const{

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
		};

		__device__ float getThetaCompt(float energy, float randomN) const {
			
			int stepsize = 200;
			int arraysize = (int) (M_PI/(1./((float)stepsize)))+1;

			float probSum=0.;
			float* prob= new float[arraysize];
			float* theta= new float[arraysize];

			float photLambda = 1. / (energy/12.39841930);


			float x =0.;
			for(int i=0; i<arraysize; i++){
				x = ((float)i)/((float)(stepsize));
				theta[i]= 2*asin(x*photLambda);
				//dcs_comp[energy_entries][angle_entries];
				//prob[i] = XRayLibAPI::DCS_Compt(z_,energy,theta[i]);

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
		};

		__device__ float getThetaRayl(float energy, float randomN) const{return 1.0;};

		__device__ int getTransition(int shell, float randomN) const{return 1.0;};/*{
			thrust::device_vector<int> line_ratios_line;
			thrust::device_vector<float> line_ratios_ratio;	

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
		};*/


		// Helper functions
		__device__ float interpolate(float arg, float stepsize, const float* vec) const{
			float x = arg/stepsize;
			int i = ceilf(x); //std::ceil(x);

			float x1 = (i-1)*stepsize;
			float x2 = i*stepsize;

			float y1 = vec[i-1];
			float y2 = vec[i];

			return y1 + (y2-y1)/(x2-x1)*(arg-x1);
		};

		__device__ float interpolate(float arg1, float arg2, float stepsize1, float stepsize2, const float** vec) const{
			float x1 = arg1/stepsize1;
			float x2 = arg2/stepsize2;

			int i1 = ceilf(x1); 
			int i2 = ceilf(x2); 

			float x1_ = (i1-1)*stepsize1;
			float x2_ = i2*stepsize2;

			float y1_ = vec[i1-1][i2-1];
			float y2_ = vec[i1][i2-1];

			float y1 = vec[i1-1][i2];
			float y2 = vec[i1][i2];

			return y1_ + (y2_-y1_)/(x2_-x1_)*(arg1-x1_) + (y2-y1)/(x2-x1)*(arg2-x2);
		};

		__host__ size_t getMemorySize() const {

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

			//std::cout	<<	"Total size: \t\t"		<<	tot_size					<<"\t MB"<<std::endl;

			// return total size
			return tot_size;
		};
};

#endif