#ifndef ChemElementGPU_H
#define ChemElementGPU_H

/** Chemical-Element-Object for GPU */

//#include <math.h>
#include <device_launch_parameters.h>
#include "../api/XRayLibAPI.hpp"

class ChemElementGPU {
	private:
		int z_;																		// Atomic number
		float a_;																	// Atomic Mass number [g/mol]
		float rho_;																	// Density at room temperature [g/cm³]

		static constexpr float energy_resolution = 0.2;								// Resolution of the energy grid [keV] -> 0.01 keV
		static constexpr float max_energy = 20.0;									// Maximum energy of the energy grid [keV] -> 40 keV
		static constexpr int energy_entries = (int)(max_energy/energy_resolution);	// Number of energy grid entries

		static constexpr float angle_resolution = 0.01;								// Resolution of the angle grid [rad] -> 0.005 rad
		static constexpr float max_angle = M_PI;									// Maximum angle of the angle grid [rad] -> 2*pi
		static constexpr int angle_entries = (int)(max_angle/angle_resolution);		// Number of angle grid entries

		static constexpr int line_entries = 382;									// Number of lines in the line grid
		static constexpr int shell_entries = 25;									// Number of shells in the shell grid
		static constexpr int max_shell = shell_entries;								// Maximum shell number

		float cs_tot[energy_entries];
		float cs_phot_prob[energy_entries];
		float cs_ray_prob[energy_entries];
		float cs_compt_prob[energy_entries];

		float cs_phot_part[shell_entries][energy_entries];

		float dcs_rayl[energy_entries][angle_entries];
		float dcs_comp[energy_entries][angle_entries];

		float line_energies[line_entries];
		float rad_rate[line_entries];

		float fluor_yield[shell_entries];
		float auger_yield[shell_entries];

		
		int shell_lines[shell_entries][2]= { 
			{0,28}, 																	// K lines
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
				cs_phot_prob[i] = XRayLibAPI::CS_Phot(z_,e) / XRayLibAPI::CS_Tot(z_,e);
				cs_ray_prob[i] = XRayLibAPI::CS_Ray(z_,e) / XRayLibAPI::CS_Tot(z_,e);
				cs_compt_prob[i] = XRayLibAPI::CS_Compt(z_,e) / XRayLibAPI::CS_Tot(z_,e);

				float dcs_rayl_sum=0., dcs_comp_sum=0.;
				for(int j = 0; j < angle_entries; j++){
					float a = j*angle_resolution;
					dcs_rayl[i][j] = XRayLibAPI::DCS_Rayl(z_,e,a);
					dcs_comp[i][j] = XRayLibAPI::DCS_Compt(z_,e,a);
					dcs_rayl_sum += dcs_rayl[i][j];
					dcs_comp_sum += dcs_comp[i][j]; 
				}
				for(int j = 0; j < angle_entries; j++){
					dcs_rayl[i][j] = dcs_rayl[i][j] / dcs_rayl_sum;
					dcs_comp[i][j] = dcs_comp[i][j] / dcs_comp_sum;

				}

			}

			for(int i = 0; i < 31; i++){
				fluor_yield[i] = XRayLibAPI::FluorY(z_,i);
				auger_yield[i] = XRayLibAPI::AugY(z_,i);

				for(int j = 0; j < energy_entries; j++){
					float e = j*energy_resolution;
					//cs_fluor_l[i][j] = XRayLibAPI::CS_FluorL(z_,j,e);
					cs_phot_part[i][j] = XRayLibAPI::CS_Phot_Part(z_,i,e) / XRayLibAPI::CS_Phot(z_,e);
				}
			}

			for(int i = 0; i < 382; i++){
				line_energies[i] = XRayLibAPI::LineE(z_,i*-1-1);
				rad_rate[i] = XRayLibAPI::RadRate(z_,i*-1-1);
			}
		};

  	public:

  		__host__ ChemElementGPU() {};

		__host__ ChemElementGPU(const int& z){
			z_ =	z;
			a_ =	XRayLibAPI::A(z_);
			rho_ = 	XRayLibAPI::Rho(z_);

			discretize();
			getMemorySize();
		};
		
		// Member-Getter
		__host__ __device__ float A() const {return a_;};
		__host__ __device__ int Z() const {return z_;};
		__host__ __device__ float Rho() const {return rho_;};	

		// Simple DB-Access
		__device__ float Fluor_Y(int shell) const {return fluor_yield[shell];};
		__device__ float Aug_Y(int shell) const {return auger_yield[shell];};
		__device__ float Rad_Rate(int line) const {return rad_rate[line];};
		__device__ float Line_Energy(int line) const {return line_energies[line];};

		__device__ float CS_Tot(float energy) const { return interpolate(energy, energy_resolution, cs_tot);};
		__device__ float CS_Phot_Prob(float energy) const { return interpolate(energy, energy_resolution, cs_phot_prob);};
		__device__ float CS_Rayl_Prob(float energy) const { return interpolate(energy, energy_resolution, cs_ray_prob);};
		__device__ float CS_Compt_Prob(float energy) const { return interpolate(energy, energy_resolution, cs_compt_prob);};
		__device__ float CS_Phot_Part_Prob(int shell, float energy) const {return interpolate(energy,energy_resolution, cs_phot_part[shell]);};

		__device__ float DCS_Rayl(float energy, float angle) const { return interpolate(angle, angle_resolution, dcs_rayl[lrintf(energy/energy_resolution)]);};
		__device__ float DCS_Compt(float energy, float angle) const { return interpolate(angle, angle_resolution, dcs_comp[lrintf(energy/energy_resolution)]);};

		// "Decisions" with random number
		__device__ int getInteractionType(float energy, float randomN) const{

			if(randomN <= CS_Phot_Prob(energy)) return 0;
			else if(randomN <= (CS_Phot_Prob(energy) + CS_Rayl_Prob(energy))) return 1;
			else return 2;
		};

		__device__ int getExcitedShell(float energy, float randomN) const{

			int shell_;
			float temp_= 0.;
			float sum_ = 0.;
			float cs_tot = CS_Phot_Prob(energy);

			if(cs_tot!=0.0)
			for(shell_ = 0; shell_ < max_shell; shell_++){
				temp_= CS_Phot_Part_Prob(shell_, energy);
				sum_ += temp_;
				if(sum_ > randomN) break;
			}

			return shell_;
		};

		__device__ float getThetaCompt(float energy, float randomN) const {
			
			int i; 
			float sum = 0.;
			for(i=0; i<angle_entries; i++){
				sum += DCS_Compt(energy,i*angle_resolution);
				if(sum >randomN) break;
			}

			return i*angle_resolution;
		};

		__device__ float getThetaRayl(float energy, float randomN) const {
			
			int i; 
			float sum = 0.;
			for(i=0; i<angle_entries; i++){
				sum += DCS_Rayl(energy,i*angle_resolution);
				if(sum > randomN) break;
			}

			return i*angle_resolution;
		};

		__device__ int getTransition(int shell, float randomN) const { 

			/*int shell_lines[shell_entries][2]= { 
				{0,28}, 																	// K lines
				{29,57},{85,112},{113,135},													// L lines
				{136,157},{158,179},{180,199},{200,218},									// M lines
				{219,236},{237,253},{254,269},{270,284},{285,298},{299,311},{312,323},		// N lines
				{321,334},{335,344},{345,353},{354,361},{362,368},{369,371},{372,373},		// O lines
				{374,377},{378,380},{381,382}												// P lines
			};*/
			int dimensions = 0;
			//int i1 = shell_lines[shell][0];
			//int i2 = shell_lines[shell][1];

			for(int i = shell_lines[shell][0]; i <= shell_lines[shell][1]; i++){
				if(Rad_Rate(i)>0.0)
					dimensions++;
			}
			if(dimensions !=0){


				
			int* line_ratios_line;
			float* line_ratios_ratio;	
			cudaMalloc(&line_ratios_line, sizeof(int)*dimensions);
			cudaMalloc(&line_ratios_ratio, sizeof(float)*dimensions);

			int c = 0;
			for(int i = shell_lines[shell][0]; i <= shell_lines[shell][1]; i++){
				float ratio = Rad_Rate(i);
				if (Rad_Rate(i)>0.0) {
					line_ratios_line[c]=i;
					line_ratios_ratio[c]=ratio;	
					c++;
				}
			}

			float mySum = 0.;
			int myLine;
					
			for(int i = 0; i < dimensions; i++){
				mySum += line_ratios_ratio[i];
				myLine = line_ratios_line[i];
				if(mySum>randomN) break; 	
			}
			
			cudaFree(line_ratios_line);
			cudaFree(line_ratios_ratio);
			//cudaFree(&shell_lines);
				return myLine;
			}
			return 0;
		};

		// Helper functions
		__device__ float interpolate(float arg, float stepsize, const float* vec) const{
			float x = arg/stepsize;
			int i = ceilf(x); //std::ceil(x);

			float x1 = (i-1)*stepsize;
			float x2 = i*stepsize;

			float y1 = vec[i-1];
			float y2 = vec[i];

			float result =  y1 + (y2-y1)/(x2-x1)*(arg-x1);

			return result;

		};





		__host__ size_t getMemorySize() const {

			size_t z_size = sizeof(int);
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

			size_t tot_size = z_size + a_size + rho_size + cs_tot_size + cs_phot_prob_size + 
				cs_ray_prob_size + cs_compt_prob_size + cs_fluor_l_size + cs_phot_part_size + 
				fluor_yield_size + aug_yield_size + line_energies_size + rad_rate_size + 
				dcs_rayl_size + dcs_compt_size;
			tot_size = tot_size/1024/1024;

			//print all the sizes
			/*
			std::cout	<<	"z_size: \t\t"			<<	z_size 						<<	"\t Byte" << std::endl;
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