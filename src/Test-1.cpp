#include <iostream>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "api/PolyCapAPI.hpp"
#include "api/Shadow3API.hpp"

#include "tracer/PrimaryBeam.hpp"

int main(int argc, const char* argv[]) {

	std::cout << "START: Test-1" << std::endl;

    int job_id = atoi( argv[1] );
	int n_sh_rays = atoi( argv[2] );
	int n_iter = atoi( argv[3] );
	int n_files = atoi( argv[4] );
	
	int rand_seed = 1;
	std::ifstream seed_file("../test-data/in/seeds.txt");
   	for (int i=0; i<job_id; i++)
        seed_file >> rand_seed;

	Shadow3API shadow_((char*) "../test-data/in/shadow3");
	PolyCapAPI pc1_((char*) "../test-data/in/polycap/pc-246-descr.txt");	

	PrimaryBeam primary_(shadow_, pc1_, job_id, rand_seed, n_sh_rays, n_iter, n_files);

	std::cout << "END: Test-1" << std::endl;

    return 0;

}