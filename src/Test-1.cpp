#include <iostream>
#include <armadillo>

#include "api/PolyCapAPI.hpp"
#include "api/Shadow3API.hpp"
//#include "api/XRayLibAPI.hpp"

//#include "base/ChemElement.hpp"
//#include "base/Material.hpp"
//#include "base/XRBeam.hpp"
//#include "base/Sample.hpp"

#include "tracer/Tracer.hpp"
#include "tracer/PrimaryBeam.hpp"

int main(int argc, const char* argv[]) {

	std::cout << "START: Test-1" << std::endl;

    int job_id = atoi( argv[1] );
	
	int rand_seed = 1;
	std::ifstream seed_file("../test-data/in/seeds.txt");
   	for (int i=0; i<job_id; i++)
        seed_file >> rand_seed;

	Shadow3API shadow_((char*) "../test-data/in/shadow3");
	PolyCapAPI pc1_((char*) "../test-data/in/polycap/pc-246-descr.txt");	

	PrimaryBeam primary_(shadow_, pc1_, job_id, rand_seed);

	std::cout << "END: Test-1" << std::endl;

    return 0;

}