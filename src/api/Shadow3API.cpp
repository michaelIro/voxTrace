/*XRayLib API*/

#include "Shadow3API.hpp"

/*Empty constructor*/
Shadow3API::Shadow3API(){}

double* Shadow3API::getBeam(std::string path){
    Source src;
    Beam beam;
    
    // load variables from start.00
    src.load( (char*) "../test-data/start.00");
    // overwrite number of rays
    src.NPOINT=100000;
    // calculate source
    beam.genSource(&src);
    
    return NULL;
}