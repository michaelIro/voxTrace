/* Shadow 3 API*/

#include "Shadow3API.hpp"

/* Empty constructor */
Shadow3API::Shadow3API(){}

/* Blah */
arma::Mat<double> Shadow3API::getBeam(int nRays){
    
    Source src;
    //OE     oe1;
    Beam beam;
    
    // load variables from start.00
    src.load((char*) "../test-data/shadow3/start.00" );

    // overwrite number of rays
    src.NPOINT=nRays;

    // calculate source
    beam.genSource(&src);

    // load start.01 into oe1
    //oe1.load( (char*) "start.01");
    
    // traces OE1
    //ray.traceOE(&oe1,1);

    // write rays to arma::mat
    arma::Mat<double> rays = arma::ones(src.NPOINT, src.NCOL);
    for(int i = 0; i < rays.n_rows; i++)
        for(int j = 0; j < rays.n_cols; j++)
            rays(i,j) = (*(beam.rays+i*18+j));

    return rays;
}

