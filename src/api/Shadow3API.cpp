/* Shadow 3 API*/
#include "Shadow3API.hpp"

/* Empty constructor */
Shadow3API::Shadow3API(){}

/* Empty constructor */
Shadow3API::Shadow3API(char* path){
    // load variables from start.00
    source_.load(path);
    source_.NPOINT=1;       
}


arma::vec Shadow3API::getSingleRay(){
    beam_.genSource(&source_);
    arma::vec ray_ = arma::ones(source_.NCOL);
    for(int j = 0; j < ray_.n_cols; j++)
            ray_(j) = (*(beam_.rays+j));
    return ray_;
}

/** Generates X-Rays from a Shadow3-Source 
 * @param nRays Number of Rays to be generated
 * @param path Path to start.00 File (Shadow3)
 * @return arma::Mat<double> with Rays generated from start.00 File
 */
arma::Mat<double> Shadow3API::getBeamFromSource(int nRays, char* path){
    
    Source src;
    Beam beam;
    
    // load variables from start.00
    src.load(path);                         // NOTE: Path when called from here is (char*) "../../test-data/shadow3/start.00" 

    // overwrite number of rays
    src.NPOINT=nRays;

    // calculate source
    beam.genSource(&src);

    // write rays to arma::mat
    arma::Mat<double> rays = arma::ones(src.NPOINT, src.NCOL);
    for(int i = 0; i < rays.n_rows; i++)
        for(int j = 0; j < rays.n_cols; j++)
            rays(i,j) = (*(beam.rays+i*18+j));

    return rays;
}

/** Generates X-Rays from a Shadow3-Source and traces them through Shadow3-Optical-Elements 
 * @param nRays Number of Rays to be generated and traced
 * @param path Path to start.** Files (Shadow3)
 * @return arma::Mat<double> with Rays generated from start.** Files
 */
arma::Mat<double> Shadow3API::getBeamFromOE(int nRays, char* path){
    
    Source src;
    Beam beam;
    
    // load variables from start.00
    src.load(path);

    // overwrite number of rays
    src.NPOINT=nRays;

    // calculate source
    beam.genSource(&src);

    //OE     oe1;                   //TODO:: Load and trace OEs

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

