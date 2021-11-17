/* Shadow 3 API*/
#include "Shadow3API.hpp"

/* Empty constructor */
Shadow3API::Shadow3API(){}

/** Constructor loading start.00 File from given Path to define Source.
* @param path Path to folder that contains at least a start.00 file defining the source and possibly start.** files defining th eoptical elements. (Shadow3)
* @return Shadow3API-Object
*/
Shadow3API::Shadow3API(char* path){
    for (const auto & entry : filesystem::directory_iterator(path)){
        if(entry.path().filename() == "start.00")
            src_.load((char *) entry.path().c_str());             // load variables from start.00

        else{
            // TODO: Load optical elements     
            //OE     oe1;                   //TODO:: Load and trace OEs

            // load start.01 into oe1
            //oe1.load( (char*) "start.01");
        }
    }   
}

/** Generates X-Rays from a Shadow3-Source and if present trace the generated rays through optical elements. 
 * @param nRays Number of Rays to be generated
 * @return arma::Mat<double> with Rays generated from Source
 */
arma::Mat<double> Shadow3API::getBeam(int nRays){
    // overwrite number of rays
    src_.NPOINT=nRays;

    // calculate source
    beam_.genSource(&src_);

    // trace through optical shadow elements if present
    if(!oe_.empty()){
        for(int i = 0; i < oe_.size(); i++ )
            beam_.traceOE(&(oe_[i]),1);         // traces OE1
    }

    // write rays to arma::mat
    arma::Mat<double> rays = arma::ones(src_.NPOINT, src_.NCOL);
    for(int i = 0; i < rays.n_rows; i++)
        for(int j = 0; j < rays.n_cols; j++)
            rays(i,j) = (*(beam_.rays+i*18+j));

    return rays;
}

/* Empty constructor */
arma::rowvec Shadow3API::getSingleRay(){
    src_.NPOINT=1;  
    beam_.genSource(&src_);
    arma::rowvec ray_(src_.NCOL);
    for(int j = 0; j < ray_.n_elem; j++)
            ray_(j) = (*(beam_.rays+j));
    return ray_;
}