/* Shadow 3 API*/
#include "Shadow3API.hpp"

/* Empty constructor */
//Shadow3API::Shadow3API(){}

/** Constructor loading start.00 File from given Path to define Source.
* @param path Path to folder that contains at least a start.00 file defining the source and possibly start.** files defining the optical elements. (Shadow3)
* @return Shadow3API-Object
*/
Shadow3API::Shadow3API(char* path){
    for (const auto & entry : filesystem::directory_iterator(path)){
        if(entry.path().filename() == "start.00")
            src_.load((char *) entry.path().c_str());             // load variables from start.00

        else{
            //std::cout<< entry.path().extension().string().substr(1) << std::endl; // TODO: Check if file is really start.** & Check correct order of OE
            
            // load start.** into oe
            OE     oe__;
            oe__.load((char *) entry.path().c_str());
            oe_.push_back(oe__);
        }
    }   
}

/** Copy Constructor
* @param shadow3api 
* @return Shadow3API-Object
*/
Shadow3API::Shadow3API(Shadow3API* shadow3api){
    src_ = (*shadow3api).get_src_();
    //oe_ = (*shadow3api).get_oe_(); // TODO: Check if NULL here
    //beam_ = (*shadow3api).get_beam_();
}

/** Generates X-Rays from a Shadow3-Source and if present trace the generated rays through optical elements. 
 * @param nRays Number of Rays to be generated
 * @return void -> Result is written to beam_ 
 */
void Shadow3API::trace(int nRays){
    // overwrite number of rays
    src_.NPOINT=nRays;
    
    stringstream ss;
    ss << "Seed: " << src_.ISTAR1 <<"\n";
    //string str = ss.str();
    //std::string =  + 
    //std::cout << "Seed: " << src_.ISTAR1 << std::endl;
    std::cout <<  ss.str();

    // calculate source
    beam_.genSource(&src_);

    // trace through optical shadow elements if present
    if(!oe_.empty()){
        for(int i = 0; i < oe_.size(); i++ )
            beam_.traceOE(&(oe_[i]),1);         // traces OE1
    }
}

/** Generates X-Rays from a Shadow3-Source and if present trace the generated rays through optical elements. 
 * @param nRays Number of Rays to be generated
 * @param seed 
 * @return void -> Result is written to beam_ 
 */
void Shadow3API::trace(int nRays, int seed){
    src_.ISTAR1=seed;
    trace(nRays);
}

/** Generates X-Rays from a Shadow3-Source and if present trace the generated rays through optical elements.
 * Each row has the values: x0,y0,z0,   xd,yd,zd,   asx,asy,asz,    flag,k,index,   opd,fs,fp,  apx,apy,apz
 * @param nRays Number of Rays to be generated
 * @return arma::Mat<double> with Rays generated from Source
 */
arma::Mat<double> Shadow3API::getBeamMatrix(){
    // write rays to arma::mat
    arma::Mat<double> rays = arma::ones(src_.NPOINT, 18);
    for(int i = 0; i < rays.n_rows; i++)
        for(int j = 0; j < rays.n_cols; j++)
            rays(i,j) = (*(beam_.rays+i*18+j));

    return rays;
}

/** Generates X-Rays from a Shadow3-Source and if present trace the generated rays through optical elements.
 * Each row has the values: x0,y0,z0,   xd,yd,zd,   asx,asy,asz,    flag,k,index,   opd,fs,fp,  apx,apy,apz
 * @param nRays Number of Rays to be generated
 * @return arma::Mat<double> with Rays generated from Source
 */
arma::Mat<double> Shadow3API::getBeamMatrix(vector<Beam>* beams){

    // write rays to arma::mat
    arma::Mat<double> rays = arma::ones(src_.NPOINT*(*beams).size(), 18); //TODO: make different beam sizes possible
    for(int k = 0; k< (*beams).size(); k++)
        for(int i = 0; i < rays.n_rows; i++)
            for(int j = 0; j < rays.n_cols; j++)
                rays(i,j+k*(*beams).size()) = (*((*beams)[k].rays+i*18+j));

    return rays;
}

/** Generates X-Rays from a Shadow3-Source and if present trace the generated rays through optical elements.
 * Each row has the values: x0,y0,z0,   xd,yd,zd,   asx,asy,asz,    flag,k,index,   opd,fs,fp,  apx,apy,apz
 * @param nRays Number of Rays to be generated
 * @return Beam
 */
Beam Shadow3API::get_beam_(){
    return beam_;
}

Source Shadow3API::get_src_(){
    return src_;
}

std::vector<OE> Shadow3API::get_oe_(){
    return oe_;
}