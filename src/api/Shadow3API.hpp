/*!
  API to Shadow3, an open source ray tracing code for modeling optical systems, maintained by Manuel Sanchez del Rio. 

  Documentation is available @ https://github.com/oasys-kit/shadow3 
  
  Repositories can be found @ https://github.com/PaNOSC-ViNYL/shadow3/tree/gfortran8-fixes.
  
  For ease of use / a better workflow a GUI is available, as Shadow3 is part of the OASYS (OrAnge SYnchrotron Suite) project, an open-source Graphical 
  Environment for optics simulation softwares used in synchrotron facilities, based on Orange 3. Installation instructions can be found @ http://ftp.esrf.eu/pub/scisoft/Oasys/readme.html. 
  However the C++-API, which voxTrace depends on, is not auomatically built/installed with this installation process (see #Installation).
*/
#ifndef Shadow3API_H
#define Shadow3API_H

//namespace myNameSpace
//{
//using namespace std;
#include <shadow_bind_cpp.hpp>
//}


#include <armadillo>
#include <filesystem>

class Shadow3API{
    private:
      Shadow3::Source src_;
      std::vector<Shadow3::OE> oe_;
      Shadow3::Beam beam_;

    public:
      Shadow3API() = delete;
      Shadow3API(char* path);
      //Shadow3API(Shadow3API* shadow3api);

      void trace(int nRays);
      void trace(int nRays, int seed);

      arma::Mat<double> getBeamMatrix();
      arma::Mat<double> getBeamMatrix(std::vector<Shadow3::Beam>* beams);

      Shadow3::Source get_src_();
      std::vector<Shadow3::OE> get_oe_();   
      Shadow3::Beam get_beam_();
};

#endif
