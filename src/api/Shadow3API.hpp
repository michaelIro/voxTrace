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

#include <shadow_bind_cpp.hpp>

#include <armadillo>
#include <filesystem>
#include <vector>

class Shadow3API{
    private:
      Source src_;
      vector<OE> oe_;
      Beam beam_;

    public:
      Shadow3API();
      Shadow3API(char* path);

      arma::Mat<double> getBeam(int nRays);
      arma::rowvec getSingleRay();
};

#endif
