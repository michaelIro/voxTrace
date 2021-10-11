/*!
  API to PolyCap, a C library to calculate X-ray transmission through polycapillaries 

  PolyCap is a C library to calculate X-ray transmission through polycapillaries, published by Pieter Tack, Tom Schoonjans and Laszlo Vincze.
  
  Documentation: https://pietertack.github.io/polycap/ \n
  Repositories: https://github.com/PieterTack/polycap
  (August-2021)


*/
#ifndef PolyCapAPI_H
#define PolyCapAPI_H

#include <polycap.h>

class PolyCapAPI{
  private: 
    polycap_error *error;
	  polycap_profile *profile;
    polycap_description *description;
    polycap_source *source;
	  polycap_transmission_efficiencies *efficiencies;

  public:
    PolyCapAPI();
    void defineSource();
    void traceSource();
    void traceSinglePhoton();
};

#endif