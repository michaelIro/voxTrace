/*!
API to PolyCap, a C library to calculate X-ray transmission through polycapillaries, published by Pieter Tack, Tom Schoonjans and Laszlo Vincze.
  
  Documentation is available @  https://pietertack.github.io/polycap
  
  Repositories can be found @ https://github.com/PieterTack/polycap

  For more general usage this Interface also implements an Interface to Rays generated/traced with the Shadow3 code.
*/
#ifndef PolyCapAPI_H
#define PolyCapAPI_H

#include <polycap.h>

class PolyCapAPI{
  private: 
    polycap_error *error;

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