//! API to PolyCap, a C library to calculate X-ray transmission through polycapillaries 
/*!
  PolyCap is a C library to calculate X-ray transmission through polycapillaries, published by Pieter Tack, Tom Schoonjans, Laszlo Vincze.
  Documentation is available @ https://pietertack.github.io/polycap/ and repositories can be found @ https://github.com/PieterTack/polycap (August-2021).
*/
#ifndef PolyCap_H
#define PolyCap_H

#include <polycap.h>

class PolyCapAPI{

    public:
        PolyCapAPI();
        void something();
        void defineSource();
};

#endif