/*!
  API to XrayLib, a library for interactions of X-rays with matter, published by Tom Schoonjans.

  Documentation is available @ https://github.com/tschoonj/xraylib/wiki 
  
  Repositories can be found @ https://github.com/tschoonj/xraylib

  Copyright (c) 2009, Tom Schoonjans
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
      in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior 
      written permission.

  THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY 
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
  THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef XRayLibAPI_H
#define XRayLibAPI_H

#include "xraylib.h"

class XRayLibAPI{

    public:
        XRayLibAPI();

        static const char* ZToSym(int z);
        static double SymToZ(const char* symbol);
        static double A(int z);
        static double Rho(int z);

        static double CS_Tot(int z, double energy);
        static double CS_Phot(int z, double energy);
        static double CS_Ray(int z, double energy);
        
        static double CS_FluorL(int z, int shell, double energy);
        static double FluorY(int z, int shell);
        static double AugY(int z, int shell);

        static double LineE(int z, int line);

        static void test();


};

#endif
