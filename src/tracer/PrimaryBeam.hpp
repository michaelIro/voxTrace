/** PrimaryBeam*/

#ifndef PrimaryBeam_H
#define PrimaryBeam_H

//#include <iostream>
//#include <vector>
//#include <armadillo>

//#include "./Ray.hpp"

#include "../api/Shadow3API.hpp"
#include "../api/PolyCapAPI.hpp"

class PrimaryBeam{

	private:
		Shadow3API shadowSource_;
        PolyCapAPI polyCap_;

	public:
  		PrimaryBeam();
		PrimaryBeam(char* shadowPath, char* polycapPath);

		Shadow3API getShadow3API();
};

#endif