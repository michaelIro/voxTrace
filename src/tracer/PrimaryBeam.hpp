/** PrimaryBeam*/

#ifndef PrimaryBeam_H
#define PrimaryBeam_H

#include "../api/Shadow3API.hpp"
#include "../api/PolyCapAPI.hpp"
#include "../base/XRBeam.hpp"

class PrimaryBeam{

	private:
		Shadow3API* shadowSource_;
		PolyCapAPI* polyCap_;

	public:

  		PrimaryBeam();
		PrimaryBeam(Shadow3API* shadowSource, PolyCapAPI* polyCap);

		//Shadow3API getShadow3API();
};

#endif