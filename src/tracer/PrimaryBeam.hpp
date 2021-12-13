/** PrimaryBeam*/

#ifndef PrimaryBeam_H
#define PrimaryBeam_H

#include "../api/Shadow3API.hpp"
#include "../api/PolyCapAPI.hpp"
#include "../base/XRBeam.hpp"

#include <omp.h>
#include <chrono>


class PrimaryBeam{

	private:
		Shadow3API* shadowSource_;
		PolyCapAPI* polyCap_;

	public:

  		PrimaryBeam();
		PrimaryBeam(Shadow3API* shadowSource, PolyCapAPI* polyCap);
};

#endif