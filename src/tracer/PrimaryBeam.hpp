/** PrimaryBeam*/

#ifndef PrimaryBeam_H
#define PrimaryBeam_H

#include "../api/Shadow3API.hpp"
#include "../api/PolyCapAPI.hpp"
#include "../base/XRBeam.hpp"

#include <omp.h>
#include <chrono>
#include <thread> 


class PrimaryBeam{

	private:
		//Shadow3API& shadowSource_;
		//PolyCapAPI& polyCap_;

	public:

  		PrimaryBeam() = delete;
		PrimaryBeam(Shadow3API& shadowSource, PolyCapAPI& polyCap, int job_id, int rand_seed, int n_sh_rays, int n_iter, int n_files);
};

#endif