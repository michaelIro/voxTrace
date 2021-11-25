#ifndef Tracer_H
#define Tracer_H

/**Tracer*/

#include <iostream>
#include <math.h> 
#include <stdlib.h>
#include <time.h>
#include <list>
#include <limits>

#include <omp.h>

#include "../api/XRayLibAPI.hpp"

#include "../base/Ray.hpp"
#include "../base/Sample.hpp"
#include "../base/XRBeam.hpp"


using namespace std;

class Tracer {

	private:
		XRBeam source_;
		Sample sample_;
		std::vector<Ray> beam_;
		
	public:
  		Tracer();
		Tracer(XRBeam source, Sample sample);

		void start();

		/*Trace the Path of a single ray.*/
		Ray* traceForward(Ray* ray, Voxel* currentVoxel, int* nextVoxel, Sample* sample, int* ia);
		std::vector<Ray> getBeam();
};

#endif