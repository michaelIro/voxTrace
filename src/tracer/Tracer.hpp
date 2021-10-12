#ifndef Tracer_H
#define Tracer_H

/**Tracer*/

#include <iostream>
#include <math.h> 
#include <stdlib.h>
#include <time.h>
#include <list>
#include <limits>

#include "../api/XRayLibAPI.hpp"

#include "../base/Ray.hpp"
#include "../base/Sample.hpp"
#include "../setup/Source.hpp"


using namespace std;

class Tracer {

	private:
		//Source source_;
		//Sample sample_;
		
	public:
  		Tracer();
		Tracer(Source source, Sample sample);

		void start();

		/*Trace the Path of a single ray.*/
		Ray* traceForward(Ray* ray, Voxel* currentVoxel, int* nextVoxel, Sample *sample);
};

#endif