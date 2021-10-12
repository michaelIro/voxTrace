#ifndef Tracer_H
#define Tracer_H

/**Tracer*/

#include <iostream>
#include <math.h> 
#include <stdlib.h>
#include <time.h>
#include <list>
#include <limits>

#include "xraylib.h"
#include "../base/Ray.h"
#include "../base/Sample.h"
#include "../setup/Source.h"
//#include "IUPAC.h"


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

