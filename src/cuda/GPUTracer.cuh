
#ifndef GPU_TRACER_H
#define GPU_TRACER_H

#include "../base/Sample.hpp"
#include "../base/XRBeam.hpp"
//#include "../api/XRayLibAPI.hpp"
//#include "../base/Ray.hpp"
#include <iostream>
#include <math.h>

class GPUTracer{	

	public:
		static void callAdd(int N, float *x, float *y, Sample *sample, XRBeam *beam);

};

#endif
