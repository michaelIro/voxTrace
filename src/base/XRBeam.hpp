#ifndef XRSource_H
#define XRSource_H

#include <iostream>
#include <vector>
#include "./Ray.hpp"

//#include "../api/Shadow3API.hpp"
//#include "../api/PolyCapAPI.hpp"

class XRBeam{

	private:
		vector<Ray> rayList_;

	public:
  		XRBeam();
		XRBeam(vector<Ray> beam);
		XRBeam(XRBeam zeroSource, double x, double y, double z);
		
		void primaryTransform(double x0, double y0, double z0, double d, double alpha);
		void secondaryTransform(double x0, double y0, double z0, double d, double alpha);

		vector<Ray> getRays() const;
		void print() const;
};

#endif

