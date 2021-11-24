#ifndef XRSource_H
#define XRSource_H

#include <iostream>
#include <vector>
#include "./Ray.hpp"

//#include "../api/Shadow3API.hpp"
//#include "../api/PolyCapAPI.hpp"

using namespace std;

class XRSource{

	private:
		double position_;
		vector<Ray> rayList_;

	public:
  		XRSource();
		XRSource(vector<Ray> beam, double x0, double y0, double d, double alpha);
		XRSource(XRSource zeroSource, double x, double y, double z);

		vector<Ray> getRayList() const;
		void print() const;
};

#endif

