#ifndef XRSource_H
#define XRSource_H

// Source

#include <iostream>
#include <list>
#include "../api/Shadow3API.hpp"
#include "./Ray.hpp"

using namespace std;

class XRSource{

	private:
		double* position_;
		list<Ray> rayList_;

	public:
  		XRSource();
		XRSource(Shadow3API shadowSource);
		//XRSource(string path, double rayNum, double workingDistance, double spotSize);
		//XRSource(XRSource zeroSource, double x, double y, double z);

		list<Ray> getRayList() const;
		void print() const;
};

#endif

