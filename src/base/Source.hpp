#ifndef Source_H
#define Source_H

// Source

#include <iostream>
#include "../api/Shadow3API.hpp"
#include "./Ray.hpp"

using namespace std;

class Source {

	private:
		double* position_;
		list<Ray> rayList_;

	public:
  		Source();
		Source(string path, double rayNum, double workingDistance, double spotSize);
		Source(Source zeroSource, double x, double y, double z);

		list<Ray> getRayList() const;
		void print() const;
};

#endif

