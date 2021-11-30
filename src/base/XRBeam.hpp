#ifndef XRSource_H
#define XRSource_H

#include <iostream>
#include <vector>
#include <armadillo>

#include "./Ray.hpp"

//#include "../api/Shadow3API.hpp"
//#include "../api/PolyCapAPI.hpp"

class XRBeam{

	private:
		vector<Ray> rayList_;

	public:
  		XRBeam();
		XRBeam(vector<Ray> rays);
		
		void shift(double x, double y, double z);
		void primaryTransform(double x0, double y0, double z0, double d, double alpha);
		void secondaryTransform(double x0, double y0, double z0, double d, double beta);

		vector<Ray> getRays() const;
		arma::Mat<double> getMatrix() const;

		void print() const;
};

#endif

