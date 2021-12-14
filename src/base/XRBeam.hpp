/** XRBeam */

#ifndef XRBeam_H
#define XRBeam_H

#include <iostream>
#include <vector>
#include <armadillo>

#include "./Ray.hpp"

class XRBeam{

	private:
		vector<Ray> rays_;

	public:
  		XRBeam();
		XRBeam(vector<Ray> rays);
		XRBeam(arma::Mat<double> rays);

		void shift(double x, double y, double z);
		void primaryTransform(double x0, double y0, double z0, double d, double alpha);
		void secondaryTransform(double x0, double y0, double z0, double d, double beta);

		static XRBeam probabilty(XRBeam beam);
		static XRBeam merge(vector<XRBeam> beams);

		vector<Ray> getRays() const;
		arma::Mat<double> getMatrix() const;

		void print() const;
};

#endif

