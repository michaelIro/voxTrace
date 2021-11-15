#ifndef Detector_H
#define Detector_H

// Detector

#include <iostream>
#include "ChemElement.h"
#include "../base/Ray.h"
#include "../base/Spectrum.h"

using namespace std;

class Detector {

	private:
		ChemElement detMat_;
		ChemElement winMat_;

		vector<double> channelEnergies_; 
		//int channels_;
		//double* windowPosition_;
		//double* detectorPosition_;
		//double activeDetectorArea_;
		//double gain_;
		//double zero_;
		//double fano_;
		//double noise_;
		//double liveTime_;
		//double pulseWidth_;
		
		

	public:
  		Detector();
		Detector(ChemElement detectorMaterial, ChemElement windowMaterial);

		Spectrum detect(vector<Ray> rays);
};

#endif

