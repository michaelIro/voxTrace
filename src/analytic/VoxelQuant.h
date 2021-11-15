#ifndef VoxelQuant_H
#define VoxelQuant_H

// Confocal Volume

#include <iostream>
#include <vector>
#include <map>
#include <armadillo>
#define OPTIM_ENABLE_ARMA_WRAPPERS
#include "optim.hpp"

#include "../base/ChemElement.h"
#include "../base/Material.h"
#include "Sensitivity.h"
#include "PolyCap.h"
#include "xraylib.h"
#include "./Calibration.h"
#include "./SingleVoxel.h"

using namespace std;

class VoxelQuant {

	SingleVoxel vox_;

	arma::mat m_;
	arma::mat b_;
	arma::vec v_;

	optim::algo_settings_t s_;

  public:
  	VoxelQuant();
  	VoxelQuant(SingleVoxel voxel);

	static double evaluate(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data);

	void makeProblem(int optType, bool stepnorm, bool useBeta, bool useBulk, bool useInvisible, double x, int rhoGuess, int wGuess, int boundType);

	void fetchProblem();
	void printProblem();
	//void set(bool bound);

	void run();
	void printSolution();

};

#endif

