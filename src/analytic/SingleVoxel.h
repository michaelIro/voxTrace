#ifndef SingleVoxel_H
#define SingleVoxel_H

// SingleVoxel

#include <iostream>
#include <armadillo>
#include "./Calibration.h"
#include "../base/ChemElement.h"

using namespace std;

class SingleVoxel {

	private:

		Calibration calib_;					// Calibration parameter of the setup

		arma::mat alpha_;					// Matrix with Rows: Z | X-Ray-Line | Intensity | (Weighted Intensity)
		arma::mat beta_;					// Matrix with Rows: Z | X-Ray-Line | Intensity | (Weighted Intensity)

		arma::mat bulk_;					// Matrix with Rows: Z | Weight-Concentration
		arma::vec invisible_;				// Vec with Entries: Z
		
		arma::vec elements_;				// Unique elements of (Alpha + Beta + Bulk + Invisible)
		arma::vec rhos_;					// Element Densities of unique Elements			

		arma::rowvec excitation_;			// Linear Attenuation coefficients of elements for Excitation Energy

		arma::mat invisibleInformation_;	// additional rows for Lines that are produced but absorbed



		int useBeta_;						// 0 = no; 1 = yes; 
		int useBulk_;						
		int useInvisible_;
		double penetrationDepth_;
		int optType_; 						// 0 = rho & w; 1 = rho; 2 = w; 3 = D in max, 4 = D & x 
		int stepNormType_;
		int rhoGuessType_;
		int wGuessType_;
		int boundType_;


		arma::mat matrix_;		// matrix for optimization
		arma::mat bound_;		// boundary conditions for optimization
		//double rhoGuess_;
		//arma::vec wGuess_;

		arma::vec getUniqueElements(arma::mat alpha, arma::mat beta, arma::vec invisible, arma::mat bulk);
		arma::Mat<double> prepareMatrix(arma::Mat<double> lines, Calibration calib);
		arma::rowvec getExcitationVec(arma::vec excitationLine);
		
	public:
  		SingleVoxel();
		SingleVoxel(arma::mat alpha, arma::mat beta, arma::vec invisible, arma::mat bulk, Calibration calib, arma::vec excitationLine);



		//void setWeights();
		//double getMuLinTot(arma::vec weights);

		void checkInvisible(arma::vec invisible, Calibration calib, arma::vec excitationLine);
		arma::mat excLines();


		void makeMatrix(int optType, bool useBeta, bool useInvisible, double x);
		void makeWeightGuess(int weightGuess, bool useBulk);
		void makeRhoGuess(int rhoGuess);

		void makeBoundaries( bool useBulk, int boundType);
		void makeBoundaries();

		arma::mat updateMatrix(int type, arma::vec current);

		void makeProblem(int optType, bool stepnorm, bool useBeta, bool useBulk, bool useInvisible, double x, int rhoGuess, int wGuess, int boundType);	
		void makeProblem(int optType, bool stepnorm, bool useBeta, bool useBulk, bool useInvisible, double x, int wGuess, int boundType, double rho);
		void makeProblem(int optType, bool stepnorm, bool useBeta, bool useBulk, bool useInvisible, double x, int rhoGuess, int boundType, arma::vec w);
		void makeProblem(int optType, bool stepnorm, bool useBeta, bool useBulk, bool useInvisible, double x, int boundType, double rho,  arma::vec w);
								
		arma::mat getMatrix();
		arma::mat getBoundaries();
		arma::vec getElements();


		int getBetaUsed();
		int getBulkUsed();
		int getInvisibleUsed();
		double getPenetrationDepth();
		int getOptType();
		int getStepNormType();
		int getWGuessType();
		int getRhoGuessType();
		int getBoundType();
		
		int setBetaUsed();
		int setBulkUsed();
		int setInvisibleUsed();
		double setPenetrationDepth();
		int setOptType();
		int setStepNormType();
		int setWGuessType();
		int setRhoGuessType();
		int setBoundType();

		void print();
		void printAlpha();
		void printBeta();
		void printBulk();
		void printInvisible();

		



};

#endif

