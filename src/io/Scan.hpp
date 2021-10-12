#ifndef Scan_H
#define Scan_H

/* ReadPoints */

#include <iostream>
#include <fstream>
#include <filesystem>
#include <regex>
#include <vector> 
#include <map>

//#include <armadillo>
//#include <algorithm>    // std::next_permutation, std::sort

#include "../base/Material.hpp"
#include "../base/ChemElement.hpp"
//#include "../setup/Sensitivity.h"


using namespace std;

class Scan{
	private:
		// Internal Variables
		vector<filesystem::path> paths_;
		vector<vector<int>> coordinates_;
		vector<vector<double>> motorpositions_;
		
		//External variables
		vector<vector<vector<Material>>> materials_;
		vector<ChemElement> elements_;
		vector<int> lines_;
		vector<double> lengths_;
		vector<vector<vector<int>>> spes_;
		vector<vector<vector<map<int,double>>>> intensities_; //counts per second
		vector<vector<vector<map<int,double>>>> uncertainties_;



	public:
		Scan ();
		Scan (filesystem::path folderPath);

		void readPaths(filesystem::path folderPath);
		void readPoints(filesystem::path pointsFilePath);
		void checkCoordOrder();
		void readSpectra();
		vector<double> readRoi(filesystem::path roiFilePath); 	//for now just to check wheter x,y,z are switched	
		void readFit(filesystem::path fitFilePath);

		void readSpectra(filesystem::path speFolderPath);

		//void intensities2Materials(Sensitivity sensitivity);
		
		//void readIntensities();
		//vector<string> getSpeFileNames() const;
		//vector<vector<int>> getCoordinates() const;
		//vector<vector<double>> getMotorpositions() const;
		//vector<vector<vector<double>>> getIntensities() const;
		//vector<vector<vector<double>>> getUncertainties() const;
		//vector<string> getLines() const;

		vector<vector<vector<Material>>> getMaterials() const;
		vector<ChemElement> getElements() const;
		vector<double> getLengths() const;

		void print() const;
};

#endif
