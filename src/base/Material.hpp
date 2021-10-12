#ifndef Material_H
#define Material_H

/*Material*/

#include <iostream>
#include <string>
#include <map>
#include <vector>

#include "../base/ChemElement.hpp"

using namespace std;

class Material {
	private:
		map<int,double> wi_;
		double rho_;
		
	public:
  		Material();
  		Material(map<int,double> massFractions, double rho);

  		map<int,double> getMassFractions() const;
		double getRho() const;

		double getMuMass(double energy, vector<ChemElement> elements) const;
		double getMuLin(double energy, vector<ChemElement> elements) const;

		ChemElement getInteractingElement(double energy, double randomN, vector<ChemElement> elements) const;

		void setRho(double rho);

		string getName(vector<ChemElement> elements) const;
		string getName() const;
		void print() const;
};

ostream& operator<<(std::ostream&, Material&);

#endif