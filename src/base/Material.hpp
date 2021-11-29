#ifndef Material_H
#define Material_H

/*Material*/

#include <iostream>
#include <string>
#include <map>

#include "../base/ChemElement.hpp"

class Material {
	private:
		map<ChemElement* const,double> wi_;
		double rho_;
		
	public:
  		Material();
  		Material(map<ChemElement* const,double> massFractions, double rho);

  		map<ChemElement* const,double> getMassFractions() const;
		double getRho() const;

		double getMuMass(double energy) const;
		double getMuLin(double energy) const;

		ChemElement* const getInteractingElement(double energy, double randomN) const;

		void setRho(double rho);

		string getName() const;
		void print() const;
};

ostream& operator<<(std::ostream&, Material&);

#endif