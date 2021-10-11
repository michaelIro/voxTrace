#ifndef ChemElement_H
#define ChemElement_H

/*Chemical Element*/

#include <iostream>
#include <map>
#include <vector>
//#include <list>
#include <math.h>
#include "xraylib.h"
//#include "IUPAC.h"

using namespace std;

class ChemElement {
	private:
		const char* symbol_;	// Chemical Symbol
		int z_;					// Atomic number
		double a_;				// Atomic Mass number [g/mol]
		double rho_;			// Density at room temperature [g/cm³]

  	public:
  		ChemElement();
  		ChemElement(const char *symbol);
		ChemElement(const int& z);
		
		/*Member-Getter*/
		double getA() const;
		int getZ() const;
		double getRho() const;	
		const char* getSymbol() const;

		/*Simple DB-Access*/
		double getMuMass(double energy) const;
		double getFluorescenceYield(int shell) const;
		double getAugerYield(int shell) const;
		double getLineEnergy(int line) const;
		double getFluorescenceCrossSection(int shell, double energy) const;	

		/*Decisions with random number*/
		int getInteractionType(double energy, double randomN) const;

		double getTheta(double energy, double randomN, int interactionType, double phi);
		double getThetaCompt(double energy, double randomN);
		double getThetaRayl(double energy, double randomN);

		/*Shell*/
		int getExcitedShell(double energy, double randomN);

		/*Transition*/
		map<int,double> getLineRatios(int shell);
		int getTransition(int shell, double randomN);


		static vector<int> getTransitionList(int shell);

};

ostream& operator<<(std::ostream&, const ChemElement&);

bool operator<(const ChemElement&, const ChemElement&);
bool operator>(const ChemElement&, const ChemElement&);
bool operator==(const ChemElement&, const ChemElement&);


#endif

