/** Chemical Element */

#include "ChemElement.hpp"

using namespace std;

/** Empty constructor */
ChemElement::ChemElement(){}

/** Constructor using Atomic number Z 
 * @param z Atomic number
 * @return ChemElement containing information 
 */
ChemElement::ChemElement(const int& z){
	z_ =	z;
	sym_ =	XRayLibAPI::ZToSym(z_);
	a_ =	XRayLibAPI::A(z_);
	rho_ = 	XRayLibAPI::Rho(z_);
}

/** Constructor using element symbol used in periodic table 
 * @param symbol Element symbol used in periodic table 
 * @return ChemElement containing 
 */
ChemElement::ChemElement(const char *symbol){
	z_=			XRayLibAPI::SymToZ(symbol);
	sym_ =		symbol;
	a_ =		XRayLibAPI::A(z_);
	rho_ = 		XRayLibAPI::Rho(z_);
}

/*Getter-Functions for member variables*/
double ChemElement::A() const {return a_;}
int ChemElement::Z() const {return z_;}
double ChemElement::Rho() const {return rho_;}
const char* ChemElement::Sym() const {return sym_;}

/** Feed symbol to ostream / Overload comparison operators*/
ostream& operator<<(std::ostream& os, const ChemElement& el){return os<<el.Sym();}
bool operator<(const ChemElement& el1, const ChemElement& el2){return (el1.Z() < el2.Z());}
bool operator>(const ChemElement& el1, const ChemElement& el2){return (el1.Z() > el2.Z());}
bool operator==(const ChemElement& el1, const ChemElement& el2){return (el1.Z() == el2.Z());}

//Simple DB-Access functions TODO: Check if this "double link" for DB access function slows down considerably
/*returns total Mass attenuation coefficient [cm²/g]*/
double ChemElement::getMuMass(double energy) const { return XRayLibAPI::CS_Tot(z_,energy);}
double ChemElement::getFluorescenceYield(int  shell) const {return XRayLibAPI::FluorY(z_,shell);}
double ChemElement::getAugerYield(int shell) const {return XRayLibAPI::AugY(z_,shell);}
double ChemElement::getFluorescenceCrossSection(int shell, double energy) const {return XRayLibAPI::CS_FluorL(z_,shell,energy);}

double ChemElement::getLineEnergy(int line) const {
	return XRayLibAPI::LineE(z_,line*-1-1); //see xraylib and IUPAC for *-1-1
}

/** Calculate interaction type based on comparison with randim number
 * @param energy Energy of interacting Ray in keV
 * @param randomN Random number between 0 and 1
 * @return type of Interaction -> 0 = Photo-Effect / 1 = Rayleigh-Scattering / 2 = Compton-Scattering
*/
int ChemElement::getInteractionType(double energy, double randomN) const{
	
	double tot = XRayLibAPI::CS_Tot(z_,energy);
	double phot = XRayLibAPI::CS_Phot(z_,energy) / tot;
	double photRayleigh = (XRayLibAPI::CS_Phot(z_,energy) + XRayLibAPI::CS_Ray(z_,energy)) / tot;

	if(randomN <= phot) return 0;
	else if(randomN <= photRayleigh) return 1;
	else return 2;
}

/*Gives back The excited shell*/
int ChemElement::getExcitedShell(double energy, double randomN){

	int myShell_;
	double temp_= 0.;
	double sum_ = 0.;
	double cs_tot = XRayLibAPI::CS_Phot(z_,energy);

	if(cs_tot!=0.0)
	for(myShell_ = 0; myShell_ < 31; myShell_++){
		temp_= XRayLibAPI::CS_Phot_Part(z_, myShell_, energy) / cs_tot;
		sum_ += temp_;
		if(sum_ > randomN) break;
	}

	return myShell_;
}

int ChemElement::getTransition(int shell, double randomN){
	map<int,double> lineRatios;
	//IUPAC iupac;
	for(int myLine =getTransitionList(shell)[0]; myLine <= getTransitionList(shell)[1]; myLine++){
		int lineInput = myLine*-1-1;
		double ratio = RadRate(z_,lineInput,NULL);
		if(ratio!=0)		
			lineRatios.insert({myLine,ratio});
	}

	double mySum =0.;
	int myLine;
					
	for(auto const& line: lineRatios){
		//cout<<line.first<<" "<<line.second<<endl;		
		mySum += line.second;
		myLine = line.first;
		if(mySum>randomN) break; 	
	}
	return myLine;
}

map<int,double> ChemElement::getLineRatios(int shell){
	map<int,double> lineRatios;
	//IUPAC iupac;
	for(int myLine = getTransitionList(shell)[0]; myLine <= getTransitionList(shell)[1]; myLine++){
		int lineInput = myLine*-1-1;
		double ratio = RadRate(z_,lineInput,NULL);
		if(ratio!=0)		
			lineRatios.insert({myLine,ratio});
	}
	return lineRatios;
}

vector<int>  ChemElement::getTransitionList(int shell){
	vector<vector<int>> shell_lines_ = {
		{0,28},																		// K lines
		{29,57},{85,112},{113,135},													// L lines	
		{136,157},{158,179},{180,199},{200,218},									// M lines
		{219,236},{237,253},{254,269},{270,284},{285,298},{299,311},{312,323},		// N lines
		{321,334},{335,344},{345,353},{354,361},{362,368},{369,371},{372,373},		// O lines
		{374,377},{378,380},{381,382}												// P lines
	};

	return shell_lines_[shell];
}

/*TODO: Maybe Integral to unprecise*/
double ChemElement::getThetaCompt(double energy, double randomN){
		int stepsize = 200;
	int arraysize = (int) (M_PI/(1./((double)stepsize)))+1;

	double probSum=0.;
	double prob[arraysize];
	double theta[arraysize];

	double photLambda = 1. / (energy/12.39841930);


	double x =0.;
	for(int i=0; i<arraysize; i++){
		x = ((double)i)/((double)(stepsize));
		theta[i]= 2*asin(x*photLambda);
		prob[i] = DCS_Compt(z_,energy,theta[i],NULL);

		if((isnan(prob[i]))) break;
		if(i!=0) probSum += prob[i-1]*(cos(theta[i]) - cos(theta[i-1]));	
	}

  	double integral = abs(probSum*2*M_PI);
	//double expected = CS_Rayl(z_,energy,NULL);

	probSum =0.;
	int j;
	for(j=0; j<arraysize; j++){
		probSum += prob[j]/integral;
		if(probSum >randomN)break;
	}
	return theta[j];
}

double ChemElement::getThetaRayl(double energy, double randomN){
	int stepsize = 200;
	int arraysize = (int) (M_PI/(1./((double)stepsize)))+1;

	double probSum=0.;
	double prob[arraysize];
	double theta[arraysize];

	double photLambda = 1. / (energy/12.39841930);


	double x =0.;
	for(int i=0; i<arraysize; i++){
		x = ((double)i)/((double)(stepsize));
		theta[i]= 2*asin(x*photLambda);
		prob[i] = DCS_Rayl(z_,energy,theta[i],NULL);

		if((isnan(prob[i]))) break;
		if(i!=0) probSum += prob[i-1]*(cos(theta[i]) - cos(theta[i-1]));	
	}

  	double integral = abs(probSum*2*M_PI);
	//double expected = CS_Rayl(z_,energy,NULL);

	probSum =0.;
	int j;
	for(j=0; j<arraysize; j++){
		probSum += prob[j]/integral;
		if(probSum >randomN)break;
	}
	return theta[j];
}