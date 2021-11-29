/** Material */
#include "Material.hpp"

/** Empty Constructor */
Material::Material (){}

/** Standard Constructor
 * @param massFractions
 * @param rho  
 * @return Material   
 */
Material::Material (map<ChemElement* const,double> massFractions, double rho){
	wi_ = massFractions;
	rho_ = rho;
}

/** Getter function for 
 * @return a 
 */
map<ChemElement* const,double> Material::getMassFractions() const {
	return wi_;
}

/** Get Density ρ of Material
 * @param
 * @return Density in ...
 */
double Material::getRho() const {
	return rho_; 
}

/** Get Total μ_Mass of Material
 * @param Energy of the Ray in keV
 * @return Total Mass absorption coefficient in ...
 */
double Material::getMuMass(double energy) const{
	double mum_= 0.;
	for(auto it: wi_)
		mum_ += it.first->getMuMass(energy)*it.second;
	return mum_;
}

/** Get Total μ_Lin of Material
 * @param energy of the ray in keV
 * @return Total Linear absorption coefficient in ...
 */
double Material::getMuLin(double energy) const { 
	return (getMuMass(energy) * rho_);
}

/** Get interacting Element
 * @param Energy of the Ray in keV
 * @param Random Number between 0 and 1 
 * @return Total Linear absorption coefficient in ...
 */
ChemElement* const Material::getInteractingElement(double energy, double randomN) const{

	double muMassTot = getMuMass(energy);
	double sum = 0.;

	for(auto it: getMassFractions()){
		sum += it.first->getMuMass(energy)*it.second /muMassTot;

		if(sum > randomN)
			return it.first;	
	}

	return getMassFractions().end()->first;
}

void Material::setRho(double rho){rho_=rho;}

string Material::getName() const {
	string name;
	for(auto const& it: wi_){
		name.append(it.first->Sym());
		name.append(" ");
		name.append(to_string(it.second));
	}
	return name;
}

ostream& operator<<(std::ostream& os, Material& mat){return os<<mat.getName();}

void Material::print() const{
	for (auto& x: wi_) {
		cout << x.first << ":" << x.second << '\t';
    }  
	cout << "ρ:" << rho_;
	cout << '\n';
}

