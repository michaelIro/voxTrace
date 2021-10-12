/*Material*/

#include "Material.hpp"

using namespace std;

/*Empty constructor*/
Material::Material (){}

/*Regular constructor*/
Material::Material (map<int,double> massFractions, double rho){
	wi_ = massFractions;
	rho_ = rho;
}

/*Member getter*/
map<int,double> Material::getMassFractions() const {return wi_;}
double Material::getRho() const{ return rho_; }

/*Get total Mass absorption coefficient */
double Material::getMuMass(double energy, vector<ChemElement> elements) const{
	double mum_= 0.;
	for(auto const& it: wi_){
		mum_ += elements[it.first].getMuMass(energy)*it.second;
		//cout<<it.first.getMuMass(energy)<<"*"<<it.second<<"="<<it.first.getMuMass(energy)*it.second<<" -> "<<muMass<<endl;
		//cout<<"Density"<<rho_<<" "<<it.first.getRho()<<endl;
	}
	return mum_;
}

/**/
double Material::getMuLin(double energy, vector<ChemElement> elements) const{ return (getMuMass(energy, elements) * rho_);}

/**/
ChemElement Material::getInteractingElement(double energy, double randomN, vector<ChemElement> elements) const{

	double muMassTot = getMuMass(energy,elements);
	double sum = 0.;

	for(auto const& it: getMassFractions()){
		sum += elements[it.first].getMuMass(energy)*it.second /muMassTot;

		if(sum > randomN)
			return it.first;	
	}

	return elements[getMassFractions().end()->first];
}

void Material::setRho(double rho){rho_=rho;}

string Material::getName(vector<ChemElement> elements) const {
	string name;
	for(auto const& it: wi_){
		name.append(elements[it.first].getSymbol());
		name.append(std::to_string(it.second));
	}
	return name;
}

string Material::getName() const {
	string name;
	for(auto const& it: wi_){
		name.append(to_string(it.first));
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

