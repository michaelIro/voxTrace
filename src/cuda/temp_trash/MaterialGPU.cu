/** Material */
#include "MaterialGPU.cuh"

/** Empty Constructor */
//__device__ Material::Material (){};

/** Standard Constructor
 * @param massFractions
 * @param rho  
 * @return Material   
 */
/*
Material::Material (thrust::host_vector<ChemElementGPU&> elements,thrust::host_vector<float> element_weights, float rho){
	elements_ = elements;
    element_weights_ = element_weights;
	rho_ = rho;
}*/

/** Get Density ρ of Material
 * @param void
 * @return Density in \f$\frac{g}{cm^3}\f$
 */
//float Material::Rho() const 

/** Get Total μ_Mass of Material
 * @param Energy of the Ray in keV
 * @return Total Mass absorption coefficient in ...
 */
/*float MaterialGPU::CS_Tot(float energy) const{
	//float mum_= 0.;
	//for(int i = 0; i < num_elements_; i++)
        //mum_ += elements_[i].CS_Tot(energy)*weights_[i];
	//	mum_ += rho_;
	return rho_;
}/*

/** Get Total μ_Lin of Material
 * @param energy of the ray in keV
 * @return Total Linear absorption coefficient in ...
 */
//__device__ float MaterialGPU::CS_Tot_Lin(float energy) const { 
//	return (CS_Tot(energy) * rho_);
//}

/** Get interacting Element
 * @param Energy of the Ray in keV
 * @param Random Number between 0 and 1 
 * @return Total Linear absorption coefficient in ...
 */
ChemElement& const Material::getInteractingElement(float energy, float randomN) const{

	float muMassTot = CS_Tot(energy);
	float sum = 0.;

	for(int i = 0; i < elements_.size(); i++){
		sum += elements_[i].CS_Tot(energy)*element_weights_/muMassTot;

		if(sum > randomN)
			return elements_[i];	
	}

	return elements_.back();
}

std::string Material::getName() const {
	std::string name;
	for(auto const& it: wi_){
		name.append(it.first->Sym());
		name.append(" ");
		name.append(std::__cxx11::to_string(it.second));
	}
	return name;
}

std::ostream& operator<<(std::ostream& os, Material& mat){return os<<mat.getName();}

void Material::print() const{
	for (auto& x: wi_) {
		std::cout << x.first << ":" << x.second << '\t';
    }  
	std::cout << "ρ:" << rho_;
	std::cout << '\n';
}
*/
