/*XRayLib API*/
#include "XRayLibAPI.hpp"

/*Empty constructor*/
XRayLibAPI::XRayLibAPI(){}

const char* XRayLibAPI::ZToSym(int z){
    return AtomicNumberToSymbol(z, NULL);
}

double XRayLibAPI::SymToZ(const char* symbol){
    return SymbolToAtomicNumber(symbol, NULL);
}

double XRayLibAPI::A(int z){
    return AtomicWeight(z, NULL);
}

double XRayLibAPI::Rho(int z){
    return ElementDensity(z, NULL);
}

double XRayLibAPI::CS_Tot(int z, double energy){
    return CS_Total(z, energy, NULL);
}

double XRayLibAPI::CS_Phot(int z, double energy){
    return CS_Photo(z, energy, NULL);
}

double XRayLibAPI::CS_Ray(int z, double energy){
    return CS_Rayl(z, energy, NULL);
}

double XRayLibAPI::LineE(int z, int line){
    return LineEnergy(z, line, NULL);
}

void XRayLibAPI::test(){
    double a = CS_Photo_Partial(29,K_SHELL,17.4,NULL);
}
