/*XRayLib API*/
#include "XRayLibAPI.hpp"

const char* XRayLibAPI::ZToSym(int z){
    return XRayLib::AtomicNumberToSymbol(z, NULL);
}

double XRayLibAPI::SymToZ(const char* symbol){
    return XRayLib::SymbolToAtomicNumber(symbol, NULL);
}

double XRayLibAPI::A(int z){
    return XRayLib::AtomicWeight(z, NULL);
}

double XRayLibAPI::Rho(int z){
    return XRayLib::ElementDensity(z, NULL);
}

double XRayLibAPI::CS_Tot(int z, double energy){
    return XRayLib::CS_Total(z, energy, NULL);
}

double XRayLibAPI::CS_Phot(int z, double energy){
    return XRayLib::CS_Photo(z, energy, NULL);
}

double XRayLibAPI::CS_Ray(int z, double energy){
    return XRayLib::CS_Rayl(z, energy, NULL);
}

double XRayLibAPI::CS_Phot_Part(int z, int shell, double energy){
    return XRayLib::CS_Photo_Partial(z, shell, energy, NULL); 
}

double XRayLibAPI::CS_FluorL(int z, int shell, double energy){
    return XRayLib::CS_FluorLine(z,shell, energy,NULL);
}

double XRayLibAPI::FluorY(int z, int shell){
    return XRayLib::FluorYield(z,shell,NULL);
}

double XRayLibAPI::AugY(int z, int shell){
    return XRayLib::AugerYield(z,shell,NULL);
}

double XRayLibAPI::LineE(int z, int line){
    return XRayLib::LineEnergy(z, line, NULL);
}
double XRayLibAPI::RadRate(int z, int line){
    return XRayLib::RadRate(z, line, NULL);
}

double XRayLibAPI::DCS_Rayl(int z, double energy, double theta){
    return XRayLib::DCS_Rayl(z, energy, theta, NULL);
}

double XRayLibAPI::DCS_Compt(int z, double energy, double theta){
    return XRayLib::DCS_Compt(z, energy, theta, NULL);
}

void XRayLibAPI::test(){
    double a = XRayLib::CS_Photo_Partial(29,K_SHELL,17.4,NULL);
}