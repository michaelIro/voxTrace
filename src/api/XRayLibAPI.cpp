/*XRayLib API*/
#include "XRayLibAPI.hpp"

/*Empty constructor*/
XRayLibAPI::XRayLibAPI(){}

const char* XRayLibAPI::AToSym(int z){
    return AtomicNumberToSymbol(z, NULL);
}

double XRayLibAPI::SymToA(const char* symbol){
    return SymbolToAtomicNumber(symbol, NULL);
}

double XRayLibAPI::A(int z){
    return AtomicWeight(z, NULL);
}

double XRayLibAPI::Rho(int z){
    return ElementDensity(z, NULL);
}

void XRayLibAPI::test(){
    double a = CS_Photo_Partial(29,K_SHELL,17.4,NULL);
}
