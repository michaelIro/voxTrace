#include <iostream>
#include "cuda/MyDummy.cuh"
#include "base/ChemElement.hpp"

int main() {
	std::cout << "START: Test-3" << std::endl;

    int N = 1<<20;
    float *x, *y;

    //CudaAccelerator::callAdd(N, x, y);
    MyDummy::callAdd(N, x, y);
    MyDummy::test();

    ChemElement cu(29);

    std::cout << "END: Test-3" << std::endl;
    return 0;
}


