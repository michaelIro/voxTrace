//!  API to global optimization library/libraries (ensmallen, OptimLib, GSL, ...)
#ifndef OptimizerAPI_H
#define OptimizerAPI_H

#include <ensmallen.hpp>

#include <gsl/gsl_sf_bessel.h>

#define OPTIM_ENABLE_ARMA_WRAPPERS
#include "optim.hpp"

class OptimizerAPI{

    public:
        OptimizerAPI();

};

#endif