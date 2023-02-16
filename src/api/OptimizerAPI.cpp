/// \dir src/api 
///      
/// \brief Brief description of the dir src/api goes here
/// 
/// \details A more detailed description goes here. 
///  

//! API to global optimization library/libraries (ensmallen, OptimLib, GSL, ...)
/*!
  An interface to 
*/

#include "OptimizerAPI.hpp"

// Ackley function
/*double evaluate(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    const double x = vals_inp(0);
    const double y = vals_inp(1);
    const double pi = arma::datum::pi;

    double obj_val = -20*std::exp( -0.2*std::sqrt(0.5*(x*x + y*y)) ) - std::exp( 0.5*(std::cos(2*pi*x) + std::cos(2*pi*y)) ) + 22.718282L;

    return obj_val;
}*/

/*Empty constructor*/
OptimizerAPI::OptimizerAPI(){

/*
 ///////////////////////////////////////////////////////////////////////////////////////////////////////
    double x0 = 5.0;
    double y0 = gsl_sf_bessel_J0 (x0);
    printf ("J0(%g) = %.18e\n", x0, y0);
 ///////////////////////////////////////////////////////////////////////////////////////////////////////
    arma::vec xn = arma::ones(2,1) + 1.0;

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

    bool success = optim::de(xn,evaluate,nullptr);
    //bool success2 = optim::pso(xn,evaluate,nullptr);

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    if (success) {
        std::cout << "de: Ackley test completed successfully.\n"
                  << "elapsed time: " << elapsed_seconds.count() << "s\n";
    } else {
        std::cout << "de: Ackley test completed unsuccessfully." << std::endl;
    }

    arma::cout << "\nde: solution to Ackley test:\n" << xn << arma::endl;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
*/
    
    //ens::RosenbrockFunction f;
    //ens::DE         optimizerDE(200, 1000, 0.6, 0.8, 1e-5);
    //ens::SA<>       optimizerSA(ens::ExponentialSchedule(), 1000000, 1000., 1000, 100, 1e-10, 3, 1.5, 0.5, 0.3);
    //ens::CNE        optimizerCNE(200, 10000, 0.2, 0.2, 0.3, 1e-5);
    //ens::LBestPSO   optimizerPSO(64, 50, 60, 3000, 400, 1e-30, 2.05, 2.05);
    //ens::SPSA       optimizerSPSA(0.1, 0.102, 0.16, 0.3, 100000, 1e-5);
    //optimizer.Optimize(ackley_fn,xn);
    
}


