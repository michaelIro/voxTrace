#ifndef OptimizerAPI_H
#define OptimizerAPI_H

#include <functional>
#include <vector>
#include <string>
#include <stdexcept>

using ObjFn     = std::function<double(const std::vector<double>&)>;
using ObjGradFn = std::function<double(const std::vector<double>&, std::vector<double>&)>;

enum class Algorithm { DE, LBFGS, NelderMead };

struct OptimizerConfig {
    size_t maxIter            = 1000;
    double tolerance          = 1e-5;
    double stepSize           = 0.1;   // NelderMead simplex / LBFGS step
    size_t populationSize     = 100;   // DE
    double crossoverRate      = 0.5;   // DE CR ∈ [0,1]
    double differentialWeight = 0.8;   // DE F  ∈ [0.4,1.0]
};

struct OptimizerResult {
    std::vector<double> x;
    double              fval;
    size_t              iters;
    bool                converged;
    std::string         algorithm;
};

class OptimizerAPI {
public:
    /** DE and NelderMead: gradient-free */
    static OptimizerResult Minimize(Algorithm algo, ObjFn obj,
                                    std::vector<double> x0,
                                    const OptimizerConfig& cfg = {});

    /** L-BFGS: combined f+grad */
    static OptimizerResult Minimize(Algorithm algo, ObjGradFn objGrad,
                                    std::vector<double> x0,
                                    const OptimizerConfig& cfg = {});
};

#endif
