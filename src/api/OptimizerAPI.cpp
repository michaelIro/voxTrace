#include "OptimizerAPI.hpp"
#include <ensmallen.hpp>
#include <gsl/gsl_multimin.h>

// ── Ensmallen adapters ────────────────────────────────────────────────────────

namespace {

std::vector<double> fromArma(const arma::mat& m) {
    return { m.memptr(), m.memptr() + m.n_elem };
}
arma::rowvec toArma(const std::vector<double>& v) {
    return arma::rowvec(v.data(), v.size());
}

struct ArbAdapter {      // for DE (gradient-free)
    ObjFn fn;
    double Evaluate(const arma::mat& x) const { return fn(fromArma(x)); }
};

struct GradAdapter {     // for L-BFGS
    ObjGradFn fn;
    mutable std::vector<double> gbuf;
    double EvaluateWithGradient(const arma::mat& x, arma::mat& g) const {
        auto xv = fromArma(x);
        gbuf.resize(xv.size());
        double val = fn(xv, gbuf);
        g = toArma(gbuf);
        return val;
    }
    double Evaluate(const arma::mat& x) const {
        auto xv = fromArma(x); gbuf.resize(xv.size()); return fn(xv, gbuf);
    }
    void Gradient(const arma::mat& x, arma::mat& g) const {
        EvaluateWithGradient(x, g);
    }
};

} // namespace

// ── DE ────────────────────────────────────────────────────────────────────────

static OptimizerResult run_DE(ObjFn obj, std::vector<double> x0,
                               const OptimizerConfig& cfg)
{
    ArbAdapter adapter{std::move(obj)};
    arma::mat x = toArma(x0);
    ens::DE opt(cfg.populationSize, cfg.maxIter,
                cfg.crossoverRate, cfg.differentialWeight, cfg.tolerance);
    double fval = opt.Optimize(adapter, x);
    return { fromArma(x), fval, cfg.maxIter, true, "DE" };
}

// ── L-BFGS ────────────────────────────────────────────────────────────────────

static OptimizerResult run_LBFGS(ObjGradFn fn, std::vector<double> x0,
                                  const OptimizerConfig& cfg)
{
    GradAdapter adapter{std::move(fn)};
    arma::mat x = toArma(x0);
    ens::L_BFGS opt;
    opt.MaxIterations()   = cfg.maxIter;
    opt.MinGradientNorm() = cfg.tolerance;
    double fval = opt.Optimize(adapter, x);
    return { fromArma(x), fval, cfg.maxIter, true, "L-BFGS" };
}

// ── Nelder-Mead (GSL) ─────────────────────────────────────────────────────────

static OptimizerResult run_NelderMead(ObjFn obj, std::vector<double> x0,
                                       const OptimizerConfig& cfg)
{
    struct Ctx { ObjFn* fn; size_t n; };
    Ctx ctx{ &obj, x0.size() };

    gsl_multimin_function func;
    func.n      = x0.size();
    func.params = &ctx;
    func.f      = [](const gsl_vector* v, void* p) -> double {
        auto* c = static_cast<Ctx*>(p);
        std::vector<double> x(c->n);
        for (size_t i = 0; i < c->n; ++i) x[i] = gsl_vector_get(v, i);
        return (*c->fn)(x);
    };

    gsl_vector* x  = gsl_vector_alloc(x0.size());
    gsl_vector* ss = gsl_vector_alloc(x0.size());
    for (size_t i = 0; i < x0.size(); ++i) {
        gsl_vector_set(x,  i, x0[i]);
        gsl_vector_set(ss, i, cfg.stepSize);
    }

    auto* s = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2, x0.size());
    gsl_multimin_fminimizer_set(s, &func, x, ss);

    size_t iter = 0;
    int status  = GSL_CONTINUE;
    while (status == GSL_CONTINUE && iter < cfg.maxIter) {
        ++iter;
        if (gsl_multimin_fminimizer_iterate(s)) break;
        status = gsl_multimin_test_size(gsl_multimin_fminimizer_size(s), cfg.tolerance);
    }

    std::vector<double> xout(x0.size());
    for (size_t i = 0; i < x0.size(); ++i) xout[i] = gsl_vector_get(s->x, i);
    double fval = s->fval;

    gsl_multimin_fminimizer_free(s);
    gsl_vector_free(x);
    gsl_vector_free(ss);

    return { xout, fval, iter, (status == GSL_SUCCESS), "NelderMead" };
}

// ── Public dispatch ───────────────────────────────────────────────────────────

OptimizerResult OptimizerAPI::Minimize(Algorithm algo, ObjFn obj,
    std::vector<double> x0, const OptimizerConfig& cfg)
{
    if (algo == Algorithm::DE)         return run_DE(std::move(obj), std::move(x0), cfg);
    if (algo == Algorithm::NelderMead) return run_NelderMead(std::move(obj), std::move(x0), cfg);
    throw std::invalid_argument("L-BFGS requires ObjGradFn overload");
}

OptimizerResult OptimizerAPI::Minimize(Algorithm algo, ObjGradFn objGrad,
    std::vector<double> x0, const OptimizerConfig& cfg)
{
    if (algo == Algorithm::LBFGS) return run_LBFGS(std::move(objGrad), std::move(x0), cfg);
    throw std::invalid_argument("DE and NelderMead require ObjFn overload");
}
