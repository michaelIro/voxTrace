#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../api/XRayLibAPI.hpp"
#include "../api/OptimizerAPI.hpp"
#include "../core/PolyCap.hpp"
#include "../core/Ray.hpp"
#include "../core/RNG.hpp"
#include "../core/Source.hpp"

namespace {

struct PolycapFixture {
    double length = 0.;
    double rExtUpstream = 0.;
    double rExtDownstream = 0.;
    double rCapUpstream = 0.;
    double rCapDownstream = 0.;
    double focalDistanceIn = 0.;
    double focalDistanceOut = 0.;
    std::vector<int> atomicNumbers;
    std::vector<double> weightPercentages;
    double density = 0.;
    double roughness = 0.;
    double numCapillaries = 0.;
};

double parseScalar(const std::string& line) {
    size_t pos = line.find(';');
    return std::stod(line.substr(0, pos));
}

std::vector<int> parseIntArray(const std::string& line) {
    size_t begin = line.find('{');
    size_t end = line.find('}');
    std::stringstream stream(line.substr(begin + 1, end - begin - 1));
    std::vector<int> values;
    std::string token;
    while (std::getline(stream, token, ',')) {
        values.push_back(std::stoi(token));
    }
    return values;
}

std::vector<double> parseDoubleArray(const std::string& line) {
    size_t begin = line.find('{');
    size_t end = line.find('}');
    std::stringstream stream(line.substr(begin + 1, end - begin - 1));
    std::vector<double> values;
    std::string token;
    while (std::getline(stream, token, ',')) {
        values.push_back(std::stod(token));
    }
    return values;
}

PolycapFixture readPolycapFixture(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open polycap fixture: " + path);
    }

    PolycapFixture fixture;
    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        ++line_num;
        if (line.empty() || line_num == 1) {
            continue;
        }
        switch (line_num) {
            case 2: fixture.length = parseScalar(line); break;
            case 3: fixture.rExtUpstream = parseScalar(line); break;
            case 4: fixture.rExtDownstream = parseScalar(line); break;
            case 5: fixture.rCapUpstream = parseScalar(line); break;
            case 6: fixture.rCapDownstream = parseScalar(line); break;
            case 7: fixture.focalDistanceIn = parseScalar(line); break;
            case 8: fixture.focalDistanceOut = parseScalar(line); break;
            case 10: fixture.atomicNumbers = parseIntArray(line); break;
            case 11: fixture.weightPercentages = parseDoubleArray(line); break;
            case 12: fixture.density = parseScalar(line); break;
            case 13: fixture.roughness = parseScalar(line); break;
            case 14: fixture.numCapillaries = parseScalar(line); break;
            default: break;
        }
    }
    return fixture;
}

Ray makeCenteredRay(float energy_keV, int index) {
    Ray ray;
    // PolyCap's optic axis is the ray's +z; aim a centred ray straight down it.
    ray.setStartCoordinates(0.f, 0.f, -100.f);
    ray.setEndCoordinates(0.f, 0.f, 1.f);
    ray.setSPol(1.f, 0.f, 0.f);
    ray.setPPol(0.f, 1.f, 0.f);
    ray.setEnergyKeV(energy_keV);
    ray.setProb(1.f);
    ray.setIAFlag(false);
    ray.setIANum(index);
    return ray;
}

}  // namespace

int main() {

    // ── XRayLib test ──────────────────────────────────────────────────────────
    std::cout << "Atomic number of Cu: " << XRayLibAPI::SymToZ("Cu") << "\n";

    // ── Optimizer test ────────────────────────────────────────────────────────
    auto result = OptimizerAPI::Minimize(
        Algorithm::NelderMead,
        [](const std::vector<double>& x){ return x[0]*x[0] + x[1]*x[1]; },
        {3.0, 3.0}
    );
    std::cout << "Nelder-Mead min: f=" << result.fval
              << "  x=(" << result.x[0] << ", " << result.x[1] << ")\n";

    // ── Source test ───────────────────────────────────────────────────────────
    Source src = Source::fromCapGeom(0.002f, 0.5f, 0.003f, 8.0f);
    src.print();

    RNG rng(42ULL);
    for (int i = 0; i < 3; ++i) {
        Ray ray = src.generate(i, rng);
        ray.print();
    }

    // ── PolyCap ray tracing test ─────────────────────────────────────────────
    PolycapFixture fixture = readPolycapFixture("test-data/api/polycap/pc-236-descr.txt");
    std::vector<int>   iz(fixture.atomicNumbers.begin(), fixture.atomicNumbers.end());
    std::vector<float> wt(fixture.weightPercentages.begin(), fixture.weightPercentages.end());
    PolyCap optic(0.f,
                  (float)fixture.length,
                  (float)fixture.rExtUpstream,   (float)fixture.rExtDownstream,
                  (float)fixture.rCapUpstream,   (float)fixture.rCapDownstream,
                  (float)fixture.focalDistanceIn,(float)fixture.focalDistanceOut,
                  PolyCap::ELLIPSOIDAL,
                  (int)iz.size(), iz.data(), wt.data(),
                  (float)fixture.density, (float)fixture.roughness,
                  (int)fixture.numCapillaries);

    std::vector<float> energies_keV = {8.0f, 12.0f, 17.4f};
    int transmitted = 0;
    for (int i = 0; i < (int)energies_keV.size(); ++i) {
        Ray ray = makeCenteredRay(energies_keV[i], i);
        optic.trace(ray);
        bool ok = ray.getIAFlag();
        transmitted += ok ? 1 : 0;
        std::cout << "PolyCap trace E=" << energies_keV[i] << " keV"
                  << " transmitted=" << ok
                  << " prob=" << ray.getProb()
                  << " reflections=" << ray.getIANum() << "\n";
    }

    if (transmitted == 0) {
        std::cerr << "PolyCap ray tracing smoke test failed: no rays transmitted\n";
        return 1;
    }

    return 0;
}
