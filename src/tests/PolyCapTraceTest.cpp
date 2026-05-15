#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <filesystem>

#include "../api/XRayLibAPI.hpp"
#include "../core/Ray.hpp"
#include "../core/RNG.hpp"
#include "../core/PolyCap.hpp"
#include "../core/Source.hpp"

// ============================================================================
// Data structures to hold parsed parameters
// ============================================================================

struct SourceParameter {
    double distanceZ;           // distance between optic entrance and source along z-axis, in cm
    double radiusX;             // source radius in x, in cm
    double radiusY;             // source radius in y, in cm
    double divergenceX;         // source divergence in x, in rad
    double divergenceY;         // source divergence in y, in rad
    double shiftX;              // source shift in x compared to optic central axis, in cm
    double shiftY;              // source shift in y compared to optic central axis, in cm
    double polarizationFactor;  // source polarisation factor
    std::vector<double> energies; // photon energies in keV
};

struct PolycapParameter {
    double length;              // optic length in cm
    double rExtUpstream;        // external radius at entrance, in cm
    double rExtDownstream;      // external radius at exit, in cm
    double rCapUpstream;        // single capillary radius at entrance, in cm
    double rCapDownstream;      // single capillary radius at exit, in cm
    double focalDistanceIn;     // focal distance on entrance side, in cm
    double focalDistanceOut;    // focal distance on exit side, in cm
    int numElements;            // number of elements in material
    std::vector<int> atomicNumbers; // atomic numbers
    std::vector<double> weightPercentages; // weight percentages
    double density;             // material density in g/cm^3
    double roughness;           // surface roughness in Angstrom
    double numCapillaries;      // number of capillaries
};

// ============================================================================
// Parser functions
// ============================================================================

/**
 * Parse a single number from a line containing format: "number; // comment"
 */
double parseLineValue(const std::string& line) {
    size_t pos = line.find_first_of("';");
    if (pos != std::string::npos) {
        std::string numStr = line.substr(0, pos);
        try {
            return std::stod(numStr);
        } catch (...) {
            return 0.0;
        }
    }
    return 0.0;
}

/**
 * Parse energy array from line with format: "[N]={e1, e2, ..., eN};"
 */
std::vector<double> parseEnergyArray(const std::string& line) {
    std::vector<double> energies;
    size_t start = line.find_first_of('{');
    size_t end = line.find_first_of('}');
    
    if (start != std::string::npos && end != std::string::npos) {
        std::string arrayStr = line.substr(start + 1, end - start - 1);
        std::stringstream ss(arrayStr);
        std::string token;
        
        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t\n\r"));
            token.erase(token.find_last_not_of(" \t\n\r") + 1);
            if (!token.empty()) {
                try {
                    energies.push_back(std::stod(token));
                } catch (...) {
                    // Skip unparseable values
                }
            }
        }
    }
    return energies;
}

/**
 * Parse atomic numbers array from line with format: "[N]={z1,z2,...,zN};"
 */
std::vector<int> parseIntArray(const std::string& line) {
    std::vector<int> values;
    size_t start = line.find_first_of('{');
    size_t end = line.find_first_of('}');
    
    if (start != std::string::npos && end != std::string::npos) {
        std::string arrayStr = line.substr(start + 1, end - start - 1);
        std::stringstream ss(arrayStr);
        std::string token;
        
        while (std::getline(ss, token, ',')) {
            token.erase(0, token.find_first_not_of(" \t\n\r"));
            token.erase(token.find_last_not_of(" \t\n\r") + 1);
            if (!token.empty()) {
                try {
                    values.push_back(std::stoi(token));
                } catch (...) {
                    // Skip unparseable values
                }
            }
        }
    }
    return values;
}

/**
 * Parse a double array from line with format: "[N]={d1,d2,...,dN};"
 */
std::vector<double> parseDoubleArray(const std::string& line) {
    std::vector<double> values;
    size_t start = line.find_first_of('{');
    size_t end = line.find_first_of('}');
    
    if (start != std::string::npos && end != std::string::npos) {
        std::string arrayStr = line.substr(start + 1, end - start - 1);
        std::stringstream ss(arrayStr);
        std::string token;
        
        while (std::getline(ss, token, ',')) {
            token.erase(0, token.find_first_not_of(" \t\n\r"));
            token.erase(token.find_last_not_of(" \t\n\r") + 1);
            if (!token.empty()) {
                try {
                    values.push_back(std::stod(token));
                } catch (...) {
                    // Skip unparseable values
                }
            }
        }
    }
    return values;
}

/**
 * Read and parse source description file
 */
SourceParameter readSourceDescription(const std::string& filepath) {
    SourceParameter source;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        exit(1);
    }
    
    std::string line;
    int lineNum = 0;
    
    while (std::getline(file, line)) {
        lineNum++;
        
        // Skip empty lines and header
        if (line.empty() || line.find("Parameter") != std::string::npos) {
            continue;
        }
        
        switch (lineNum) {
            case 2: source.distanceZ = parseLineValue(line); break;
            case 3: source.radiusX = parseLineValue(line); break;
            case 4: source.radiusY = parseLineValue(line); break;
            case 5: source.divergenceX = parseLineValue(line); break;
            case 6: source.divergenceY = parseLineValue(line); break;
            case 7: source.shiftX = parseLineValue(line); break;
            case 8: source.shiftY = parseLineValue(line); break;
            case 9: source.polarizationFactor = parseLineValue(line); break;
            case 11: source.energies = parseEnergyArray(line); break;
        }
    }
    
    file.close();
    return source;
}

/**
 * Read and parse polycap description file
 */
PolycapParameter readPolycapDescription(const std::string& filepath) {
    PolycapParameter polycap;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        exit(1);
    }
    
    std::string line;
    int lineNum = 0;
    
    while (std::getline(file, line)) {
        lineNum++;
        
        // Skip empty lines and header
        if (line.empty() || line.find("Parameter") != std::string::npos) {
            continue;
        }
        
        switch (lineNum) {
            case 2: polycap.length = parseLineValue(line); break;
            case 3: polycap.rExtUpstream = parseLineValue(line); break;
            case 4: polycap.rExtDownstream = parseLineValue(line); break;
            case 5: polycap.rCapUpstream = parseLineValue(line); break;
            case 6: polycap.rCapDownstream = parseLineValue(line); break;
            case 7: polycap.focalDistanceIn = parseLineValue(line); break;
            case 8: polycap.focalDistanceOut = parseLineValue(line); break;
            case 9: polycap.numElements = (int)parseLineValue(line); break;
            case 10: polycap.atomicNumbers = parseIntArray(line); break;
            case 11: polycap.weightPercentages = parseDoubleArray(line); break;
            case 12: polycap.density = parseLineValue(line); break;
            case 13: polycap.roughness = parseLineValue(line); break;
            case 14: polycap.numCapillaries = parseLineValue(line); break;
        }
    }
    
    file.close();
    return polycap;
}

// ============================================================================
// Ray generation function
// ============================================================================

/**
 * Generate rays from the source as Ray objects
 * Uses realistic source parameters with Ray structure
 */
std::vector<Ray> generateRaysFromSourceGPU(
    const SourceParameter& source,
    int numRaysPerEnergy = 10)
{
    std::vector<Ray> rays;
    
    // Random number generators
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<> radiusDist(0.0, 1.0);
    std::uniform_real_distribution<> angleDist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<> divergenceDist(-1.0, 1.0);
    
    // For each energy level
    for (double energy : source.energies) {
        // Generate rays for this energy
        for (int i = 0; i < numRaysPerEnergy; i++) {
            // Random position within source (circular disk)
            double r = source.radiusX * std::sqrt(radiusDist(gen));
            double theta = angleDist(gen);
            float posX = source.shiftX + r * std::cos(theta);
            float posY = source.shiftY + r * std::sin(theta);
            float posZ = -source.distanceZ; // Source is before optic entrance
            
            // Random direction within divergence cone
            double divX = source.divergenceX * divergenceDist(gen);
            double divY = source.divergenceY * divergenceDist(gen);
            
            // Direction towards optic (positive Z)
            double dz = std::sqrt(1.0 - divX * divX - divY * divY);
            if (dz < 0) dz = 0.0;
            
            float dirX = (float)divX;
            float dirY = (float)divY;
            float dirZ = (float)dz;
            
            // Normalize direction
            float norm = std::sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
            if (norm > 0.0f) {
                dirX /= norm;
                dirY /= norm;
                dirZ /= norm;
            }
            
            // Create Ray object
            // Parameters: startX, startY, startZ, dirX, dirY, dirZ, 
            //            asX, asY, asZ, flag, k, q, opd, fS, fP, apX, apY, apZ, prob
            Ray ray(
                posX, posY, posZ,           // start position
                dirX, dirY, dirZ,           // direction
                0.0f, 0.0f, 0.0f,          // s-polarization vector
                false,                      // flag
                (float)energy,              // photon energy (k parameter)
                (int)i,                     // q parameter (ray index)
                0.0f,                       // optical path difference
                1.0f,                       // s-phase
                1.0f,                       // p-phase
                0.0f, 0.0f, 0.0f,          // p-polarization vector
                1.0f / numRaysPerEnergy     // probability/weight
            );
            
            rays.push_back(ray);
        }
    }
    
    return rays;
}

/**
 * Trace rays through polycap using GPU accelerated Ray and PolyCap classes
 */
std::vector<Ray> traceRaysThrooughPolycapGPU(
    const std::vector<Ray>& inputRays,
    const PolycapParameter& polycap)
{
    std::cout << "\n=== Ray Tracing Through Polycap (GPU) ===" << std::endl;
    std::cout << "Input rays: " << inputRays.size() << std::endl;
    
    // Create PolyCap optics object
    int Z[] = {0, 0};
    float wt[] = {0.0f, 0.0f};
    
    // Convert atomic numbers and weights
    for (int i = 0; i < std::min(2, (int)polycap.atomicNumbers.size()); i++) {
        Z[i] = polycap.atomicNumbers[i];
        if (i < polycap.weightPercentages.size()) {
            wt[i] = polycap.weightPercentages[i];
        }
    }
    
    // Create PolyCap object with parameters from description file
    // Constructor: posY, length, rExtIn, rExtOut, rCapIn, rCapOut, 
    //             focalIn, focalOut, nElem, Z, wt, density, roughness, nCap
    PolyCap optic(
        0.0f,                               // posY (center on axis)
        (float)polycap.length,              // length in cm
        (float)polycap.rExtUpstream,        // external radius at entrance
        (float)polycap.rExtDownstream,      // external radius at exit
        (float)polycap.rCapUpstream,        // capillary radius at entrance
        (float)polycap.rCapDownstream,      // capillary radius at exit
        (float)polycap.focalDistanceIn,     // focal distance entrance
        (float)polycap.focalDistanceOut,    // focal distance exit
        (int)polycap.numElements,           // number of elements
        Z,                                  // atomic numbers
        wt,                                 // weight percentages
        (float)polycap.density,             // material density
        (float)polycap.roughness,           // surface roughness (Angstrom)
        (float)polycap.numCapillaries       // number of capillaries
    );
    
    std::cout << "PolyCap optics created:\n";
    optic.print();
    
    // Trace each ray through the polycap
    std::vector<Ray> outputRays;
    int transmittedCount = 0;
    
    for (size_t i = 0; i < inputRays.size(); i++) {
        Ray ray = inputRays[i];
        
        // Trace single ray through polycap
        // The ray interacts with the optics; modification is in-place
        optic.trace(ray);
        
        // Check if ray was transmitted (probability > 0)
        if (ray.getIAFlag()) {
            outputRays.push_back(ray);
            transmittedCount++;
        }
        
        // Print progress for first and last rays, and every 1000th ray
        if (i == 0 || i == inputRays.size() - 1 || (i + 1) % 1000 == 0) {
            std::cout << "  Traced ray " << (i + 1) << "/" << inputRays.size() 
                      << ", transmitted so far: " << transmittedCount << std::endl;
        }
    }
    
    std::cout << "\nTracing complete:" << std::endl;
    std::cout << "Output rays: " << outputRays.size() << std::endl;
    std::cout << "Transmission efficiency: " 
              << (100.0 * outputRays.size() / inputRays.size()) << "%" << std::endl;
    
    return outputRays;
}

// ============================================================================
// Main test function
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Polycapillary Ray Tracing Test" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Define paths
    std::string testDataDir = "test-data/in/polycap";
    std::string sourceFile = testDataDir + "/source-descr.txt";
    
    // Try to find a pc-*.txt file
    std::string polycapFile;
    bool foundPolycap = false;
    
    for (const auto& entry : std::filesystem::directory_iterator(testDataDir)) {
        std::string filename = entry.path().filename().string();
        if (filename.find("pc-") == 0 && filename.find(".txt") != std::string::npos) {
            polycapFile = entry.path().string();
            foundPolycap = true;
            break;
        }
    }
    
    if (!foundPolycap) {
        std::cerr << "Error: Could not find any pc-*.txt file in " << testDataDir << std::endl;
        return 1;
    }
    
    std::cout << "Source file:    " << sourceFile << std::endl;
    std::cout << "Polycap file:   " << polycapFile << "\n" << std::endl;
    
    // ========================================================================
    // Step 1: Parse description files
    // ========================================================================
    
    std::cout << "=== Reading Source Description ===" << std::endl;
    SourceParameter source = readSourceDescription(sourceFile);
    std::cout << "Distance from source to optic: " << source.distanceZ << " cm" << std::endl;
    std::cout << "Source radius X/Y: " << source.radiusX << " / " << source.radiusY << " cm" << std::endl;
    std::cout << "Source divergence X/Y: " << source.divergenceX << " / " << source.divergenceY << " rad" << std::endl;
    std::cout << "Number of energies: " << source.energies.size() << std::endl;
    if (source.energies.size() > 0) {
        std::cout << "Energy range: " << source.energies.front() << " - " 
                  << source.energies.back() << " keV" << std::endl;
    }
    
    std::cout << "\n=== Reading Polycap Description ===" << std::endl;
    PolycapParameter polycap = readPolycapDescription(polycapFile);
    std::cout << "Polycap length: " << polycap.length << " cm" << std::endl;
    std::cout << "External aperture (entrance/exit): " << polycap.rExtUpstream 
              << " / " << polycap.rExtDownstream << " cm" << std::endl;
    std::cout << "Capillary radius (entrance/exit): " << polycap.rCapUpstream 
              << " / " << polycap.rCapDownstream << " cm" << std::endl;
    std::cout << "Material: ";
    for (int i = 0; i < polycap.atomicNumbers.size(); i++) {
        std::cout << XRayLibAPI::ZToSym(polycap.atomicNumbers[i]);
        if (i < polycap.atomicNumbers.size() - 1) std::cout << " + ";
    }
    std::cout << " (density: " << polycap.density << " g/cm³)" << std::endl;
    std::cout << "Number of capillaries: " << polycap.numCapillaries << std::endl;
    
    // ========================================================================
    // Step 2: Generate rays from source using Ray
    // ========================================================================
    
    int numRaysPerEnergy = 10;
    std::cout << "\n=== Generating Rays from Source (Ray) ===" << std::endl;
    std::vector<Ray> sourceRays = generateRaysFromSourceGPU(source, numRaysPerEnergy);
    std::cout << "Total rays generated: " << sourceRays.size() << std::endl;
    
    // Print some example ray statistics
    if (sourceRays.size() > 0) {
        double totalWeight = 0;
        for (const auto& ray : sourceRays) {
            totalWeight += ray.getProb();
        }
        std::cout << "Total ray weight (intensity): " << totalWeight << std::endl;
        
        // Group by energy
        std::cout << "\nRays per energy:" << std::endl;
        for (const auto& energy : source.energies) {
            int count = 0;
            for (const auto& ray : sourceRays) {
                if (ray.getEnergyKeV() == (float)energy) count++;
            }
            if (count > 0) {
                std::cout << "  " << energy << " keV: " << count << " rays" << std::endl;
            }
        }
    }
    
    // ========================================================================
    // Step 3: Trace rays through polycap using GPU acceleration
    // ========================================================================
    
    std::vector<Ray> exitRays = traceRaysThrooughPolycapGPU(sourceRays, polycap);
    
    // ========================================================================
    // Step 4: Output results
    // ========================================================================
    
    std::cout << "\n=== Results Summary ===" << std::endl;
    std::cout << "Input rays:        " << sourceRays.size() << std::endl;
    std::cout << "Output rays:       " << exitRays.size() << std::endl;
    std::cout << "Transmission:      " << (100.0 * exitRays.size() / sourceRays.size()) << "%" << std::endl;
    
    if (exitRays.size() > 0) {
        double outputWeight = 0;
        for (const auto& ray : exitRays) {
            outputWeight += ray.getProb();
        }
        std::cout << "Output intensity:  " << outputWeight << std::endl;
        
        // Sample some exit rays
        std::cout << "\nSample exit rays (first 5):" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), exitRays.size()); i++) {
            const auto& ray = exitRays[i];
            std::cout << "  Ray " << i << ": pos=(" << ray.getStartX() << ", " << ray.getStartY() 
                      << ", " << ray.getStartZ() << "), energy=" << ray.getEnergyKeV() << " keV, prob=" << ray.getProb() << std::endl;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test completed successfully!" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return 0;
}
