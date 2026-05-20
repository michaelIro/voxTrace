#ifndef SimulationParameter_H
#define SimulationParameter_H

/*Simulation Parameter*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

struct MaterialPoint {
    float x, y, z;
    int n_elements;
    std::vector<int> elements;
    std::vector<float> mass_fractions;
};

class SimulationParameter {
public:
    // Constructor to read simulation parameters from a file
    SimulationParameter(const std::string& directory);

    // Getters for simulation parameters
    int getNumRays() const;
    const float* getPrimCapGeom() const;
    const float* getPrimTransParam() const;
    const float* getSecTransParam() const;
    const float* getSampleStart() const;
    const float* getSampleLength() const;
    const float* getSampleVoxelLength() const;
    int getSampleType() const;
    int getNumVoxels() const;
    int getNumMeasPoints() const;
    const std::vector<std::vector<float>>& getMeasurementPoints() const;
    const std::vector<MaterialPoint>& getMaterialPoints() const;
    const std::vector<int>& getUniqueElements() const;

    const std::string getDirectory() const;

private:
    const std::string directory_;                           // Directory of the simulation parameter files
    float prim_cap_geom[4];                                 // Primary cap geometry parameters
    float prim_trans_param[5];                              // Primary transmission parameters
    float sec_trans_param[6];                               // Secondary transmission parameters

    float sample_start[3];                                  // Start position of the sample in x, y, z direction
    float sample_length[3];                                 // Length of the sample in x, y, z direction
    float sample_voxel_length[3];                           // Length of a voxel in x, y, z direction
    int sample_type;                                        // 0: box, 1: cylinder, 2: sphere
    int num_voxels;                                         // Number of voxels in the sample

    int num_meas_points;                                    // Number of simulated measurement points
    int num_rays;                                           // Number of rays per measurement point
    std::vector<std::vector<float>> measurement_points;     // Measurement points

    std::vector<MaterialPoint> material_points;             // Material points
    std::vector<int> unique_elements;                       // Unique elements in the material points

};

#endif