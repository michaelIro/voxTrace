#include <iostream>
#include <filesystem>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "api/PolyCapAPI.hpp"
#include "base/XRBeam.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "No Input Folder given!" << std::endl;
        return 1;
    }
    
    std::cout << "START: CapillaryTracer (polycap)" << std::endl;

    // Set the base directory
    std::string base_dir = argv[1];

    // Construct the input and output directories
    std::string input_dir = base_dir + "/post-sample";
    std::string output_dir = base_dir + "/detector";

    // Create output directory if it does not exist
    std::filesystem::create_directory(output_dir);


    // Construct the path to the Polycapillary.txt file
    //std::string poly_cap_file = base_dir + "Polycapillary.txt";

    // Iterate over all files in the input directory
    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            std::string path_in = entry.path();
            std::string filename = entry.path().filename().string();
            filename.replace(0, 2, "de");
            std::string path_out = output_dir + "/" + filename;

            arma::Mat<double> sec_beam_mat;
            sec_beam_mat.load(arma::hdf5_name(path_in, "my_data"));

            /**********************************************************************/

            auto transformMatrix = [](arma::Mat<double>& matrix, double x_shift, double y_shift, double z_shift, double rot_x, double rot_y, double rot_z) {
                matrix.col(0) += x_shift;
                matrix.col(1) += y_shift;
                matrix.col(2) += z_shift;

                // Rot around x,y and z axis
                arma::Col<double> c3 = cosf(rot_z)*cosf(rot_y)*matrix.col(3)    - (cosf(rot_z)*sinf(rot_y)*sinf(rot_x)-sinf(rot_z)*cosf(rot_x))*matrix.col(4)   + (cosf(rot_z)*sinf(rot_y)*cosf(rot_x)+sinf(rot_z)*sinf(rot_x))*matrix.col(5) ;
                arma::Col<double> c4 = sinf(rot_z)*cosf(rot_y)*matrix.col(3)    - (sinf(rot_z)*sinf(rot_y)*sinf(rot_x)-cosf(rot_z)*cosf(rot_x))*matrix.col(4)   + (sinf(rot_z)*sinf(rot_y)*cosf(rot_x)-cosf(rot_z)*sinf(rot_x))*matrix.col(5) ;  
                arma::Col<double> c5 = -sinf(rot_y)*matrix.col(3)               + (cosf(rot_y)*sinf(rot_x))*matrix.col(4)                                       + (cosf(rot_y)*cosf(rot_x))*matrix.col(5) ;
                matrix.col(3) = c3;
                matrix.col(4) = c4;
                matrix.col(5) = c5;
            };

            transformMatrix(sec_beam_mat, 0.0,0.0125,0.0, 0.002,0.000,0.000);
            
            /**********************************************************************/       
            
            PolyCapAPI mySecondaryPolycap(base_dir);
            XRBeam myDetectorBeam(mySecondaryPolycap.traceFast(sec_beam_mat));

            std::cout << "Detector size: " << myDetectorBeam.getRays().size() << std::endl;
            myDetectorBeam.getMatrix().save(arma::hdf5_name(path_out, "my_data"));
        }
    }

    std::cout << "END: CapillaryTracer (polycap)" << std::endl << std::endl;
    return 0;
}