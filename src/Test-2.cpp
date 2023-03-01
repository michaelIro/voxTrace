#include <iostream>
#include <filesystem> // include the filesystem library
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "api/PolyCapAPI.hpp"
#include "base/XRBeam.hpp"

int main() {
    
    std::cout << "START: Test-2" << std::endl;
    
    // set input and output directories
    std::string input_dir = "/media/miro/Data-1TB/simulation/triple-cross/post-sample";
    std::string output_dir = "/media/miro/Data-1TB/simulation/triple-cross/detector";

    // create output directory if it does not exist
    std::filesystem::create_directory(output_dir);

    // iterate over all files in the input directory
    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            std::string path_in = entry.path(); // get path of input file
            std::string filename = entry.path().filename().string(); // get filename
			filename.replace(0, 2, "de"); // replace first two characters
            std::string path_out = output_dir + "/" + filename; // create path for output file

            arma::Mat<double> sec_beam_mat;
            sec_beam_mat.load(arma::hdf5_name(path_in,"my_data"));
            
            PolyCapAPI mySecondaryPolycap((char*) "/home/miro/Software/1st-party/voxTrace/test-data/in/polycap/pc-236-descr.txt");
            XRBeam myDetectorBeam(mySecondaryPolycap.traceFast(sec_beam_mat));
            
            std::cout << "Detector size:" << myDetectorBeam.getRays().size() << std::endl;
            myDetectorBeam.getMatrix().save(arma::hdf5_name(path_out,"my_data"));
        }
    }
    
    std::cout << "END: Test-2" << std::endl << std::endl;
    return 0;
}