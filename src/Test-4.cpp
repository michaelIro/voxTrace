#include <iostream>
#include <armadillo>
#include <filesystem>

int main() {
	std::cout << "START: Test-4" << std::endl;

	std::string path = "/media/miro/Data/pos-0-new";
	arma::Mat<double> complete_;
	int raysN= 0;
  	for (const auto & file : std::filesystem::directory_iterator(path)){
    	arma::Mat<double> rays__;
		rays__.load(arma::hdf5_name(file.path(),"my_data"));
		raysN += rays__.n_rows;
		//std::cout << rays__.n_rows << std::endl;
		//std::cout << rays__.n_cols << std::endl;
	}
	std::cout << raysN << std::endl;

	complete_.set_size(raysN,19);
	int counter = 0;
  	for (const auto & file : std::filesystem::directory_iterator(path)){
    	arma::Mat<double> rays__;
		rays__.load(arma::hdf5_name(file.path(),"my_data"));
		for(int i = 0; i < rays__.n_rows; i++){
			for(int j = 0; j < rays__.n_cols; j++){
				complete_(i+counter,j) = rays__(i,j);
			}
		}
		counter+= rays__.n_rows;
		//raysN += rays__.n_rows;
		//std::cout << rays__.n_rows << std::endl;
		//std::cout << rays__.n_cols << std::endl;
	}
	complete_.save(arma::hdf5_name("/media/miro/Data/pos-0-new.h5","my_data"));

	std::cout << "END: Test-4" << std::endl << std::endl;
    return 0;
}


