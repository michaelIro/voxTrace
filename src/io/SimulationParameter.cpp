#include "SimulationParameter.hpp"

SimulationParameter::SimulationParameter(const std::string& directory) : directory_(directory) {
    std::string filepath = directory + "/Capillaries.txt";
    std::ifstream in(filepath);
    if (!in) {
        std::cerr << "Could not open file: " << filepath << std::endl;
        exit(1);
    }
    std::string line;
    std::getline(in, line);
	std::getline(in, line);
    // Read primary capillary geometry
    for(int i = 0; i < 4; i++){
		std::getline(in, line);
    	size_t pos = line.find_first_of("#");
    	line = line.substr(0, pos);
    	std::stringstream ss(line);
		std::string temp;
		ss >> temp;
		prim_cap_geom[i] = std::stof(temp);
		//std::cout << prim_cap_geom[i] << std::endl;
	}


	std::getline(in, line);
	std::getline(in, line);
	for(int i = 0; i < 5; i++){
		std::getline(in, line);
    	size_t pos = line.find_first_of("#");
    	line = line.substr(0, pos);
    	std::stringstream ss2(line);
		std::string temp;
		ss2 >> temp;
		prim_trans_param[i] = std::stof(temp);
		//std::cout << prim_trans_param[i] << std::endl;
	}
	std::getline(in, line);
	std::getline(in, line);
	
	for(int i = 0; i < 6; i++){
		std::getline(in, line);
    	size_t pos = line.find_first_of("#");
    	line = line.substr(0, pos);
    	std::stringstream ss3(line);
		std::string temp;
		ss3 >> temp;
		sec_trans_param[i] = std::stof(temp);
		//std::cout << sec_trans_param[i] << std::endl;
	}
	in.close();
	
	std::string filepath2 = directory + "/Sample.txt";
    std::ifstream in2(filepath2);

    std::getline(in2, line);
    std::getline(in2, line);

    // Read start position
    for(int i = 0; i < 3; i++){
        std::getline(in2, line);
        size_t pos = line.find_first_of("#");
        line = line.substr(0, pos);
        std::stringstream ss(line);
        std::string temp;
        ss >> temp;
        sample_start[i] = std::stof(temp);
        //std::cout << sample_start[i] << std::endl;
    }
    std::getline(in2, line);
    std::getline(in2, line);
    // Read sample dimensions
    for(int i = 0; i < 3; i++){
        std::getline(in2, line);
        size_t pos = line.find_first_of("#");
        line = line.substr(0, pos);
        std::stringstream ss(line);
        std::string temp;
        ss >> temp;
        sample_length[i] = std::stof(temp);
        //std::cout << sample_length[i] << std::endl;
    }
    std::getline(in2, line);
    std::getline(in2, line);
    // Read voxel dimensions
    for(int i = 0; i < 3; i++){
        std::getline(in2, line);
        size_t pos = line.find_first_of("#");
        line = line.substr(0, pos);
        std::stringstream ss(line);
        std::string temp;
        ss >> temp;
        sample_voxel_length[i] = std::stof(temp);
        //std::cout << sample_voxel_length[i] << std::endl;
    }
    std::getline(in2, line);
    std::getline(in2, line);
    // Read sample type
    std::getline(in2, line);
    size_t pos = line.find_first_of("#");
    line = line.substr(0, pos);
    sample_type = std::stoi(line);
    //std::cout << sample_type << std::endl;

	num_voxels = sample_length[0] / sample_voxel_length[0] * sample_length[1] / sample_voxel_length[1] * sample_length[2] / sample_voxel_length[2];

	std::string filepath3 = directory + "/Simulation.txt";
	std::ifstream in3(filepath3);

	std::getline(in3, line);
	std::getline(in3, line);

	// Read number of measurement points
	std::getline(in3, line);
	pos = line.find_first_of("#");
	line = line.substr(0, pos);
	num_meas_points = std::stoi(line);
	//std::cout << num_meas_points << std::endl;

	// Read number of rays per measurement point
	std::getline(in3, line);
	pos = line.find_first_of("#");
	line = line.substr(0, pos);
	num_rays = std::stoi(line);
	//std::cout << num_rays << std::endl;

	std::getline(in3, line);
	std::getline(in3, line);

	std::vector<std::vector<float>> measurement_points_(num_meas_points, std::vector<float>(3));
	// Read measurement point scan path
	for(int i = 0; i < num_meas_points; i++){
   	 	std::getline(in3, line);
    	pos = line.find_first_of("#");
    	line = line.substr(0, pos);
    	std::stringstream ss(line);
    	std::string temp;
    	for(int j = 0; j < 3; j++){
        	std::getline(ss, temp, ',');
        	measurement_points_[i][j] = std::stof(temp);
        	//std::cout << measurement_points_[i][j] << " ";
   	 	}
    	//std::cout << std::endl;
	}
	measurement_points = measurement_points_;
	in3.close();

	std::string filepath4 = directory + "/Materials.txt";
	std::ifstream in4(filepath4);
	std::getline(in4, line);
 	
	while (std::getline(in4, line)) {
		if (line.empty() || line.find('#') == 0) std::getline(in4, line);
    	MaterialPoint point;
    	std::stringstream ss(line);
    	std::string temp;

		//std::cout << line << std::endl;
    
    	// Read x, y, z coordinates

        std::getline(ss, temp, ',');
		//std::cout << temp << std::endl;
        point.x = std::stof(temp);
        std::getline(ss, temp, ',');
		//std::cout << temp << std::endl;
        point.y = std::stof(temp);
        std::getline(ss, temp, '#');
		//std::cout << temp << std::endl;
        point.z = std::stof(temp);
		//std::cout << point.x << " " << point.y << " " << point.z << std::endl;

    	// Read number of elements
    	std::getline(in4, line);
		//std::cout << line << std::endl;
		pos = line.find_first_of("#");
    	line = line.substr(0, pos);
    	point.n_elements = std::stoi(line);
		//std::cout << point.n_elements << std::endl;

		// Read element Z numbers
    	std::getline(in4, line);
    	ss = std::stringstream(line);
    	for (int i = 0; i < point.n_elements; i++) {
        	std::getline(ss, temp, ',');
        	point.elements.push_back(std::stoi(temp));
			//std::cout << point.elements[i] << std::endl;
    	}

    	// Read element mass fractions
    	std::getline(in4, line);
    	ss = std::stringstream(line);
		//std::cout << line << std::endl;
    	for (int i = 0; i < point.n_elements; i++) {
        	std::getline(ss, temp, ',');
        	point.mass_fractions.push_back(std::stof(temp));
			//std::cout << point.mass_fractions[i] << std::endl;
    	}

    	material_points.push_back(point);
		std::getline(in4, line);
	}
	in4.close();

	for (auto& point : material_points) {
    	for (auto& element : point.elements) {
        	// Check if element is already in unique_elements
        	bool found = false;
        	int index = 0;
        	for (size_t i = 0; i < unique_elements.size(); i++) {
            	if (element == unique_elements[i]) {
                	found = true;
                	index = i;
                	break;
            	}
        	}

        	if (!found) {
            	// Element not found in unique_elements, add it and record its index
            	unique_elements.push_back(element);
            	index = unique_elements.size() - 1;
        	}

        	// Replace element with its corresponding index in unique_elements
        	element = index;
    	}
	}
}

int SimulationParameter::getNumRays() const {
    return num_rays;
}

const float* SimulationParameter::getPrimCapGeom() const {
    return prim_cap_geom;
}

const float* SimulationParameter::getPrimTransParam() const {
    return prim_trans_param;
}

const float* SimulationParameter::getSecTransParam() const {
    return sec_trans_param;
}

const float* SimulationParameter::getSampleStart() const {
    return sample_start;
}

const float* SimulationParameter::getSampleLength() const {
    return sample_length;
}

const float* SimulationParameter::getSampleVoxelLength() const {
    return sample_voxel_length;
}

const std::vector<std::vector<float>>& SimulationParameter::getMeasurementPoints() const {
	return measurement_points;
}

const std::vector<MaterialPoint>& SimulationParameter::getMaterialPoints() const {
	return material_points;
}

int SimulationParameter::getNumVoxels() const {
	return num_voxels;
}

int SimulationParameter::getNumMeasPoints() const {
	return num_meas_points;
}

int SimulationParameter::getSampleType() const {
	return sample_type;
}

const std::vector<int>& SimulationParameter::getUniqueElements() const {
	return unique_elements;
}

// getDirectory
const std::string SimulationParameter::getDirectory() const {
	return directory_;
}