/*Scan*/

#include "Scan.hpp"

using namespace std;

/*Empty constructor*/
Scan::Scan (){}

/*Empty constructor*/
Scan::Scan (filesystem::path folderPath){

	readPaths(folderPath);

	readPoints(paths_[0]);
	checkCoordOrder();
	readSpectra();
	//readRoi(paths_[1]);
	readFit(paths_[2]);

	Sensitivity mySens;
	intensities2Materials(mySens);

}

void Scan::readPaths(filesystem::path folderPath){

	filesystem::path myPath(folderPath);
	filesystem::path my2Path(folderPath);
	filesystem::path my3Path(folderPath);
	
	//cout<<myPath.filename()<<endl;

	myPath /= "points.txt"; 
	my2Path /= my2Path.filename().string()+".txt";
	my3Path /= "fit"; 
	my3Path /= "IMAGES";
	for( const auto& dir_entry : std::filesystem::directory_iterator(my3Path) ) {
		if(dir_entry.path().string().rfind(".dat")!=std::string::npos){
			my3Path = dir_entry;
			break;
		}
	}	

	//cout << myPath<<endl;
	//cout << my2Path <<endl;
	//cout << my3Path <<endl;

	paths_.push_back(myPath);
	paths_.push_back(my2Path);
	paths_.push_back(my3Path);
}

void Scan::readPoints(filesystem::path pointsFilePath){
	//string test_line = "400: [1,2,3] : (3500.000,17700.000,6070.000) 000400";

	regex lineRegEx("\\d+");
	smatch sm;
    
	vector<int> x_, y_, z_;
	vector<double> mx_, my_, mz_;

	string tmp_;
	int row_ =0, col_=0;
	ifstream ifs_(pointsFilePath);
	while ( getline(ifs_, tmp_) ) {
		string myStrings[11];
        while(std::regex_search(tmp_, sm, lineRegEx))
        {
            //std::cout <<"Row: "<< row_ << " Column: " << col_<<" Value: " << stoi(sm.str()) << '\n';
            myStrings[col_] =  sm.str();
            tmp_ = sm.suffix();
			
			if(col_== 10){
				col_ = 0;
				x_.push_back(stoi(myStrings[1]));
				y_.push_back(stoi(myStrings[2]));
				z_.push_back(stoi(myStrings[3]));

				mx_.push_back(stod(myStrings[4]+"."+myStrings[5]));
				my_.push_back(stod(myStrings[6]+"."+myStrings[7]));
				mz_.push_back(stod(myStrings[8]+"."+myStrings[9]));

				//speFilenames_.push_back(myStrings[10]);
			} 
			else ++col_;
        }
		++row_;
	}

	coordinates_.push_back(x_);
	coordinates_.push_back(y_);
	coordinates_.push_back(z_);

	//cout << coordinates_[0].size() << endl;

	motorpositions_.push_back(mx_);
	motorpositions_.push_back(my_);
	motorpositions_.push_back(mz_);

	//for(int i = 0; i<x_.size();i++)
	//	cout << coordinates_[0][i] <<endl;
}

void Scan::readSpectra(){
	vector<int> a_(*max_element(begin(coordinates_[2]),end(coordinates_[2])),0);
	vector<vector<int>> b_(*max_element(begin(coordinates_[1]),end(coordinates_[1])),a_);
	vector<vector<vector<int>>> c_(*max_element(begin(coordinates_[0]),end(coordinates_[0])),b_);

	for(size_t l =0; l< coordinates_[0].size();l++)
		c_[coordinates_[0][l]-1][coordinates_[1][l]-1][coordinates_[2][l]-1] = l;
	
	spes_ = c_;
	/*
	for(int i = 0; i<c_.size(); i++){
		for(int j = 0; j<c_[i].size(); j++){
			for(int k = 0; k<c_[i][j].size(); k++){
				cout << i << " " << j << " " << k  << " " << c_[i][j][k] << endl;
	*/
}

void Scan::readFit(filesystem::path fitFilePath){

	vector<vector<double>> values__;
	vector<string> names__;

	string tmp1_,tmp2_;
	int row_ = 0;
	int col_ = 0;

	ifstream ifs(fitFilePath);

	// Read the first line
	getline(ifs, tmp1_);
	istringstream iss (tmp1_);
	while(iss >> tmp2_) {
		vector<double> vector_;
		values__.push_back(vector_);
		names__.push_back(tmp2_);
	}

	// Extract which X-ray-lines were fitted
	vector<int> intCols__;
	vector<int> uncCols__;
	vector<int> shellVec__;
	for(size_t j= 0; j<names__.size();j++){

		std::size_t found_ = names__[j].find_first_of("_");
		std::size_t foundBrac_ = names__[j].find_first_of("(");

		if(found_ != std::string::npos){
			if(found_ < 3){
				for(int i = 1; i<108; i++){
					ChemElement myEl(i);
					
					if(names__[j].substr(0,found_) == myEl.getSymbol()){
						//int myShell;
						if( names__[j].substr(found_+1,names__[j].length()-found_) == "K")
							shellVec__.push_back(K_SHELL);
						else if( names__[j].substr(found_+1,names__[j].length()-found_) == "L")
							shellVec__.push_back(L3_SHELL);
						else if( names__[j].substr(found_+1,names__[j].length()-found_) == "M")
							shellVec__.push_back(M3_SHELL);
						
						//cout<<names_[j].substr(0,found)<<" "<<names_[j].substr(found+1,names_[j].length()-found)<<endl;
						intCols__.push_back(j);
						elements_.push_back(myEl);
					}
				}
			}
			else if(foundBrac_ !=std::string::npos){
				uncCols__.push_back(j);
			}
		}
	}


	// Read all Data from File
	while ( getline(ifs, tmp1_) ){
		istringstream iss (tmp1_);
		while(iss >> tmp2_) {
			if(tmp2_ == "nan") tmp2_="0.0";
			values__[col_].push_back(stod(tmp2_));
			++col_;
		}
		col_=0;
		++row_;
	} 


	double measurementTime = 100.; // in seconds FIXME: read measurement time from spe-File

	//FIXME: Problems with multiple lines (e.g. Au-L, Au-M)
	// Create 3D-Material-Vector for 
	for(size_t i = 0; i<spes_.size(); i++){
		vector<vector<Material>> aMat_; 
		vector<vector<map<int,double>>> aInt_; 
		vector<vector<map<int,double>>> aUnc_; 
		for(size_t j = 0; j<spes_[i].size(); j++){
			vector<Material> bMat_; 
			vector<map<int,double>> bInt_;
			vector<map<int,double>> bUnc_;
			for(size_t k = 0; k<spes_[i][j].size(); k++){
				map<int,double> intMap_;
				map<int,double> uncMap_;
				for(size_t l = 0; l < elements_.size(); l++){
					intMap_.insert(std::pair<int,double>(elements_[l].getZ(),values__[intCols__[l]][spes_[i][j][k]]/measurementTime));
					uncMap_.insert(std::pair<int,double>(elements_[l].getZ(),values__[uncCols__[l]][spes_[i][j][k]]/measurementTime));
				}
				Material c_(intMap_,1.0);
				bMat_.push_back(c_);
				bInt_.push_back(intMap_);
				bUnc_.push_back(uncMap_);
			}
			aMat_.push_back(bMat_);
			aInt_.push_back(bInt_);
			aUnc_.push_back(bUnc_);
		}
		materials_.push_back(aMat_);
		intensities_.push_back(aInt_);
		uncertainties_.push_back(aUnc_);
	}
}

vector<double >Scan::readRoi(filesystem::path roiFilePath){
	
	vector<double> startpositions_;

	string tmp1_, tmp2_;
	int row_ = 0;
	int col_ = 0;

	ifstream ifs(roiFilePath);

	for (int i =0; i < 8; i++) //Skip Header lines right into first data line
		getline(ifs, tmp1_); 
	
	istringstream iss (tmp1_);
	while(iss >> tmp2_) {
		if(col_ == 2) startpositions_.push_back(stod(tmp2_));
		if(col_ == 3) startpositions_.push_back(stod(tmp2_));
		if(col_ == 4) startpositions_.push_back(stod(tmp2_));
		++col_;
	}

	//Read the rest
	/*
	while ( getline(ifs, tmp1_) ){
		istringstream iss (tmp1_);
		while(iss >> tmp2_) {
			//cout << tmp2_ << " ";
			++col_;
		}
		++row_;
		//cout << endl;
	} 
	*/

	return startpositions_;
}

void Scan::checkCoordOrder(){

	// manually order  
	vector<vector<double>> tmp_mot_pos_;
	tmp_mot_pos_.push_back(motorpositions_[2]);
	tmp_mot_pos_.push_back(motorpositions_[0]);
	tmp_mot_pos_.push_back(motorpositions_[1]);
	motorpositions_ = tmp_mot_pos_;

	// should be in order
	vector<vector<int>> tmp_coord_;
	tmp_coord_.push_back(coordinates_[0]);
	tmp_coord_.push_back(coordinates_[2]);
	tmp_coord_.push_back(coordinates_[1]);
	coordinates_ = tmp_coord_;


	// TODO: automaticvally order 

	/*
	vector<vector<double>> borders_;
	vector<double> xborders_;
	vector<double> yborders_;
	vector<double> zborders_; 

	xborders_.push_back( *min_element(motorpositions_[0].begin(),motorpositions_[0].end()) );
	xborders_.push_back( *max_element(motorpositions_[0].begin(),motorpositions_[0].end()) );
	yborders_.push_back( *min_element(motorpositions_[1].begin(),motorpositions_[1].end()) );
	yborders_.push_back( *max_element(motorpositions_[1].begin(),motorpositions_[1].end()) );
	zborders_.push_back( *min_element(motorpositions_[2].begin(),motorpositions_[2].end()) );
	zborders_.push_back( *max_element(motorpositions_[2].begin(),motorpositions_[2].end()) );

	borders_.push_back(xborders_);
	borders_.push_back(yborders_);
	borders_.push_back(zborders_);

	vector<double> directions_;
	for(int i = 0; i < 3; i++)
		if ( abs(borders_[i][0] - motorpositions_[i][0]) < abs(borders_[i][1] - motorpositions_[i][0]) )
			directions_.push_back(1);
		else
			directions_.push_back(-1);

	vector<double> lengths_;

	for(int i = 0; i < 3; i++)
		lengths_.push_back((borders_[i][1] -borders_[i][0]));

	for(int i = 0; i < 3; i++)
		cout << directions_[i] << " "<< borders_[i][0] << " " << borders_[i][1] <<" " <<lengths_[i] <<  endl;	

	double a[3][3]={{0,0,0},{0,0,0},{0,0,0}};

	cout << coordinates_.size() << motorpositions_.size() << endl;
	
	
	for (int k =1; k < coordinates_[0].size();k++){
		int b[3];
		b[0] = (coordinates_[0][k]-coordinates_[0][k-1]);
		b[1] = (coordinates_[1][k]-coordinates_[1][k-1]);
		b[2] = (coordinates_[2][k]-coordinates_[2][k-1]);
			
		if ( (b[0]+b[1]+b[2]) == 1 ){

			for(int i=0; i< 3; i++){
				vector<double> c;
				c.push_back((motorpositions_[0][k]-motorpositions_[0][k-1]));
				c.push_back(motorpositions_[1][k]-motorpositions_[1][k-1]);
				c.push_back(motorpositions_[2][k]-motorpositions_[2][k-1]);
				

				for(int j =0; j< 3;j++){
					//if(c[j] == *max_element(c.begin(), c.end()))
					//	if( b[i] == 1 ) 
							//cout << k << " b: " << b[0] << " " << b[1] << " " << b[2] << " c: " <<  c[0] << " " << c[1] << " " << c[2] << endl;
					 
					//cout << k << endl; 
					//a[i][j] += (coordinates_[i][k]-coordinates_[i][k-1]);//*(motorpositions_[j][k]-motorpositions_[j][k-1]);
					//cout << (coordinates_[i][k]-coordinates_[i][k-1]) << endl;
				}
			}
		}
	}

	for(int i=0; i< 3; i++){
		for(int j =0; j< 3;j++){
			cout << i << " " << j << " " << a[i][j] << endl;
		}
	}

	vector<double> mstart_ = readRoi(paths_[1]);
	vector<double> coord_start_;

	coord_start_.push_back(motorpositions_[0][0]);
	coord_start_.push_back(motorpositions_[1][0]);
	coord_start_.push_back(motorpositions_[2][0]);


	vector<vector<double>> coordinates_unsrt_;
	coordinates_unsrt_.push_back(motorpositions_[0]);
	coordinates_unsrt_.push_back(motorpositions_[1]);
	coordinates_unsrt_.push_back(motorpositions_[2]);


	//cout << mstart_[0] << " " << mstart_[1] << " " << mstart_[2] << " " <<  endl;
	//cout << coord_start_[0] << " " << coord_start_[1] << " " << coord_start_[2] << " " <<  endl;
	//cout << endl;
	
	//cout << Subtract(mstart_, coord_start_)[0] << endl;
	//cout << coordinates_unsrt_[0][0] << " " << coordinates_unsrt_[1][0] << " " << coordinates_unsrt_[2][0] << " " <<  endl;
	do {
    	std::cout << coord_start_[0] << ' ' << coord_start_[1] << ' ' << coord_start_[2] << '\n';
  	} while ( std::prev_permutation(coord_start_.begin(),coord_start_.end()) );
	  cout << endl;

	do {
    	std::cout << coord_start_[0] << ' ' << coord_start_[1] << ' ' << coord_start_[2] << '\n';
  	} while ( std::next_permutation(coord_start_.begin(),coord_start_.end()) );

	
	cout<<"Delta xx: "<<mstart_[0]-coordinates_unsrt_[0][0]<<endl;
	cout<<"Delta xy: "<<mstart_[0]-coordinates_unsrt_[0][0]<<endl;
	cout<<"Delta xz: "<<mstart_[0]-coordinates_unsrt_[0][0]<<endl;
	cout << endl;
	cout<<"Delta xx: "<<mstart_[0]-coordinates_unsrt_[1][0]<<endl;
	cout<<"Delta xy: "<<mstart_[0]-coordinates_unsrt_[1][0]<<endl;
	cout<<"Delta xz: "<<mstart_[0]-coordinates_unsrt_[1][0]<<endl;
	cout << endl;
	cout<<"Delta xx: "<<mstart_[0]-coordinates_unsrt_[2][0]<<endl;
	cout<<"Delta xy: "<<mstart_[0]-coordinates_unsrt_[2][0]<<endl;
	cout<<"Delta xz: "<<mstart_[0]-coordinates_unsrt_[2][0]<<endl;

	//for(int i = 0; i < 3; i++){
	//	for(int j =0; j<3; j++){
	//		cout<<"Delta "<<i<<" "<<j<<" "<<mstart_[i]-coordinates_unsrt_[j][0]<<endl;	
	//		cout << mstart_[i]<<endl;	
	//		cout << coordinates_unsrt_[j][0]<<endl;	
	//	}
	//}
	*/
}

vector<vector<vector<Material>>>  Scan::getMaterials() const {return materials_;}
vector<ChemElement> Scan::getElements() const {return elements_;}
vector<double> Scan::getLengths() const {
	vector<double> lengths_;
	lengths_.push_back(*max_element(begin(motorpositions_[0]),end(motorpositions_[0])) - *min_element(begin(motorpositions_[0]),end(motorpositions_[0])));
	lengths_.push_back(*max_element(begin(motorpositions_[1]),end(motorpositions_[1])) - *min_element(begin(motorpositions_[1]),end(motorpositions_[1])));
	lengths_.push_back(*max_element(begin(motorpositions_[2]),end(motorpositions_[2])) - *min_element(begin(motorpositions_[2]),end(motorpositions_[2])));

	return lengths_;
}

/*
void Scan::intensities2Materials(Sensitivity sensitivity){
	vector<vector<vector<Material>>> myMaterials__;
	for(size_t i =0; i < intensities_.size(); i++){
		vector<vector<Material>> aMat_; 
		for(size_t j =0; j < intensities_[i].size(); j++){
			vector<Material> bMat_; 
			for(size_t k =0; k < intensities_[i][j].size(); k++){
				map<int,double> intMap__;
				for(auto map: intensities_[i][j][k]){
					intMap__.insert(std::pair<int,double>(map.first,map.second*sensitivity.getSensitivity(map.first)));
				}
				Material myMaterial__(intMap__,1.0);
				bMat_.push_back(myMaterial__);
			}
			aMat_.push_back(bMat_);
		}
		myMaterials__.push_back(aMat_);
	}
	materials_ = myMaterials__;

	

}
*/
void Scan::print() const {

	for(size_t i = 0; i<spes_.size(); i++){
		for(size_t j = 0; j<spes_[i].size(); j++){
			for(size_t k = 0; k<spes_[i][j].size(); k++){
				cout <<spes_[i][j][k]<<" "<< i << " " << j << " " << k  << " ";
				//for(int l = 0; l < elements_.size(); l++){
					//cout << cols[j] << " " << elements_[j].getSymbol() <<" "<< elements_[j].getZ() <<" " <<shellVec[j] << endl;
				//	cout << materials_[i][j][k].getMassFractions()[elements_[l].getZ() ] << " ";
				//}
				for(auto const& it: intensities_[i][j][k]){
					cout << it.first << ": " << it.second << " ";
				}
				cout <<endl;
				for(auto const& it: uncertainties_[i][j][k]){
					cout << it.first << ": " << it.second << " ";
				}
				cout <<endl<<endl;

				
			}
		}
	}

	for(size_t l = 0; l < elements_.size(); l++){
		cout <<  elements_[l].getSymbol() <<" ";
	}
	cout << endl;

}