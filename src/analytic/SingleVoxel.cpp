/**SingleVoxel*/

#include "SingleVoxel.h"

using namespace std;

SingleVoxel::SingleVoxel(){}

SingleVoxel::SingleVoxel(arma::mat alpha, arma::mat beta, arma::vec invisible, arma::mat bulk, Calibration calib, arma::vec excitationLine){

	bulk_ = bulk;
	invisible_ = invisible;

	elements_ = getUniqueElements(alpha,beta,invisible,bulk);

	rhos_ = arma::ones(elements_.n_elem);
	for(size_t i = 0; i < elements_.n_elem; i++)
		rhos_(i) = ElementDensity(elements_(i),NULL);

	alpha_ = prepareMatrix(alpha, calib);

	if(!beta.is_empty())
		beta_ = prepareMatrix(beta, calib);

	excitation_ = getExcitationVec(excitationLine);

	//checkInvisible(invisible, calib, excitationLine);
}

arma::Mat<double> SingleVoxel::prepareMatrix(arma::Mat<double> lines, Calibration calib){

    arma::Mat<double> matrix__ = lines;
    //arma::vec elements_ = unique(matrix__.col(0));
    
    matrix__.resize( lines.n_rows, lines.n_cols + elements_.n_elem + 2 );

    for(size_t i = 0; i < matrix__.n_rows; i++){
        for(size_t j = 0; j < elements_.n_elem; j++){
            if(elements_(j) == matrix__(i,0)) matrix__(i, 3) = j;
            
			matrix__( i , lines.n_cols + j + 2 ) = CS_Total( elements_(j) , LineEnergy( matrix__(i,0) , matrix__(i,1) , NULL ) , NULL );
			matrix__( i , 4 ) = calib.getSigX( ((int)matrix__(i,0)) , ((int)matrix__(i,1)) );
        }
        if(calib.getCompleteCalFactor( matrix__(i,0) , matrix__(i,1) ) == 0.0)  matrix__( i , lines.n_cols - 1 ) = 0;
        else matrix__( i , lines.n_cols - 1 ) = matrix__( i , lines.n_cols - 1 ) / calib.getCompleteCalFactor( matrix__(i,0) , matrix__(i,1) );
    }

    return matrix__;
}

arma::rowvec SingleVoxel::getExcitationVec(arma::vec excitationLine){
	double excEnergy = LineEnergy( excitationLine(0) , excitationLine(1) , NULL );
	arma::vec excVec_  = arma::ones(elements_.n_elem);
    for(size_t i = 0; i < elements_.n_elem; i++){
        excVec_(i)=CS_Total( elements_(i) , excEnergy , NULL );
    }
	return excVec_.t();
}

arma::vec SingleVoxel::getUniqueElements(arma::mat alpha, arma::mat beta, arma::vec invisible, arma::mat bulk){
	arma::vec temp_ = alpha.col(0);

	if(!beta.is_empty())
		temp_ = arma::join_cols(temp_, beta.col(0));

	if(!invisible.is_empty())
			temp_ = arma::join_cols(temp_, invisible);

	if(!bulk.is_empty())
			temp_ = arma::join_cols(temp_, bulk.col(0));		
	
	return arma::unique(temp_);
}

void SingleVoxel::checkInvisible(arma::vec invisible, Calibration calib, arma::vec excitationLine){


	arma::vec lin = arma::zeros(383);



double excEnergy = LineEnergy( excitationLine(0) , excitationLine(1) , NULL );

cout << endl << "Alpha: " << endl;
	for(size_t i = 0; i < alpha_.n_rows; i++)
		cout << alpha_(i,0) << " " << alpha_(i,1) << " " <<  calib.getCompleteCalFactor(alpha_(i,0),alpha_(i,1)) << endl;

cout << endl << "Beta: " << endl;	
	for(size_t i = 0; i < beta_.n_rows; i++)
		cout << beta_(i,0) << " " << beta_(i,1) << " " <<  calib.getCompleteCalFactor(beta_(i,0),beta_(i,1)) << endl;

cout << endl << "Invisible: " << endl;	
	arma::mat exc_=excLines();
	exc_.resize(exc_.n_rows,exc_.n_cols+2);
	exc_.swap_cols(0,2);

	for(size_t i = 0; i < invisible.n_rows; i++){

		arma::mat el_= exc_;
		arma::vec thisel_ = arma::vec(el_.n_rows);
		thisel_.fill(invisible(i));
		el_.col(0)=thisel_;

		for(size_t j = 0; j < el_.n_rows; j++ ){
			//cout << EdgeEnergy(invisible(i),j,NULL) << endl;
			if( EdgeEnergy(invisible(i),el_(j,1),NULL) >  excEnergy) el_(j,3) = 0;
			else el_(j,3) = calib.getCompleteCalFactor(invisible(i),el_(j,2));
		}	
		
		arma::mat a = el_;
		int shed =0;
		for(size_t j = 0; j < el_.n_rows; j++ ){
			if(el_(j,3)==0){
				a.shed_row(j-shed);
				shed++;
			}		
		}


			
		//arma::sort( (el_.col(3)), "descend").print();
		//arma::sort_index( (el_.col(3)), "descend");
		//cout << arma::sort_index( (el_.col(3)), "descend")(0) << endl;
		//(arma::sort_index(el_.col(3)))).print();
		a.print();
		//cout << invisible(i) << " " << KL3_LINE << " " <<  calib.getCompleteCalFactor(invisible(i),KL3_LINE) << endl;
		//cout << invisible(i) << " " << KM3_LINE << " " <<  calib.getCompleteCalFactor(invisible(i),KM3_LINE) << endl;
		//cout << invisible(i) << " " << L3M5_LINE << " " <<  calib.getCompleteCalFactor(invisible(i),L3M5_LINE) << endl;
		//cout << invisible(i) << " " << L2M4_LINE << " " <<  calib.getCompleteCalFactor(invisible(i),L2M4_LINE) << endl;
	}

}

arma::mat SingleVoxel::excLines(){

	arma::mat shell_lines_ = {
		{0,28},																		// K lines
		{29,57},{58,84},{85,112},													// L lines	
		{113,135},{136,157},{158,179},{180,199},{200,218},							// M lines
	};

	arma::mat lines = arma::zeros(219,2);

	for(int i = 0; i < 219; i++ )
		lines(i,0) =  i*(-1)-1;

	for(size_t i = 0; i < shell_lines_.n_rows; i ++)
		for(int j = shell_lines_(i,0); j < shell_lines_(i,1) + 1 ; j++)
			lines(j,1) = i;
	
	return lines;
}



void SingleVoxel::makeMatrix(int optType, bool useBeta, bool useInvisible, double x){
	if(useBeta)
		matrix_ = arma::join_cols(alpha_,beta_);
	
	else 
		matrix_ = alpha_;

	if(useInvisible && !invisibleInformation_.is_empty())
		matrix_ = arma::join_cols(matrix_, invisibleInformation_);

	matrix_.resize(matrix_.n_rows + 1 ,matrix_.n_cols);			// add row at bottom
	matrix_.insert_rows(0,1);									// add row at top

	for(size_t i = 0; i < elements_.n_elem; i++)					
		matrix_(matrix_.n_rows-1, i+5) = excitation_(i);		// write exc mus to last line
	
	matrix_(0,0) = optType;										
	matrix_(0,1) = x;
}

void SingleVoxel::makeWeightGuess(int weightGuess, bool useBulk){
	
	double totVarWeights = 1.0;
	double nVarElem = elements_.n_elem;
	arma::vec weightGuessVec = arma::zeros(elements_.n_elem);

	if(useBulk && !bulk_.is_empty()){
		totVarWeights = totVarWeights - arma::sum(bulk_.col(1));
		nVarElem -= bulk_.n_rows;	
	}
		
	
	if( weightGuess == 0 ){			// equi
		for(size_t i = 0; i < elements_.n_elem; i++ )
			weightGuessVec(i) = totVarWeights / nVarElem;
	}
	else if (weightGuess == 1 ){	//alpha
		double totAlphaCounts = arma::sum(alpha_.col(2));
		for(size_t i =0; i < alpha_.n_rows; i++)
			weightGuessVec(alpha_(i,3)) = (alpha_(i,2) / totAlphaCounts) * totVarWeights;
	}	

	if(useBulk && !bulk_.is_empty()){
		for(size_t i =0; i < elements_.n_elem; i++){
			for(size_t j = 0; j < bulk_.n_rows; j++){
				if(elements_(i) == bulk_(j,0)){
					weightGuessVec(i) = bulk_(j,1);
				}
			}
		}
	}	

	for(size_t i = 0; i < weightGuessVec.n_elem; i++)
		matrix_(0,i+5) = weightGuessVec(i);	
}

void SingleVoxel::makeRhoGuess(int rhoGuess){
	
	double r = 0.0;
	
	if( rhoGuess == 0 ){
		r = (rhos_.min() + rhos_.max())/2.0;
	}

	else if(rhoGuess == 1){
		arma::vec wG = matrix_.row(0).subvec(5, matrix_.n_cols-1).as_col();
		r = arma::dot(rhos_,wG);
	}

	matrix_(0,2) = r;
}

void SingleVoxel::makeBoundaries( bool useBulk, int boundType){
	
	if( matrix_(0,0) == 0 ){
		bound_ = arma::zeros(elements_.n_elem + 1, 3);
		bound_.col(1).subvec(0,bound_.n_rows-2) = matrix_.row(0).subvec(5, matrix_.n_cols-1).as_col();
		bound_.col(1).back() = matrix_(0,2);
	}
		
	else if( matrix_(0,0) == 1 ){
		bound_ = arma::zeros(1,3);
		bound_(1) = matrix_(0,2);
	}

	else if( matrix_(0,0) == 2 ){
		bound_ = arma::zeros(elements_.n_elem,3);
		bound_.col(1) = matrix_.row(0).subvec(5, matrix_.n_cols-1).as_col();
	}

	if( boundType == 0 ){
		arma::vec t = arma::ones(bound_.n_rows);
		bound_.col(2) = t;
		if( (matrix_(0,0) == 0 ) || (matrix_(0,0) == 1) )
			bound_.col(2).back() = 22.0;

	}
	else if(boundType == 1){
		bound_.col(2) = bound_.col(1)*2;	
	}
	else if(boundType == 2){
		bound_.col(0) = bound_.col(1)*0.9;	
		bound_.col(2) = bound_.col(1)*1.1;	
	}

	if( !(matrix_(0,0) == 1) ){
		if(useBulk && !bulk_.is_empty()){
			for(size_t i =0; i < elements_.n_elem; i++){
				for(size_t j = 0; j < bulk_.n_rows; j++){
					if(elements_(i) == bulk_(j,0)){
						bound_(i,0) = bound_(i,1)*0.99;
						bound_(i,2) = bound_(i,1)*1.01;
					}
				}
			}
		}
	}
	



		//bound_(0) = rhos_.min();
		//bound_(1) = (rhos_.min()+rhos_.max())/2;
		//bound_(2) = rhos_.max();
}

void SingleVoxel::makeProblem(int optType, bool stepnorm, bool useBeta, bool useBulk, bool useInvisible, double x, int rhoGuess, int wGuess, int boundType){
	makeMatrix(optType, useBeta, useInvisible, x);
	makeWeightGuess(wGuess, useBulk);
	makeRhoGuess(rhoGuess);
	makeBoundaries(useBulk, boundType);
	matrix_(0,3) = ((int)stepnorm);
}

void SingleVoxel::makeProblem(int optType, bool stepnorm, bool useBeta, bool useBulk, bool useInvisible, double x, int wGuess, int boundType, double rho){
	makeMatrix(optType, useBeta, useInvisible, x);
	makeWeightGuess(wGuess, useBulk);
	matrix_(0,2) = rho;
	makeBoundaries(useBulk, boundType);
	matrix_(0,3) = ((int)stepnorm);
}

void SingleVoxel::makeProblem(int optType, bool stepnorm, bool useBeta, bool useBulk, bool useInvisible, double x, int rhoGuess, int boundType, arma::vec w){
	makeMatrix(optType, useBeta, useInvisible, x);
	for(size_t i = 0; i < w.n_elem; i++)
		matrix_(0,i+5) = w(i);
	makeRhoGuess(rhoGuess);
	makeBoundaries(useBulk, boundType);
	matrix_(0,3) = ((int)stepnorm);
}

void SingleVoxel::makeProblem(int optType, bool stepnorm, bool useBeta, bool useBulk, bool useInvisible, double x, int boundType, double rho,  arma::vec w){
	makeMatrix(optType, useBeta, useInvisible, x);
	for(size_t i = 0; i < w.n_elem; i++)
		matrix_(0,i+5) = w(i);
	matrix_(0,2) = rho;
	makeBoundaries(useBulk, boundType);
	matrix_(0,3) = ((int)stepnorm);
}

arma::mat SingleVoxel::updateMatrix(int type, arma::vec current){

	if(type == 0){
		matrix_(0,2) = current.back();
		double norm = arma::sum(current.subvec(0, current.n_elem-2));
		for(size_t i = 0; i < current.n_rows -1 ; i++)
			matrix_(0,i+5) = current(i)/norm;	
	}

	if(type == 1 ){
		matrix_(0,2) = current.back();
	}


	if(type == 2){
		double norm = arma::sum(current);
		for(size_t i = 0; i < current.n_rows; i++)
			matrix_(0,i+5) = current(i) / norm;	
	}

	return matrix_;
}

arma::mat SingleVoxel::getMatrix(){ return matrix_;}
arma::mat SingleVoxel::getBoundaries(){ return bound_;}
arma::vec SingleVoxel::getElements(){return elements_;}