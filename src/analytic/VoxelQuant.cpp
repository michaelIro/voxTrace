/**Test Quantificatione*/

#include "VoxelQuant.h"

using namespace std;

VoxelQuant::VoxelQuant(){}

VoxelQuant::VoxelQuant(SingleVoxel voxel){ vox_ = voxel; }

double VoxelQuant::evaluate(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data){
    SingleVoxel* m0_ = static_cast<SingleVoxel*> (opt_data); 
    arma::Mat<double> m1_ = m0_->getMatrix();
    arma::Mat<double>* m_ = &m1_;
    //arma::Mat<double>* m_ = static_cast<arma::Mat<double> *> (opt_data); 
    arma::vec w_;
    double rho_;

    if( (*m_)(0,0) == 0 ){
        rho_ = vals_inp.back();
        w_ = vals_inp;
        w_.shed_row( w_.n_elem - 1 );
    }
    else if ( (*m_)(0,0) == 1 ){
        rho_ = vals_inp(0);
        w_ = (*m_).row(0).as_col();
        w_ = w_.subvec(5,w_.n_elem-1);
    }
    else if ( (*m_)(0,0) == 2 ){
        w_= vals_inp;
        rho_ = (*m_)(0,1);
    }

    arma::mat mus_ = (*m_).submat(1,5,(*m_).n_rows-2, (*m_).n_cols-1);
    arma::vec phis_ = (*m_).col(2).subvec(1, (*m_).n_rows-2);
    arma::rowvec exc_ = (*m_).row((*m_).n_rows-1).subvec(5,(*m_).n_cols-1);

    arma::vec mu_tot_ = (mus_*w_ + arma::as_scalar(exc_*w_)) * rho_ / sqrt(2); 
    arma::vec absorptions_ = arma::ones(mu_tot_.n_elem);

    for (size_t i = 0; i < mu_tot_.n_elem; i++)
        absorptions_(i) = Sensitivity::getAbs(mu_tot_(i), 0.000860395460856518, 0.000860395460856518, INFINITY);

    for (size_t i = 0; i < mu_tot_.n_elem; i++)
        absorptions_(i) = ( absorptions_(i) * rho_ ) * w_( (*m_)(i+1,3) );


    arma::vec epsilon_ = (phis_-absorptions_);
    for (size_t i = 0; i < epsilon_.n_elem; i++)
        epsilon_(i) = epsilon_(i); //* (*m_)(i+1,2); 

    //for (int i = 0; i < epsilon_.n_elem; i++)
    //    epsilon__(i) = epsilon__ * calib.getCompleteCalFactor( matrix__(i,0) , matrix__(i,1) )

    double chisq = dot(epsilon_,epsilon_);
    if((*m_)(0,3) == 1)
        chisq += pow(arma::sum(w_)-1,2);

    return chisq;
}

void VoxelQuant::makeProblem(int optType, bool stepnorm, bool useBeta, bool useBulk, bool useInvisible, double x, int rhoGuess, int wGuess, int boundType){
   
    vox_.makeProblem(optType, stepnorm, useBeta, useBulk, useInvisible, x, rhoGuess,wGuess, boundType);

    fetchProblem();
    printProblem();

    s_.vals_bound = true;
    s_.lower_bounds = b_.col(0);
    s_.upper_bounds = b_.col(2);
    s_.de_settings.initial_lb = b_.col(0);
    s_.de_settings.initial_ub = b_.col(2);
    s_.de_settings.mutation_method = 1; 
    s_.de_settings.max_fn_eval = 1000000000;   
    s_.print_level = 0;
    s_.iter_max = 100000;
    s_.de_settings.n_gen = 1000;
}

void VoxelQuant::fetchProblem(){
    m_ = vox_.getMatrix();
    b_ = vox_.getBoundaries();
    v_= b_.col(1);
}

void VoxelQuant::printProblem(){
    cout << endl << "Problem-Matrix: " << endl; 
    m_.print();

    cout << endl << "Boundary-Matrix: " << endl;  
    b_.print();
}

void VoxelQuant::run(){

    const clock_t begin_time = clock();
    bool success = optim::de( v_, VoxelQuant::evaluate, &vox_, s_ );
    //bool success = optim::bfgs(v_, VoxelQuant::evaluate, &m_, s_ );
    const double seconds =  float( clock () - begin_time ) /  CLOCKS_PER_SEC;

    cout << endl << "Calculation Time: " << seconds << endl;
}

void VoxelQuant::printSolution(){
    cout << endl << "Result: " << endl;
    /*for(int i=0; i < vox_.getElements().n_elem; i++){
        ChemElement curEl(vox_.getElements()(i));
        cout << "\t" << curEl.getSymbol() << " - " << vox_.getElements()(i) << "\t : \t" << v_(i) << endl;
    }
    if(vox_.getElements().n_elem < v_.n_elem)
        cout << "Rho: " << v_.back() << endl;*/
    cout << endl << endl;
}