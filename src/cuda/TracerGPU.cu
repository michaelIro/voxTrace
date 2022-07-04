
//!  TracerGPU
#include "TracerGPU.cuh"

/** GPU kernel function to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) 
{
  curand_init(seed, 					      // the seed can be the same for each core, here we pass the time in from the CPU 
              blockIdx.x, 				  // the sequence number should be different for each core (unless you want all cores to get the same sequence of numbers for some reason - use thread id!
              0, 						        // the offset is how much extra we advance in the sequence for each call, can be 0 
              &states[blockIdx.x]);
}

// Kernel function to trace rays
__global__ void trace(RayGPU *rays, SampleGPU* sample, curandState_t *states)
{

  //int pos = (blockIdx.x-1)*256 + threadIdx.x;
	RayGPU*	currentRay = &rays[blockIdx.x];
  currentRay->primaryTransform(300000.0, 300000.0,0.0, 5100.0, 45.0); // 300 used to be 70
  currentRay->setStartCoordinates(currentRay->getStartX(), currentRay->getStartY(), currentRay->getStartZ()+60.0);

	VoxelGPU* currentVoxel = sample->findStartVoxel(currentRay);

  int c=0;
  do{

    if(sample->isOOB(currentVoxel->getNN(currentRay->getNextVoxel()))){
      c=c;
    }
    if(c!=0)
      currentVoxel = currentVoxel->getNN(currentRay->getNextVoxel());

    // Check if ray is already out of bounds of sample -> If so, no interaction is possible -> return ray.
	  if(sample->isOOB(currentVoxel))
      currentRay->setOOBFlag(true); 

    else
      TracerGPU::traceForward(currentRay, currentVoxel, &states[blockIdx.x]);

    c++;
    if(currentRay->getIAFlag()){
      c=c;
    }
  }while(!(currentRay->getIAFlag() || currentRay->getOOBFlag()));

  currentRay->setStartCoordinates(currentRay->getStartX(), currentRay->getStartY(), currentRay->getStartZ()-60.0);
  currentRay->secondaryTransform(300000.0, 300000.0, 0.0, 4900.0, 45.0, 950.0); //rin actually 0.095
}

__device__  void TracerGPU::traceForward(RayGPU* ray, VoxelGPU* currentVoxel, curandState_t *localState){
	
	float rayEnergy = ray->getEnergyKeV();
	float muLin = currentVoxel->getMaterial()->CS_Tot_Lin(rayEnergy)/10000.;
	float intersectionLength = currentVoxel->intersect(ray);
  float randomN = curand_uniform(localState);

	// Interaction in this Voxel?
	if(expf(-muLin*intersectionLength) < randomN){

		// Selection of chemical Element to interact with	
    randomN = curand_uniform(localState);
		ChemElementGPU* interactingElement = currentVoxel->getMaterial()->getInteractingElement(rayEnergy,randomN);

		// Selection of interaction type
    randomN = curand_uniform(localState);
		int interactionType = interactingElement->getInteractionType(rayEnergy,randomN);

		if(interactionType == 0){                                                                   // Photo-Absorption
			
			// Selection of excited shell
      randomN = curand_uniform(localState);
			int myShell = interactingElement->getExcitedShell(rayEnergy,randomN);
			
      randomN = curand_uniform(localState);

			if(randomN < interactingElement->Fluor_Y(myShell)){                                      // X-ray-Fluorescence (in case of Auger no further action, except for setting the IA flag                                                                
        randomN = curand_uniform(localState);
				int myLine = interactingElement->getTransition(myShell, randomN);
        randomN = curand_uniform(localState);
				float phi = 2*M_PI*randomN;
        randomN = curand_uniform(localState);
				float theta = acosf(2*randomN-1);
        randomN = curand_uniform(localState);
				float l = intersectionLength*randomN + ray->getTIn();

				float xNew = ray->getStartX()+ray->getDirX()*l;
				float yNew = ray->getStartY()+ray->getDirY()*l;
				float zNew = ray->getStartZ()+ray->getDirZ()*l;
				
				ray->rotate(phi,theta);
				ray->setStartCoordinates(xNew,yNew,zNew);
				ray->setEnergyKeV(interactingElement->Line_Energy(myLine));
			}
		}
		else if(interactionType == 1){	                                                           // Rayleigh-Scattering TODO: Polarized-Unpolarized
      randomN = curand_uniform(localState);
			float phi = 2*M_PI*randomN;
      randomN = curand_uniform(localState);
			float theta = interactingElement->getThetaRayl(rayEnergy,randomN);	
			ray->rotate(phi,theta);
      randomN = curand_uniform(localState);
			float l = intersectionLength*randomN+ ray->getTIn();

			float xNew = ray->getStartX()+ray->getDirX()*l;
			float yNew = ray->getStartY()+ray->getDirY()*l;
			float zNew = ray->getStartZ()+ray->getDirZ()*l;

			ray->setStartCoordinates(xNew,yNew,zNew);
		}
		else if(interactionType == 2){                                                               // Compton-Scattering
      randomN = curand_uniform(localState);
			float phi = 2*M_PI*randomN;
      randomN = curand_uniform(localState);
			float theta = interactingElement->getThetaCompt(rayEnergy,randomN);	

			ray->rotate(phi,theta);
      randomN = curand_uniform(localState);
			float l = intersectionLength*randomN+ ray->getTIn();

			float xNew = ray->getStartX()+ray->getDirX()*l;
			float yNew = ray->getStartY()+ray->getDirY()*l;
			float zNew = ray->getStartZ()+ray->getDirZ()*l;

			ray->setStartCoordinates(xNew,yNew,zNew);
		}
		
    ray->setIANum(ray->getIANum()+1);
		ray->setIAFlag(true);
	}	
}

void TracerGPU::callTrace(){

  static constexpr int n_elements = 6;

  float x_=0.0, y_=0.0,z_=0.0;
  float xL_=600000.0,yL_=600000.0,zL_=3000.0;
  float xLV_=60000.0,yLV_=60000.0,zLV_=10.0;

  int xN_ = (int)(xL_/xLV_); // (...)+1 ????
  int yN_ = (int)(yL_/yLV_);
  int zN_ = (int)(zL_/zLV_);

  xLV_ = xL_/((float)(xN_)); // (*N_-1) ???
  yLV_ = yL_/((float)(yN_));
  zLV_ = zL_/((float)(zN_));

  ChemElementGPU* elements;
  cudaMallocManaged(&elements, sizeof(ChemElementGPU)*n_elements);

  float* weights;
  cudaMallocManaged(&weights, sizeof(float)*n_elements*xN_*yN_*zN_);

  MaterialGPU* materials;
  cudaMallocManaged(&materials, sizeof(MaterialGPU)*xN_*yN_*zN_);
  
  VoxelGPU* voxels; 
  cudaMallocManaged(&voxels, sizeof(VoxelGPU)*xN_*yN_*zN_);

  VoxelGPU* oobVoxel; 
  cudaMallocManaged(&oobVoxel, sizeof(VoxelGPU));

  SampleGPU* sample;
  cudaMallocManaged(&sample, sizeof(SampleGPU));

  ChemElementGPU fe(26);
  ChemElementGPU ni(28);
  ChemElementGPU cu(29);
  ChemElementGPU zn(30);
  ChemElementGPU sn(50);
  ChemElementGPU pb(82);

  size_t a = cu.getMemorySize();

  elements[0] = cu;
  elements[1] = fe;
  elements[2] = pb;
  elements[3] = ni;
  elements[4] = sn;
  elements[5] = zn;
  
  for(int i = 0; i < xN_; i++){
    for(int j = 0; j < yN_; j++){
      for(int k = 0; k < zN_; k++){
        weights[i*yN_*zN_*n_elements+j*zN_*n_elements+k*n_elements+0] = 0.6119;
        weights[i*yN_*zN_*n_elements+j*zN_*n_elements+k*n_elements+1] = 0.0004;
        weights[i*yN_*zN_*n_elements+j*zN_*n_elements+k*n_elements+2] = 0.0019;
        weights[i*yN_*zN_*n_elements+j*zN_*n_elements+k*n_elements+3] = 0.0010;
        weights[i*yN_*zN_*n_elements+j*zN_*n_elements+k*n_elements+4] = 0.0107;
        weights[i*yN_*zN_*n_elements+j*zN_*n_elements+k*n_elements+5] = 0.3741;

        materials[i*yN_*zN_+j*zN_+k] = MaterialGPU(n_elements, elements, &weights[i*yN_*zN_*n_elements+j*zN_*n_elements+k*n_elements+0]);
        voxels[i*yN_*zN_+j*zN_+k] = VoxelGPU(x_+i*xLV_, y_+j*yLV_, z_+k*zLV_, xLV_, yLV_, zLV_,&materials[i*yN_*zN_+j*zN_+k]);
      }
    }
  }

  *oobVoxel = VoxelGPU(-1.,-1.,-1.,-1.,-1.,-1.,&(materials[0]));
  *sample = SampleGPU(x_, y_,  z_, xL_,  yL_, zL_,  xLV_,  yLV_,  zLV_, xN_, yN_,  zN_, voxels, oobVoxel);

  std::string path = "/tank/data/";

  for (const auto & file : std::filesystem::directory_iterator(path)){

	  std::string pathname = file.path();
	  std::cout << pathname << std::endl;
    std::string path_out = "/media/miro/Data/" + file.path().filename().string() + "-pos-5.h5";
    if(std::filesystem::exists(path_out)){
      std::cout << "File already exists" << std::endl;
    }

    else{
      clock_t begin = clock();

	    arma::Mat<double> beam_;
      beam_.load(arma::hdf5_name(pathname, "my_data")); 

      std::cout << "beam_.n_rows: " << beam_.n_rows << std::endl;

      int N = beam_.n_rows;

      //int threads = 1024;
      //int blocks= N/threads;

      curandState_t* states;
      cudaMallocManaged(&states, N*sizeof(curandState_t));

      init<<<N, 1>>>(775289, states); //time(0)

      RayGPU* rays;
      cudaMallocManaged(&rays, beam_.n_rows*sizeof(RayGPU));

     for(int i = 0; i < beam_.n_rows; i++){
	  	  rays[i] = RayGPU(beam_(i,0),beam_(i,1),beam_(i,2),beam_(i,3),beam_(i,4),beam_(i,5),beam_(i,6),beam_(i,7),beam_(i,8),beam_(i,9),
	  	  	beam_(i,10),beam_(i,11),beam_(i,12),beam_(i,13),beam_(i,14),beam_(i,15),beam_(i,16),beam_(i,17),beam_(i,18));
      }

      clock_t middle = clock();

      trace<<<N,1>>>(rays,sample, states);


      cudaDeviceSynchronize();

      clock_t end = clock();
      double time_spent_to_read = (double)(middle - begin) / CLOCKS_PER_SEC;
      double time_spent_to_trace= (double)(end - middle) / CLOCKS_PER_SEC;
      printf("Read / Prepare: %f s \t Trace: %f s\n", time_spent_to_read, time_spent_to_trace);

      int success=0;
	    for(int i = 0; i < beam_.n_rows; i++){
        if(rays[i].getIAFlag())
          success++;
	    }

	    arma::Mat<double> rays__;
      rays__ = arma::ones(success+1, 19);
      success=0;
      
      for(int i = 0; i < beam_.n_rows; i++){
        if(rays[i].getIAFlag()){
          rays__.row(success++)=arma::rowvec({
			      rays[i].getStartX(),rays[i].getStartY(),rays[i].getStartZ(),
			      rays[i].getDirX(),rays[i].getDirY(),rays[i].getDirZ(),
			      rays[i].getSPolX(),rays[i].getSPolY(),rays[i].getSPolZ(),
			      (double) rays[i].getFlag(), rays[i].getWaveNumber(),(double) rays[i].getIndex(),
			      rays[i].getOpticalPath(),rays[i].getSPhase(),rays[i].getPPhase(),
			      rays[i].getPPolX(), rays[i].getPPolY(),rays[i].getPPolZ(),
			      rays[i].getProb()});
        }
	    }

      rays__.save(arma::hdf5_name(path_out,"my_data"));

      std::cout << "Size at 2nd-Polycap-Entry: " << success << std::endl;

	    cudaFree(rays);
      cudaFree(states);
    }
  }
 
  //free all memory cuda
  cudaFree(elements);
  cudaFree(materials);
  cudaFree(voxels);
  cudaFree(oobVoxel);
  cudaFree(sample);
  cudaFree(weights);
  
}



// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
  //for (int i = 0; i < n; i++)
    y[threadIdx.x] = x[threadIdx.x] + y[threadIdx.x];
}

// Kernel function to trace rays
__global__ void test(RayGPU *primary_rays,float *x, float *y, SampleGPU* sample, curandState_t *states)
{
  y[threadIdx.x] = sample->getOOBVoxel()->getMaterial()->CS_Tot(0.5);
  // generate a random number between 0 and 1 with cuRand
	//float r = curand_uniform (&states[threadIdx.x]);
  //y[threadIdx.x] = r;
}

void TracerGPU::callAdd(){

  int N = 16;
  float *x, *y;

  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    	x[i] = 1.0f;
    	y[i] = 2.0f;
  }

  std::cout<< "Before:"<<std::endl;
  for (int i = 0; i < N; i++) 
	  std::cout << "\t x[i]: " << x[i] << "\t y[i]:" << y[i] << std::endl;

  add<<<1,16>>>(16,x,y);

  cudaDeviceSynchronize();

  std::cout<<std::endl<<std::endl<< "After:"<<std::endl;
  for (int i = 0; i < N; i++) 
	  std::cout << "\t x[i]: " << x[i] << "\t y[i]:" << y[i] << std::endl;

  //free all memory cuda
  cudaFree(x);
  cudaFree(y);
  
}

void TracerGPU::callTest(){
    //std::cout << "beam.ptr" << beam->memptr() << std::endl;

	//const arma::Mat<double> & beam_ = *beam;    
	arma::Mat<double> beam_;// = new arma::Mat<double>();
    
    beam_.load(arma::hdf5_name("/tank/data/PrimaryBeam-2-0.h5", "my_data")); 
    std::cout << "beam_.n_rows: " << beam_.n_rows << std::endl;
    //std::cout << "beam.ptr" << beam.memptr() << std::endl;


  RayGPU* primary_rays;
  //RayGPU* secondary_rays;

  cudaMallocManaged(&primary_rays, beam_.n_rows*sizeof(RayGPU));
  //cudaMallocManaged(&secondary_rays, beam_.n_rows*sizeof(RayGPU));

	std::cout<< "HIER" << beam_.n_rows<<std::endl;

		double  tetstl= beam_(0,0);
  for(int i = 0; i < beam_.n_rows; i++){
	primary_rays[i] = RayGPU(beam_(i,0),beam_(i,1),beam_(i,2),beam_(i,3),beam_(i,4),beam_(i,5),beam_(i,6),beam_(i,7),beam_(i,8),beam_(i,9),
	beam_(i,10),beam_(i,11),beam_(i,12),beam_(i,13),beam_(i,14),beam_(i,15),beam_(i,16),beam_(i,17),beam_(i,18));
  }

  for(int i = 0; i < 7; i++)
	primary_rays[i].print();


  // Test-Area
  int N = beam_.n_rows;
  float *x, *y;

  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    	x[i] = 1.0f;
    	y[i] = 2.0f;
  }

  //std::cout<< "Before:"<<std::endl;
  //for (int i = 0; i < N; i++) 
//	  std::cout << "\t x[i]: " << x[i] << "\t y[i]:" << y[i] << std::endl;
  


  int n_elements = 3;
  //int n_threads = 3;

  float x_=0.0, y_=0.0,z_=0.0;
  float xL_=150.0,yL_=150.0,zL_=5.0;
  float xLV_=15.0,yLV_=15.0,zLV_=0.5;
  int xN_ = (int)(xL_/xLV_)+1;
  int yN_ = (int)(yL_/yLV_)+1;
  int zN_ = (int)(zL_/zLV_)+1;
  //int voxN_= xN_*yN_*zN_;

  xLV_ = xL_/((float)(xN_-1));
  yLV_ = yL_/((float)(yN_-1));
  zLV_ = zL_/((float)(zN_-1));

  ChemElementGPU* elements;
  cudaMallocManaged(&elements, sizeof(ChemElementGPU)*n_elements);

  float* weights;
  cudaMallocManaged(&weights, sizeof(float)*n_elements*xN_*yN_*zN_);

  MaterialGPU* materials;
  cudaMallocManaged(&materials, sizeof(MaterialGPU)*n_elements*xN_*yN_*zN_);
  
  VoxelGPU* voxels; 
  cudaMallocManaged(&voxels, sizeof(VoxelGPU)*n_elements*xN_*yN_*zN_);

  VoxelGPU* oobVoxel; 
  cudaMallocManaged(&oobVoxel, sizeof(VoxelGPU));

  SampleGPU* sample;
  cudaMallocManaged(&sample, sizeof(SampleGPU));

  curandState_t* states;
  cudaMallocManaged(&states, N*sizeof(curandState_t));

  init<<<N, 1>>>(time(0), states);

  ChemElementGPU cu(29);
  ChemElementGPU sn(50);
  ChemElementGPU pb(82);

  elements[0] = cu;
  elements[1] = sn;
  elements[2] = pb;
  
  for(int i = 0; i < xN_; i++){
    for(int j = 0; j < yN_; j++){
      for(int k = 0; k < zN_; k++){
        weights[i*yN_*zN_*3+j*zN_*3+k*3+0] = i*yN_*zN_*3+j*zN_*3+k*3+0;
        weights[i*yN_*zN_*3+j*zN_*3+k*3+1] = i*yN_*zN_*3+j*zN_*3+k*3+1;
        weights[i*yN_*zN_*3+j*zN_*3+k*3+2] = i*yN_*zN_*3+j*zN_*3+k*3+2;

        materials[i*yN_*zN_+j*zN_+k] = MaterialGPU(n_elements, elements, &weights[i*yN_*zN_*3+j*zN_*3+k*3+0]);
        voxels[i*yN_*zN_+j*zN_+k] = VoxelGPU(x_+i*xLV_, y_+j*yLV_, z_+k*zLV_, xLV_, yLV_, zLV_,&materials[i*yN_*zN_+j*zN_+k]);
      }
    }
  }

  *oobVoxel = VoxelGPU(-1.,-1.,-1.,-1.,-1.,-1.,&(materials[0]));
  *sample = SampleGPU(x_, y_,  z_, xL_,  yL_, zL_,  xLV_,  yLV_,  zLV_, xN_, yN_,  zN_, voxels, oobVoxel);

  //test<<<1,16>>>(primary_rays,x,y,sample, states);
  trace<<<N,1>>>(primary_rays,sample, states);
  cudaDeviceSynchronize();

  //std::cout<<std::endl<<std::endl<< "After:"<<std::endl;
  //for (int i = 0; i < N; i++) 
	//  std::cout << "\t x[i]: " << x[i] << "\t y[i]:" << y[i] << std::endl;

	for(int i = 0; i < beam_.n_rows; i++){
		if(primary_rays[i].getIANum() != 0)
			primary_rays[i].print();

	}


  //free all memory cuda
  cudaFree(x);
  cudaFree(y);
  cudaFree(elements);
  cudaFree(materials);
  cudaFree(voxels);
  cudaFree(oobVoxel);
  cudaFree(sample);
  cudaFree(weights);
  
}