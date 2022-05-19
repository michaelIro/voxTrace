
//!  TracerGPU
#include "TracerGPU.cuh"


/*#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}*/

/** GPU kernel function to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) 
{
  curand_init(seed, 					      // the seed can be the same for each core, here we pass the time in from the CPU 
              blockIdx.x, 				  // the sequence number should be different for each core (unless you want all cores to get the same sequence of numbers for some reason - use thread id!
              0, 						        // the offset is how much extra we advance in the sequence for each call, can be 0 
              &states[blockIdx.x]);
}

/** GPU kernel function to create RayGPU objects from armadillo matrix 
__global__ void raygen(RayGPU* rays,  arma::Mat<double> beam_) 
{
      int i = blockIdx
		  rays[i] = RayGPU(beam_(i,0),beam_(i,1),beam_(i,2),beam_(i,3),beam_(i,4),beam_(i,5),beam_(i,6),beam_(i,7),beam_(i,8),beam_(i,9),
	  		beam_(i,10),beam_(i,11),beam_(i,12),beam_(i,13),beam_(i,14),beam_(i,15),beam_(i,16),beam_(i,17),beam_(i,18));
}*/ 

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
  //for (int i = 0; i < n; i++)
    y[threadIdx.x] = x[threadIdx.x] + y[threadIdx.x];
}

// Kernel function to trace rays
__global__ void trace(RayGPU *rays, SampleGPU* sample, curandState_t *states)
{
	RayGPU*	currentRay = &rays[blockIdx.x];
  currentRay->primaryTransform(70.0, 70.0,0.0, 0.51, 45.0);
	VoxelGPU* currentVoxel = sample->findStartVoxel(currentRay);
	int nextVoxel = 13;	
	*currentRay = *TracerGPU::traceForward(currentRay, currentVoxel,&nextVoxel, sample,&states[blockIdx.x]);
  currentRay->secondaryTransform(70.0, 70.0,0.0, 0.49, 45.0);
	//printf("BlockIDX: %p ThreadIDX: %p\n",blockIdx.x,threadIdx.x);
	//printf("ThreadIDX %p\n",threadIdx.x);
}

// Kernel function to trace rays
__global__ void test(RayGPU *primary_rays,float *x, float *y, SampleGPU* sample, curandState_t *states)
{
    y[threadIdx.x] = sample->getOOBVoxel()->getMaterial()->CS_Tot(0.5);
    // generate a random number between 0 and 1 with cuRand
	//float r = curand_uniform (&states[threadIdx.x]);
    //y[threadIdx.x] = r;

	//traceForward(primary_rays[threadIdx.x], sample, states[threadIdx.x]);

	RayGPU*	currentRay = &primary_rays[threadIdx.x];
	VoxelGPU* currentVoxel = sample->findStartVoxel(currentRay);
	int nextVoxel = 13;	
	currentRay = TracerGPU::traceForward(currentRay, currentVoxel,&nextVoxel, sample,&states[threadIdx.x]);

	//RayGPU* aNewRay 
	//tracedRays[i++]=(*aNewRay);

    //printf("BlockIDX %f\n",r);
    //cuRandGenerateUniform(x, y, 1);
}

__device__  RayGPU* TracerGPU::traceForward(RayGPU* ray, VoxelGPU* currentVoxel, int* nextVoxel, SampleGPU *sample, curandState_t *localState){

	// Check if ray is already out of bounds of sample -> If so, no interaction is possible -> return ray.
	if(sample->isOOB(currentVoxel))
		return ray;
		
	float tIn;
	float rayEnergy = ray->getEnergyKeV();
	float muLin = currentVoxel->getMaterial()->CS_Tot_Lin(rayEnergy);
	float intersectionLength = currentVoxel->intersect(ray,nextVoxel,&tIn);
	float randomN = curand_uniform(localState);

	// Interaction in this Voxel?
	if(expf(-muLin*intersectionLength) < curand_uniform(localState)){

		// Selection of chemical Element to interact with	
		ChemElementGPU* interactingElement = currentVoxel->getMaterial()->getInteractingElement(rayEnergy,curand_uniform(localState));

		// Selection of interaction type
		int interactionType = interactingElement->getInteractionType(rayEnergy,curand_uniform(localState));

		if(interactionType == 0){ // Photo-Absorption
			
			// Selection of excited shell
			randomN = curand_uniform (localState);
			int myShell = interactingElement->getExcitedShell(rayEnergy,randomN);
			
			randomN = curand_uniform (localState);
			if(randomN > interactingElement->Fluor_Y(myShell)){ // Auger-Effect

				ray->setIANum(ray->getIANum()+1);
				ray->setIAFlag(true);
				ray->setFlag(false);
			}
			else{ // X-ray-Fluorescence

				randomN = curand_uniform (localState);
				int myLine = interactingElement->getTransition(myShell, randomN);

				randomN = curand_uniform (localState);
				float phi = 2*M_PI*randomN;

				randomN = curand_uniform (localState);
				float theta = acosf(2*randomN-1);

				randomN = curand_uniform (localState);
				float l = intersectionLength*randomN + tIn;

				float xNew = ray->getStartX()+ray->getDirX()*l;
				float yNew = ray->getStartY()+ray->getDirY()*l;
				float zNew = ray->getStartZ()+ray->getDirZ()*l;
				
				ray->rotate(phi,theta);
				ray->setStartCoordinates(xNew,yNew,zNew);
				ray->setEnergyKeV(interactingElement->Line_Energy(myLine));
				ray->setIANum(ray->getIANum()+1);
				ray->setIAFlag(true);
			}
		}
		else if(interactionType == 1){	// Rayleigh-Scattering TODO: Polarized-Unpolarized

			randomN = curand_uniform (localState);
			float phi = 2*M_PI*randomN;
				
			randomN = curand_uniform (localState);
			float theta = interactingElement->getThetaRayl(rayEnergy,randomN);	
	
			ray->rotate(phi,theta);

			randomN = curand_uniform (localState);
			float l = intersectionLength*randomN + tIn;

			float xNew = ray->getStartX()+ray->getDirX()*l;
			float yNew = ray->getStartY()+ray->getDirY()*l;
			float zNew = ray->getStartZ()+ray->getDirZ()*l;
			ray->setStartCoordinates(xNew,yNew,zNew);
			ray->setIANum(ray->getIANum()+1);
			ray->setIAFlag(true);
		}
		else if(interactionType == 2){ // Compton-Scattering

			randomN = curand_uniform (localState);
			float phi = 2*M_PI*randomN;

			randomN = curand_uniform (localState);
			float theta = interactingElement->getThetaCompt(rayEnergy,randomN);	

			ray->rotate(phi,theta);

			randomN = curand_uniform (localState);
			float l = intersectionLength*randomN + tIn;

			float xNew = ray->getStartX()+ray->getDirX()*l;
			float yNew = ray->getStartY()+ray->getDirY()*l;
			float zNew = ray->getStartZ()+ray->getDirZ()*l;
			ray->setStartCoordinates(xNew,yNew,zNew);
			ray->setIANum(ray->getIANum()+1);
			ray->setIAFlag(true);
		}
	}
	else{ // No interaction happening in this Voxel.
		currentVoxel =  currentVoxel->getNN(*nextVoxel);
		if(sample->isOOB(currentVoxel)) 
			ray->setFlag(false);
	}


	if((*ray).getFlag()){
		return traceForward(ray, currentVoxel,nextVoxel,sample,localState);
	}
	else {
		return ray;
	}
												
}

void TracerGPU::callTrace(){

  //int n_threads = 3;
  //int n_blocks = 3;

  int n_elements = 3;

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

  ChemElementGPU cu(29);
  ChemElementGPU sn(50);
  ChemElementGPU pb(82);

  elements[0] = cu;
  elements[1] = sn;
  elements[2] = pb;
  
  for(int i = 0; i < xN_; i++){
    for(int j = 0; j < yN_; j++){
      for(int k = 0; k < zN_; k++){
        weights[i*yN_*zN_*3+j*zN_*3+k*3+0] = 0.9;
        weights[i*yN_*zN_*3+j*zN_*3+k*3+1] = 0.05;
        weights[i*yN_*zN_*3+j*zN_*3+k*3+2] = 0.05;

        materials[i*yN_*zN_+j*zN_+k] = MaterialGPU(n_elements, elements, &weights[i*yN_*zN_*3+j*zN_*3+k*3+0]);
        voxels[i*yN_*zN_+j*zN_+k] = VoxelGPU(x_+i*xLV_, y_+j*yLV_, z_+k*zLV_, xLV_, yLV_, zLV_,&materials[i*yN_*zN_+j*zN_+k]);
      }
    }
  }

  *oobVoxel = VoxelGPU(-1.,-1.,-1.,-1.,-1.,-1.,&(materials[0]));
  *sample = SampleGPU(x_, y_,  z_, xL_,  yL_, zL_,  xLV_,  yLV_,  zLV_, xN_, yN_,  zN_, voxels, oobVoxel);

  //std::string path = "/tank/data/";
  std::string path = "/media/miro/Data/Documents/TU Wien/VSC-BEAM/";

  for (const auto & file : std::filesystem::directory_iterator(path)){
	  std::string pathname = file.path();
	  std::cout << pathname << std::endl;

	  arma::Mat<double> beam_;	// = new arma::Mat<double>();
    beam_.load(arma::hdf5_name(pathname, "my_data")); 
    std::cout << "beam_.n_rows: " << beam_.n_rows << std::endl;
    int N = beam_.n_rows;


    curandState_t* states;
    cudaMallocManaged(&states, N*sizeof(curandState_t));

    init<<<N, 1>>>(time(0), states);

    RayGPU* rays;
    int success=0;

    cudaMallocManaged(&rays, beam_.n_rows*sizeof(RayGPU));
    //cudaMallocManaged(&s_rays, beam_.n_rows*sizeof(RayGPU));
	  clock_t begin = clock();

    for(int i = 0; i < beam_.n_rows; i++){
		  rays[i] = RayGPU(beam_(i,0),beam_(i,1),beam_(i,2),beam_(i,3),beam_(i,4),beam_(i,5),beam_(i,6),beam_(i,7),beam_(i,8),beam_(i,9),
	  		beam_(i,10),beam_(i,11),beam_(i,12),beam_(i,13),beam_(i,14),beam_(i,15),beam_(i,16),beam_(i,17),beam_(i,18));
    }

    clock_t middle = clock();


    //std::chrono::steady_clock::time_point t1_ = std::chrono::steady_clock::now();

    //std::cout << "READ FILE FOR: " << t1_-t0_ << std::endl;

    trace<<<N,1>>>(rays,sample, states);


    cudaDeviceSynchronize();

    clock_t end = clock();
    double time_spent = (double)(end - middle) / CLOCKS_PER_SEC;
    double time_spent1 = (double)(middle - begin) / CLOCKS_PER_SEC;
    printf("Read: %f Trace: %f\n", time_spent, time_spent1);

	  for(int i = 0; i < beam_.n_rows; i++){
			//p_rays[i].secondaryTransform(70.0, 70.0,0.0, 0.49, 45.0);
      if(rays[i].getIAFlag())
        success++;
	  }
    std::cout << "sucess: " << success << std::endl;

	  cudaFree(rays);
  }
 
  //free all memory cuda
  cudaFree(elements);
  cudaFree(materials);
  cudaFree(voxels);
  cudaFree(oobVoxel);
  cudaFree(sample);
  cudaFree(weights);
  
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