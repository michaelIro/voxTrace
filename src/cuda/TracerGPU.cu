
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
__global__ void tracePreBeam(RayGPU *rays, SampleGPU* sample, curandState_t *states)
{

  RayGPU*	currentRay = &rays[blockIdx.x];

  currentRay->primaryTransform(300000.0, 300000.0,0.0, 5100.0, 45.0);
  currentRay->setStartCoordinates(currentRay->getStartX(), currentRay->getStartY(), currentRay->getStartZ());

	VoxelGPU* currentVoxel = sample->findStartVoxel(currentRay);

  int c=0;
  do{

    if(c!=0)
      currentVoxel = currentVoxel->getNN(currentRay->getNextVoxel());

    // Check if ray is already out of bounds of sample -> If so, no interaction is possible -> return ray.
	  if(sample->isOOB(currentVoxel) || currentRay->getEnergyKeV() < 1.0|| currentRay->getEnergyKeV() > 19.0)
      currentRay->setOOBFlag(true); 

    else
      TracerGPU::traceForward(currentRay, currentVoxel, &states[blockIdx.x]);

    c++;

  }while(!(currentRay->getOOBFlag())); //currentRay->getIAFlag() currentRay->getIANum() < 5 || 

  currentRay->setStartCoordinates(currentRay->getStartX(), currentRay->getStartY(), currentRay->getStartZ());
  currentRay->secondaryTransform(300000.0, 300000.0, 0.0, 4900.0, 45.0, 950.0); //rin actually 0.095
}

// Kernel function to trace rays
__global__ void traceNewBeam(RayGPU *rays, SampleGPU* sample, curandState_t *states, float* offset, float* prim_trans_param, float* sec_trans_param, float* prim_cap_geom)
{
  RayGPU*	currentRay = &rays[blockIdx.x];
  do{
    currentRay->raiseRespawnCounter();
    currentRay->generateRayGPU(&states[blockIdx.x], prim_cap_geom[0], prim_cap_geom[1], prim_cap_geom[2], prim_cap_geom[3]);
    currentRay->primaryTransform(prim_trans_param[0], prim_trans_param[1], prim_trans_param[2], prim_trans_param[3], prim_trans_param[4]);
    currentRay->setStartCoordinates(currentRay->getStartX()+offset[0], currentRay->getStartY()+offset[1], currentRay->getStartZ()+offset[2]);

	  VoxelGPU* currentVoxel = sample->findStartVoxel(currentRay);

    int c=0;
    do{

      if(c!=0)
        currentVoxel = currentVoxel->getNN(currentRay->getNextVoxel());

      // Check if ray is already out of bounds of sample -> If so, no interaction is possible -> return ray.
	    if(sample->isOOB(currentVoxel) || currentRay->getEnergyKeV() < 1.0|| currentRay->getEnergyKeV() > 19.0)
        currentRay->setOOBFlag(true); 

      else
        TracerGPU::traceForward(currentRay, currentVoxel, &states[blockIdx.x]);

      c++;

      if(currentRay->getOOBFlag()) 
        c=c;

    }while(!(currentRay->getOOBFlag()));

    currentRay->setStartCoordinates(currentRay->getStartX()-offset[0], currentRay->getStartY()-offset[1], currentRay->getStartZ()-offset[2]);
    currentRay->secondaryTransform(sec_trans_param[0], sec_trans_param[1], sec_trans_param[2], sec_trans_param[3], sec_trans_param[4], sec_trans_param[5]);
    
    if(currentRay->getIAFlag() && currentRay->getIndex()==0) 
        c=c;
  }while(!(currentRay->getIAFlag()));
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
				float l = intersectionLength*randomN + ray->getTIn();
        randomN = curand_uniform(localState);
				int myLine = interactingElement->getTransition(myShell, randomN);
        randomN = curand_uniform(localState);
				float phi = 2*M_PI*randomN;
        randomN = curand_uniform(localState);
				float theta = acosf(2*randomN-1);
        //float theta = M_PI* randomN; 

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
      float l = intersectionLength*randomN + ray->getTIn();

      float xNew = ray->getStartX()+ray->getDirX()*l;
			float yNew = ray->getStartY()+ray->getDirY()*l;
			float zNew = ray->getStartZ()+ray->getDirZ()*l;

      ray->setStartCoordinates(xNew,yNew,zNew);

      randomN = curand_uniform(localState);
			float phi = 2*M_PI*randomN;
      randomN = curand_uniform(localState);
			float theta = interactingElement->getThetaRayl(rayEnergy,randomN);	
			ray->rotate(phi,theta);

		}
		else if(interactionType == 2){                                    // Compton-Scattering
      randomN = curand_uniform(localState);
      float l = intersectionLength*randomN + ray->getTIn();       

      float xNew = ray->getStartX()+ray->getDirX()*l;
			float yNew = ray->getStartY()+ray->getDirY()*l;
			float zNew = ray->getStartZ()+ray->getDirZ()*l;

			ray->setStartCoordinates(xNew,yNew,zNew);    


      randomN = curand_uniform(localState);
			float phi = 2*M_PI*randomN;
      randomN = curand_uniform(localState);
			float theta = interactingElement->getThetaCompt(rayEnergy,randomN);	

			ray->rotate(phi,theta);

      float eNew = interactingElement->getComptEnergy(ray->getEnergyKeV(),theta);


      ray->setEnergyKeV(eNew);
		}
		
    ray->setIANum(ray->getIANum()+1);
		ray->setIAFlag(true);
	}	
}

void TracerGPU::callTracePreBeam(){

  static constexpr int n_elements = 6;

  float x_=0.0, y_=0.0,z_=0.0;
  float xL_=600000.0,yL_=600000.0,zL_=3000.0;
  float xLV_=60000.0,yLV_=60000.0,zLV_=10.0;

  int xN_ = (int)(xL_/xLV_);
  int yN_ = (int)(yL_/yLV_);
  int zN_ = (int)(zL_/zLV_);

  xLV_ = xL_/((float)(xN_)); 
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
  //std::string path = "/gpfs/data/fs71764/miro/work/out";

  for (const auto & file : std::filesystem::directory_iterator(path)){

	  std::string pathname = file.path();
	  std::cout << pathname << std::endl;
    std::string path_out = "/media/miro/Data/" + file.path().filename().string() + "-pos-5.h5";
    //std::string path_out = "/gpfs/data/fs71764/miro/work/nist-1107/pos-0/" + file.path().filename().string() + ".h5";
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

      tracePreBeam<<<N,1>>>(rays,sample, states);


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

void TracerGPU::callTraceNewBeam(float* offset, int n_rays, int  n_el, int* els, float* wgt, float* prim_trans_param, float* sec_trans_param, float* prim_cap_geom, std::string path_out){

  float x_=0.0, y_=0.0,z_=0.0;
  float xL_=600.0,yL_=600.0,zL_=3000.0;
  float xLV_=6.0,yLV_=6.0,zLV_=10.0;

  int xN_ = (int)(xL_/xLV_);
  int yN_ = (int)(yL_/yLV_);
  int zN_ = (int)(zL_/zLV_);

  xLV_ = xL_/((float)(xN_)); 
  yLV_ = yL_/((float)(yN_));
  zLV_ = zL_/((float)(zN_));

  ChemElementGPU* elements;
  cudaMallocManaged(&elements, sizeof(ChemElementGPU)*n_el);

  float* weights;
  cudaMallocManaged(&weights, sizeof(float)*n_el*xN_*yN_*zN_);

  MaterialGPU* materials;
  cudaMallocManaged(&materials, sizeof(MaterialGPU)*xN_*yN_*zN_);
  
  VoxelGPU* voxels; 
  cudaMallocManaged(&voxels, sizeof(VoxelGPU)*xN_*yN_*zN_);

  VoxelGPU* oobVoxel; 
  cudaMallocManaged(&oobVoxel, sizeof(VoxelGPU));

  SampleGPU* sample;
  cudaMallocManaged(&sample, sizeof(SampleGPU));

  float* ofst;
  cudaMallocManaged(&ofst, sizeof(float)*3);
  for(int i = 0; i< 3; i++)
    ofst[i] = offset[i];

  float* prim_trans;
  cudaMallocManaged(&prim_trans, sizeof(float)*5);
  for(int i = 0; i< 5; i++)
    prim_trans[i] = prim_trans_param[i];

  float* sec_trans;
  cudaMallocManaged(&sec_trans, sizeof(float)*6);
  for(int i = 0; i< 6; i++)
    sec_trans[i] = sec_trans_param[i];

  float* prim_geom;
  cudaMallocManaged(&prim_geom, sizeof(float)*4);
  for(int i = 0; i< 4; i++)
    prim_geom[i] = prim_cap_geom[i];

  for(int i = 0; i< n_el; i++)
    elements[i] = ChemElementGPU(els[i]); 
  
  for(int i = 0; i < xN_; i++){
    for(int j = 0; j < yN_; j++){
      for(int k = 0; k < zN_; k++){
        for(int l = 0; l< n_el; l++)
          weights[i*yN_*zN_*n_el+j*zN_*n_el+k*n_el+l] = wgt[l]; 

        materials[i*yN_*zN_+j*zN_+k] = MaterialGPU(n_el, elements, &weights[i*yN_*zN_*n_el+j*zN_*n_el+k*n_el+0]);
        voxels[i*yN_*zN_+j*zN_+k] = VoxelGPU(x_+i*xLV_, y_+j*yLV_, z_+k*zLV_, xLV_, yLV_, zLV_,&materials[i*yN_*zN_+j*zN_+k]);
      }
    }
  }

  *oobVoxel = VoxelGPU(-1.,-1.,-1.,-1.,-1.,-1.,&(materials[0]));
  *sample = SampleGPU(x_, y_,  z_, xL_,  yL_, zL_,  xLV_,  yLV_,  zLV_, xN_, yN_,  zN_, voxels, oobVoxel);

  clock_t begin = clock();

  curandState_t* states;
  cudaMallocManaged(&states, n_rays*sizeof(curandState_t));

  init<<<n_rays, 1>>>(775289, states); //time(0)

  RayGPU* rays;
  cudaMallocManaged(&rays, n_rays*sizeof(RayGPU));

  for(int i = 0; i < n_rays; i++){
	  	  rays[i] = RayGPU( 
                          0.0f,0.0f,0.0f,
                          0.0f,0.0f,0.0f,
                          0.0f,0.0f,0.0f,
                          false, 17.4*50677300.0,i,
                          3.94f,0.0f,0.0f,
                          0.0f,0.0f,0.0f,
                          1.0f
                        );
  }

  traceNewBeam<<<n_rays,1>>>(rays,sample, states, ofst, prim_trans, sec_trans, prim_geom);

  cudaDeviceSynchronize();

  clock_t end = clock();
  double time_spent_to_trace= (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Trace: %f s\n", time_spent_to_trace);

	arma::Mat<double> rays__;
  rays__ = arma::ones(n_rays, 21);
  
  int total_respawns = 0;
  for(int i = 0; i < n_rays; i++){
    rays__.row(i)=arma::rowvec({
			rays[i].getStartX(),rays[i].getStartY(),rays[i].getStartZ(),
			rays[i].getDirX(),rays[i].getDirY(),rays[i].getDirZ(),
			rays[i].getSPolX(),rays[i].getSPolY(),rays[i].getSPolZ(),
			(double) rays[i].getFlag(), rays[i].getWaveNumber(),(double) rays[i].getIndex(),
			rays[i].getOpticalPath(),rays[i].getSPhase(),rays[i].getPPhase(),
			rays[i].getPPolX(), rays[i].getPPolY(),rays[i].getPPolZ(),
			rays[i].getProb(), (double) rays[i].getIANum(),(double) rays[i].getRespawnCounter()});  
    total_respawns += rays[i].getRespawnCounter();
	}

  printf("Generated Rays: %i\n", total_respawns);
  rays__.save(arma::hdf5_name(path_out,"my_data"));

  //free all memory cuda
  cudaFree(rays);
  cudaFree(states);
  cudaFree(elements);
  cudaFree(materials);
  cudaFree(voxels);
  cudaFree(oobVoxel);
  cudaFree(sample);
  cudaFree(weights);
}