
//!  GPUTracer 
#include "GPUTracer.cuh"

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

// Kernel function to trace rays
__global__ void trace(int n, float *x, float *y)
{
  /*for (Ray ray: primary_.getRays()) {
		//std::cout << i << std::endl;
		Ray*	currentRay = &ray;
		Voxel* 	currentVoxel = sample_.findStartVoxel(currentRay);
		int nextVoxel = 13;	
		Ray* aNewRay= traceForward(currentRay, currentVoxel,&nextVoxel, &sample_,&ia);
		tracedRays[i++]=(*aNewRay);
	}*/
}

/*********************************/
Ray* traceForward(Ray* ray, Voxel* currentVoxel, int* nextVoxel, Sample *sample, int* ia){

	// Check if ray is already out of bounds of sample -> If so, no interaction is possible -> return ray.
	if(sample->isOOB(currentVoxel))
		return ray;
		
	double tIn;
	double rayEnergy = (*ray).getEnergyKeV() /50677300.0;	 // FIXME: SOMETHING WRONG WITH ENERGY / 50677... should not be here
	double muLin = (*currentVoxel).getMaterial().getMuLin(rayEnergy);
	double intersectionLength = (*currentVoxel).intersect(ray,nextVoxel,&tIn);
	double randomN = ((double) rand()) / ((double) RAND_MAX);

	//cout << "  Coordinates: " << (*currentVoxel).getX0()<<" "<<(*currentVoxel).getY0()<<" "<<(*currentVoxel).getZ0()<<endl;
	//cout << "  Next Voxel: " << (*nextVoxel) << endl;
	//cout << "  Intersection length: " << intersectionLength << "µm" << endl;
	//cout << "  Linear attenuation coefficient: " << muLin << endl;
	//cout << "  Interaction Probability: " << (1.-exp(-muLin*intersectionLength))*100 << "%" << endl<<endl;
	//(*currentVoxel).print();

	/*Interaction in this Voxel?*/
	if(exp(-muLin*intersectionLength) < randomN){
		//cout << "Interaction"<<endl;
		(*ia)++;

		randomN = ((double) rand()) / ((double) RAND_MAX);
		/*Selection of chemical Element to interact with*/			
		ChemElement* interactingElement = (*currentVoxel).getMaterial().getInteractingElement(rayEnergy,randomN);

		//cout<<"\t Interacting Element: "<<interactingElement<<endl;
		//cout<<"  Next: "<<(*nextVoxel).getXPos0()<<" "<<(*nextVoxel).getYPos0()<<" "<<(*nextVoxel).getYPos0()<<endl;

		/*Selection of interaction type*/
		randomN = ((double) rand()) / ((double) RAND_MAX);
		int interactionType = (*interactingElement).getInteractionType(rayEnergy,randomN);
		//cout<<"Interaction Type: "<<interactionType<<endl;

		if(interactionType == 0){ 
			//cout<<"\t Photo-Absorption"<<endl;
			randomN = ((double) rand()) / ((double) RAND_MAX);
			int myShell = (*interactingElement).getExcitedShell(rayEnergy,randomN);
			//cout<<"\t Excited Shell: "<< myShell << " \n";

			randomN = ((double) rand()) / ((double) RAND_MAX);
			if(randomN > (*interactingElement).getFluorescenceYield(myShell)){
				//cout<<"\t Auger-Effect: ";
				//cout<<interactingElement.getAugerYield(myShell)<<endl;
				(*ray).setIANum((*ray).getIANum()+1);
				(*ray).setIAFlag(true);
				(*ray).setFlag(false);
			}
			else{
				//int myShell1 = (int) myShell;
				//cout<<"\t Fluorescence-Yield: "<<interactingElement.getFluorescenceYield(myShell)<<endl;

				randomN = ((double) rand()) / ((double) RAND_MAX);
				int myLine = (*interactingElement).getTransition(myShell, randomN);
				//cout<<"\t Line: "<<myLine<<" Energy: "<<interactingElement.getLineEnergy(myLine)<<endl;

				randomN = ((double) rand()) / ((double) RAND_MAX);
				double phi = 2*M_PI*randomN;

				randomN = ((double) rand()) / ((double) RAND_MAX);
				double theta = acos(2*randomN-1);

				randomN = ((double) rand()) / ((double) RAND_MAX);
				double l = intersectionLength*randomN + tIn;

				double xNew = (*ray).getStartX()+(*ray).getDirX()*l;
				double yNew = (*ray).getStartY()+(*ray).getDirY()*l;
				double zNew = (*ray).getStartZ()+(*ray).getDirZ()*l;
				
				//cout<< "OLD COORDINATES:"<<(*ray).getStartX()<< " "<<(*ray).getStartY()<<" "<<(*ray).getStartZ()<<endl;
				//cout<< "OLD DIRECTION:"<<(*ray).getDirX()<< " "<<(*ray).getDirY()<<" "<<(*ray).getDirZ()<<endl;
				//cout<< "NEW COORDINATES:"<<xNew<< " "<<yNew<<" "<<zNew<<endl;

				(*ray).rotate(phi,theta);
				(*ray).setStartCoordinates(xNew,yNew,zNew);
				(*ray).setEnergy((*interactingElement).getLineEnergy(myLine));
				(*ray).setIANum((*ray).getIANum()+1);
				(*ray).setIAFlag(true);
			}
		}
		else if(interactionType == 1){
			//cout<<"\t Rayleigh-Scattering"<<endl; 			//TODO: Polarized -Unpolarized

			randomN = ((double) rand()) / ((double) RAND_MAX);
			double phi = 2*M_PI*randomN;
				
			randomN = ((double) rand()) / ((double) RAND_MAX);
			double theta = (*interactingElement).getThetaRayl(rayEnergy,randomN);	
	
			//cout<<"\t Theta: "<<theta<<endl;
			(*ray).rotate(phi,theta);

			randomN = ((double) rand()) / ((double) RAND_MAX);
			double l = intersectionLength*randomN + tIn;

			double xNew = (*ray).getStartX()+(*ray).getDirX()*l;
			double yNew = (*ray).getStartY()+(*ray).getDirY()*l;
			double zNew = (*ray).getStartZ()+(*ray).getDirZ()*l;
			(*ray).setStartCoordinates(xNew,yNew,zNew);
			(*ray).setIANum((*ray).getIANum()+1);
			(*ray).setIAFlag(true);
		}
		else if(interactionType == 2){
			//cout<<"\t Compton-Scattering"<<endl;

			randomN = ((double) rand()) / ((double) RAND_MAX);
			double phi = 2*M_PI*randomN;

			randomN = ((double) rand()) / ((double) RAND_MAX);
			double theta = (*interactingElement).getThetaCompt(rayEnergy,randomN);	

			//cout<<"\t Theta: "<<theta<<endl;
			(*ray).rotate(phi,theta);

			randomN = ((double) rand()) / ((double) RAND_MAX);
			double l = intersectionLength*randomN + tIn;

			double xNew = (*ray).getStartX()+(*ray).getDirX()*l;
			double yNew = (*ray).getStartY()+(*ray).getDirY()*l;
			double zNew = (*ray).getStartZ()+(*ray).getDirZ()*l;
			(*ray).setStartCoordinates(xNew,yNew,zNew);
			(*ray).setIANum((*ray).getIANum()+1);
			(*ray).setIAFlag(true);
		}
	}
	else{
		/*No interaction happening in this Voxel.*/
		//cout<<"No Interaction"<<endl;
		currentVoxel =  (*currentVoxel).getNN(*nextVoxel);
		if((*sample).isOOB(currentVoxel)) 
			(*ray).setFlag(false);
	}
	//cout<<"Flag"<<(*ray).getFlag()<<endl;
	if((*ray).getFlag()){
		return traceForward(ray, currentVoxel,nextVoxel,sample,ia);
	}
	else {
		return ray;
	}
												
}
/*********************************/

void GPUTracer::callAdd(int N, float *x, float *y, Sample *sample, XRBeam *beam){
  
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  std::cout << sizeof(XRBeam) << " " << sizeof(*beam) << " " << sizeof(&beam) << std::endl;	
  std::cout << sizeof(Sample) << " " << sizeof(*sample) << " " << sizeof(&sample) << std::endl;	

  int nRays = beam->getRays().size();
  int sizeOfRay = sizeof(double)*19; 
  XRBeam *d_beam;
  cudaMallocManaged((void **) &d_beam, nRays*sizeOfRay);
  cudaMemcpy(d_beam, beam, nRays*sizeOfRay, cudaMemcpyHostToDevice);

  d_beam->getRays()[0].print(5);
  //Ray* prim_ = ((*beam).getRays()).data();
  //prim_[0].print(8);
  //((*beam).getRays())[0].print(7);
  //(b_.getRays())[0].print(9);
  //std::cout << beam << std::endl;

  std::cout << "Hallo" << sizeof(float) << " " << sizeof(sample) << std::endl;
  //sample.print();

  //cudaMallocManaged(&sample, N*sizeof(sample));
  //cudaMallocManaged(&beam, sizeof(beam));
  //((*beam).getRays())[0].print(5);

    // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<256, 256>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
}