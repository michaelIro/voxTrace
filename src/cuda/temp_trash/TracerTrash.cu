  //test<<<16,16>>>(weights);

  //VoxelGPU oobVoxel(-1.,-1.,-1.,-1.,-1.,-1.,&(materials[0][0][0]));
  //SampleGPU mySample(xN_,yN_,zN_,voxels,&oobVoxel);
//(float)0.,(float)0.,(float)0.,(float)150.,(float)150.,(float)5.,(float)15.,(float)15.,(float)0.5
  //for

  /*




  weights_[0] = 0.5;
  weights_[1] = 0.3;
  weights_[2] = 0.2;



  //thrust::device_vector<VoxelGPU> voxels_thrust(voxN_);
  //SampleGPU* samples;

  //float **A, **B, **C;
  // unified memory allocation, B and C analogous
  //cudaMallocManaged(&A, N * sizeof(float *));
  //for (i = 0; i < N; i++) {
  //  cudaMallocManaged(&A[i], N * sizeof(float));
  //}
// kernel invocation
  //allocate Memory for 4D array
 // gpuErrchk( cudaMallocManaged(&weights, sizeof(float***)*xN_));
  //cudaMallocManaged(&weights, sizeof(float***)*xN_);
  //for(int i=0;i<xN_;i++){
   // cudaMallocManaged(&weights[i], sizeof(float**)*yN_);
    //for(int j=0;j<yN_;j++){
     // cudaMallocManaged(&weights[i][j], sizeof(float*)*zN_);
     // for(int k=0;k<zN_;k++){
     //   cudaMallocManaged(&weights[i][j][k], sizeof(float)*n_elements);
     // }
   // }
 // }

  //cudaMallocManaged(&weights, xN_ * sizeof(float ***));
  //for (int i = 0; i < xN_; i++) {
  //  cudaMallocManaged(&A[i], N * sizeof(float));
  //}

  //cudaMallocManaged((void**)&weights, sizeof(float)*xN_*yN_*zN_*n_elements);
  //cudaMallocManaged((void**)&materials, sizeof(MaterialGPU**)*voxN_);

  //cudaMalloc3DArray((VoxelGPU***)voxels, sizeof(VoxelGPU), xN_, yN_, zN_);
  //cudaMallocManaged((void**)&voxels, sizeof(VoxelGPU**)*voxN_);
  //cudaMallocManaged((void**)&samples, sizeof(SampleGPU*)*voxN_);




  //thrust::device_vector<ChemElementGPU> elements_thrust(n_elements);
  //elements_thrust.push_back(cu);
  //elements_thrust.push_back(sn);
  //elements_thrust.push_back(pb);
  
  //cudaMallocManaged(&, n_elements*sizeof(ChemElementGPU));
  //cudaMallocManaged(&weights, n_elements*voxN_*sizeof(float));
  //cudaMallocManaged(&materials, voxN_*sizeof(MaterialGPU));
  //cudaMallocManaged(&voxels,voxN_*sizeof(VoxelGPU));
  //cudaMallocManaged(&samples, n_threads*sizeof(SampleGPU));



 
  //std::cout << << std::endl;
  

*/




void TracerGPU::callAdd(int N, int M, float *x, float *y, RayGPU* rays_1, RayGPU* rays_2, ChemElementGPU* elements, MaterialGPU* materials,ChemElementGPU* myElements, float* myElements_weights){ //

  //print adress of elements
  std::cout << "elements: " << elements << std::endl;

  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  cudaMallocManaged(&rays_1, M*sizeof(RayGPU));
  cudaMallocManaged(&rays_2, M*sizeof(RayGPU));
  cudaMallocManaged(&elements, M*sizeof(ChemElementGPU));
  cudaMallocManaged(&myElements, 3*M*sizeof(ChemElementGPU));
  cudaMallocManaged(&myElements_weights, 3*M*sizeof(float));
  cudaMallocManaged(&materials, M*sizeof(MaterialGPU));

  std::cout << "elements: " << elements << std::endl;
	std::cout<< "Before:"<<std::endl;
  ChemElementGPU cu(29);
  ChemElementGPU sn(50);
	ChemElementGPU pb(82);

  //ChemElementGPU myElements[3] = {cu, sn, pb};
  myElements[0] = cu;
  myElements[1] = sn;
  myElements[2] = pb;

  myElements_weights[0] = 0.1;
  myElements_weights[1] = 0.1;
  myElements_weights[2] = 0.1;

  //thrust::device_vector<ChemElementGPU> d_elements_managed(myElements, myElements + M);

  MaterialGPU cu_mat(7.0,myElements,myElements_weights,3);
  //cu_mat.setElementsAdress(elements);

	for (int i = 0; i < M; i++) {
		rays_1[i] = RayGPU(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,false, i*100., i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
		rays_2[i] = RayGPU(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,false, i*200., i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
		rays_1[i].print();
		rays_2[i].print();
		elements[i] = cu;
		materials[i]= cu_mat;

    std::cout<< &elements[i] <<std::endl;
	}

  //std::cout << "Material" << materials[1].CS_Tot(4.4) << std::endl;
	// initialize x and y arrays on the host
  	for (int i = 0; i < N; i++) {
    	x[i] = 1.0f;
    	y[i] = 2.0f;
  	}

  // Run kernel on 1M elements on the GPU
  //add<<<1, 5>>>(N, d_x, d_y);
  //trace<<<1,5>>>(M, d_rays_1, d_rays_2,d_elements_managed, materials);
  add<<<16, 16>>>(N, x, y);
  trace<<<1,5>>>(M, rays_1, rays_2, elements, materials);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Copy device memory to host
  //cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  //cudaMemcpy(rays_1, d_rays_1, M*sizeof(RayGPU), cudaMemcpyDeviceToHost);
  //cudaMemcpy(rays_2, d_rays_2, M*sizeof(RayGPU), cudaMemcpyDeviceToHost);

	  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++){
    maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << "\t Result: " <<  y[i] << std::endl;
  }

  std::cout<< "After:"<<std::endl;
	for (int i = 0; i < M; i++) {
		rays_1[i].print();
		rays_2[i].print();
	}
  
  // Free memory
  cudaFree(x);
  cudaFree(y);
  cudaFree(rays_1);
  cudaFree(rays_2);
  cudaFree(elements);
  
}


  //float *d_x, *d_y;
  //RayGPU* d_rays_1, *d_rays_2;
  //ChemElementGPU* d_elements_managed;

  /**cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));
  cudaMalloc(&d_rays_1, M*sizeof(RayGPU));
  cudaMalloc(&d_rays_2, M*sizeof(RayGPU));
  cudaMalloc(&d_elements_managed, M*sizeof(ChemElementGPU));

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rays_1, rays_1, M*sizeof(RayGPU), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rays_2, rays_2, M*sizeof(RayGPU), cudaMemcpyHostToDevice);
  cudaMemcpy(d_elements_managed, elements, M*sizeof(ChemElementGPU), cudaMemcpyHostToDevice);**/

  //std::cout << "Size of ChemElementGPU: " << sizeof(ChemElementGPU) << std::endl;
  //std::cout << "Location of d_elements_managed: " << d_elements_managed << std::endl;
  //std::cout << "Location of elements: " << elements << std::endl;
  // std::cout << "Line Energy Inside: " << d_elements_managed[0].Line_Energy(0) << std::endl;
  //std::cout << sizeof(RayGPU) << std::endl;

    //cudaFree(d_x);
  //cudaFree(d_y);
  //cudaFree(d_rays_1);
  //cudaFree(d_rays_2);
  //cudaFree(d_elements_managed);
  
  
  
  
  
  

/**
 * @brief 
 * 
 * 
	int n_threads = 16; 
	int n_blocks = 16;
	int n_samples = n_blocks*n_threads;
	int n_elements = 3;

	int n_x_voxels = 1;
	int n_y_voxels = 1;
	int n_z_voxels = 1;
	int n_voxels = n_x_voxels*n_y_voxels*n_z_voxels;

	int M = 4;

	ChemElementGPU elements[n_elements];
	MaterialGPU materials[n_voxels];
	VoxelGPU voxels[n_voxels];
	SampleGPU sample;

	float x[n_threads], y[n_threads];
	RayGPU rays_1[M], rays_2[M];

	ChemElementGPU myElements[3];
	float myElements_weights[3];


	int myArray[7][9];
	//map 2d array elements to 1d pointer array
	int *myArrayPtr = (int*)myArray;
	//map 1d pointer array to 2d array
	int **myArrayPtrPtr = (int**)myArrayPtr;
	//map 2d array elements to 1d pointer array
	int *myArrayPtr2 = (int*)myArrayPtrPtr;

	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 9; j++) {
			myArray[i][j] = i*9 + j;
		}
	}

	//TracerGPU::callAdd(n_threads,M,x,y,rays_1,rays_2,elements,materials,myElements, myElements_weights); 
 * 
 * 
 
	//map<ChemElement* const,double> bronzeMap{{&cu,0.7},{&sn,0.2},{&pb,0.1}};

	//std::cout << "Memory size of ChemElementGPU: " << cu.getMemorySize() << " Compiler says: " << sizeof(ChemElementGPU) << std::endl;
	//std::cout <<"Line Energy: "<< cu.Line_Energy(0) << std::endl;
	//int* runVers;
	//cudaRuntimeGetVersion(runVers);
	//cudaDriverGetVersion(runVers);
	//std::cout <<"Cuda-Runtime-Version: "<<  runVers << std::endl;
*/

/*
	//arma::field<Material> myMaterials(11,11,11); TODO: change vec<vec<vec>> to field
	vector<vector<vector<Material>>> myMat;	
	for(int i = 0; i < 11; i++){
		vector<vector<Material>> myMat1;
		for(int j = 0; j < 11; j++){
			vector<Material> myMat2;
			for(int k = 0; k < 11; k++){
				myMat2.push_back(Material(bronzeMap,8.96));
				//myMaterials(i,j,k) = Material(cuMatMap,8.96);
			}
			myMat1.push_back(myMat2);
		}
		myMat.push_back(myMat1);
	} 

//---------------------------------------------------------------------------------------------

	Sample sample_ (0.,0.,0.,150.,150.,5.,15.,15.,0.5,myMat);
	//sample_.print();
//---------------------------------------------------------------------------------------------

	arma::Mat<double> myPrimaryCapBeam;
    myPrimaryCapBeam.load(arma::hdf5_name("/media/miro/Data/Documents/TU Wien/VSC-BEAM/PrimaryBeam-1-0.h5","my_data"));
	XRBeam myPrimaryBeam(myPrimaryCapBeam);
//---------------------------------------------------------------------------------------------
    (myPrimaryBeam.getRays())[0].print(9);

    int N = 1<<20;
    float *x, *y;

    GPUTracer::callAdd(N, x, y,&sample_, &myPrimaryBeam);

    //arma::Mat<double> dummy;
    //PlotAPI::scatter_plot((char*) "../test-data/out/plots/example-sine-functions.pdf",true,true, dummy);
*/
