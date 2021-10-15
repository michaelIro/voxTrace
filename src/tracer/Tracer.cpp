/**Tracer*/

#include "Tracer.h"

using namespace std;

Tracer::Tracer(){}

Tracer::Tracer(Source source, Sample sample){
	cout<<"START: voxTrace - Tracer()"<<endl;

	srand(time(NULL)); 		// TODO: is this the correct place 

	list<Ray> tracedRays;
	int i = 0;
	
	for (auto ray : source.getRayList()) {
    	ray.print(i++);
		Ray*	currentRay = &ray;
		Voxel* 	currentVoxel = sample.findStartVoxel(currentRay);
		int 	nextVoxel = 13;	

		tracedRays.push_back(*traceForward(currentRay, currentVoxel,&nextVoxel, &sample));	
	}


	cout<<"END: voxTrace - Tracer()"<<endl;

	cout<<"START: RESULTING RAYS"<<endl;
	i=0;
	for(auto ray: tracedRays){
		ray.print(i++);
	}
	cout<<"END: RESULTING RAYS"<<endl;
}

/*********************************/
Ray* Tracer::traceForward(Ray* ray, Voxel* currentVoxel, int* nextVoxel, Sample *sample){


	double tIn;
	double rayEnergy = (*ray).getEnergyKeV();	
	double muLin = (*currentVoxel).getMaterial().getMuLin(rayEnergy, (*sample).getElements());
	double intersectionLength = (*currentVoxel).intersect(ray,nextVoxel,&tIn);
	double randomN = ((double) rand()) / ((double) RAND_MAX);



	cout << "  Coordinates: " << (*currentVoxel).getX0()<<" "<<(*currentVoxel).getY0()<<" "<<(*currentVoxel).getY0()<<endl;
	cout << "  Next Voxel: " << (*nextVoxel) << endl;
	cout << "  Intersection length: " << intersectionLength << "µm" << endl;
	cout << "  Linear attenuation coefficient: " << muLin << endl;
	cout << "  Interaction Probability: " << (1.-exp(-muLin*intersectionLength))*100 << "%" << endl<<endl;
	//(*currentVoxel).print();

	/*Interaction in this Voxel?*/
	if(exp(-muLin*intersectionLength) < randomN){
					cout << "Interaction"<<endl;

		/*Selection of chemical Element to interact with*/			
		ChemElement interactingElement = (*currentVoxel).getMaterial().getInteractingElement(rayEnergy,randomN,(*sample).getElements());

		cout<<"\t Interacting Element: "<<interactingElement<<endl;
		//cout<<"  Next: "<<(*nextVoxel).getXPos0()<<" "<<(*nextVoxel).getYPos0()<<" "<<(*nextVoxel).getYPos0()<<endl;

		/*Selection of interaction type*/
		int interactionType = interactingElement.getInteractionType(rayEnergy,randomN);
		//cout<<"Interaction Type: ";//<<interactionType<<endl;

		if(interactionType == 0){ 
			cout<<"\t Photo-Absorption"<<endl;

			int myShell = interactingElement.getExcitedShell(rayEnergy,randomN);
			cout<<"\t Excited Shell: "<< myShell << " \n";

			randomN = ((double) rand()) / ((double) RAND_MAX);
			if(randomN > interactingElement.getFluorescenceYield(myShell)){
				cout<<"\t Auger-Effect: ";
				cout<<interactingElement.getAugerYield(myShell)<<endl;
				(*ray).setIANum((*ray).getIANum()+1);
				(*ray).setIAFlag(true);
				(*ray).setFlag(false);
			}
			else{
				//int myShell1 = (int) myShell;
				cout<<"\t Fluorescence-Yield: "<<interactingElement.getFluorescenceYield(myShell)<<endl;

				randomN = ((double) rand()) / ((double) RAND_MAX);
				int myLine = interactingElement.getTransition(myShell, randomN);
				cout<<"\t Line: "<<myLine<<" Energy: "<<interactingElement.getLineEnergy(myLine)<<endl;

				randomN = ((double) rand()) / ((double) RAND_MAX);
				double phi = 2*M_PI*randomN;

				randomN = ((double) rand()) / ((double) RAND_MAX);
				double theta = acos(2*randomN-1);

				randomN = ((double) rand()) / ((double) RAND_MAX);
				double l = intersectionLength*randomN + tIn;

				double xNew = (*ray).getStartX()+(*ray).getDirX()*l;
				double yNew = (*ray).getStartY()+(*ray).getDirY()*l;
				double zNew = (*ray).getStartZ()+(*ray).getDirZ()*l;
				cout<< "OLD COORDINATES:"<<(*ray).getStartX()<< " "<<(*ray).getStartY()<<" "<<(*ray).getStartZ()<<endl;
				cout<< "OLD DIRECTION:"<<(*ray).getDirX()<< " "<<(*ray).getDirY()<<" "<<(*ray).getDirZ()<<endl;
				cout<< "NEW COORDINATES:"<<xNew<< " "<<yNew<<" "<<zNew<<endl;

							(*ray).rotate(phi,theta);

				(*ray).setStartCoordinates(xNew,yNew,zNew);
				(*ray).setEnergy(interactingElement.getLineEnergy(myLine));
				(*ray).setIANum((*ray).getIANum()+1);
				(*ray).setIAFlag(true);
			}
		}
		else if(interactionType == 1){
			cout<<"\t Rayleigh-Scattering"<<endl; 			//TODO: Polarized -Unpolarized

			randomN = ((double) rand()) / ((double) RAND_MAX);
			double phi = 2*M_PI*randomN;
				
			randomN = ((double) rand()) / ((double) RAND_MAX);
			double theta = interactingElement.getThetaRayl(rayEnergy,randomN);	
	
			cout<<"\t Theta: "<<theta<<endl;
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
			cout<<"\t Compton-Scattering"<<endl;

			randomN = ((double) rand()) / ((double) RAND_MAX);
			double phi = 2*M_PI*randomN;

			randomN = ((double) rand()) / ((double) RAND_MAX);
			double theta = interactingElement.getThetaCompt(rayEnergy,randomN);	

			cout<<"\t Theta: "<<theta<<endl;
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
		cout<<"No Interaction"<<endl;
		currentVoxel =  (*currentVoxel).getNN(*nextVoxel);
		if((*sample).isOOB(currentVoxel)) 
			(*ray).setFlag(false);
	}

	//cout<<"Flag"<<(*ray).getFlag()<<endl;
	if((*ray).getFlag())
		return traceForward(ray, currentVoxel,nextVoxel,sample);
	else 
		return ray;
}
/*********************************/
void Tracer::start(){

}
