#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#include"param.h"
#include"ofxMarchingCubes.h"



// Storing a single particle's state. 
class Particle{
public:
	float3 pos;
	float3 vel;
	float3 hvel;	// velocity at half time
	float3 acc;
	float dens;
	float press;
};







class ParticleSystem {
protected:

	// CPU data
	float3 *hPos;
	float3 *hVel;

	
	// GPU data

	
	uint2 *dHash;	// x: hash of cell that the particle resides in  y: index of the particle

	uint *dIndex;

	uint *dStart;
	uint *dEnd;
	Particle *dParticles;

	float3 *dPos;
	float3 *dVel;

	float *dIsoval;
	
		
	cudaGraphicsResource *cudaVboResource;

	
	


protected:



	// Add a particle into system
	void addParticle(float3 pos, float3 vel);

	

	
public:
	// Parameters on the host side
	Param param;

	// Compute isovalues for marching cubes
	bool bIsoval = true;

	// Simulation paused
	bool bPaused = false;



public:
	
	~ParticleSystem();
	

	// Initial system
	void init();

	// Prepare some necessary information for GPU simulation, such as number of threads, etc
	void initSimulation();

	// Update the system state after dt, refleshing one frame
	void update();


	float3* getPos() const;

	void addSphere(float3 pos, float3 vel, int r, float spacing);
	void addCuboid(float3 posMin, float3 posMax, float3 vel, float spacing);

	uint getNum() const;
	
	float getRadius() const;
	void setupScene();

	void setIsoValue(ofxMarchingCubes &mc);


};





#endif
