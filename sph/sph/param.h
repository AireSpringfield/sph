#ifndef PARAM_H
#define PARAM_H

#include<vector_types.h>
#include<vector_functions.h>

#ifndef UINT
#define UINT
typedef unsigned int uint;
#endif

class Param {

public:
	// Parameters can be manually set by user
	uint numParticles;		// Number of particles
	uint numMax;				// Maximum number
	bool bAllowAdding;		// Whether adding new particles is allowed
	
	float h;				// "Smooth kernel radius" 
	float m;				// Mass of a single particle

	uint blockDim;			// How many threads in a block

	float dt;				// Time step

	float3 worldMin;		// Particles can only exist in a limited cuboid space
	float3 worldMax;

	float3 gravity;			

	float gasConstant;
	float refDensity;


	float viscosity;
	float coeCollision;		// Elastic collision coefficient between particles and walls

	float res;	//	Marching cubes resolution [dimRes] is [res] times as [dimCell]
	float r;	// Radius for computing isovalue




public:
	// Parameters should not be directly set by users

	float3 sizeWorld;
	float sizeCell;
	uint3 dimCell;
	uint numCell;

	uint numThreads;		// Number of threads
	uint gridDim;			// Number of blocks


	uint3 dimRes;				// Resolution for marching cubes 
	float3 sizeRes;				
	uint numPoints_Isoval;
	uint numThreads_Isoval;	// Using when calculating iso values (density) at grid points
	uint gridDim_Isoval;
	int r2h;				// Proportion : ceil(r/h)
	float r2;


	float kDensity;			// Coefficients for density, pressure and viscosity computations
	float kPressure;
	float kViscosity;

	float selfDensity;

	float h2;

	bool bIsSimulating;

public:
	void setDefault();
	void init();
};


#endif