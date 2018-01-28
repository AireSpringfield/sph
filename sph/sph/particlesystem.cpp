#include<Windows.h>
#include"particlesystem.h"
#include"particlesystem.cuh"
#include<cuda_runtime.h>
#include<iostream>


inline void allocateArray(void **dPtr, size_t size) {
	cudaMalloc(dPtr, size);
}

inline void freeArray(void *dPtr) {
	cudaFree(dPtr);
}



inline void copyArray(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
	cudaMemcpy(dst, src, count, kind);
}




ParticleSystem::~ParticleSystem() {
	delete[] hPos;
	delete[] hVel;

	freeArray(dParticles);
	freeArray(dStart);
	freeArray(dEnd);
	freeArray(dHash);
	freeArray(dPos);
	freeArray(dVel);
	freeArray(dIsoval);


}

void ParticleSystem::init() {

	if (param.bAllowAdding) {
		hPos = new float3[sizeof(float3)*param.numMax];
		hVel = new float3[sizeof(float3)*param.numMax];

		allocateArray((void **)&dPos, sizeof(float3)*param.numMax);
		allocateArray((void **)&dVel, sizeof(float3)*param.numMax);
		allocateArray((void **)&dParticles, sizeof(Particle)*param.numMax);
		allocateArray((void **)&dHash, sizeof(uint2)*param.numMax);
		allocateArray((void **)&dStart, sizeof(uint)*param.numCell);
		allocateArray((void **)&dEnd, sizeof(uint)*param.numCell);

		if(bIsoval)
			allocateArray((void **)&dIsoval, sizeof(float)*param.numPoints_Isoval);

	}

	else {
		hPos = new float3[sizeof(float3)*param.numParticles];
		hVel = new float3[sizeof(float3)*param.numParticles];

		allocateArray((void **)&dPos, sizeof(float3)*param.numParticles);
		allocateArray((void **)&dVel, sizeof(float3)*param.numParticles);
		allocateArray((void **)&dParticles, sizeof(Particle)*param.numParticles);
		allocateArray((void **)&dHash, sizeof(uint2)*param.numParticles);
		allocateArray((void **)&dStart, sizeof(uint)*param.numCell);
		allocateArray((void **)&dEnd, sizeof(uint)*param.numCell);
		allocateArray((void **)&dEnd, sizeof(uint)*param.numCell);

		if (bIsoval)
			allocateArray((void **)&dIsoval, sizeof(float)*param.numPoints_Isoval);
	}


}




void ParticleSystem::addParticle(float3 pos, float3 vel) {
	/*
	if (!param.bAllowAdding) {
		std::cout << "Adding new particle is not permitted!" << std::endl;
		return;
	}
	*/
	if (param.numParticles >= param.numMax) {
		std::cout << "Particle number limit exceeded!" << std::endl;
		return;
	}

	hPos[param.numParticles] = pos;
	hVel[param.numParticles] = vel;

	++param.numParticles;

	// Adding device data
	// ......
	
}

inline float frand()
{
	return rand() / (float)RAND_MAX;
}

void ParticleSystem::addCuboid(float3 posMin, float3 posMax, float3 vel, float spacing)
{

	float jitter = spacing*0.05f;

	for (float z = posMin.z; z <= posMax.z; z+=spacing)
	{
		for (float y = posMin.y; y <= posMax.y; y+=spacing)
		{
			for (float x = posMin.x; x <= posMax.x; x+=spacing)
			{
						
					addParticle(
						make_float3(x + (frand()*2.0f - 1.0f)*jitter,
							y + (frand()*2.0f - 1.0f)*jitter,
							z + (frand()*2.0f - 1.0f)*jitter),
						vel);
				
			}
		}
	}
}

void ParticleSystem::addSphere(float3 pos, float3 vel, int r, float spacing)
{

	float jitter = spacing*0.01f;
	for (int z = -r; z <= r; z++)
	{
		for (int y = -r; y <= r; y++)
		{
			for (int x = -r; x <= r; x++)
			{
				float dx = x*spacing;
				float dy = y*spacing;
				float dz = z*spacing;
				float l = sqrtf(dx*dx + dy*dy + dz*dz);
				

				if (l <= spacing*r)
				{
					addParticle(
						make_float3(pos.x + dx + (frand()*2.0f - 1.0f)*jitter,
							pos.y + dy + (frand()*2.0f - 1.0f)*jitter,
							pos.z + dz + (frand()*2.0f - 1.0f)*jitter),
						vel);
				}
			}
		}
	}
}

void ParticleSystem::initSimulation() {

	param.init();
	
}

void ParticleSystem::update() {
	

	param.numThreads = param.blockDim > param.numParticles ? param.blockDim : param.numParticles;
	param.gridDim = param.numThreads%param.blockDim == 0 ? param.numThreads / param.blockDim : param.numThreads / param.blockDim + 1;


	// Fresh parameters in device memory
	copyParamToDevice(&param);

	// Update state after dt
	setupIdx(&param, dHash, dStart, dEnd, dParticles);
	computeAcc(&param, dHash, dStart, dEnd, dParticles);
	leapfrogIntegrate(&param, dHash, dStart, dEnd, dParticles);

	// Extract position information to a continous device array
	copyPosToDeviceVector(&param, dParticles, dPos);

	// Consider VBO for instead here..
	copyArray(hPos, dPos, sizeof(float3)*param.numParticles, cudaMemcpyDeviceToHost);
	
}









void ParticleSystem::setupScene() {
	/*
	addCuboid(
		make_float3(-0.4f,-0.499f,-0.4f),
		make_float3(0.4f, -0.40f, 0.4f),
		make_float3(0.0f, 0.0f, 0.0f),
		param.h);
	*/
	addCuboid(
		make_float3(-0.25f, -0.499f, -0.25f),
		make_float3(0.25f, -0.4f, 0.25f),
		make_float3(0.0f, 0.0f, 0.0f),
		param.h);
		
	
	/*addCuboid(
		make_float3(-0.499f, -0.499f, -0.499f),
		make_float3(0.499f, -0.40f, 0.499f),
		make_float3(0.0f, 0.0f, 0.0f),
		param.h);*/
		

	addSphere(make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), 5, param.h);

	addSphere(make_float3(0.2f, 0.3f, 0.0f), make_float3(0.0f, -0.5f, 0.0f), 4, param.h);

	//addSphere(make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, -0.5f, 0.0f), 5, param.h);

	copyArray(dPos, hPos, param.numParticles*sizeof(float3), cudaMemcpyHostToDevice);
	copyPosFromDeviceVector(&param, dParticles, dPos, 0, param.numParticles);
	copyArray(dVel, hVel, param.numParticles*sizeof(float3), cudaMemcpyHostToDevice);
	copyVelFromDeviceVector(&param, dParticles, dVel, 0, param.numParticles);

}

float3* ParticleSystem::getPos() const {
	return hPos;
}

uint ParticleSystem::getNum() const {
	return param.numParticles;
}

float ParticleSystem::getRadius() const {
	return param.h;
}


void ParticleSystem::setIsoValue(ofxMarchingCubes &mc) {

	computeIsoValue(&param, dHash, dStart, dEnd, dParticles, dIsoval);
	copyArray(&mc.isoVals[0], dIsoval, param.numPoints_Isoval * sizeof(float), cudaMemcpyDeviceToHost);


}