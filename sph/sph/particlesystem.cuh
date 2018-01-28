#ifndef PARTICLESYSTEM_CUH
#define PARTICLESYSTEM_CUH

#include"particlesystem.h"


// Copy parameters to the device memory for faster accessing


extern void copyParamToDevice(Param *pParam);

void allocateArray(void **dPtr, size_t size);
void freeArray(void *dPtr);
void copyParamToDevice(Param *param);
void copyArray(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);


void setupIdx(Param *pParam, uint2 *dHash, uint *dStart, uint *dEnd, Particle *dParticle);
void computeAcc(Param *pParam, uint2 *dHash, uint *dStart, uint *dEnd, Particle *dParticles);
void leapfrogIntegrate(Param *pParam, uint2 *dHash, uint *dStart, uint *dEnd, Particle *dParticles);

void computeIsoValue(Param *pParam, uint2 *dHash, uint *dStart, uint *dEnd, Particle *dParticles, float *dIsoval);


// Data transfer between dPos, dVel and dParticles
void copyPosToDeviceVector(Param *pParam, Particle *dParticles, float3 *dPos);
void copyVelToDeviceVector(Param *pParam, Particle *dParticles, float3 *dVel);

void copyPosFromDeviceVector(Param *pParam, Particle *dParticles, float3 *dPos, uint start, uint end);
void copyVelFromDeviceVector(Param *pParam, Particle *dParticles, float3 *dVel, uint start, uint end);


#endif