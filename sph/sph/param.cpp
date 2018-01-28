#include"param.h"
#include<cmath>

void Param::setDefault() {
	
	numParticles = 0;
	numMax = 500000;
	bAllowAdding = true;

	h = 0.012f;
	m = 0.005f;

	blockDim = 512;

	dt = 0.0008;

	worldMin = make_float3(-0.5f, -0.5f, -0.5f);
	worldMax = make_float3(0.5f, 0.5f, 0.5f);

	gasConstant = 1.0f;
	refDensity = 500.0f;

	viscosity = 4.0f;
	coeCollision = -0.2f;

	gravity = make_float3(0.0f, -10.0f, 0.0f);

	res = 2.0f;
	r = 0.035f;


}

void Param::init() {
	static const float pi = 3.1415926535f;


	sizeCell = h;
	sizeWorld = make_float3(worldMax.x - worldMin.x, worldMax.y - worldMin.y, worldMax.z - worldMin.z);
	dimCell = make_uint3((uint)ceilf(sizeWorld.x/sizeCell),(uint)ceilf(sizeWorld.y/sizeCell),(uint)ceilf(sizeWorld.z/sizeCell));
	numCell = dimCell.x*dimCell.y*dimCell.z;

	numThreads = blockDim > numParticles ? blockDim : numParticles;
	gridDim = numThreads%blockDim == 0 ? numThreads / blockDim : numThreads / blockDim + 1;

	dimRes = make_uint3((uint)round(res*dimCell.x), (uint)round(res*dimCell.y), (uint)round(res*dimCell.z));
	sizeRes = make_float3(sizeWorld.x / dimRes.x, sizeWorld.y / dimRes.y, sizeWorld.z / dimRes.z);
	numPoints_Isoval = (dimRes.x + 1)*(dimRes.y + 1)*(dimRes.z + 1);
	numThreads_Isoval= blockDim > numPoints_Isoval ? blockDim : numPoints_Isoval;
	gridDim_Isoval = numThreads_Isoval%blockDim == 0 ? numThreads_Isoval / blockDim : numThreads_Isoval / blockDim + 1;
	r2h = (int)ceil(r / h);


	kDensity = 315.f / (64.f*pi*pow(h, 9));
	kPressure = 45.f / (pi*pow(h, 6));
	kViscosity = -45.f / (pi*pow(h, 6));

	selfDensity = m*315.f / (64.f*pi*h*h*h);

	//refDensity = selfDensity;

	bIsSimulating = false;

	h2 = h*h;
	r2 = r*r;

}