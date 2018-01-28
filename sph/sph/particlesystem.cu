#include<cmath>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<device_functions.h>
#include<curand.h>

#include<device_launch_parameters.h>
#include<thrust/sort.h>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
typedef unsigned int uint;

#include"particlesystem.cuh"

__constant__ Param dParam;
//__device__ Param dParam;


inline void copyParamToDevice(Param *pParam) {
	cudaMemcpyToSymbol(dParam, pParam, sizeof(Param));
	//cudaMemcpy(&dParam, pParam, sizeof(Param), cudaMemcpyHostToDevice);
}


__device__
uint3 getIdx(float3 pos) {
	return make_uint3(
		floor((pos.x - dParam.worldMin.x) / dParam.sizeCell),
		floor((pos.y - dParam.worldMin.y) / dParam.sizeCell),
		floor((pos.z - dParam.worldMin.z) / dParam.sizeCell)
		);
};

__device__
uint getHash(uint3 idx) {
	return idx.z*dParam.dimCell.y*dParam.dimCell.x + idx.y*dParam.dimCell.x + idx.x;
};


__global__
void calcHash_kernel(Particle *dParticles, uint2 *dHash) {
	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= dParam.numParticles) return;

	uint3 posIdx = getIdx(dParticles[index].pos);
	dHash[index].x = getHash(posIdx);
	dHash[index].y = index;


};

void calcHash(Param *pParam, Particle *dParticles, uint2 *dHash) {
	calcHash_kernel <<<pParam->gridDim, pParam->blockDim >>> (dParticles, dHash);
}

/*
__device__ 
bool cmp(const uint2 &left, const uint2 &right) { 
	return left.x < right.x; 
}
*/
struct cmp {
	__host__  __device__
		bool operator()(const uint2 &left, const uint2 &right) {
		return left.x < right.x;
	}

};


void sortByHash(Param *pParam, uint2 *dHash){
	thrust::sort(
		thrust::device_ptr<uint2>(dHash),
		thrust::device_ptr<uint2>(dHash) + pParam->numParticles,
		cmp()
	);
	/*
	thrust::sort_by_key(thrust::device_ptr<uint>(dHash),// key Ö¸µÄÊÇhash
		thrust::device_ptr<uint>(dHash + pParam->numParticles),
		thrust::device_ptr<uint>(dIndex));
		*/
}


__global__
void calcStartEnd_kernel(uint2 *dHash, uint *dStart, uint *dEnd) {
	extern __shared__ uint sharedHash[]; 
	uint index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= dParam.numParticles) return;


	// copy dHash to shared memory, offset one position

	uint hash = dHash[index].x;

	sharedHash[threadIdx.x + 1] = hash;

	if (threadIdx.x == 0 && blockIdx.x > 0) {
		sharedHash[0] = dHash[index - 1].x;
	}

	__syncthreads();

	if (index == 0) {
		dStart[hash] = 0;
	}

	else if (sharedHash[threadIdx.x] != hash) {
		dStart[hash] = index;
		dEnd[sharedHash[threadIdx.x]] = index;
	}


	if (index == dParam.numParticles - 1) {
		dEnd[hash] = index + 1;
	}


};

void calcStartEnd(Param *pParam, uint2 *dHash, uint *dStart, uint *dEnd) {
	cudaMemset(dStart, 0xffu, pParam->numCell * sizeof(uint));
	cudaMemset(dEnd, 0, pParam->numCell * sizeof(uint));

	uint sharedMemSize = (pParam->blockDim + 1)*sizeof(uint);

	calcStartEnd_kernel <<<pParam->gridDim, pParam->blockDim, sharedMemSize >>> (dHash, dStart, dEnd);

}




void setupIdx(Param *pParam, uint2 *dHash, uint *dStart, uint *dEnd, Particle *dParticles) {
	calcHash(pParam, dParticles, dHash);
	sortByHash(pParam, dHash);
	calcStartEnd(pParam, dHash, dStart, dEnd);
}

// Compute density that all particles in a cell contribute to the target particle
__device__
float computeCellDensity(uint2 *dHash, uint *dStart, uint *dEnd, uint3 cellIdx, uint index, Particle *dParticles) {

	float density = 0.f;

	uint cellHash = getHash(cellIdx);

	uint startIdx = dStart[cellHash], endIdx = dEnd[cellHash];

	if (startIdx == 0xffffffff) return density;

	for (uint idx = startIdx;idx != endIdx;++idx) {
		if (index == dHash[idx].y) continue;

		Particle particle = dParticles[index];
		float3 neighborPos = dParticles[dHash[idx].y].pos;
		float3 particlePos = particle.pos;
		float r2 = (particlePos.x - neighborPos.x)*(particlePos.x - neighborPos.x) +
			(particlePos.y - neighborPos.y)*(particlePos.y - neighborPos.y) +
			(particlePos.z - neighborPos.z)*(particlePos.z - neighborPos.z);
		if (r2 >= dParam.h2) continue;
		density += dParam.m*dParam.kDensity*pow(dParam.h2 - r2, 3);

	}
	return density;
}

__global__
void computeDensityPressure_kernel(uint2 *dHash, uint *dStart, uint *dEnd, Particle *dParticles) {

	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= dParam.numParticles) return;

	// Compute density
	float density = dParam.selfDensity;
	uint3 cellIdx = getIdx(dParticles[index].pos);

	for (int dx = -1;dx <= 1;++dx) {
		int x = cellIdx.x + dx;
		if (x<0 || x >= dParam.dimCell.x)continue;

		for (int dy = -1;dy <= 1;++dy) {
			int y = cellIdx.y + dy;
			if (y<0 || y >= dParam.dimCell.y)continue;

			for (int dz = -1;dz <= 1;++dz) {
				int z = cellIdx.z + dz;
				if (z<0 || z >= dParam.dimCell.z)continue;

				uint3 neighborCellIdx = make_uint3(x, y, z);

				density += computeCellDensity(dHash, dStart, dEnd, neighborCellIdx, index, dParticles);

			}
		}
	}

	dParticles[index].dens = density;

	// Compute pressure

	dParticles[index].press = dParam.gasConstant*(density - dParam.refDensity);

}








// Compute forces exerted by particles in a cell on the given particle
__device__
float3 computeCellForce(uint2 *dHash, uint *dStart, uint *dEnd, uint3 cellIdx, uint index, Particle *dParticles) {

	float3 force = make_float3(0.f, 0.f, 0.f);

	// If two particles are too close, ignore their interatcion
	float r2min = dParam.h2*0.0001f;

	uint cellHash = getHash(cellIdx);

	uint startIdx = dStart[cellHash], endIdx = dEnd[cellHash];

	if (startIdx == 0xffffffff) return force;

	
	for (uint idx = startIdx;idx != endIdx;++idx) {
		if (index == dHash[idx].y) continue;	// Do not interact with itself

		Particle particle = dParticles[index];
		Particle neighbor = dParticles[dHash[idx].y];
		
		float3 rji = make_float3(particle.pos.x - neighbor.pos.x, particle.pos.y - neighbor.pos.y, particle.pos.z - neighbor.pos.z);

		float r2 = rji.x * rji.x + rji.y * rji.y + rji.z * rji.z;
		if (r2 < r2min) continue;

		float r = sqrt(r2);
		float3 vji = make_float3(particle.vel.x - neighbor.vel.x, particle.vel.y - neighbor.vel.y, particle.vel.z - neighbor.vel.z);
		if (r2 >= dParam.h2)continue;

		// Pressure force
		float coePressure = dParam.m*dParam.kPressure*(particle.press + neighbor.press) / (2 * neighbor.dens*r)*(dParam.h - r)*(dParam.h - r);
		force.x += coePressure*(rji.x/r);
		force.y += coePressure*(rji.y/r);
		force.z += coePressure*(rji.z/r);


		// Viscosity force
		float coeViscosity = dParam.m*dParam.viscosity*dParam.kViscosity / neighbor.dens*(dParam.h - r);
		force.x += coeViscosity*vji.x;
		force.y += coeViscosity*vji.y;
		force.z += coeViscosity*vji.z;


	}

	return force;


}








__global__
void computeAcc_kernel(uint2 *dHash, uint *dStart, uint *dEnd, Particle *dParticles) {


	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= dParam.numParticles) return;

	uint3 cellIdx = getIdx(dParticles[index].pos);

	dParticles[index].acc = make_float3(0.f, 0.f, 0.f);

	for (int dx = -1;dx <= 1;++dx) {
		int x = cellIdx.x + dx;
		if (x < 0 || x >= dParam.dimCell.x)continue;

		for (int dy = -1;dy <= 1;++dy) {
			int y = cellIdx.y + dy;
			if (y < 0 || y >= dParam.dimCell.y)continue;

			for (int dz = -1;dz <= 1;++dz) {
				int z = cellIdx.z + dz;
				if (z < 0 || z >= dParam.dimCell.z)continue;

				uint3 neighborCellIdx = make_uint3(x, y, z);
				float3 neigoborCellForce = computeCellForce(dHash, dStart, dEnd, neighborCellIdx, index, dParticles);
				dParticles[index].acc.x += neigoborCellForce.x / dParticles[index].dens;
				dParticles[index].acc.y += neigoborCellForce.y / dParticles[index].dens;
				dParticles[index].acc.z += neigoborCellForce.z / dParticles[index].dens;

			}
		}
	}
	dParticles[index].acc.x += dParam.gravity.x ;
	dParticles[index].acc.y += dParam.gravity.y ;
	dParticles[index].acc.z += dParam.gravity.z ;

}

void computeAcc(Param *pParam, uint2 *dHash, uint *dStart, uint *dEnd, Particle *dParticles) {
	computeDensityPressure_kernel << <pParam->gridDim, pParam->blockDim >>>(dHash, dStart, dEnd, dParticles);
	computeAcc_kernel <<< pParam->gridDim, pParam->blockDim >>>(dHash, dStart, dEnd, dParticles);
}

__device__
inline void clamp(float &val) {
	float valMax = 100.0f;
	float valMin = -valMax;
	val = val > valMax ? valMax : val;
	val = val < valMin ? valMin : val;
}



__global__
void leapfrogIntegrate_kernel(uint2 *dHash, uint *dStart, uint *dEnd, Particle *dParticles) {

	

	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= dParam.numParticles) return;

	// Upgrade half-time velocity
	dParticles[index].hvel.x += dParticles[index].acc.x*dParam.dt/2.f;
	dParticles[index].hvel.y += dParticles[index].acc.y*dParam.dt/2.f;
	dParticles[index].hvel.z += dParticles[index].acc.z*dParam.dt/2.f;

	// Upgrade position according to half-time velocity
	dParticles[index].pos.x += dParticles[index].hvel.x*dParam.dt;
	dParticles[index].pos.y += dParticles[index].hvel.y*dParam.dt;
	dParticles[index].pos.z += dParticles[index].hvel.z*dParam.dt;

	// Upgrade velocity 
	dParticles[index].vel.x += dParticles[index].acc.x*dParam.dt;
	dParticles[index].vel.y += dParticles[index].acc.y*dParam.dt;
	dParticles[index].vel.z += dParticles[index].acc.z*dParam.dt;

	/*
	// This is for numerical stability
	clamp(dParticles[index].vel.x);
	clamp(dParticles[index].vel.y);
	clamp(dParticles[index].vel.z);

	clamp(dParticles[index].hvel.x);
	clamp(dParticles[index].hvel.y);
	clamp(dParticles[index].hvel.z);
	*/

	// Handle collision against the wall

	float jitter = dParam.h*0.01f;
	if (dParticles[index].pos.x > dParam.worldMax.x) {
		dParticles[index].pos.x = dParam.worldMax.x - jitter;
		dParticles[index].vel.x *= dParam.coeCollision;
	}
	if (dParticles[index].pos.x < dParam.worldMin.x) {
		dParticles[index].pos.x = dParam.worldMin.x + jitter;
		dParticles[index].vel.x *= dParam.coeCollision;
	}

	if (dParticles[index].pos.y > dParam.worldMax.y) {
		dParticles[index].pos.y = dParam.worldMax.y - jitter;
		dParticles[index].vel.y *= dParam.coeCollision;
	}
	if (dParticles[index].pos.y < dParam.worldMin.y) {
		dParticles[index].pos.y = dParam.worldMin.y + jitter;
		dParticles[index].vel.y *= dParam.coeCollision;
	}

	if (dParticles[index].pos.z > dParam.worldMax.z) {
		dParticles[index].pos.z = dParam.worldMax.z - jitter;
		dParticles[index].vel.z *= dParam.coeCollision;
	}
	if (dParticles[index].pos.z < dParam.worldMin.z) {
		dParticles[index].pos.z = dParam.worldMin.z + jitter;
		dParticles[index].vel.z *= dParam.coeCollision;
	}

}

void leapfrogIntegrate(Param *pParam, uint2 *dHash, uint *dStart, uint *dEnd, Particle *dParticles) {
	leapfrogIntegrate_kernel << <pParam->gridDim, pParam->blockDim >> > (dHash, dStart, dEnd, dParticles);
	
}

__global__
void copyPosToDeviceVector_kernel(Particle *dParticles, float3 *dPos) {
	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= dParam.numParticles) return;
	
	dPos[index] = dParticles[index].pos;
}

void copyPosToDeviceVector(Param *pParam, Particle *dParticles, float3 *dPos) {
	copyPosToDeviceVector_kernel <<<pParam->gridDim, pParam->blockDim >>> (dParticles, dPos);
}


__global__
void copyVelToDeviceVector_kernel(Particle *dParticles, float3 *dVel) {
	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= dParam.numParticles) return;

	dVel[index] = dParticles[index].vel;
}

void copyVelToDeviceVector(Param *pParam, Particle *dParticles, float3 *dVel) {
	copyVelToDeviceVector_kernel <<<pParam->gridDim, pParam->blockDim >> > (dParticles, dVel);
}

__global__
void copyPosFromDeviceVector_kernel(Particle *dParticles, float3 *dPos, uint start, uint end) {
	uint index = blockIdx.x*blockDim.x + threadIdx.x + start;
	if (index >= end) return;

	dParticles[index].pos = dPos[index];
}

void copyPosFromDeviceVector(Param *pParam, Particle *dParticles, float3 *dPos, uint start, uint end) {
	uint blockDim = pParam->blockDim;
	uint numThreads = end - start > pParam->blockDim ? end - start : pParam->blockDim;
	uint gridDim= numThreads%blockDim == 0 ? numThreads / blockDim : numThreads / blockDim + 1;

	copyPosFromDeviceVector_kernel << <gridDim, blockDim >> > (dParticles, dPos, start, end);
}

__global__
void copyVelFromDeviceVector_kernel(Particle *dParticles, float3 *dVel, uint start, uint end) {
	uint index = blockIdx.x*blockDim.x + threadIdx.x + start;
	if (index >= end) return;

	dParticles[index].vel = dVel[index];
	dParticles[index].hvel = dVel[index];
}

void copyVelFromDeviceVector(Param *pParam, Particle *dParticles, float3 *dVel, uint start, uint end) {
	uint blockDim = pParam->blockDim;
	uint numThreads = end - start > pParam->blockDim ? end - start : pParam->blockDim;
	uint gridDim = numThreads%blockDim == 0 ? numThreads / blockDim : numThreads / blockDim + 1;
	
	copyVelFromDeviceVector_kernel << <gridDim, blockDim >> > (dParticles, dVel, start, end);
}


// Compute density that all particles in a cell contribute to the target point
// Using only for isovalue computing
__device__
float computeCellIsoval(uint2 *dHash, uint *dStart, uint *dEnd, uint3 cellIdx, float3 pointPos, Particle *dParticles) {

	float val = 0.f;


	uint cellHash = getHash(cellIdx);

	uint startIdx = dStart[cellHash], endIdx = dEnd[cellHash];

	if (startIdx == 0xffffffff) return val;

	for (uint idx = startIdx;idx != endIdx;++idx) {
		
		
		float3 neighborPos = dParticles[dHash[idx].y].pos;
		
		float r2 = (pointPos.x - neighborPos.x)*(pointPos.x - neighborPos.x) +
			(pointPos.y - neighborPos.y)*(pointPos.y - neighborPos.y) +
			(pointPos.z - neighborPos.z)*(pointPos.z - neighborPos.z);
		if (r2 >= dParam.r2) continue;
		float _r = sqrt(r2) / dParam.r;	// r scaled

		// Metaball model
		// See http://www.geisswerks.com/ryan/BLOBS/blobs.html
		val += 1.0f - _r*_r*_r*(_r*(_r*6.0f - 15.0f) + 10.0f);

	}
	return val;
}


__global__
void computeIsoValue_kernel(uint2 *dHash, uint *dStart, uint *dEnd, Particle *dParticles, float *dIsoval) {
	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= dParam.numPoints_Isoval) return;

	uint resY = dParam.dimRes.y + 1, resZ = dParam.dimRes.z + 1;
	uint index_temp = index;
	uint x, y, z;
	z = index_temp % resZ;
	index_temp = (index_temp - z) / resZ;
	y = index_temp % resY;
	x = (index_temp - y) / resY;

	float3 pointPos = make_float3(
		x*dParam.sizeRes.x+dParam.worldMin.x,
		y*dParam.sizeRes.y+dParam.worldMin.y, 
		z*dParam.sizeRes.z+dParam.worldMin.z
	);
	uint3 cellIdx = getIdx(pointPos);

	float val = 0.0f;

	for (int dx = -dParam.r2h;dx <= dParam.r2h;++dx) {
		int _x = cellIdx.x + dx;
		if (_x < 0 || _x >= dParam.dimCell.x)continue;

		for (int dy = -dParam.r2h;dy <= dParam.r2h;++dy) {
			int _y = cellIdx.y + dy;
			if (_y < 0 || _y >= dParam.dimCell.y)continue;

			for (int dz = -dParam.r2h;dz <= dParam.r2h;++dz) {
				int _z = cellIdx.z + dz;
				if (_z < 0 || _z >= dParam.dimCell.z)continue;

				uint3 neighborCellIdx = make_uint3(_x, _y, _z);

				val += computeCellIsoval(dHash, dStart, dEnd, neighborCellIdx, pointPos, dParticles);
			

			}
		}
	}

	dIsoval[index] = val;

}

void computeIsoValue(Param *pParam, uint2 *dHash, uint *dStart, uint *dEnd, Particle *dParticles, float *dIsoval) {
	computeIsoValue_kernel <<< pParam->gridDim_Isoval, pParam->blockDim >>> (dHash, dStart, dEnd, dParticles, dIsoval);
}