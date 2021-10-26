// DEM force computation related custom kernels
#include <granular/DataStructs.h>
#include <cub/device/device_reduce.cuh>

inline __device__ void cleanUpForces(unsigned int thisClump, sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData) {
    // Actually, h should be JITCed into the kernel itself
    granData->h2aX[thisClump] = 0;
    granData->h2aY[thisClump] = 0;
    granData->h2aZ[thisClump] = 0;
}

inline __device__ void applyGravity(unsigned int thisClump, sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData) {
    // Actually, h should be JITCed into the kernel itself
    granData->h2aX[thisClump] += simParams->h * simParams->h * simParams->Gx;
    granData->h2aY[thisClump] += simParams->h * simParams->h * simParams->Gy;
    granData->h2aZ[thisClump] += simParams->h * simParams->h * simParams->Gz;
}

__global__ void deriveClumpForces(sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData) {
    unsigned int thisClump = blockIdx.x * blockDim.x + threadIdx.x;
    if (thisClump < simParams->nClumpBodies) {
        cleanUpForces(thisClump, simParams, granData);
        applyGravity(thisClump, simParams, granData);
    }
}
