// Granular force computation related custom kernels
#include <granular/DataStructs.h>
#include "cub/device/device_reduce.cuh"

inline __device__ void applyGravity(unsigned int thisClump, sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData) {
    // Actually, h should be JITCed into the kernel itself
    granData->h2aX[thisClump] += simParams->h * simParams->h * simParams->Gx;
}

__global__ void deriveClumpForces(sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData) {
    unsigned int thisClump = blockIdx.x * blockDim.x + threadIdx.x;
    if (thisClump < simParams->nClumpBodies) {
        applyGravity(thisClump, simParams, granData);
    }
}
