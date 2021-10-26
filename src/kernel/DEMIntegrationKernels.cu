// DEM integration related custom kernels
#include <granular/DataStructs.h>

// For now, write a custom kernel (instead of cub-based), and change it later
inline __device__ void integrateLinVel(unsigned int thisClump,
                                       sgps::DEMSimParams* simParams,
                                       sgps::DEMDataDT* granData) {
    // Explicit update
    granData->hvX[thisClump] += granData->h2aX[thisClump];
    granData->hvY[thisClump] += granData->h2aY[thisClump];
    granData->hvZ[thisClump] += granData->h2aZ[thisClump];
}

inline __device__ void integrateLinPos(unsigned int thisClump,
                                       sgps::DEMSimParams* simParams,
                                       sgps::DEMDataDT* granData) {
    // TODO: call a kernel here to determine whether loc is in a new voxel now
    granData->locX[thisClump] += granData->hvX[thisClump];
    granData->locY[thisClump] += granData->hvY[thisClump];
    granData->locZ[thisClump] += granData->hvZ[thisClump];
}

__global__ void integrateClumps(sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData) {
    unsigned int thisClump = blockIdx.x * blockDim.x + threadIdx.x;
    if (thisClump < simParams->nClumpBodies) {
        integrateLinVel(thisClump, simParams, granData);
        integrateLinPos(thisClump, simParams, granData);
    }
}
