// DEM force computation related custom kernels
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>

inline __device__ void cleanUpContactForces(size_t thisContact,
                                            sgps::DEMSimParams* simParams,
                                            sgps::DEMDataDT* granData) {
    // Actually, h should be JITCed into the kernel itself
    granData->contactForces[thisContact].x = 0;
    granData->contactForces[thisContact].y = 0;
    granData->contactForces[thisContact].z = 0;
}

inline __device__ void cleanUpAcc(size_t thisClump, sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData) {
    // Actually, h should be JITCed into the kernel itself
    granData->h2aX[thisClump] = 0;
    granData->h2aY[thisClump] = 0;
    granData->h2aZ[thisClump] = 0;
    granData->h2AlphaX[thisClump] = 0;
    granData->h2AlphaY[thisClump] = 0;
    granData->h2AlphaZ[thisClump] = 0;
}

// Gravity creates no torque about CoM so we can do this
inline __device__ void applyGravity(size_t thisClump, sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData) {
    // Actually, l should be JITCed into the kernel itself
    granData->h2aX[thisClump] += simParams->h * simParams->h * simParams->Gx / simParams->l;
    granData->h2aY[thisClump] += simParams->h * simParams->h * simParams->Gy / simParams->l;
    granData->h2aZ[thisClump] += simParams->h * simParams->h * simParams->Gz / simParams->l;
}

__global__ void prepareForceArrays(sgps::DEMSimParams* simParams,
                                   sgps::DEMDataDT* granData,
                                   sgps::DEMTemplate* granTemplates) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < simParams->nContactPairs) {
        cleanUpContactForces(myID, simParams, granData);
    }
    if (myID < simParams->nClumpBodies) {
        cleanUpAcc(myID, simParams, granData);
        applyGravity(myID, simParams, granData);
    }
}