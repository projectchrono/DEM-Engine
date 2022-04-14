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
    granData->h2aX[thisClump] = simParams->h * simParams->h * _Gx_ / _l_;
    granData->h2aY[thisClump] = simParams->h * simParams->h * _Gy_ / _l_;
    granData->h2aZ[thisClump] = simParams->h * simParams->h * _Gz_ / _l_;
    granData->h2AlphaX[thisClump] = 0;
    granData->h2AlphaY[thisClump] = 0;
    granData->h2AlphaZ[thisClump] = 0;
}

__global__ void prepareForceArrays(sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData, size_t nContactPairs) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        cleanUpContactForces(myID, simParams, granData);
    }
    if (myID < _nClumpBodies_) {
        cleanUpAcc(myID, simParams, granData);
    }
}