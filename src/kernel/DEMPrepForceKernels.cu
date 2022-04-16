// DEM force computation related custom kernels
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>

inline __device__ void cleanUpContactForces(const size_t thisContact,
                                            sgps::DEMSimParams* simParams,
                                            sgps::DEMDataDT* granData) {
    granData->contactForces[thisContact].x = 0;
    granData->contactForces[thisContact].y = 0;
    granData->contactForces[thisContact].z = 0;
}

inline __device__ void cleanUpAcc(const size_t thisClump, sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData) {
    granData->aX[thisClump] = _Gx_;
    granData->aY[thisClump] = _Gy_;
    granData->aZ[thisClump] = _Gz_;
    granData->alphaX[thisClump] = 0;
    granData->alphaY[thisClump] = 0;
    granData->alphaZ[thisClump] = 0;
}

__global__ void prepareForceArrays(sgps::DEMSimParams* simParams,
                                   sgps::DEMDataDT* granData,
                                   const size_t nContactPairs) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        cleanUpContactForces(myID, simParams, granData);
    }
    if (myID < _nClumpBodies_) {
        cleanUpAcc(myID, simParams, granData);
    }
}