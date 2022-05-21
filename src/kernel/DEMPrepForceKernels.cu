// DEM force computation related custom kernels
#include <DEM/DEMDefines.h>
#include <kernel/DEMHelperKernels.cu>

inline __device__ void cleanUpContactForces(size_t thisContact,
                                            sgps::DEMSimParams* simParams,
                                            sgps::DEMDataDT* granData) {
    granData->contactForces[thisContact].x = 0;
    granData->contactForces[thisContact].y = 0;
    granData->contactForces[thisContact].z = 0;
    // TODO: Prescribed forces to be added here
}

inline __device__ void cleanUpAcc(size_t thisClump, sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData) {
    granData->aX[thisClump] = _Gx_;
    granData->aY[thisClump] = _Gy_;
    granData->aZ[thisClump] = _Gz_;
    granData->alphaX[thisClump] = 0;
    granData->alphaY[thisClump] = 0;
    granData->alphaZ[thisClump] = 0;
    // TODO: Prescribed accelerations to be added here
}

__global__ void prepareForceArrays(sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData, size_t nContactPairs) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        cleanUpContactForces(myID, simParams, granData);
    }
    if (myID < _nOwnerBodies_) {
        cleanUpAcc(myID, simParams, granData);
    }
}

__global__ void rearrangeContactHistory(sgps::contactPairs_t* contactMapping,
                                        float3* oldHistory,
                                        float3* newHistory,
                                        sgps::notStupidBool_t* sentry,
                                        size_t nContactPairs) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        sgps::contactPairs_t map_from = contactMapping[myID];
        if (map_from == sgps::DEM_NULL_MAPPING_PARTNER) {
            // If it is a NULL ID then kT says this contact is new
            newHistory[myID] = make_float3(0, 0, 0);
        } else {
            // Not a new contact, need to map it from somewhere in the old history array
            newHistory[myID] = oldHistory[map_from];
            // This sentry trys to make sure that all `alive' contacts got mapped to some place
            sentry[map_from] = 0;
        }
    }
}

__global__ void markAliveContacts(float3* history, sgps::notStupidBool_t* sentry, size_t nContactPairs) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        float3 myHistory = history[myID];
        // If this is alive then mark it
        if (length(myHistory) > SGPS_DEM_TINY_FLOAT) {
            sentry[myID] = 1;
        } else {
            sentry[myID] = 0;
        }
    }
}
