// DEM misc. kernels
#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>

__global__ void markOwnerToChange(deme::notStupidBool_t* idBool,
                                  float* ownerFactors,
                                  deme::bodyID_t* dIDs,
                                  float* dFactors,
                                  size_t n) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        deme::bodyID_t myOwner = dIDs[myID];
        float myFactor = dFactors[myID];
        idBool[myOwner] = 1;
        ownerFactors[myOwner] = myFactor;
    }
}

template <typename DEMData>
__global__ void modifyComponents(DEMData* granData, deme::notStupidBool_t* idBool, float* factors, size_t n) {
    size_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < n) {
        // Get my owner ID
        deme::bodyID_t myOwner = granData->ownerClumpBody[sphereID];
        // If not marked, we have nothing to do
        if (idBool[myOwner]) {
            float factor = factors[myOwner];
            // Expand radius and relPos
            granData->relPosSphereX[sphereID] *= factor;
            granData->relPosSphereY[sphereID] *= factor;
            granData->relPosSphereZ[sphereID] *= factor;
            granData->radiiSphere[sphereID] *= factor;
        }
    }
}

__global__ void computeMarginFromAbsv(deme::DEMSimParams* simParams,
                                      deme::DEMDataKT* granData,
                                      double* ts,
                                      unsigned int* maxDrift,
                                      size_t n) {
    size_t ownerID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ownerID < n) {
        float absv = granData->marginSize[ownerID];
        unsigned int my_family = granData->familyID[ownerID];
        if (!isfinite(absv)) {
            // May produce messy error messages, but it's still good to know what entities went wrong
            DEME_ABORT_KERNEL("Absolute velocity for ownerID %llu is not finite. This happened at time %.9g.\n",
                              static_cast<unsigned long long>(ownerID), simParams->timeElapsed);
        }
        if (absv > simParams->approxMaxVel) {
            absv = simParams->approxMaxVel;
        }
        // User-specified extra margin also needs to be added here. This marginSize is used for bin--sph or bin--tri
        // contacts but not entirely the same as the one used for sph--sph or sph--tri contacts, since the latter is
        // stricter.
        granData->marginSize[ownerID] =
            (double)(absv * simParams->expSafetyMulti + simParams->expSafetyAdder) * (*ts) * (*maxDrift) +
            granData->familyExtraMarginSize[my_family];
    }
}

__global__ void fillMarginValues(deme::DEMSimParams* simParams, deme::DEMDataKT* granData, size_t n) {
    size_t ownerID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ownerID < n) {
        unsigned int my_family = granData->familyID[ownerID];
        granData->marginSize[ownerID] = simParams->beta + granData->familyExtraMarginSize[my_family];
    }
}
