//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <algorithms/DEMStaticDeviceSubroutines.h>
#include <algorithms/DEMStaticDeviceUtilities.cuh>

#include <kernel/DEMHelperKernels.cuh>

namespace deme {

////////////////////////////////////////////////////////////////////////////////
// Misc kernels implementations
////////////////////////////////////////////////////////////////////////////////

__global__ void markOwnerToChange_impl(notStupidBool_t* idBool,
                                       float* ownerFactors,
                                       bodyID_t* dIDs,
                                       float* dFactors,
                                       size_t n) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        bodyID_t myOwner = dIDs[myID];
        float myFactor = dFactors[myID];
        idBool[myOwner] = 1;
        ownerFactors[myOwner] = myFactor;
    }
}

void markOwnerToChange(notStupidBool_t* idBool,
                       float* ownerFactors,
                       bodyID_t* dIDs,
                       float* dFactors,
                       size_t n,
                       cudaStream_t& this_stream) {
    size_t blocks_needed = (n + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        markOwnerToChange_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            idBool, ownerFactors, dIDs, dFactors, n);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
}

template <typename DEMData>
__global__ void modifyComponents_impl(DEMData* granData,
                                      notStupidBool_t* idBool,
                                      float* factors,
                                      size_t n) {
    size_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < n) {
        // Get my owner ID
        bodyID_t myOwner = granData->ownerClumpBody[sphereID];
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

void modifyComponents(DEMDataDT* granData,
                      notStupidBool_t* idBool,
                      float* factors,
                      size_t n,
                      cudaStream_t& this_stream) {
    size_t blocks_needed = (n + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        modifyComponents_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, idBool, factors, n);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
}

void modifyComponents(DEMDataKT* granData,
                      notStupidBool_t* idBool,
                      float* factors,
                      size_t n,
                      cudaStream_t& this_stream) {
    size_t blocks_needed = (n + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        modifyComponents_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, idBool, factors, n);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
}

__global__ void computeMarginFromAbsv_impl(DEMSimParams* simParams,
                                           DEMDataKT* granData,
                                           float* ts,
                                           unsigned int* maxDrift,
                                           size_t n) {
    size_t ownerID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ownerID < n) {
        float absv = granData->marginSize[ownerID];
        unsigned int my_family = granData->familyID[ownerID];
        if (!isfinite(absv)) {
            // May produce messy error messages, but it's still good to know what entities went wrong
            DEME_ABORT_KERNEL(
                "Absolute velocity for ownerID %llu is infinite (and it's a worse version of "
                "max-velocity-exceeded-allowance).\n",
                static_cast<unsigned long long>(ownerID));
        }
        if (absv > simParams->approxMaxVel) {
            absv = simParams->approxMaxVel;
        }
        // User-specified extra margin also needs to be added here. This marginSize is used for bin--sph or bin--tri
        // contacts but not entirely the same as the one used for sph--sph or sph--tri contacts, since the latter is
        // stricter.
        granData->marginSize[ownerID] =
            (double)(absv * simParams->expSafetyMulti + simParams->expSafetyAdder) * (*ts) * (*maxDrift) +
            // Temp artificial margin for mesh contact
            0.05 + granData->familyExtraMarginSize[my_family];
    }
}

void computeMarginFromAbsv(DEMSimParams* simParams,
                           DEMDataKT* granData,
                           float* ts,
                           unsigned int* maxDrift,
                           size_t n,
                           cudaStream_t& this_stream) {
    size_t blocks_needed = (n + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        computeMarginFromAbsv_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            simParams, granData, ts, maxDrift, n);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
}

__global__ void fillMarginValues_impl(DEMSimParams* simParams, DEMDataKT* granData, size_t n) {
    size_t ownerID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ownerID < n) {
        unsigned int my_family = granData->familyID[ownerID];
        granData->marginSize[ownerID] = simParams->beta + granData->familyExtraMarginSize[my_family];
    }
}

void fillMarginValues(DEMSimParams* simParams, DEMDataKT* granData, size_t n, cudaStream_t& this_stream) {
    size_t blocks_needed = (n + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        fillMarginValues_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(simParams, granData, n);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
}

}  // namespace deme
