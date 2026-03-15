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
        markOwnerToChange_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(idBool, ownerFactors,
                                                                                              dIDs, dFactors, n);
    }
}

template <typename DEMData>
__global__ void modifyComponents_impl(DEMData* granData, notStupidBool_t* idBool, float* factors, size_t n) {
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

template <typename DEMData>
void modifyComponents(DEMData* granData, notStupidBool_t* idBool, float* factors, size_t n, cudaStream_t& this_stream) {
    size_t blocks_needed = (n + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        modifyComponents_impl<DEMData>
            <<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(granData, idBool, factors, n);
    }
}
template void modifyComponents<DEMDataDT>(DEMDataDT* granData,
                                          notStupidBool_t* idBool,
                                          float* factors,
                                          size_t n,
                                          cudaStream_t& this_stream);
template void modifyComponents<DEMDataKT>(DEMDataKT* granData,
                                          notStupidBool_t* idBool,
                                          float* factors,
                                          size_t n,
                                          cudaStream_t& this_stream);

__global__ void fillMarginValues_impl(DEMSimParams* simParams,
                                      DEMDataKT* granData,
                                      float* marginSizeArr,
                                      bodyID_t* ownerIDArr,
                                      size_t n) {
    size_t ID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ID < n) {
        bodyID_t ownerID = ownerIDArr[ID];
        unsigned int my_family = granData->familyID[ownerID];
        marginSizeArr[ID] = simParams->dyn.beta + granData->familyExtraMarginSize[my_family];
    }
}

void fillMarginValues(DEMSimParams* simParams,
                      DEMDataKT* granData,
                      size_t nSphere,
                      size_t nTri,
                      size_t nAnal,
                      cudaStream_t& this_stream) {
    size_t blocks_needed = (nSphere + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        fillMarginValues_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            simParams, granData, granData->marginSizeSphere, granData->ownerClumpBody, nSphere);
    }
    blocks_needed = (nTri + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        fillMarginValues_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            simParams, granData, granData->marginSizeTriangle, granData->ownerTriMesh, nTri);
    }
    blocks_needed = (nAnal + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        fillMarginValues_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            simParams, granData, granData->marginSizeAnalytical, granData->ownerAnalBody, nAnal);
    }
}

}  // namespace deme
