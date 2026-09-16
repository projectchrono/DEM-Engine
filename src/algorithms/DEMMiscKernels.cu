//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include "algorithms/DEMStaticDeviceSubroutines.h"
#include "algorithms/DEMStaticDeviceUtilities.cuh"

#include "kernel/DEMHelperKernels.cuh"

namespace deme {

////////////////////////////////////////////////////////////////////////////////
// Misc kernels implementations
////////////////////////////////////////////////////////////////////////////////

// granData is a device pointer: select its primitive arrays inside the kernel, never in the host launch wrapper.
template <ownerType_t OwnerType>
__global__ void fillMarginValues_impl(DEMSimParams* simParams, DEMDataKT* granData, size_t n) {
    size_t ID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ID < n) {
        float* marginSizeArr;
        bodyID_t* ownerIDArr;
        if constexpr (OwnerType == OWNER_T_CLUMP) {
            marginSizeArr = granData->marginSizeSphere;
            ownerIDArr = granData->ownerClumpBody;
        } else if constexpr (OwnerType == OWNER_T_MESH) {
            marginSizeArr = granData->marginSizeTriangle;
            ownerIDArr = granData->ownerTriMesh;
        } else {
            marginSizeArr = granData->marginSizeAnalytical;
            ownerIDArr = granData->ownerAnalBody;
        }
        bodyID_t ownerID = ownerIDArr[ID];
        unsigned int my_family = granData->familyID[ownerID];
        marginSizeArr[ID] = simParams->dyn.beta + granData->familyExtraMarginSize[my_family];
    }
}

// Fill each primitive kind with its fixed margin and family allowance using the device-resident metadata.
void fillMarginValues(DEMSimParams* simParams,
                      DEMDataKT* granData,
                      size_t nSphere,
                      size_t nTri,
                      size_t nAnal,
                      cudaStream_t& this_stream) {
    constexpr unsigned int FIXED_MARGIN_BLOCK = 256;
    size_t blocks_needed = (nSphere + FIXED_MARGIN_BLOCK - 1) / FIXED_MARGIN_BLOCK;
    if (blocks_needed > 0) {
        fillMarginValues_impl<OWNER_T_CLUMP>
            <<<blocks_needed, FIXED_MARGIN_BLOCK, 0, this_stream>>>(simParams, granData, nSphere);
        DEME_GPU_DEBUG_SYNC(this_stream);
    }
    blocks_needed = (nTri + FIXED_MARGIN_BLOCK - 1) / FIXED_MARGIN_BLOCK;
    if (blocks_needed > 0) {
        fillMarginValues_impl<OWNER_T_MESH>
            <<<blocks_needed, FIXED_MARGIN_BLOCK, 0, this_stream>>>(simParams, granData, nTri);
        DEME_GPU_DEBUG_SYNC(this_stream);
    }
    blocks_needed = (nAnal + FIXED_MARGIN_BLOCK - 1) / FIXED_MARGIN_BLOCK;
    if (blocks_needed > 0) {
        fillMarginValues_impl<OWNER_T_ANALYTICAL>
            <<<blocks_needed, FIXED_MARGIN_BLOCK, 0, this_stream>>>(simParams, granData, nAnal);
        DEME_GPU_DEBUG_SYNC(this_stream);
    }
}

}  // namespace deme
