//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include "DEMAnalyticalBoundaryConstants.cuh"
#include <core/utils/GpuError.h>
#include <cuda_runtime.h>

namespace deme {

// Definition of constant memory arrays
__constant__ objType_t d_objType[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ bodyID_t d_objOwner[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ float d_objNormal[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ materialsOffset_t d_objMaterial[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ float d_objRelPosX[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ float d_objRelPosY[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ float d_objRelPosZ[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ float d_objRotX[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ float d_objRotY[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ float d_objRotZ[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ float d_objSize1[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ float d_objSize2[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ float d_objSize3[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ float d_objMass[DEME_MAX_ANALYTICAL_COMPONENTS];

void copyAnalyticalBoundaryDataToConstantMemory(
    const objType_t* h_objType,
    const bodyID_t* h_objOwner,
    const float* h_objNormal,
    const materialsOffset_t* h_objMaterial,
    const float* h_objRelPosX,
    const float* h_objRelPosY,
    const float* h_objRelPosZ,
    const float* h_objRotX,
    const float* h_objRotY,
    const float* h_objRotZ,
    const float* h_objSize1,
    const float* h_objSize2,
    const float* h_objSize3,
    const float* h_objMass,
    unsigned int nAnalGM) {
    
    if (nAnalGM > DEME_MAX_ANALYTICAL_COMPONENTS) {
        DEME_ERROR("Number of analytical components (%u) exceeds maximum allowed (%u). "
                   "Please increase DEME_MAX_ANALYTICAL_COMPONENTS in DEMAnalyticalBoundaryConstants.cuh",
                   nAnalGM, DEME_MAX_ANALYTICAL_COMPONENTS);
        return;
    }

    // Copy each array to its corresponding constant memory symbol
    if (nAnalGM > 0) {
        GPU_CALL(cudaMemcpyToSymbol(d_objType, h_objType, nAnalGM * sizeof(objType_t)));
        GPU_CALL(cudaMemcpyToSymbol(d_objOwner, h_objOwner, nAnalGM * sizeof(bodyID_t)));
        GPU_CALL(cudaMemcpyToSymbol(d_objNormal, h_objNormal, nAnalGM * sizeof(float)));
        GPU_CALL(cudaMemcpyToSymbol(d_objMaterial, h_objMaterial, nAnalGM * sizeof(materialsOffset_t)));
        GPU_CALL(cudaMemcpyToSymbol(d_objRelPosX, h_objRelPosX, nAnalGM * sizeof(float)));
        GPU_CALL(cudaMemcpyToSymbol(d_objRelPosY, h_objRelPosY, nAnalGM * sizeof(float)));
        GPU_CALL(cudaMemcpyToSymbol(d_objRelPosZ, h_objRelPosZ, nAnalGM * sizeof(float)));
        GPU_CALL(cudaMemcpyToSymbol(d_objRotX, h_objRotX, nAnalGM * sizeof(float)));
        GPU_CALL(cudaMemcpyToSymbol(d_objRotY, h_objRotY, nAnalGM * sizeof(float)));
        GPU_CALL(cudaMemcpyToSymbol(d_objRotZ, h_objRotZ, nAnalGM * sizeof(float)));
        GPU_CALL(cudaMemcpyToSymbol(d_objSize1, h_objSize1, nAnalGM * sizeof(float)));
        GPU_CALL(cudaMemcpyToSymbol(d_objSize2, h_objSize2, nAnalGM * sizeof(float)));
        GPU_CALL(cudaMemcpyToSymbol(d_objSize3, h_objSize3, nAnalGM * sizeof(float)));
        GPU_CALL(cudaMemcpyToSymbol(d_objMass, h_objMass, nAnalGM * sizeof(float)));
    }
}

}  // namespace deme
