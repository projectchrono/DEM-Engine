//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_ANALYTICAL_BOUNDARY_CONSTANTS_CUH
#define DEME_ANALYTICAL_BOUNDARY_CONSTANTS_CUH

#include <DEM/Defines.h>

// Maximum number of analytical boundary components that can be stored in constant memory
// This limit exists because CUDA constant memory is limited (typically 64KB)
#define DEME_MAX_ANALYTICAL_COMPONENTS 4096

namespace deme {

// Analytical boundary component data stored in constant memory
// These arrays are populated at initialization time with analytical boundary information
__constant__ extern objType_t d_objType[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ extern bodyID_t d_objOwner[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ extern float d_objNormal[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ extern materialsOffset_t d_objMaterial[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ extern float d_objRelPosX[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ extern float d_objRelPosY[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ extern float d_objRelPosZ[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ extern float d_objRotX[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ extern float d_objRotY[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ extern float d_objRotZ[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ extern float d_objSize1[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ extern float d_objSize2[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ extern float d_objSize3[DEME_MAX_ANALYTICAL_COMPONENTS];
__constant__ extern float d_objMass[DEME_MAX_ANALYTICAL_COMPONENTS];

// Host-side function to copy analytical boundary data to constant memory
// This should be called once during initialization
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
    unsigned int nAnalGM);

}  // namespace deme

#endif  // DEME_ANALYTICAL_BOUNDARY_CONSTANTS_CUH
