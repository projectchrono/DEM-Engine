//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cub/cub.cuh>
#include <core/utils/JitHelper.h>

#include <algorithms/DEMCubHelperFunctions.h>

#include <core/utils/GpuError.h>

namespace sgps {

void cubCollectForces(sgps::clumpBodyInertiaOffset_t* inertiaPropOffsets,
                      sgps::bodyID_t* idA,
                      sgps::bodyID_t* idB,
                      float3* contactForces,
                      float* clump_h2aX,
                      float* clump_h2aY,
                      float* clump_h2aZ,
                      sgps::bodyID_t* ownerClumpBody,
                      float* massClumpBody,
                      double h,
                      size_t n,
                      double l,
                      GpuManager::StreamInfo& streamInfo,
                      sgps::DEMSolverStateData& scratchPad) {
    // Firstly, prepare the owner ID array (nContactPairs * bodyID_t) for usage in final reduction by key

    // Secondly, prepare the owner mass array (nContactPairs * float) for usage in final reduction by key

    // Finally, do the reduction by key

    for (size_t i = 0; i < n; i++) {
        bodyID_t bodyA = idA[i];
        bodyID_t bodyB = idB[i];
        const float3 F = contactForces[i];
        bodyID_t AOwner = ownerClumpBody[bodyA];
        float AMass = massClumpBody[inertiaPropOffsets[AOwner]];
        clump_h2aX[AOwner] += (double)F.x / AMass / l * h * h;
        clump_h2aY[AOwner] += (double)F.y / AMass / l * h * h;
        clump_h2aZ[AOwner] += (double)F.z / AMass / l * h * h;

        bodyID_t BOwner = ownerClumpBody[bodyB];
        float BMass = massClumpBody[inertiaPropOffsets[BOwner]];
        clump_h2aX[BOwner] += -(double)F.x / BMass / l * h * h;
        clump_h2aY[BOwner] += -(double)F.y / BMass / l * h * h;
        clump_h2aZ[BOwner] += -(double)F.z / BMass / l * h * h;
    }
}

}  // namespace sgps
