//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <granular/DataStructs.h>
#include <granular/GranularStructs.h>
#include <granular/GranularDefines.h>
#include <core/utils/GpuManager.h>

namespace sgps {

void cubPrefixScan(binsSphereTouches_t* d_in,
                   binsSphereTouchesScan_t* d_out,
                   size_t n,
                   GpuManager::StreamInfo& streamInfo,
                   DEMSolverStateData& scratchPad);

void cubSortByKeys(binID_t* d_keys,
                   bodyID_t* d_vals,
                   size_t n,
                   GpuManager::StreamInfo& streamInfo,
                   DEMSolverStateData& scratchPad);

void cubCollectForces(clumpBodyInertiaOffset_t* inertiaPropOffsets,
                      bodyID_t* idA,
                      bodyID_t* idB,
                      float3* contactForces,
                      float* clump_h2aX,
                      float* clump_h2aY,
                      float* clump_h2aZ,
                      bodyID_t* ownerClumpBody,
                      float* massClumpBody,
                      double h,
                      size_t nContactPairs,
                      size_t nClumps,
                      double l,
                      bool contactPairArr_isFresh,
                      GpuManager::StreamInfo& streamInfo,
                      DEMSolverStateData& scratchPad,
                      clumpBodyInertiaOffset_t nDistinctClumpBodyTopologies);

}  // namespace sgps
