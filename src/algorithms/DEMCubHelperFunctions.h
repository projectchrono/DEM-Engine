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
                   cudaStream_t& this_stream,
                   DEMSolverStateData& scratchPad);

void cubSortByKeys(binID_t* d_keys_in,
                   binID_t* d_keys_out,
                   bodyID_t* d_vals_in,
                   bodyID_t* d_vals_out,
                   size_t n,
                   cudaStream_t& this_stream,
                   DEMSolverStateData& scratchPad);

void cubCollectForces(clumpBodyInertiaOffset_t* inertiaPropOffsets,
                      bodyID_t* idA,
                      bodyID_t* idB,
                      float3* contactForces,
                      float3* contactPointA,
                      float3* contactPointB,
                      float* clump_h2aX,
                      float* clump_h2aY,
                      float* clump_h2aZ,
                      float* clump_h2AlphaX,
                      float* clump_h2AlphaY,
                      float* clump_h2AlphaZ,
                      bodyID_t* ownerClumpBody,
                      float* massClumpBody,
                      float* mmiXX,
                      float* mmiYY,
                      float* mmiZZ,
                      double h,
                      size_t nContactPairs,
                      size_t nClumps,
                      double l,
                      bool contactPairArr_isFresh,
                      cudaStream_t& this_stream,
                      DEMSolverStateData& scratchPad,
                      clumpBodyInertiaOffset_t nDistinctClumpBodyTopologies);

}  // namespace sgps
