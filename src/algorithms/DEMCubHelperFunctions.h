//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <granular/DataStructs.h>
#include <granular/GranularStructs.h>
#include <granular/GranularDefines.h>
#include <core/utils/GpuManager.h>

namespace sgps {

// This file should not be visible to gcc so it's difficult to make functions here templated. We probably have to bear
// with writing each version of the same functions individually.
void cubPrefixScan_binSphere(binsSphereTouches_t* d_in,
                             binSphereTouchPairs_t* d_out,
                             size_t n,
                             cudaStream_t& this_stream,
                             DEMSolverStateDataKT& scratchPad);

void cubPrefixScan_sphereGeo(objID_t* d_in,
                             binSphereTouchPairs_t* d_out,
                             size_t n,
                             cudaStream_t& this_stream,
                             DEMSolverStateDataKT& scratchPad);

void cubPrefixScan_contacts(spheresBinTouches_t* d_in,
                            contactPairs_t* d_out,
                            size_t n,
                            cudaStream_t& this_stream,
                            DEMSolverStateDataKT& scratchPad);

// template <typename T1, typename T2>
// void cubPrefixScan(T1* d_in,
//                    T2* d_out,
//                    size_t n,
//                    cudaStream_t& this_stream,
//                    DEMSolverStateData& scratchPad);

void cubSortByKeys(binID_t* d_keys_in,
                   binID_t* d_keys_out,
                   bodyID_t* d_vals_in,
                   bodyID_t* d_vals_out,
                   size_t n,
                   cudaStream_t& this_stream,
                   DEMSolverStateDataKT& scratchPad);

void cubUnique(binID_t* d_in,
               binID_t* d_out,
               size_t* d_num_out,
               size_t n,
               cudaStream_t& this_stream,
               DEMSolverStateDataKT& scratchPad);

void cubRunLengthEncode(binID_t* d_in,
                        binID_t* d_unique_out,
                        spheresBinTouches_t* d_counts_out,
                        size_t* d_num_out,
                        size_t n,
                        cudaStream_t& this_stream,
                        DEMSolverStateDataKT& scratchPad);

void cubSum(float* d_in, float* d_out, size_t n, cudaStream_t& this_stream, DEMSolverStateDataDT& scratchPad);

void cubCollectForces(std::shared_ptr<jitify::Program>& collect_force,
                      clumpBodyInertiaOffset_t* inertiaPropOffsets,
                      bodyID_t* idA,
                      bodyID_t* idB,
                      contact_t* contactType,
                      float3* contactForces,
                      float3* contactPointA,
                      float3* contactPointB,
                      float* clump_aX,
                      float* clump_aY,
                      float* clump_aZ,
                      float* clump_alphaX,
                      float* clump_alphaY,
                      float* clump_alphaZ,
                      bodyID_t* ownerClumpBody,
                      const size_t nContactPairs,
                      const size_t nClumps,
                      bool contactPairArr_isFresh,
                      cudaStream_t& this_stream,
                      DEMSolverStateDataDT& scratchPad);

}  // namespace sgps
