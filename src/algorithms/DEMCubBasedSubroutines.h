//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <DEM/DEMStructs.h>
#include <DEM/DEMDefines.h>
#include <core/utils/GpuManager.h>
#include <core/utils/ManagedAllocator.hpp>

namespace sgps {

void sumReduce(float* d_in, float* d_out, size_t n, cudaStream_t& this_stream, DEMSolverStateDataDT& scratchPad);

void contactDetection(std::shared_ptr<jitify::Program>& bin_occupation_kernels,
                      std::shared_ptr<jitify::Program>& contact_detection_kernels,
                      DEMDataKT* granData,
                      DEMSimParams* simParams,
                      SolverFlags& solverFlags,
                      DEM_VERBOSITY& verbosity,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& idGeometryA,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& idGeometryB,
                      std::vector<contact_t, ManagedAllocator<contact_t>>& contactType,
                      cudaStream_t& this_stream,
                      DEMSolverStateDataKT& scratchPad);

void collectForces(std::shared_ptr<jitify::Program>& collect_force_kernels,
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
