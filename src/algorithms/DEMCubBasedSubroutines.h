//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <DEM/DEMStructs.h>
#include <DEM/DEMDefines.h>
#include <core/utils/GpuManager.h>
#include <core/utils/ManagedAllocator.hpp>

namespace sgps {

void doubleSumReduce(double* d_in, double* d_out, size_t n, cudaStream_t& this_stream, DEMSolverStateData& scratchPad);

void floatSumReduce(float* d_in, float* d_out, size_t n, cudaStream_t& this_stream, DEMSolverStateData& scratchPad);

void boolSumReduce(notStupidBool_t* d_in,
                   size_t* d_out,
                   size_t n,
                   cudaStream_t& this_stream,
                   DEMSolverStateData& scratchPad);

void boolMaxReduce(notStupidBool_t* d_in,
                   notStupidBool_t* d_out,
                   size_t n,
                   cudaStream_t& this_stream,
                   DEMSolverStateData& scratchPad);

void floatMaxReduce(float* d_in, float* d_out, size_t n, cudaStream_t& this_stream, DEMSolverStateData& scratchPad);

void floatMaxReduceByKey(notStupidBool_t* d_keys_in,
                         notStupidBool_t* d_unique_out,
                         float* d_vals_in,
                         float* d_aggregates_out,
                         size_t* d_num_out,
                         size_t n,
                         cudaStream_t& this_stream,
                         DEMSolverStateData& scratchPad);

void contactDetection(std::shared_ptr<jitify::Program>& bin_occupation_kernels,
                      std::shared_ptr<jitify::Program>& contact_detection_kernels,
                      std::shared_ptr<jitify::Program>& history_kernels,
                      DEMDataKT* granData,
                      DEMSimParams* simParams,
                      SolverFlags& solverFlags,
                      DEM_VERBOSITY& verbosity,
                      // The following arrays may need to change sizes, so we can't pass pointers
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& idGeometryA,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& idGeometryB,
                      std::vector<contact_t, ManagedAllocator<contact_t>>& contactType,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& previous_idGeometryA,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& previous_idGeometryB,
                      std::vector<contact_t, ManagedAllocator<contact_t>>& previous_contactType,
                      std::vector<contactPairs_t, ManagedAllocator<contactPairs_t>>& contactMapping,
                      cudaStream_t& this_stream,
                      DEMSolverStateData& scratchPad,
                      SolverTimers& timers);

void collectContactForces(std::shared_ptr<jitify::Program>& collect_force_kernels,
                          DEMDataDT* granData,
                          const size_t nContactPairs,
                          const size_t nClumps,
                          bool contactPairArr_isFresh,
                          cudaStream_t& this_stream,
                          DEMSolverStateData& scratchPad,
                          SolverTimers& timers);

}  // namespace sgps
