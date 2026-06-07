//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_STATIC_DEVICE_SUBROUTINES_H
#define DEME_STATIC_DEVICE_SUBROUTINES_H

#include <DEM/Defines.h>
#include <DEM/Structs.h>
#include <core/utils/JitHelper.h>
#include <core/utils/GpuManager.h>
#include <core/utils/CudaAllocator.hpp>

namespace deme {

// ========================================================================
// Typically, these are CUB-based relatively heavy-duty subroutines.
// Other files of this project may link against this .h file, without knowing
// it's in fact about CUDA, as they are C++ only.
// Functions here are statically compiled against available CUDA versions,
// unlike other just-in-time compiled kernels you see in other files of
// this project.
// ========================================================================

////////////////////////////////////////////////////////////////////////////////
// Cub utilities that may be needed by some user actions
////////////////////////////////////////////////////////////////////////////////

template <typename T1, typename T2>
void cubSumReduce(T1* d_in, T2* d_out, size_t n, cudaStream_t& this_stream, DEMSolverScratchData& scratchPad);
template <typename T1, typename T2>
void cubSumReduceByKey(T1* d_keys_in,
                       T1* d_unique_out,
                       T2* d_vals_in,
                       T2* d_aggregates_out,
                       size_t* d_num_out,
                       size_t n,
                       cudaStream_t& this_stream,
                       DEMSolverScratchData& scratchPad);

template <typename T1>
void cubMaxReduce(T1* d_in, T1* d_out, size_t n, cudaStream_t& this_stream, DEMSolverScratchData& scratchPad);
template <typename T1, typename T2>
void cubMaxReduceByKey(T1* d_keys_in,
                       T1* d_unique_out,
                       T2* d_vals_in,
                       T2* d_aggregates_out,
                       size_t* d_num_out,
                       size_t n,
                       cudaStream_t& this_stream,
                       DEMSolverScratchData& scratchPad);

template <typename T1, typename T2>
void cubMaxNegativeReduceByKey(T1* d_keys_in,
                               T1* d_unique_out,
                               T2* d_vals_in,
                               T2* d_aggregates_out,
                               size_t* d_num_out,
                               size_t n,
                               cudaStream_t& this_stream,
                               DEMSolverScratchData& scratchPad);

template <typename T1>
void cubMinReduce(T1* d_in, T1* d_out, size_t n, cudaStream_t& this_stream, DEMSolverScratchData& scratchPad);
template <typename T1, typename T2>
void cubMinReduceByKey(T1* d_keys_in,
                       T1* d_unique_out,
                       T2* d_vals_in,
                       T2* d_aggregates_out,
                       size_t* d_num_out,
                       size_t n,
                       cudaStream_t& this_stream,
                       DEMSolverScratchData& scratchPad);

template <typename T1, typename T2>
void cubSortByKey(T1* d_keys_in,
                  T1* d_keys_out,
                  T2* d_vals_in,
                  T2* d_vals_out,
                  size_t n,
                  cudaStream_t& this_stream,
                  DEMSolverScratchData& scratchPad);

template <typename T1, typename T2>
void cubRunLengthEncode(T1* d_in,
                        T1* d_unique_out,
                        T2* d_counts_out,
                        size_t* d_num_out,
                        size_t n,
                        cudaStream_t& this_stream,
                        DEMSolverScratchData& scratchPad);

template <typename T1, typename T2>
void cubPrefixScan(T1* d_in, T2* d_out, size_t n, cudaStream_t& this_stream, DEMSolverScratchData& scratchPad);
template <typename T1, typename T2>
inline void cubDEMInclusiveScan(T1* d_in,
                                T2* d_out,
                                size_t n,
                                cudaStream_t& this_stream,
                                DEMSolverScratchData& scratchPad);

////////////////////////////////////////////////////////////////////////////////
// For kT and dT's private usage
////////////////////////////////////////////////////////////////////////////////

void contactDetection(std::shared_ptr<JitHelper::CachedProgram>& bin_sphere_kernels,
                      std::shared_ptr<JitHelper::CachedProgram>& bin_triangle_kernels,
                      std::shared_ptr<JitHelper::CachedProgram>& sphere_contact_kernels,
                      std::shared_ptr<JitHelper::CachedProgram>& sphTri_contact_kernels,
                      DualStruct<DEMDataKT>& granData,
                      DualStruct<DEMSimParams>& simParams,
                      SolverFlags& solverFlags,
                      verbosity_t& verbosity,
                      // The following arrays may need to change sizes, so we can't pass pointers
                      DualArray<bodyID_t>& idPrimitiveA,
                      DualArray<bodyID_t>& idPrimitiveB,
                      DualArray<contact_t>& contactTypePrimitive,
                      DualArray<bodyID_t>& previous_idPrimitiveA,
                      DualArray<bodyID_t>& previous_idPrimitiveB,
                      DualArray<contact_t>& previous_contactTypePrimitive,
                      DualArray<bodyID_t>& previous_primitivePatchIsland,
                      DualArray<notStupidBool_t>& contactPersistency,
                      DualArray<contactPairs_t>& contactMapping,
                      // NEW: Separate patch ID arrays and mapping
                      DualArray<bodyID_t>& idPatchA,
                      DualArray<bodyID_t>& idPatchB,
                      DualArray<bodyID_t>& previous_idPatchA,
                      DualArray<bodyID_t>& previous_idPatchB,
                      DualArray<contact_t>& contactTypePatch,
                      DualArray<contact_t>& previous_contactTypePatch,
                      DualArray<bodyID_t>& contactPatchIsland,
                      DualArray<bodyID_t>& previous_contactPatchIsland,
                      ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>>& typeStartCountPatchMap,
                      DualArray<contactPairs_t>& geomToPatchMap,
                      cudaStream_t& this_stream,
                      DEMSolverScratchData& scratchPad,
                      SolverTimers& timers,
                      kTStateParams& stateParams);

void collectContactForcesThruCub(std::shared_ptr<JitHelper::CachedProgram>& collect_force_kernels,
                                 DualStruct<DEMDataDT>& granData,
                                 const size_t nContactPairs,
                                 const size_t nClumps,
                                 bool contactPairArr_isFresh,
                                 cudaStream_t& this_stream,
                                 DEMSolverScratchData& scratchPad,
                                 SolverTimers& timers);

void overwritePrevContactArrays(DualStruct<DEMDataKT>& kT_data,
                                DualStruct<DEMDataDT>& dT_data,
                                DualArray<bodyID_t>& previous_idPatchA,
                                DualArray<bodyID_t>& previous_idPatchB,
                                DualArray<contact_t>& previous_contactTypePatch,
                                DualArray<bodyID_t>& previous_contactPatchIsland,
                                ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>>& typeStartCountPatchMap,
                                DualStruct<DEMSimParams>& simParams,
                                DEMSolverScratchData& scratchPad,
                                cudaStream_t& this_stream,
                                size_t nContacts);

void getContactForcesConcerningOwners(float3* d_points,
                                      float3* d_forces,
                                      float3* d_torques,
                                      size_t* d_numUsefulCnt,
                                      bodyID_t* d_ownerIDs,
                                      size_t IDListSize,
                                      DEMSimParams* simParams,
                                      DEMDataDT* granData,
                                      size_t numCnt,
                                      bool need_torque,
                                      bool torque_in_local,
                                      cudaStream_t& this_stream);

////////////////////////////////////////////////////////////////////////////////
// Patch-based voting wrappers for mesh contact correction
////////////////////////////////////////////////////////////////////////////////

// Prepares weighted normals (normal * area) for voting.
// Keys are sourced directly from granData->geomToPatchMap + startOffsetPrimitive by the caller,
// avoiding a temporary key buffer and a separate kernel write for keys/areas.
void prepareWeightedNormalsForVoting(DEMDataDT* granData,
                                     float3* weightedNormals,
                                     contactPairs_t startOffset,
                                     contactPairs_t count,
                                     cudaStream_t& this_stream);

// Normalizes voted normals by total area and scatters to output
// If total area is 0, output is (0,0,0) indicating no contact
void normalizeAndScatterVotedNormals(float3* votedWeightedNormals,
                                     float3* output,
                                     contactPairs_t count,
                                     cudaStream_t& this_stream);

// Computes four per-primitive weighted quantities in a single fused kernel pass:
//   projectedAreas[i] = area_i * dot(normal_i, votedNormal)   (clamped to >= 0)
//   projectedPens[i]  = pen_i  * dot(normal_i, votedNormal)   (clamped to >= 0)
//   weights[i]        = projectedArea_i * projectedPen_i       (= projArea * projPen)
//   weightedCPs[i]    = contactPoint_i * weights[i]
// projectedAreas/weights/weightedCPs are sum-reduced per patch; projectedPens are max-reduced per patch.
void computePerPrimitiveWeightedQuantities(DEMDataDT* granData,
                                           const float3* votedNormals,
                                           const contactPairs_t* keys,
                                           double* projectedAreas,
                                           double* projectedPens,
                                           double* weights,
                                           double3* weightedCPs,
                                           contactPairs_t startOffsetPrimitive,
                                           contactPairs_t startOffsetPatch,
                                           contactPairs_t count,
                                           cudaStream_t& this_stream);

// Extracts primitive penetrations from contactPointGeometryA for max-reduce operation
void extractPrimitivePenetrations(DEMDataDT* granData,
                                  double* penetrations,
                                  contactPairs_t startOffset,
                                  contactPairs_t count,
                                  cudaStream_t& this_stream);

// Finds the primitive with max penetration for zero-area patches and extracts its normal, penetration, and contact
// point
void findMaxPenetrationPrimitiveForZeroAreaPatches(DEMDataDT* granData,
                                                   double* maxPenetrations,
                                                   float3* zeroAreaNormals,
                                                   double* zeroAreaPenetrations,
                                                   double3* zeroAreaContactPoints,
                                                   contactPairs_t* keys,
                                                   contactPairs_t startOffsetPrimitive,
                                                   contactPairs_t startOffsetPatch,
                                                   contactPairs_t countPrimitive,
                                                   cudaStream_t& this_stream);

// Finalizes patch results: finalPen = max projected penetration (from cubMaxReduceByKey);
// finalCP = weight-averaged contact point (weight = projArea * projPen).
// Zero-area patches fall back to the max-penetration primitive's values.
void finalizePatchResults(double* totalProjectedAreas,
                          double* maxProjPens,
                          double* totalWeights,
                          float3* votedNormals,
                          double3* totalWeightedCPs,
                          float3* zeroAreaNormals,
                          double* zeroAreaPenetrations,
                          double3* zeroAreaContactPoints,
                          double* finalAreas,
                          float3* finalNormals,
                          double* finalPenetrations,
                          double3* finalContactPoints,
                          contactPairs_t count,
                          cudaStream_t& this_stream);

// Compute per-patch normal-force magnitude and tangential slip speed used by per-triangle diagnostics.
void computePatchPVScalars(DEMSimParams* simParams,
                           DEMDataDT* granData,
                           const float3* finalNormals,
                           const double3* finalContactPoints,
                           contactPairs_t startOffsetPatch,
                           contactPairs_t countPatch,
                           float* patchNormalForce,
                           float* patchSlipSpeed,
                           cudaStream_t& this_stream);

// Redistribute patch-level normal force to participating triangles using primitive contact weight share, and
// accumulate per-triangle P and P*V contributions over the current output window.
void accumulateTrianglePVFromPatchContacts(DEMSimParams* simParams,
                                           DEMDataDT* granData,
                                           const contactPairs_t* keys,
                                           const double* primitiveWeights,
                                           const double* patchWeights,
                                           const float* patchNormalForce,
                                           const float* patchSlipSpeed,
                                           contactPairs_t startOffsetPrimitive,
                                           contactPairs_t startOffsetPatch,
                                           contactPairs_t countPrimitive,
                                           const int* triGlobalToLocal,
                                           float* triAccumP,
                                           float* triAccumPV,
                                           cudaStream_t& this_stream);

////////////////////////////////////////////////////////////////////////////////
// Prep force kernels declaration
////////////////////////////////////////////////////////////////////////////////

void prepareForceArrays(DEMSimParams* simParams,
                        DEMDataDT* granData,
                        size_t nPrimitiveContactPairs,
                        cudaStream_t& this_stream);
void prepareAccArrays(DEMSimParams* simParams, DEMDataDT* granData, bodyID_t nOwnerBodies, cudaStream_t& this_stream);
void rearrangeContactWildcards(DEMDataDT* granData,
                               float* wildcard,
                               notStupidBool_t* sentry,
                               unsigned int nWildcards,
                               size_t nContactPairs,
                               cudaStream_t& this_stream);
void markAliveContacts(float* wildcard, notStupidBool_t* sentry, size_t nContactPairs, cudaStream_t& this_stream);

////////////////////////////////////////////////////////////////////////////////
// Misc kernels declarations
////////////////////////////////////////////////////////////////////////////////

void markOwnerToChange(notStupidBool_t* idBool,
                       float* ownerFactors,
                       bodyID_t* dIDs,
                       float* dFactors,
                       size_t n,
                       cudaStream_t& this_stream);

template <typename DEMData>
void modifyComponents(DEMData* granData, notStupidBool_t* idBool, float* factors, size_t n, cudaStream_t& this_stream);

void fillMarginValues(DEMSimParams* simParams,
                      DEMDataKT* granData,
                      size_t nSphere,
                      size_t nTri,
                      size_t nAnal,
                      cudaStream_t& this_stream);

}  // namespace deme

#endif
