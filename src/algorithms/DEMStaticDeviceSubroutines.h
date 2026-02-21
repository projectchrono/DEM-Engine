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

// Prepares weighted normals (normal * area / penetration) for voting.
//
// The weighted normal magnitude represents the voting power. The subsequent normalization step only
// needs the *direction*, therefore any positive scalar multiple of the weight yields the same
// voted direction.  The current implementation follows the existing, validated semantics.
void prepareWeightedNormalsForVoting(DEMDataDT* granData,
                                     float3* weightedNormals,
                                     double* areas,
                                     contactPairs_t* keys,
                                     contactPairs_t startOffset,
                                     contactPairs_t count,
                                     contact_t contactType,
                                     cudaStream_t& this_stream);

// Optimized overload: prepares weighted normals only.
//
// This avoids materializing temporary areas/keys buffers. Keys can be sourced directly from
// granData->geomToPatchMap + startOffsetPrimitive in the caller.
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

// Patch-level accumulator used to fuse multiple ReduceByKey passes.
//
// The reduction operator is component-wise associative (sum + max), therefore it can safely be used
// with CUB ReduceByKey.
struct PatchContactAccum {
    double sumProjArea;     ///< Sum of projected contact areas (per patch)
    double maxProjPen;      ///< Max projected penetration (per patch)
    double sumWeight;       ///< Sum of weights w = projectedPenetration * projectedArea (per patch)
    double3 sumWeightedCP;  ///< Sum of (contactPoint * w) (per patch)

    __host__ __device__ __forceinline__ PatchContactAccum operator+(const PatchContactAccum& other) const {
        PatchContactAccum out;
        out.sumProjArea = sumProjArea + other.sumProjArea;
        out.maxProjPen = (maxProjPen > other.maxProjPen) ? maxProjPen : other.maxProjPen;
        out.sumWeight = sumWeight + other.sumWeight;
        out.sumWeightedCP =
            make_double3(sumWeightedCP.x + other.sumWeightedCP.x, sumWeightedCP.y + other.sumWeightedCP.y,
                         sumWeightedCP.z + other.sumWeightedCP.z);
        return out;
    }
};

// Computes per-primitive patch accumulators:
//   - sumProjArea: projected area contribution
//   - maxProjPen:  projected penetration contribution (to be reduced by max)
//   - sumWeight:   weight contribution (for contact point averaging)
//   - sumWeightedCP: weighted contact point contribution
void computePatchContactAccumulators(DEMDataDT* granData,
                                     const float3* votedNormals,
                                     const contactPairs_t* keys,
                                     PatchContactAccum* accumulators,
                                     contactPairs_t startOffsetPrimitive,
                                     contactPairs_t startOffsetPatch,
                                     contactPairs_t count,
                                     cudaStream_t& this_stream);

// Finalizes patch results by combining patch-accumulator voting with zero-area / SAT-fail fallback.
//
// Semantics match finalizePatchResults(), but avoids materializing intermediate arrays
// (totalProjectedAreas, votedPenetrations, votedContactPoints).
void finalizePatchResultsFromAccumulators(const PatchContactAccum* patchAccumulators,
                                          const float3* votedNormals,
                                          const float3* zeroAreaNormals,
                                          const double* zeroAreaPenetrations,
                                          const double3* zeroAreaContactPoints,
                                          double* finalAreas,
                                          float3* finalNormals,
                                          double* finalPenetrations,
                                          double3* finalContactPoints,
                                          contactPairs_t count,
                                          cudaStream_t& this_stream);

// Computes projected penetration and area for each primitive contact
// Both the penetration and area are projected onto the voted normal
// If the projected penetration becomes negative, both are set to 0
void computeWeightedUsefulPenetration(DEMDataDT* granData,
                                      float3* votedNormals,
                                      contactPairs_t* keys,
                                      double* areas,
                                      double* projectedPenetrations,
                                      double* projectedAreas,
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

// Finalizes patch results by combining normal voting with zero-area case handling
void finalizePatchResults(double* totalProjectedAreas,
                          float3* votedNormals,
                          double* votedPenetrations,
                          double3* votedContactPoints,
                          float3* zeroAreaNormals,
                          double* zeroAreaPenetrations,
                          double3* zeroAreaContactPoints,
                          double* finalAreas,
                          float3* finalNormals,
                          double* finalPenetrations,
                          double3* finalContactPoints,
                          contactPairs_t count,
                          cudaStream_t& this_stream);

// Finalizes patch contact points by combining voting with zero-area case handling
void finalizePatchContactPoints(double* totalAreas,
                                double3* votedContactPoints,
                                double3* zeroAreaContactPoints,
                                double3* finalContactPoints,
                                contactPairs_t count,
                                cudaStream_t& this_stream);

// Computes weighted contact points for each primitive contact
// The weight is: projected_penetration * projected_area
void computeWeightedContactPoints(DEMDataDT* granData,
                                  double3* weightedContactPoints,
                                  double* weights,
                                  double* projectedPenetrations,
                                  double* projectedAreas,
                                  contactPairs_t startOffsetPrimitive,
                                  contactPairs_t count,
                                  cudaStream_t& this_stream);

// Computes final contact points per patch by dividing by total weight
// If total weight is 0, contact point is set to (0,0,0)
void computeFinalContactPointsPerPatch(double3* totalWeightedContactPoints,
                                       double* totalWeights,
                                       double3* finalContactPoints,
                                       contactPairs_t count,
                                       cudaStream_t& this_stream);

// Compute per-patch normal-force magnitude and tangential slip speed used by per-triangle diagnostics.
void computePatchPVScalars(DEMSimParams* simParams,
                           DEMDataDT* granData,
                           const float3* finalNormals,
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
                                           const PatchContactAccum* primitiveAccumulators,
                                           const PatchContactAccum* patchAccumulators,
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
