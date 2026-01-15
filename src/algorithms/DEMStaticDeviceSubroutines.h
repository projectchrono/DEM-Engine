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

// Prepares weighted normals (normal * area) for voting
void prepareWeightedNormalsForVoting(DEMDataDT* granData,
                                     float3* weightedNormals,
                                     contactPairs_t startOffset,
                                     contactPairs_t count,
                                     cudaStream_t& this_stream);

// Normalize voted normals using unique patch keys and scatter to the local patch array
void normalizeAndScatterVotedNormalsFromUniqueKeys(float3* votedWeightedNormals,
                                                   contactPairs_t* uniqueKeys,
                                                   float3* output,
                                                   contactPairs_t startOffsetPatch,
                                                   contactPairs_t count,
                                                   cudaStream_t& this_stream);

// Fused accumulator carrying area-weighted normals, projected metrics, and max-penetration data.
struct FusedPatchAccum {
    double sumProjArea;      // sum of projected areas (>=0)
    double maxProjPen;       // max projected penetration (>=0)
    double sumWeight;        // sum of projectedPenetration*projectedArea (>=0)
    double3 sumWeightedCP;   // weighted contact point accumulator
    float3 sumWeightedNormal;  // area-weighted normal (normal*area), used for voted normal
    double maxPenRaw;        // raw penetration (can be negative)
    float3 maxPenNormal;     // normal associated with maxPenRaw
    double3 maxPenCP;        // contact point associated with maxPenRaw

    __host__ __device__ __forceinline__ FusedPatchAccum operator+(const FusedPatchAccum& other) const {
        FusedPatchAccum out;
        out.sumProjArea = sumProjArea + other.sumProjArea;
        out.maxProjPen = (maxProjPen > other.maxProjPen) ? maxProjPen : other.maxProjPen;
        out.sumWeight = sumWeight + other.sumWeight;
        out.sumWeightedCP =
            make_double3(sumWeightedCP.x + other.sumWeightedCP.x, sumWeightedCP.y + other.sumWeightedCP.y,
                         sumWeightedCP.z + other.sumWeightedCP.z);
        out.sumWeightedNormal =
            make_float3(sumWeightedNormal.x + other.sumWeightedNormal.x, sumWeightedNormal.y + other.sumWeightedNormal.y,
                        sumWeightedNormal.z + other.sumWeightedNormal.z);

        // Max-negative preference (equivalent to CubOpMaxNegative): prefer negatives closest to zero; otherwise most
        // negative positive (smallest positive)
        double a = maxPenRaw;
        double b = other.maxPenRaw;
        bool pick_other = false;
        if (a < 0 && b < 0) {
            pick_other = (b > a);  // closer to zero negative
        } else if (a < 0) {
            pick_other = false;
        } else if (b < 0) {
            pick_other = true;
        } else {
            pick_other = (b < a);  // both non-negative: pick smaller one
        }
        if (pick_other) {
            out.maxPenRaw = b;
            out.maxPenNormal = other.maxPenNormal;
            out.maxPenCP = other.maxPenCP;
        } else {
            out.maxPenRaw = a;
            out.maxPenNormal = maxPenNormal;
            out.maxPenCP = maxPenCP;
        }
        return out;
    }
};

// Compute fused per-primitive accumulators (projected metrics + max-penetration + area-weighted normal)
void computeFusedPatchContactAccumulators(DEMDataDT* granData,
                                          float3* votedNormals,
                                          const contactPairs_t* keys,
                                          FusedPatchAccum* accumulators,
                                          contactPairs_t startOffsetPrimitive,
                                          contactPairs_t startOffsetPatch,
                                          contactPairs_t count,
                                          cudaStream_t& this_stream);

// Scatter fused accumulators to patch-local arrays expected by finalizePatchResults.
void scatterFusedPatchAccumulators(const FusedPatchAccum* accumulators,
                                   const contactPairs_t* uniqueKeys,
                                   double* totalProjectedAreas,
                                   double* maxProjectedPenetrations,
                                   double3* votedContactPoints,
                                   float3* votedNormals,
                                   float3* zeroAreaNormals,
                                   double* zeroAreaPenetrations,
                                   double3* zeroAreaContactPoints,
                                   contactPairs_t startOffsetPatch,
                                   contactPairs_t count,
                                   cudaStream_t& this_stream);

struct PatchContactAccum {
    double sumProjArea;
    double maxProjPen;
    double sumWeight;
    double3 sumWeightedCP;

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

// Computes projected penetration/area and weighted contact point accumulators per primitive
// Checks if any primitive in each patch satisfies SAT (for tri-tri contacts)
// Outputs a flag per patch: 1 if at least one SAT-satisfying primitive exists, 0 otherwise
void checkPatchHasSATSatisfyingPrimitive(DEMDataDT* granData,
                                         notStupidBool_t* patchHasSAT,
                                         contactPairs_t* keys,
                                         contactPairs_t startOffsetPrimitive,
                                         contactPairs_t startOffsetPatch,
                                         contactPairs_t countPrimitive,
                                         contactPairs_t countPatch,
                                         cudaStream_t& this_stream);

// Finalizes patch results by combining normal voting with zero-area case handling
void finalizePatchResults(double* totalProjectedAreas,
                          float3* votedNormals,
                          double* votedPenetrations,
                          double3* votedContactPoints,
                          float3* zeroAreaNormals,
                          double* zeroAreaPenetrations,
                          double3* zeroAreaContactPoints,
                          notStupidBool_t* patchHasSAT,
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
                                notStupidBool_t* patchHasSAT,
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
