//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <algorithms/DEMStaticDeviceSubroutines.h>
#include <algorithms/DEMStaticDeviceUtilities.cuh>

#include <kernel/DEMHelperKernels.cuh>

namespace deme {

__global__ void getContactForcesConcerningOwners_impl(float3* d_points,
                                                      float3* d_forces,
                                                      float3* d_torques,
                                                      unsigned long long* d_numUsefulCnt,
                                                      bodyID_t* d_ownerIDs,
                                                      size_t IDListSize,
                                                      DEMSimParams* simParams,
                                                      DEMDataDT* granData,
                                                      size_t numCnt,
                                                      bool need_torque,
                                                      bool torque_in_local) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numCnt) {
        contact_t typeContact = granData->contactTypePatch[i];
        bodyID_t geoA = granData->idPatchA[i];
        bodyID_t ownerA = DEME_GET_PATCH_OWNER_ID(geoA, decodeTypeA(typeContact));
        bodyID_t geoB = granData->idPatchB[i];
        bodyID_t ownerB = DEME_GET_PATCH_OWNER_ID(geoB, decodeTypeB(typeContact));
        bool AorB;  // true for A, false for B
        if (cuda_binary_search<bodyID_t, ssize_t>(d_ownerIDs, ownerA, 0, IDListSize - 1)) {
            AorB = true;
        } else if (cuda_binary_search<bodyID_t, ssize_t>(d_ownerIDs, ownerB, 0, IDListSize - 1)) {
            AorB = false;
        } else {
            return;
        }

        float3 force, torque;
        force = granData->contactForces[i];
        // Note torque, like force, is in global
        if (need_torque)
            torque = granData->contactTorque_convToForce[i];
        {
            float mag = (need_torque) ? length(force) + length(torque) : length(force);
            if (mag < DEME_TINY_FLOAT)
                return;
        }

        // It's a contact we need to output...
        unsigned long long writeIndex = atomicAdd(d_numUsefulCnt, 1);
        float3 cntPnt;
        double3 CoM;
        float4 oriQ;
        bodyID_t ownerID;
        if (AorB) {
            cntPnt = granData->contactPointGeometryA[i];
            ownerID = ownerA;
        } else {
            cntPnt = granData->contactPointGeometryB[i];
            ownerID = ownerB;
            // Force dir flipped
            force = -force;
            if (need_torque)
                torque = -torque;
        }
        oriQ.w = granData->oriQw[ownerID];
        oriQ.x = granData->oriQx[ownerID];
        oriQ.y = granData->oriQy[ownerID];
        oriQ.z = granData->oriQz[ownerID];
        // Must derive torque in local...
        if (need_torque) {
            applyOriQToVector3<float, oriQ_t>(torque.x, torque.y, torque.z, oriQ.w, -oriQ.x, -oriQ.y, -oriQ.z);
            // Force times point...
            torque = cross(cntPnt, torque);
            if (!torque_in_local) {  // back to global if needed
                applyOriQToVector3<float, oriQ_t>(torque.x, torque.y, torque.z, oriQ.w, oriQ.x, oriQ.y, oriQ.z);
            }
        }

        voxelID_t voxel = granData->voxelID[ownerID];
        subVoxelPos_t subVoxX = granData->locX[ownerID];
        subVoxelPos_t subVoxY = granData->locY[ownerID];
        subVoxelPos_t subVoxZ = granData->locZ[ownerID];
        voxelIDToPosition<double, voxelID_t, subVoxelPos_t>(CoM.x, CoM.y, CoM.z, voxel, subVoxX, subVoxY, subVoxZ,
                                                            simParams->nvXp2, simParams->nvYp2, simParams->voxelSize,
                                                            simParams->l);
        CoM.x += simParams->LBFX;
        CoM.y += simParams->LBFY;
        CoM.z += simParams->LBFZ;
        applyFrameTransformLocalToGlobal<float3, double3, float4>(cntPnt, CoM, oriQ);
        d_points[writeIndex] = cntPnt;
        d_forces[writeIndex] = force;
        if (need_torque)
            d_torques[writeIndex] = torque;
    }
}

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
                                      cudaStream_t& this_stream) {
    size_t blocks_needed = (numCnt + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    getContactForcesConcerningOwners_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
        d_points, d_forces, d_torques, reinterpret_cast<unsigned long long*>(d_numUsefulCnt), d_ownerIDs, IDListSize,
        simParams, granData, numCnt, need_torque, torque_in_local);
}

////////////////////////////////////////////////////////////////////////////////
// Patch-based voting kernels for mesh contact correction
////////////////////////////////////////////////////////////////////////////////

// Kernel to compute weighted normals (normal * area / penetration) for voting
// Also prepares the area values for reduction and extracts the keys (geomToPatchMap values)

// Optimized overload: prepare weighted normals only (no temporary areas/keys arrays).
__global__ void prepareWeightedNormalsForVoting_impl(DEMDataDT* granData,
                                                          float3* weightedNormals,
                                                          contactPairs_t startOffset,
                                                          contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t myContactID = startOffset + idx;

        // Normal and geometric quantities were produced by the primitive contact kernels.
        const float3 normal = granData->contactForces[myContactID];
        const float3 areaStorage = granData->contactPointGeometryB[myContactID];
        float area = float3StorageToDouble(areaStorage);
        // But primitive contacts that do not respect the patch general direction have no right in deciding the contact
        // normal
        notStupidBool_t directionRespected = granData->contactPatchDirectionRespected[myContactID];
        if (!directionRespected) {
            area = 0.0;
        }

        weightedNormals[idx] = make_float3(normal.x * area, normal.y * area, normal.z * area);
    }
}

void prepareWeightedNormalsForVoting(DEMDataDT* granData,
                                     float3* weightedNormals,
                                     contactPairs_t startOffset,
                                     contactPairs_t count,
                                     cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        prepareWeightedNormalsForVoting_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, weightedNormals, startOffset, count);
    }
}

// Kernel to normalize the voted normals by dividing by total area and scatter to output
// If total area is 0, set result to (0,0,0)
// Assumes uniqueKeys are sorted (CUB's ReduceByKey maintains sort order)
// Uses contactPairs_t keys (geomToPatchMap values)
__global__ void normalizeAndScatterVotedNormals_impl(float3* votedWeightedNormals,
                                                     float3* output,
                                                     contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float3 votedNormal = votedWeightedNormals[idx];
        float len2 = length2(votedNormal);
        if (len2 > 0.f) {
            // Normalize votedNormal
            votedNormal *= rsqrtf(len2);
        } else {
            // If total area is 0, set to (0,0,0) to mark no real contact
            votedNormal = make_float3(0.0f, 0.0f, 0.0f);
        }

        // Write to output at the correct position
        output[idx] = votedNormal;
    }
}

void normalizeAndScatterVotedNormals(float3* votedWeightedNormals,
                                     float3* output,
                                     contactPairs_t count,
                                     cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        normalizeAndScatterVotedNormals_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            votedWeightedNormals, output, count);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Fused patch aggregation kernels (projected area, penetration, contact point)
////////////////////////////////////////////////////////////////////////////////

// Per-primitive accumulator generation.
//
// This replaces the former pipeline:
//   computeWeightedUsefulPenetration -> ReduceByKey(sum projArea)
//   ReduceByKey(max projPen)
//   computeWeightedContactPoints -> ReduceByKey(sum weightedCP) -> ReduceByKey(sum weight)
//
// It produces the same patch-level quantities, but materializes only one array
// (PatchContactAccum) and performs a single ReduceByKey.
__global__ void computePatchContactAccumulators_impl(DEMDataDT* granData,
                                                     const float3* votedNormals,
                                                     const contactPairs_t* keys,
                                                     PatchContactAccum* accumulators,
                                                     contactPairs_t startOffsetPrimitive,
                                                     contactPairs_t startOffsetPatch,
                                                     contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const contactPairs_t myContactID = startOffsetPrimitive + idx;

        // Map this primitive to its patch-pair index, then to local [0, countPatch) index.
        const contactPairs_t patchIdx = keys[idx];
        const contactPairs_t localPatchIdx = patchIdx - startOffsetPatch;

        const float3 votedNormal = votedNormals[localPatchIdx];
        const float3 originalNormal = granData->contactForces[myContactID];

        // Penetration depth (positive means overlap/contact); negative is non-contact and does not contribute.
        const float3 penStorage = granData->contactPointGeometryA[myContactID];
        double originalPenetration = float3StorageToDouble(penStorage);
        originalPenetration = (originalPenetration > 0.0) ? originalPenetration : 0.0;

        // Contact area (non-negative; fake contacts have 0 area and thus contribute 0).
        const float3 areaStorage = granData->contactPointGeometryB[myContactID];
        const double area = float3StorageToDouble(areaStorage);

        // Projection factor: clamp negative dot products to 0 (tangential/opposing contributions do not participate).
        const float dotProduct = dot(originalNormal, votedNormal);
        const double cospos = (dotProduct > 0.f) ? (double)dotProduct : 0.0;

        const double projectedPenetration = originalPenetration * cospos;
        const double projectedArea = area * cospos;

        const double weight = projectedPenetration * projectedArea;

        const double3 contactPoint = to_double3(granData->contactTorque_convToForce[myContactID]);
        const double3 weightedCP = make_double3(contactPoint.x * weight, contactPoint.y * weight, contactPoint.z * weight);

        PatchContactAccum acc;
        acc.sumProjArea = projectedArea;
        acc.maxProjPen = projectedPenetration;
        acc.sumWeight = weight;
        acc.sumWeightedCP = weightedCP;
        accumulators[idx] = acc;
    }
}

void computePatchContactAccumulators(DEMDataDT* granData,
                                     const float3* votedNormals,
                                     const contactPairs_t* keys,
                                     PatchContactAccum* accumulators,
                                     contactPairs_t startOffsetPrimitive,
                                     contactPairs_t startOffsetPatch,
                                     contactPairs_t count,
                                     cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        computePatchContactAccumulators_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, votedNormals, keys, accumulators, startOffsetPrimitive, startOffsetPatch, count);
    }
}

// Finalization from patch accumulators (no intermediate per-patch arrays).
__global__ void finalizePatchResultsFromAccumulators_impl(const PatchContactAccum* patchAccumulators,
                                                          const float3* votedNormals,
                                                          const float3* zeroAreaNormals,
                                                          const double* zeroAreaPenetrations,
                                                          const double3* zeroAreaContactPoints,
                                                          double* finalAreas,
                                                          float3* finalNormals,
                                                          double* finalPenetrations,
                                                          double3* finalContactPoints,
                                                          contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const PatchContactAccum acc = patchAccumulators[idx];
        const double projectedArea = acc.sumProjArea;

        // Use voted results only if projectedArea > 0
        if (projectedArea > 0.0) {
            finalAreas[idx] = projectedArea;
            finalNormals[idx] = votedNormals[idx];
            finalPenetrations[idx] = acc.maxProjPen;

            if (acc.sumWeight > 0.0) {
                const double invW = 1.0 / acc.sumWeight;
                finalContactPoints[idx] = make_double3(acc.sumWeightedCP.x * invW,
                                                      acc.sumWeightedCP.y * invW,
                                                      acc.sumWeightedCP.z * invW);
            } else {
                // If total weight is 0, contact point is set to (0,0,0)
                finalContactPoints[idx] = make_double3(0.0, 0.0, 0.0);
            }
        } else {
            // Zero-area case: fallback to max-penetration primitive's results
            finalAreas[idx] = 0.0;
            finalNormals[idx] = zeroAreaNormals[idx];
            finalPenetrations[idx] = zeroAreaPenetrations[idx];
            finalContactPoints[idx] = zeroAreaContactPoints[idx];
        }
    }
}

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
                                          cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        finalizePatchResultsFromAccumulators_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            patchAccumulators, votedNormals, zeroAreaNormals, zeroAreaPenetrations, zeroAreaContactPoints,
            finalAreas, finalNormals, finalPenetrations, finalContactPoints, count);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Penetration depth computation kernels for mesh contact correction
////////////////////////////////////////////////////////////////////////////////

// Kernel to compute weighted useful penetration for each primitive contact
// The "useful" penetration is the original penetration projected onto the voted normal.
// If the projection makes penetration negative (tangential contact), it's clamped to 0.
// Each primitive's useful penetration is then weighted by its contact area.
__global__ void computeWeightedUsefulPenetration_impl(DEMDataDT* granData,
                                                      float3* votedNormals,
                                                      contactPairs_t* keys,
                                                      double* areas,
                                                      double* projectedPenetrations,
                                                      double* projectedAreas,
                                                      contactPairs_t startOffsetPrimitive,
                                                      contactPairs_t startOffsetPatch,
                                                      contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t myContactID = startOffsetPrimitive + idx;

        // Get the patch pair index for this primitive (absolute index)
        contactPairs_t patchIdx = keys[idx];

        // Get the voted normalized normal for this patch pair
        // Subtract startOffsetPatch to get the local index into votedNormals
        contactPairs_t localPatchIdx = patchIdx - startOffsetPatch;
        float3 votedNormal = votedNormals[localPatchIdx];
        // If voted normal is (0,0,0), meaning all primitive contacts agree on no contact, then the end result must be
        // 0, no special handling needed

        // Get the original contact normal (stored in contactForces during primitive force calc)
        float3 originalNormal = granData->contactForces[myContactID];

        // Get the original penetration depth from contactPointGeometryA (stored as double in float3)
        float3 penetrationStorage = granData->contactPointGeometryA[myContactID];
        double originalPenetration = float3StorageToDouble(penetrationStorage);
        // Negative penetration does not participate
        if (originalPenetration <= 0.0) {
            originalPenetration = 0.0;
        }

        // Get the contact area from storage that is not yet freed. Note the index is idx not myContactID, as areas is a
        // type-specific vector.
        double area = areas[idx];

        // Compute the projected penetration and area by projecting onto the voted normal
        // Projected penetration: originalPenetration * dot(originalNormal, votedNormal)
        // Projected area: area * dot(originalNormal, votedNormal)
        // If dot product is negative (opposite directions), set both to 0
        float dotProduct = dot(originalNormal, votedNormal);
        double projectedPenetration = originalPenetration * (double)dotProduct;
        double projectedArea = area * (double)dotProduct;

        // If projected values becomes negative, set both area and penetration to 0
        if (projectedPenetration <= 0.0) {
            projectedPenetration = 0.0;
        }
        if (projectedArea <= 0.0) {
            projectedArea = 0.0;
        }

        projectedPenetrations[idx] = projectedPenetration;
        projectedAreas[idx] = projectedArea;

        // printf(
        //     "voted normal: (%f, %f, %f), original normal: (%f, %f, %f), original pen: %f, dot: %f, projected pen: %f,
        //     " "area: %f, projected area: %f\n", votedNormal.x, votedNormal.y, votedNormal.z, originalNormal.x,
        //     originalNormal.y, originalNormal.z, originalPenetration, dotProduct, projectedPenetration, area,
        //     projectedArea);
    }
}

void computeWeightedUsefulPenetration(DEMDataDT* granData,
                                      float3* votedNormals,
                                      contactPairs_t* keys,
                                      double* areas,
                                      double* projectedPenetrations,
                                      double* projectedAreas,
                                      contactPairs_t startOffsetPrimitive,
                                      contactPairs_t startOffsetPatch,
                                      contactPairs_t count,
                                      cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        computeWeightedUsefulPenetration_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, votedNormals, keys, areas, projectedPenetrations, projectedAreas, startOffsetPrimitive,
            startOffsetPatch, count);

    }
}

////////////////////////////////////////////////////////////////////////////////
// Special case handling: zero-area patches (no positive-penetration primitives)
////////////////////////////////////////////////////////////////////////////////

// Kernel to extract primitive penetrations for max-reduce operation
// For zero-area case handling, we need the max (biggest/least-negative) penetration per patch
__global__ void extractPrimitivePenetrations_impl(DEMDataDT* granData,
                                                  double* penetrations,
                                                  contactPairs_t startOffset,
                                                  contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t myContactID = startOffset + idx;

        // Extract penetration from contactPointGeometryA (stored as double in float3)
        float3 penetrationStorage = granData->contactPointGeometryA[myContactID];
        penetrations[idx] = float3StorageToDouble(penetrationStorage);
    }
}

void extractPrimitivePenetrations(DEMDataDT* granData,
                                  double* penetrations,
                                  contactPairs_t startOffset,
                                  contactPairs_t count,
                                  cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        extractPrimitivePenetrations_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, penetrations, startOffset, count);
    }
}

// Kernel to handle zero-area patches by finding the primitive with max penetration
// and using its penetration, normal, and contact point for the patch result.
// For each primitive, check if it has the max penetration for its patch.
// Note: Race condition when multiple primitives have the same max penetration is acceptable
// since any of them produces a valid result.
__global__ void findMaxPenetrationPrimitiveForZeroAreaPatches_impl(DEMDataDT* granData,
                                                                   double* maxPenetrations,
                                                                   float3* zeroAreaNormals,
                                                                   double* zeroAreaPenetrations,
                                                                   double3* zeroAreaContactPoints,
                                                                   contactPairs_t* keys,
                                                                   contactPairs_t startOffsetPrimitive,
                                                                   contactPairs_t startOffsetPatch,
                                                                   contactPairs_t countPrimitive) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < countPrimitive) {
        contactPairs_t myContactID = startOffsetPrimitive + idx;
        contactPairs_t patchIdx = keys[idx];
        contactPairs_t localPatchIdx = patchIdx - startOffsetPatch;

        // In fact, we just need to proceed if area is zero. But these no-contact cases are so
        // common, that we don't do an early termination here.

        // Get this primitive's penetration
        float3 penetrationStorage = granData->contactPointGeometryA[myContactID];
        double myPenetration = float3StorageToDouble(penetrationStorage);

        // Check if this primitive has the max penetration for its patch
        // Use a relative tolerance for floating-point comparison
        double maxPen = maxPenetrations[localPatchIdx];
        double absTol = 1e-15;  // Absolute tolerance for very small values
        double relTol = 1e-12;  // Relative tolerance for larger values
        double tolerance = fmax(absTol, fabs(maxPen) * relTol);
        if (fabs(myPenetration - maxPen) <= tolerance) {
            // This primitive has the max penetration - use its normal, penetration, and contact point
            // Note: if multiple primitives have the same max, any one of them is fine
            // The race condition is acceptable since all competing values are valid
            float3 myNormal = granData->contactForces[myContactID];
            zeroAreaNormals[localPatchIdx] = myNormal;
            zeroAreaPenetrations[localPatchIdx] = myPenetration < 0.0 ? myPenetration : -DEME_HUGE_FLOAT;
            // This zeroAreaPenetrations should store a negative number, as when it is needed, it's usually the
            // separation case (all zero-area primitives). But for the no-SAT case, which can resemble cross-particle
            // erroneous detection, we could have a positive max here (search for CubOpMaxNegative to understand how
            // this max is derived). In that case, we give it a very negative number, so in the patch-based force
            // calculation, this one is considered a non-contact.

            // Also store the contact point from this max-penetration primitive
            double3 myContactPoint = to_double3(granData->contactTorque_convToForce[myContactID]);
            zeroAreaContactPoints[localPatchIdx] = myContactPoint;
        }
    }
}

void findMaxPenetrationPrimitiveForZeroAreaPatches(DEMDataDT* granData,
                                                   double* maxPenetrations,
                                                   float3* zeroAreaNormals,
                                                   double* zeroAreaPenetrations,
                                                   double3* zeroAreaContactPoints,
                                                   contactPairs_t* keys,
                                                   contactPairs_t startOffsetPrimitive,
                                                   contactPairs_t startOffsetPatch,
                                                   contactPairs_t countPrimitive,
                                                   cudaStream_t& this_stream) {
    size_t blocks_needed = (countPrimitive + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        findMaxPenetrationPrimitiveForZeroAreaPatches_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0,
                                                             this_stream>>>(
            granData, maxPenetrations, zeroAreaNormals, zeroAreaPenetrations, zeroAreaContactPoints, keys,
            startOffsetPrimitive, startOffsetPatch, countPrimitive);
    }
}

// Kernel to finalize patch results by combining normal voting results with zero-area case handling
__global__ void finalizePatchResults_impl(double* totalProjectedAreas,
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
                                          contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        double projectedArea = totalProjectedAreas[idx];

        // Use voted results only if projectedArea > 0
        if (projectedArea > 0.0) {
            // Normal case: use voted results
            finalAreas[idx] = projectedArea;
            finalNormals[idx] = votedNormals[idx];
            finalPenetrations[idx] = votedPenetrations[idx];
            finalContactPoints[idx] = votedContactPoints[idx];
        } else {
            // Zero-area case: use max-penetration primitive's results (Step 8 fallback)
            // Set finalArea to 0 for these cases
            finalAreas[idx] = 0.0;
            finalNormals[idx] = zeroAreaNormals[idx];
            finalPenetrations[idx] = zeroAreaPenetrations[idx];
            finalContactPoints[idx] = zeroAreaContactPoints[idx];
        }
    }
}

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
                          cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        finalizePatchResults_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            totalProjectedAreas, votedNormals, votedPenetrations, votedContactPoints, zeroAreaNormals,
            zeroAreaPenetrations, zeroAreaContactPoints, finalAreas, finalNormals, finalPenetrations,
            finalContactPoints, count);
    }
}

// Kernel to compute weighted contact points for each primitive contact
// The weight is: projected_penetration * projected_area
// This prepares data for reduction to get patch-based contact points
__global__ void computeWeightedContactPoints_impl(DEMDataDT* granData,
                                                  double3* weightedContactPoints,
                                                  double* weights,
                                                  double* projectedPenetrations,
                                                  double* projectedAreas,
                                                  contactPairs_t startOffsetPrimitive,
                                                  contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t myContactID = startOffsetPrimitive + idx;

        // Get the contact point from contactTorque_convToForce (stored as float3)
        double3 contactPoint = to_double3(granData->contactTorque_convToForce[myContactID]);

        // Get the projected penetration and area
        double penetration = projectedPenetrations[idx];
        double area = projectedAreas[idx];

        // Compute weight = projected_penetration * projected_area
        double weight = penetration * area;

        // Compute weighted contact point (multiply each component by weight)
        weightedContactPoints[idx] = contactPoint * weight;

        // Store weight for later normalization
        weights[idx] = weight;
    }
}

void computeWeightedContactPoints(DEMDataDT* granData,
                                  double3* weightedContactPoints,
                                  double* weights,
                                  double* projectedPenetrations,
                                  double* projectedAreas,
                                  contactPairs_t startOffsetPrimitive,
                                  contactPairs_t count,
                                  cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        computeWeightedContactPoints_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, weightedContactPoints, weights, projectedPenetrations, projectedAreas, startOffsetPrimitive,
            count);
    }
}

// Kernel to compute final contact points per patch by dividing by total weight
// If total weight is 0, contact point is set to (0,0,0)
__global__ void computeFinalContactPointsPerPatch_impl(double3* totalWeightedContactPoints,
                                                       double* totalWeights,
                                                       double3* finalContactPoints,
                                                       contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        double totalWeight = totalWeights[idx];
        if (totalWeight > 0.0) {
            // Normalize by dividing by total weight
            double invTotalWeight = (1.0 / totalWeight);
            finalContactPoints[idx] = totalWeightedContactPoints[idx] * invTotalWeight;
        } else {
            // No valid contact point, set to (0,0,0)
            finalContactPoints[idx] = make_double3(0, 0, 0);
        }
    }
}

void computeFinalContactPointsPerPatch(double3* totalWeightedContactPoints,
                                       double* totalWeights,
                                       double3* finalContactPoints,
                                       contactPairs_t count,
                                       cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        computeFinalContactPointsPerPatch_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            totalWeightedContactPoints, totalWeights, finalContactPoints, count);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Prep force kernels
////////////////////////////////////////////////////////////////////////////////

inline __device__ void cleanUpContactForces(size_t thisContact, DEMSimParams* simParams, DEMDataDT* granData) {
    const float3 zeros = make_float3(0, 0, 0);
    granData->contactForces[thisContact] = zeros;
    granData->contactTorque_convToForce[thisContact] = zeros;
}

inline __device__ void cleanUpAcc(size_t thisClump, DEMSimParams* simParams, DEMDataDT* granData) {
    // If should not clear acc arrays, then just mark it to be clear in the next ts
    if (granData->accSpecified[thisClump]) {
        granData->accSpecified[thisClump] = 0;
    } else {
        granData->aX[thisClump] = 0;
        granData->aY[thisClump] = 0;
        granData->aZ[thisClump] = 0;
    }
    if (granData->angAccSpecified[thisClump]) {
        granData->angAccSpecified[thisClump] = 0;
    } else {
        granData->alphaX[thisClump] = 0;
        granData->alphaY[thisClump] = 0;
        granData->alphaZ[thisClump] = 0;
    }
}

__global__ void prepareAccArrays_impl(DEMSimParams* simParams, DEMDataDT* granData) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < simParams->nOwnerBodies) {
        cleanUpAcc(myID, simParams, granData);
    }
}

__global__ void prepareForceArrays_impl(DEMSimParams* simParams, DEMDataDT* granData, size_t nContactPairs) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        cleanUpContactForces(myID, simParams, granData);
    }
}

void prepareForceArrays(DEMSimParams* simParams,
                        DEMDataDT* granData,
                        size_t nPrimitiveContactPairs,
                        cudaStream_t& this_stream) {
    size_t blocks_needed_for_force_prep =
        (nPrimitiveContactPairs + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed_for_force_prep > 0) {
        prepareForceArrays_impl<<<blocks_needed_for_force_prep, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            simParams, granData, nPrimitiveContactPairs);
    }
}

void prepareAccArrays(DEMSimParams* simParams, DEMDataDT* granData, bodyID_t nOwnerBodies, cudaStream_t& this_stream) {
    size_t blocks_needed_for_acc_prep = (nOwnerBodies + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed_for_acc_prep > 0) {
        prepareAccArrays_impl<<<blocks_needed_for_acc_prep, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(simParams,
                                                                                                          granData);
    }
}

__global__ void rearrangeContactWildcards_impl(DEMDataDT* granData,
                                               float* newWildcards,
                                               notStupidBool_t* sentry,
                                               unsigned int nWildcards,
                                               size_t nContactPairs) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        contactPairs_t map_from = granData->contactMapping[myID];
        if (map_from == NULL_MAPPING_PARTNER) {
            // If it is a NULL ID then kT says this contact is new. Initialize all wildcard arrays.
            for (size_t i = 0; i < nWildcards; i++) {
                newWildcards[nContactPairs * i + myID] = 0;
            }
        } else {
            // Not a new contact, need to map it from somewhere in the old history array
            for (size_t i = 0; i < nWildcards; i++) {
                newWildcards[nContactPairs * i + myID] = granData->contactWildcards[i][map_from];
            }
            // This sentry trys to make sure that all `alive' contacts got mapped to some place
            sentry[map_from] = 0;
        }
    }
}

void rearrangeContactWildcards(DEMDataDT* granData,
                               float* wildcard,
                               notStupidBool_t* sentry,
                               unsigned int nWildcards,
                               size_t nContactPairs,
                               cudaStream_t& this_stream) {
    size_t blocks_needed = (nContactPairs + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        rearrangeContactWildcards_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, wildcard, sentry, nWildcards, nContactPairs);
    }
}

__global__ void markAliveContacts_impl(float* wildcard, notStupidBool_t* sentry, size_t nContactPairs) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        float myEntry = abs(wildcard[myID]);
        // If this is alive then mark it
        if (myEntry > DEME_TINY_FLOAT) {
            sentry[myID] = 1;
        } else {
            sentry[myID] = 0;
        }
    }
}

void markAliveContacts(float* wildcard, notStupidBool_t* sentry, size_t nContactPairs, cudaStream_t& this_stream) {
    size_t blocks_needed = (nContactPairs + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        markAliveContacts_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(wildcard, sentry,
                                                                                              nContactPairs);
    }
}

}  // namespace deme
