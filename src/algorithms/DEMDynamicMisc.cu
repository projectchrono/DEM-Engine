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
        const bool patch_space = (granData->contactTypePatch && granData->idPatchA && granData->idPatchB);
        if (!patch_space) {
            // IMPORTANT: Patch and primitive contact arrays do not share the same index space.
            // This kernel is intended for PATCH contacts only. Handling primitive contacts must use a separate kernel
            // launch.
            return;
        }
        const contact_t typeContact = granData->contactTypePatch[i];
        if (typeContact == NOT_A_CONTACT) {
            return;
        }
        bodyID_t geoA = granData->idPatchA[i];
        bodyID_t geoB = granData->idPatchB[i];
        bodyID_t ownerA = DEME_GET_PATCH_OWNER_ID(geoA, decodeTypeA(typeContact));
        bodyID_t ownerB = DEME_GET_PATCH_OWNER_ID(geoB, decodeTypeB(typeContact));
        bool AorB;  // true for A, false for B
        if (cuda_binary_search<bodyID_t, ssize_t>(d_ownerIDs, ownerA, 0, IDListSize - 1)) {
            AorB = true;
        } else if (cuda_binary_search<bodyID_t, ssize_t>(d_ownerIDs, ownerB, 0, IDListSize - 1)) {
            AorB = false;
        } else {
            return;
        }

        float3 force = granData->contactForces[i];
        float3 torque_only_force = make_float3(0.f, 0.f, 0.f);
        if (need_torque) {
            torque_only_force = granData->contactTorque_convToForce[i];
        }
        {
            float mag = length(force) + length(torque_only_force);
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
                torque_only_force = -torque_only_force;
        }
        oriQ.w = granData->oriQw[ownerID];
        oriQ.x = granData->oriQx[ownerID];
        oriQ.y = granData->oriQy[ownerID];
        oriQ.z = granData->oriQz[ownerID];
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
        if (need_torque) {
            // This is `extra torque', not including force-generated torque. The user computes the latter by themselves
            // using the contact point and force output. The reason we separate this `extra torque' from the
            // force-generated torque is that some contact models (e.g. tangential spring) may generate torque without
            // generating any net force.
            applyOriQToVector3(torque_only_force, make_float4(-oriQ.x, -oriQ.y, -oriQ.z, oriQ.w));
            float3 torque = cross(cntPnt, torque_only_force);
            if (!torque_in_local) {
                // Back to global
                applyOriQToVector3(torque, oriQ);
            }
            d_torques[writeIndex] = torque;
        }

        applyFrameTransformLocalToGlobal<float3, double3, float4>(cntPnt, CoM, oriQ);
        d_points[writeIndex] = cntPnt;
        d_forces[writeIndex] = force;
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

// Kernel to compute weighted normals (normal * area) for voting
// Also prepares the area values for reduction and extracts the keys (geomToPatchMap values)
__global__ void prepareWeightedNormalsForVoting_impl(DEMDataDT* granData,
                                                     float3* weightedNormals,
                                                     double* areas,
                                                     contactPairs_t* keys,
                                                     contactPairs_t startOffset,
                                                     contactPairs_t count,
                                                     contact_t contactType) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t myContactID = startOffset + idx;

        // Get the contact normal from contactForces
        float3 normal = granData->contactForces[myContactID];

        // Extract the area (double) from contactPointGeometryB (stored as float3)
        float3 areaStorage = granData->contactPointGeometryB[myContactID];
        double area = float3StorageToDouble(areaStorage);

        // Compute weighted normal (normal * area)
        // Note that fake contacts do not affect as their area is 0
        weightedNormals[idx] = make_float3((double)normal.x * area, (double)normal.y * area, (double)normal.z * area);

        // Store area for reduction
        areas[idx] = area;

        // Extract key from geomToPatchMap
        keys[idx] = granData->geomToPatchMap[myContactID];
    }
}

void prepareWeightedNormalsForVoting(DEMDataDT* granData,
                                     float3* weightedNormals,
                                     double* areas,
                                     contactPairs_t* keys,
                                     contactPairs_t startOffset,
                                     contactPairs_t count,
                                     contact_t contactType,
                                     cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        prepareWeightedNormalsForVoting_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, weightedNormals, areas, keys, startOffset, count, contactType);
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

        // In fact, we just need to proceed if area is zero or the SAT check failed. But these no-contact cases are so
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
// PatchContactAccum-based single-pass reduction kernels
////////////////////////////////////////////////////////////////////////////////

// Compute one PatchContactAccum per primitive contact.
// This fuses computeWeightedUsefulPenetration and computeWeightedContactPoints into one kernel pass,
// allowing the caller to use a single cubSumReduceByKey<PatchContactAccum> instead of four separate
// ReduceByKey passes.
__global__ void computePatchContactAccumulators_impl(DEMDataDT* granData,
                                                     const float3* votedNormals,
                                                     const contactPairs_t* keys,
                                                     PatchContactAccum* accumulators,
                                                     contactPairs_t startOffsetPrimitive,
                                                     contactPairs_t startOffsetPatch,
                                                     contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t myContactID = startOffsetPrimitive + idx;

        // Read area, penetration, original normal, and contact point from granData
        double area = float3StorageToDouble(granData->contactPointGeometryB[myContactID]);
        double originalPenetration = float3StorageToDouble(granData->contactPointGeometryA[myContactID]);
        if (originalPenetration <= 0.0) {
            originalPenetration = 0.0;
        }
        float3 originalNormal = granData->contactForces[myContactID];
        double3 contactPoint = to_double3(granData->contactTorque_convToForce[myContactID]);

        // Project onto the patch-voted normal
        contactPairs_t localPatchIdx = keys[idx] - startOffsetPatch;
        float3 votedNormal = votedNormals[localPatchIdx];
        float dotProduct = dot(originalNormal, votedNormal);

        double projectedPenetration = originalPenetration * (double)dotProduct;
        double projectedArea = area * (double)dotProduct;
        if (projectedPenetration <= 0.0) {
            projectedPenetration = 0.0;
        }
        if (projectedArea <= 0.0) {
            projectedArea = 0.0;
        }

        double weight = projectedPenetration * projectedArea;

        PatchContactAccum accum;
        accum.sumProjArea = projectedArea;
        accum.maxProjPen = projectedPenetration;
        accum.sumWeight = weight;
        accum.sumWeightedCP = contactPoint * weight;
        accum.sumAreaWeightedCP = contactPoint * projectedArea;
        accumulators[idx] = accum;
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

// Finalize per-patch results from PatchContactAccum reduced values.
// Mirrors finalizePatchResults but reads all patch-level quantities from a single PatchContactAccum array,
// avoiding the need to materialize totalProjectedAreas, votedPenetrations, and votedContactPoints arrays.
// When sumProjArea > 0 but sumWeight ~ 0 (near-zero projected penetrations), falls back to the
// area-weighted contact point (sumAreaWeightedCP / sumProjArea) instead of returning (0,0,0).
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
        const PatchContactAccum& accum = patchAccumulators[idx];

        if (accum.sumProjArea > 0.0) {
            // Normal case: use voted results
            finalAreas[idx] = accum.sumProjArea;
            finalNormals[idx] = votedNormals[idx];
            finalPenetrations[idx] = accum.maxProjPen;
            if (accum.sumWeight > 0.0) {
                finalContactPoints[idx] = accum.sumWeightedCP * (1.0 / accum.sumWeight);
            } else {
                // Near-zero penetration fallback: use area-weighted contact point
                finalContactPoints[idx] = accum.sumAreaWeightedCP * (1.0 / accum.sumProjArea);
            }
        } else {
            // Zero-area case: use max-penetration primitive's results
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
            patchAccumulators, votedNormals, zeroAreaNormals, zeroAreaPenetrations, zeroAreaContactPoints, finalAreas,
            finalNormals, finalPenetrations, finalContactPoints, count);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Per-triangle P / V / P*V accumulation from patch contacts
////////////////////////////////////////////////////////////////////////////////

__global__ void computePatchPVScalars_impl(const DEMSimParams* simParams,
                                           DEMDataDT* granData,
                                           const float3* finalNormals,
                                           const double3* finalContactPoints,
                                           contactPairs_t startOffsetPatch,
                                           contactPairs_t countPatch,
                                           float* patchNormalForce,
                                           float* patchSlipSpeed) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= countPatch || !patchNormalForce || !patchSlipSpeed || !finalContactPoints) {
        return;
    }

    patchNormalForce[idx] = 0.f;
    patchSlipSpeed[idx] = 0.f;

    const contactPairs_t patchContactID = startOffsetPatch + idx;
    const contact_t patchType = granData->contactTypePatch[patchContactID];
    if (patchType == NOT_A_CONTACT) {
        return;
    }

    float3 normal = finalNormals[idx];
    const float n2 = dot(normal, normal);
    if (!(n2 > DEME_TINY_FLOAT)) {
        return;
    }
    normal *= rsqrtf(n2);

    const float3 patchForce = granData->contactForces[patchContactID];
    const float normalForce = fabsf(dot(patchForce, normal));
    if (!(normalForce > DEME_TINY_FLOAT)) {
        return;
    }

    const bodyID_t geoA = granData->idPatchA[patchContactID];
    const bodyID_t geoB = granData->idPatchB[patchContactID];
    const bodyID_t ownerA = DEME_GET_PATCH_OWNER_ID(geoA, decodeTypeA(patchType));
    const bodyID_t ownerB = DEME_GET_PATCH_OWNER_ID(geoB, decodeTypeB(patchType));

    float3 velCPA = make_float3(0.f, 0.f, 0.f);
    float3 velCPB = make_float3(0.f, 0.f, 0.f);
    const double3 cp_d = finalContactPoints[idx];
    if (!isfinite(cp_d.x) || !isfinite(cp_d.y) || !isfinite(cp_d.z)) {
        return;
    }
    const float3 cp_global = make_float3((float)cp_d.x, (float)cp_d.y, (float)cp_d.z);

    if (ownerA != NULL_BODYID && ownerA < simParams->nOwnerBodies) {
        float3 linVelA = make_float3(granData->vX[ownerA], granData->vY[ownerA], granData->vZ[ownerA]);
        float3 angVelA_local =
            make_float3(granData->omgBarX[ownerA], granData->omgBarY[ownerA], granData->omgBarZ[ownerA]);
        const float4 oriA = make_float4(granData->oriQx[ownerA], granData->oriQy[ownerA], granData->oriQz[ownerA],
                                        granData->oriQw[ownerA]);
        if (isfinite(linVelA.x) && isfinite(linVelA.y) && isfinite(linVelA.z) && isfinite(angVelA_local.x) &&
            isfinite(angVelA_local.y) && isfinite(angVelA_local.z)) {
            float3 angVelA_global = angVelA_local;
            applyOriQToVector3(angVelA_global, oriA);

            double3 comA;
            voxelIDToPosition<double, voxelID_t, subVoxelPos_t>(
                comA.x, comA.y, comA.z, granData->voxelID[ownerA], granData->locX[ownerA], granData->locY[ownerA],
                granData->locZ[ownerA], simParams->nvXp2, simParams->nvYp2, simParams->voxelSize, simParams->l);
            comA.x += simParams->LBFX;
            comA.y += simParams->LBFY;
            comA.z += simParams->LBFZ;

            float3 rA_global =
                make_float3(cp_global.x - (float)comA.x, cp_global.y - (float)comA.y, cp_global.z - (float)comA.z);
            velCPA = linVelA + cross(angVelA_global, rA_global);
        }
    }

    if (ownerB != NULL_BODYID && ownerB < simParams->nOwnerBodies) {
        float3 linVelB = make_float3(granData->vX[ownerB], granData->vY[ownerB], granData->vZ[ownerB]);
        float3 angVelB_local =
            make_float3(granData->omgBarX[ownerB], granData->omgBarY[ownerB], granData->omgBarZ[ownerB]);
        const float4 oriB = make_float4(granData->oriQx[ownerB], granData->oriQy[ownerB], granData->oriQz[ownerB],
                                        granData->oriQw[ownerB]);
        if (isfinite(linVelB.x) && isfinite(linVelB.y) && isfinite(linVelB.z) && isfinite(angVelB_local.x) &&
            isfinite(angVelB_local.y) && isfinite(angVelB_local.z)) {
            float3 angVelB_global = angVelB_local;
            applyOriQToVector3(angVelB_global, oriB);

            double3 comB;
            voxelIDToPosition<double, voxelID_t, subVoxelPos_t>(
                comB.x, comB.y, comB.z, granData->voxelID[ownerB], granData->locX[ownerB], granData->locY[ownerB],
                granData->locZ[ownerB], simParams->nvXp2, simParams->nvYp2, simParams->voxelSize, simParams->l);
            comB.x += simParams->LBFX;
            comB.y += simParams->LBFY;
            comB.z += simParams->LBFZ;

            float3 rB_global =
                make_float3(cp_global.x - (float)comB.x, cp_global.y - (float)comB.y, cp_global.z - (float)comB.z);
            velCPB = linVelB + cross(angVelB_global, rB_global);
        }
    }

    const float3 relVel = velCPA - velCPB;
    const float vRelN = dot(relVel, normal);
    const float3 vRelT = relVel - vRelN * normal;
    const float slipSpeed = length(vRelT);
    if (!isfinite(slipSpeed) || !(slipSpeed >= 0.f)) {
        return;
    }
    patchNormalForce[idx] = normalForce;
    patchSlipSpeed[idx] = slipSpeed;
}

void computePatchPVScalars(DEMSimParams* simParams,
                           DEMDataDT* granData,
                           const float3* finalNormals,
                           const double3* finalContactPoints,
                           contactPairs_t startOffsetPatch,
                           contactPairs_t countPatch,
                           float* patchNormalForce,
                           float* patchSlipSpeed,
                           cudaStream_t& this_stream) {
    size_t blocks_needed = (countPatch + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        computePatchPVScalars_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            simParams, granData, finalNormals, finalContactPoints, startOffsetPatch, countPatch, patchNormalForce,
            patchSlipSpeed);
    }
}

__global__ void accumulateTrianglePVFromPatchContacts_impl(const DEMSimParams* simParams,
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
                                                           float* triAccumPV) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= countPrimitive || !triGlobalToLocal || !triAccumP || !triAccumPV) {
        return;
    }

    const contactPairs_t primContactID = startOffsetPrimitive + idx;
    const contactPairs_t patchContactID = keys[idx];
    if (patchContactID < startOffsetPatch) {
        return;
    }
    const contactPairs_t localPatchIdx = patchContactID - startOffsetPatch;

    const double patchWeight = patchAccumulators[localPatchIdx].sumWeight;
    const double primitiveWeight = primitiveAccumulators[idx].sumWeight;
    if (patchWeight <= 0.0 || primitiveWeight <= 0.0) {
        return;
    }

    float share = static_cast<float>(primitiveWeight / patchWeight);
    if (!(share > 0.f)) {
        return;
    }
    share = fminf(share, 1.f);

    const float normalForce = patchNormalForce[localPatchIdx];
    if (!(normalForce > DEME_TINY_FLOAT)) {
        return;
    }
    const float pContribution = normalForce * share;
    const float slipSpeed = patchSlipSpeed[localPatchIdx];
    const float pvContribution = pContribution * slipSpeed;

    const contact_t primType = granData->contactTypePrimitive[primContactID];
    if (primType == NOT_A_CONTACT) {
        return;
    }

    const geoType_t typeA = decodeTypeA(primType);
    const geoType_t typeB = decodeTypeB(primType);

    // Track any contact contribution on triangle sides (sphere-triangle and triangle-triangle).
    if (typeA == GEO_T_TRIANGLE) {
        const bodyID_t triA = granData->idPrimitiveA[primContactID];
        if (triA < simParams->nTriGM) {
            const int localIdx = triGlobalToLocal[triA];
            if (localIdx >= 0) {
                atomicAdd(triAccumP + localIdx, pContribution);
                atomicAdd(triAccumPV + localIdx, pvContribution);
            }
        }
    }
    if (typeB == GEO_T_TRIANGLE) {
        const bodyID_t triB = granData->idPrimitiveB[primContactID];
        if (triB < simParams->nTriGM) {
            const int localIdx = triGlobalToLocal[triB];
            if (localIdx >= 0) {
                atomicAdd(triAccumP + localIdx, pContribution);
                atomicAdd(triAccumPV + localIdx, pvContribution);
            }
        }
    }
}

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
                                           cudaStream_t& this_stream) {
    size_t blocks_needed = (countPrimitive + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        accumulateTrianglePVFromPatchContacts_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            simParams, granData, keys, primitiveAccumulators, patchAccumulators, patchNormalForce, patchSlipSpeed,
            startOffsetPrimitive, startOffsetPatch, countPrimitive, triGlobalToLocal, triAccumP, triAccumPV);
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
