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
            // Patch and primitive contact arrays do not share the same index space. This kernel is intentionally
            // patch-contact-only; primitive contact queries need a separate launch.
            return;
        }
        contact_t typeContact = granData->contactTypePatch[i];
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
            // This is the contact-model "extra torque" represented as a force-like vector, not r x contact force.
            // Force-generated torque can be reconstructed from the reported point and force by the caller.
            applyOriQToVector3(torque_only_force, make_float4(-oriQ.x, -oriQ.y, -oriQ.z, oriQ.w));
            float3 torque = cross(cntPnt, torque_only_force);
            if (!torque_in_local) {
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
    DEME_GPU_DEBUG_SYNC(this_stream);
}

////////////////////////////////////////////////////////////////////////////////
// Patch-based voting kernels for mesh contact correction
////////////////////////////////////////////////////////////////////////////////

// Optimized kernel: prepares weighted normals only (normal * area).
// Keys are sourced directly from granData->geomToPatchMap by the caller; no key/area output arrays needed.
__global__ void prepareWeightedNormalsForVoting_impl(DEMDataDT* granData,
                                                     float3* weightedNormals,
                                                     contactPairs_t startOffset,
                                                     contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t myContactID = startOffset + idx;

        const float3 normal = granData->contactForces[myContactID];
        const float3 areaStorage = granData->contactPointGeometryB[myContactID];
        const double area = float3StorageToDouble(areaStorage);

        // Compute weighted normal (normal * area).
        // Fake contacts do not contribute since their area is 0.
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
        DEME_GPU_DEBUG_SYNC(this_stream);
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
        DEME_GPU_DEBUG_SYNC(this_stream);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Per-primitive weighted quantity computation (fused kernel)
////////////////////////////////////////////////////////////////////////////////

// Fused kernel that computes four per-primitive quantities in a single pass:
//   projectedAreas[i]  = area_i * dot(normal_i, votedNormal)   (clamped to >= 0)
//   projectedPens[i]   = pen_i  * dot(normal_i, votedNormal)   (clamped to >= 0)
//   weights[i]         = projectedArea_i * projectedPen_i       (w = projArea * projPen)
//   weightedCPs[i]     = contactPoint_i * weights[i]
// These arrays are later reduced per-patch via CUB:
//   projectedAreas and weights and weightedCPs via cubSumReduceByKey,
//   projectedPens via cubMaxReduceByKey (to get per-patch max projected penetration).
__global__ void computePerPrimitiveWeightedQuantities_impl(DEMDataDT* granData,
                                                           const float3* votedNormals,
                                                           const contactPairs_t* keys,
                                                           double* projectedAreas,
                                                           double* projectedPens,
                                                           double* weights,
                                                           double3* weightedCPs,
                                                           contactPairs_t startOffsetPrimitive,
                                                           contactPairs_t startOffsetPatch,
                                                           contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t myContactID = startOffsetPrimitive + idx;

        // Read area, penetration, normal, and contact point stored by the primitive kernel
        double area = float3StorageToDouble(granData->contactPointGeometryB[myContactID]);
        double pen = float3StorageToDouble(granData->contactPointGeometryA[myContactID]);
        if (pen <= 0.0) {
            pen = 0.0;
        }
        float3 normal = granData->contactForces[myContactID];
        double3 cp = to_double3(granData->contactTorque_convToForce[myContactID]);

        // Project onto patch-voted normal
        contactPairs_t localPatchIdx = keys[idx] - startOffsetPatch;
        float3 vn = votedNormals[localPatchIdx];
        float dp = dot(normal, vn);

        double projArea = area * (double)dp;
        double projPen = pen * (double)dp;
        if (projArea <= 0.0) {
            projArea = 0.0;
        }
        if (projPen <= 0.0) {
            projPen = 0.0;
        }

        double w = projArea * projPen;

        projectedAreas[idx] = projArea;
        projectedPens[idx] = projPen;
        weights[idx] = w;
        weightedCPs[idx] = cp * w;

        // printf(
        //     "voted normal: (%f, %f, %f), original normal: (%f, %f, %f), original pen: %f, dot: %f, "
        //     "projected pen: %f, area: %f, projected area: %f\n", vn.x, vn.y, vn.z, normal.x,
        //     normal.y, normal.z, pen, dp, projPen, area, projArea);
    }
}

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
                                           cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        computePerPrimitiveWeightedQuantities_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, votedNormals, keys, projectedAreas, projectedPens, weights, weightedCPs, startOffsetPrimitive,
            startOffsetPatch, count);
        DEME_GPU_DEBUG_SYNC(this_stream);
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
        DEME_GPU_DEBUG_SYNC(this_stream);
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
        DEME_GPU_DEBUG_SYNC(this_stream);
    }
}

// Kernel to finalize patch results by combining normal voting results with zero-area case handling.
// finalPen is the max projected penetration among all primitives in the patch (from cubMaxReduceByKey).
// The contact point is still the weight-averaged contact point (weight = projArea * projPen).
__global__ void finalizePatchResults_impl(double* totalProjectedAreas,
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
                                          contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        double projArea = totalProjectedAreas[idx];

        if (projArea > 0.0) {
            // Normal case: use voted results.
            // Penetration is the max projected penetration among all primitives in this patch.
            finalAreas[idx] = projArea;
            finalNormals[idx] = votedNormals[idx];
            finalPenetrations[idx] = maxProjPens[idx];
            double totalWeight = totalWeights[idx];
            if (totalWeight > 0.0) {
                finalContactPoints[idx] = totalWeightedCPs[idx] * (1.0 / totalWeight);
            } else {
                // No positive-weight primitive contributed; set CP to zero (pen = 0 means no force anyway).
                finalContactPoints[idx] = make_double3(0.0, 0.0, 0.0);
            }
        } else {
            // Zero-area case: use max-penetration primitive's results (zero-area fallback).
            finalAreas[idx] = 0.0;
            finalNormals[idx] = zeroAreaNormals[idx];
            finalPenetrations[idx] = zeroAreaPenetrations[idx];
            finalContactPoints[idx] = zeroAreaContactPoints[idx];
        }
    }
}

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
                          cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        finalizePatchResults_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            totalProjectedAreas, maxProjPens, totalWeights, votedNormals, totalWeightedCPs, zeroAreaNormals,
            zeroAreaPenetrations, zeroAreaContactPoints, finalAreas, finalNormals, finalPenetrations,
            finalContactPoints, count);
        DEME_GPU_DEBUG_SYNC(this_stream);
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
        DEME_GPU_DEBUG_SYNC(this_stream);
    }
}

__global__ void accumulateTrianglePVFromPatchContacts_impl(const DEMSimParams* simParams,
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

    const double patchWeight = patchWeights[localPatchIdx];
    const double primitiveWeight = primitiveWeights[idx];
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
                                           cudaStream_t& this_stream) {
    size_t blocks_needed = (countPrimitive + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        accumulateTrianglePVFromPatchContacts_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            simParams, granData, keys, primitiveWeights, patchWeights, patchNormalForce, patchSlipSpeed,
            startOffsetPrimitive, startOffsetPatch, countPrimitive, triGlobalToLocal, triAccumP, triAccumPV);
        DEME_GPU_DEBUG_SYNC(this_stream);
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
        DEME_GPU_DEBUG_SYNC(this_stream);
    }
}

void prepareAccArrays(DEMSimParams* simParams, DEMDataDT* granData, bodyID_t nOwnerBodies, cudaStream_t& this_stream) {
    size_t blocks_needed_for_acc_prep = (nOwnerBodies + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed_for_acc_prep > 0) {
        prepareAccArrays_impl<<<blocks_needed_for_acc_prep, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(simParams,
                                                                                                          granData);
        DEME_GPU_DEBUG_SYNC(this_stream);
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
        DEME_GPU_DEBUG_SYNC(this_stream);
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
        DEME_GPU_DEBUG_SYNC(this_stream);
    }
}

}  // namespace deme
