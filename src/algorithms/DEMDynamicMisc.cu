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
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
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
                                                     contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t myContactID = startOffset + idx;

        // Get the contact normal from contactForces
        float3 normal = granData->contactForces[myContactID];

        // Extract the area (double) from contactPointGeometryB (stored as float3)
        float3 areaStorage = granData->contactPointGeometryB[myContactID];
        double area = float3StorageToDouble(areaStorage);

        // Compute weighted normal (normal * area)
        weightedNormals[idx] = make_float3(normal.x * area, normal.y * area, normal.z * area);

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
                                     cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        prepareWeightedNormalsForVoting_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, weightedNormals, areas, keys, startOffset, count);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
}

// Kernel to normalize the voted normals by dividing by total area and scatter to output
// If total area is 0, set result to (0,0,0)
// Assumes uniqueKeys are sorted (CUB's ReduceByKey maintains sort order)
// Uses contactPairs_t keys (geomToPatchMap values)
__global__ void normalizeAndScatterVotedNormals_impl(float3* votedWeightedNormals,
                                                     double* totalAreas,
                                                     float3* output,
                                                     contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float3 votedNormal = make_float3(0, 0, 0);
        double totalArea = totalAreas[idx];
        if (totalArea > 0.0) {
            // Normalize by dividing by total area (use reciprocal multiplication for efficiency)
            double invTotalArea = 1.0 / totalArea;
            votedNormal.x = votedWeightedNormals[idx].x * invTotalArea;
            votedNormal.y = votedWeightedNormals[idx].y * invTotalArea;
            votedNormal.z = votedWeightedNormals[idx].z * invTotalArea;
        }
        // else: votedNormal remains (0,0,0)

        // Write to output at the correct position
        output[idx] = votedNormal;
    }
}

void normalizeAndScatterVotedNormals(float3* votedWeightedNormals,
                                     double* totalAreas,
                                     float3* output,
                                     contactPairs_t count,
                                     cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        normalizeAndScatterVotedNormals_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            votedWeightedNormals, totalAreas, output, count);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
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
                                                      float3* votedNormalizedNormals,
                                                      contactPairs_t* keys,
                                                      double* weightedPenetrations,
                                                      contactPairs_t startOffsetPrimitive,
                                                      contactPairs_t startOffsetPatch,
                                                      contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t myContactID = startOffsetPrimitive + idx;

        // Get the patch pair index for this primitive (absolute index)
        contactPairs_t patchIdx = keys[idx];

        // Get the voted normalized normal for this patch pair
        // Subtract startOffsetPatch to get the local index into votedNormalizedNormals
        contactPairs_t localPatchIdx = patchIdx - startOffsetPatch;
        float3 votedNormal = votedNormalizedNormals[localPatchIdx];
        // If voted normal is (0,0,0), meaning all primitive contacts agree on no contact, then the end result must be
        // 0, no special handling needed

        // Get the original contact normal (stored in contactForces during primitive force calc)
        float3 originalNormal = granData->contactForces[myContactID];

        // Get the original penetration depth from contactPointGeometryA (stored as double in float3)
        float3 penetrationStorage = granData->contactPointGeometryA[myContactID];
        double originalPenetration = float3StorageToDouble(penetrationStorage);
        // Negative penetration does not participate in useful penetration
        if (originalPenetration <= 0.0) {
            originalPenetration = 0.0;
        }

        // Get the contact area from contactPointGeometryB (stored as double in float3)
        float3 areaStorage = granData->contactPointGeometryB[myContactID];
        double area = float3StorageToDouble(areaStorage);

        // Compute the "useful" penetration by projecting onto the voted normal
        // This is: originalPenetration * dot(originalNormal, votedNormal)
        // If dot product is negative (opposite directions), useful penetration becomes negative
        // which we clamp to 0
        float dotProduct = dot(originalNormal, votedNormal);
        double usefulPenetration = originalPenetration * (double)dotProduct;
        if (usefulPenetration <= 0.0) {
            usefulPenetration = 0.0;
        }

        // Weight the useful penetration by area
        double weightedPenetration = usefulPenetration * area;
        weightedPenetrations[idx] = weightedPenetration;

        // printf(
        //     "voted normal: (%f, %f, %f), original normal: (%f, %f, %f), original pen: %f, dot: %f, useful pen: %f, "
        //     "area: %f, weighted pen: %f\n",
        //     votedNormal.x, votedNormal.y, votedNormal.z, originalNormal.x, originalNormal.y, originalNormal.z,
        //     originalPenetration, dotProduct, usefulPenetration, area, weightedPenetration);
    }
}

void computeWeightedUsefulPenetration(DEMDataDT* granData,
                                      float3* votedNormalizedNormals,
                                      contactPairs_t* keys,
                                      double* weightedPenetrations,
                                      contactPairs_t startOffsetPrimitive,
                                      contactPairs_t startOffsetPatch,
                                      contactPairs_t count,
                                      cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        computeWeightedUsefulPenetration_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, votedNormalizedNormals, keys, weightedPenetrations, startOffsetPrimitive, startOffsetPatch,
            count);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
}

// Kernel to compute total penetration per patch pair by dividing by total area
// If total area is 0, total penetration is 0
__global__ void computeTotalPenetrationPerPatch_impl(double* totalWeightedPenetrations,
                                                     double* totalAreas,
                                                     double* totalPenetrations,
                                                     contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        double totalArea = totalAreas[idx];
        if (totalArea > 0.0) {
            totalPenetrations[idx] = totalWeightedPenetrations[idx] / totalArea;
        } else {
            totalPenetrations[idx] = 0.0;
        }
    }
}

void computeTotalPenetrationPerPatch(double* totalWeightedPenetrations,
                                     double* totalAreas,
                                     double* totalPenetrations,
                                     contactPairs_t count,
                                     cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        computeTotalPenetrationPerPatch_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            totalWeightedPenetrations, totalAreas, totalPenetrations, count);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
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
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
}

// Kernel to handle zero-area patches by finding the primitive with max penetration
// and using its penetration and normal for the patch result.
// For each primitive, check if it has the max penetration for its patch.
// Note: Race condition when multiple primitives have the same max penetration is acceptable
// since any of them produces a valid result.
__global__ void findMaxPenetrationPrimitiveForZeroAreaPatches_impl(DEMDataDT* granData,
                                                                   double* totalAreas,
                                                                   double* maxPenetrations,
                                                                   float3* zeroAreaNormals,
                                                                   double* zeroAreaPenetrations,
                                                                   contactPairs_t* keys,
                                                                   contactPairs_t startOffsetPrimitive,
                                                                   contactPairs_t startOffsetPatch,
                                                                   contactPairs_t countPrimitive) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < countPrimitive) {
        contactPairs_t myContactID = startOffsetPrimitive + idx;
        contactPairs_t patchIdx = keys[idx];
        contactPairs_t localPatchIdx = patchIdx - startOffsetPatch;

        // Only process if this patch has zero total area
        double totalArea = totalAreas[localPatchIdx];
        if (totalArea > 0.0) {
            return;  // Normal voting logic applies, skip
        }

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
            // This primitive has the max penetration - use its normal
            // Note: if multiple primitives have the same max, any one of them is fine
            // The race condition is acceptable since all competing values are valid
            float3 myNormal = granData->contactForces[myContactID];
            zeroAreaNormals[localPatchIdx] = myNormal;
            zeroAreaPenetrations[localPatchIdx] = myPenetration;
        }
    }
}

void findMaxPenetrationPrimitiveForZeroAreaPatches(DEMDataDT* granData,
                                                   double* totalAreas,
                                                   double* maxPenetrations,
                                                   float3* zeroAreaNormals,
                                                   double* zeroAreaPenetrations,
                                                   contactPairs_t* keys,
                                                   contactPairs_t startOffsetPrimitive,
                                                   contactPairs_t startOffsetPatch,
                                                   contactPairs_t countPrimitive,
                                                   cudaStream_t& this_stream) {
    size_t blocks_needed = (countPrimitive + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        findMaxPenetrationPrimitiveForZeroAreaPatches_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0,
                                                             this_stream>>>(
            granData, totalAreas, maxPenetrations, zeroAreaNormals, zeroAreaPenetrations, keys, startOffsetPrimitive,
            startOffsetPatch, countPrimitive);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
}

// Kernel to check if any primitive in each patch satisfies SAT (for tri-tri contacts)
// Uses atomic CAS to safely set patchHasSAT[patchIdx] = 1 if any primitive has contactSATSatisfied = 1
__global__ void checkPatchHasSATSatisfyingPrimitive_impl(DEMDataDT* granData,
                                                         notStupidBool_t* patchHasSAT,
                                                         contactPairs_t* keys,
                                                         contactPairs_t startOffsetPrimitive,
                                                         contactPairs_t startOffsetPatch,
                                                         contactPairs_t countPrimitive) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < countPrimitive) {
        contactPairs_t myContactID = startOffsetPrimitive + idx;
        contactPairs_t patchIdx = keys[idx];
        contactPairs_t localPatchIdx = patchIdx - startOffsetPatch;
        
        // Check if this primitive satisfies SAT
        notStupidBool_t satisfiesSAT = granData->contactSATSatisfied[myContactID];
        
        // If this primitive satisfies SAT, mark the patch as having at least one SAT-satisfying primitive
        // Since we only need to set 0 -> 1, a simple write is safe (multiple threads writing 1 is idempotent)
        if (satisfiesSAT) {
            patchHasSAT[localPatchIdx] = 1;
        }
    }
}

void checkPatchHasSATSatisfyingPrimitive(DEMDataDT* granData,
                                        notStupidBool_t* patchHasSAT,
                                        contactPairs_t* keys,
                                        contactPairs_t startOffsetPrimitive,
                                        contactPairs_t startOffsetPatch,
                                        contactPairs_t countPrimitive,
                                        contactPairs_t countPatch,
                                        cudaStream_t& this_stream) {
    // Initialize patchHasSAT to 0
    DEME_GPU_CALL(cudaMemsetAsync(patchHasSAT, 0, countPatch * sizeof(notStupidBool_t), this_stream));
    
    size_t blocks_needed = (countPrimitive + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        checkPatchHasSATSatisfyingPrimitive_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, patchHasSAT, keys, startOffsetPrimitive, startOffsetPatch, countPrimitive);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
}

// Kernel to finalize patch results by combining normal voting results with zero-area case handling
// For patches with totalArea > 0 AND patchHasSAT = 1: use voted normal and weighted penetration
// For patches with totalArea == 0 OR patchHasSAT = 0: use max-penetration primitive's normal and penetration (Step 8 fallback)
__global__ void finalizePatchResults_impl(double* totalAreas,
                                          float3* votedNormals,
                                          double* votedPenetrations,
                                          float3* zeroAreaNormals,
                                          double* zeroAreaPenetrations,
                                          notStupidBool_t* patchHasSAT,
                                          float3* finalNormals,
                                          double* finalPenetrations,
                                          contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        double totalArea = totalAreas[idx];
        notStupidBool_t hasSAT = (patchHasSAT != nullptr) ? patchHasSAT[idx] : 1;
        
        // Use voted results only if totalArea > 0 AND at least one primitive satisfies SAT
        if (totalArea > 0.0 && hasSAT) {
            // Normal case: use voted results
            finalNormals[idx] = votedNormals[idx];
            finalPenetrations[idx] = votedPenetrations[idx];
        } else {
            // Zero-area case OR no SAT-satisfying primitives: use max-penetration primitive's results (Step 8 fallback)
            finalNormals[idx] = zeroAreaNormals[idx];
            finalPenetrations[idx] = zeroAreaPenetrations[idx];
        }
    }
}

void finalizePatchResults(double* totalAreas,
                          float3* votedNormals,
                          double* votedPenetrations,
                          float3* zeroAreaNormals,
                          double* zeroAreaPenetrations,
                          notStupidBool_t* patchHasSAT,
                          float3* finalNormals,
                          double* finalPenetrations,
                          contactPairs_t count,
                          cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        finalizePatchResults_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            totalAreas, votedNormals, votedPenetrations, zeroAreaNormals, zeroAreaPenetrations, patchHasSAT,
            finalNormals, finalPenetrations, count);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
}

// Kernel to compute weighted contact points for each primitive contact
// The weight is: penetration * area
// This prepares data for reduction to get patch-based contact points
__global__ void computeWeightedContactPoints_impl(DEMDataDT* granData,
                                                  double3* weightedContactPoints,
                                                  double* weights,
                                                  contactPairs_t startOffsetPrimitive,
                                                  contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t myContactID = startOffsetPrimitive + idx;

        // Get the contact point from contactTorque_convToForce (stored as float3)
        double3 contactPoint = to_double3(granData->contactTorque_convToForce[myContactID]);

        // Get the penetration depth from contactPointGeometryA (stored as double in float3)
        float3 penetrationStorage = granData->contactPointGeometryA[myContactID];
        double penetration = float3StorageToDouble(penetrationStorage);
        // Only positive penetration contributes
        if (penetration < 0.0) {
            penetration = 0.0;
        }

        // Get the contact area from contactPointGeometryB (stored as double in float3)
        float3 areaStorage = granData->contactPointGeometryB[myContactID];
        double area = float3StorageToDouble(areaStorage);

        // Compute weight = penetration * area
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
                                  contactPairs_t startOffsetPrimitive,
                                  contactPairs_t count,
                                  cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        computeWeightedContactPoints_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, weightedContactPoints, weights, startOffsetPrimitive, count);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
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
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
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
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
}

void prepareAccArrays(DEMSimParams* simParams, DEMDataDT* granData, bodyID_t nOwnerBodies, cudaStream_t& this_stream) {
    size_t blocks_needed_for_acc_prep = (nOwnerBodies + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed_for_acc_prep > 0) {
        prepareAccArrays_impl<<<blocks_needed_for_acc_prep, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(simParams,
                                                                                                          granData);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
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
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
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
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
}

}  // namespace deme
