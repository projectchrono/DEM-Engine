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
                                                      bodyID_t* d_contact_owner,
                                                      unsigned long long* d_numUsefulCnt,
                                                      bodyID_t* d_ownerIDs,
                                                      size_t IDListSize,
                                                      DEMSimParams* simParams,
                                                      DEMDataDT* granData,
                                                      size_t numCnt,
                                                      size_t capacity,
                                                      bool need_torque,
                                                      bool torque_in_local) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numCnt) {
        contact_t typeContact = granData->contactTypePatch[i];
        bodyID_t geoA = granData->idPatchA[i];
        bodyID_t ownerA = DEME_GET_PATCH_OWNER_ID(geoA, decodeTypeA(typeContact));
        bodyID_t geoB = granData->idPatchB[i];
        bodyID_t ownerB = DEME_GET_PATCH_OWNER_ID(geoB, decodeTypeB(typeContact));
        const bool A_in = cuda_binary_search<bodyID_t, ssize_t>(d_ownerIDs, ownerA, 0, IDListSize - 1);
        const bool B_in = cuda_binary_search<bodyID_t, ssize_t>(d_ownerIDs, ownerB, 0, IDListSize - 1);
        if (!A_in && !B_in) {
            return;
        }

        const float3 base_force = granData->contactForces[i];
        float3 base_torque;
        if (need_torque)
            base_torque = granData->contactTorque_convToForce[i];
        {
            float mag = (need_torque) ? length(base_force) + length(base_torque) : length(base_force);
            if (mag < DEME_TINY_FLOAT)
                return;
        }

        if (A_in) {
            float3 force = base_force;
            float3 torque;
            if (need_torque)
                torque = base_torque;
            unsigned long long writeIndex = atomicAdd(d_numUsefulCnt, 1);
            if (writeIndex < capacity) {
                float3 cntPnt = granData->contactPointGeometryA[i];
                bodyID_t ownerID = ownerA;
                double3 CoM;
                float4 oriQ;
                oriQ.w = granData->oriQw[ownerID];
                oriQ.x = granData->oriQx[ownerID];
                oriQ.y = granData->oriQy[ownerID];
                oriQ.z = granData->oriQz[ownerID];
                if (need_torque) {
                    applyOriQToVector3<float, oriQ_t>(torque.x, torque.y, torque.z, oriQ.w, -oriQ.x, -oriQ.y, -oriQ.z);
                    torque = cross(cntPnt, torque);
                    if (!torque_in_local) {
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
                if (d_contact_owner)
                    d_contact_owner[writeIndex] = ownerID;
                if (need_torque)
                    d_torques[writeIndex] = torque;
            }
        }

        if (B_in && ownerB != ownerA) {
            float3 force = make_float3(-base_force.x, -base_force.y, -base_force.z);
            float3 torque;
            if (need_torque)
                torque = make_float3(-base_torque.x, -base_torque.y, -base_torque.z);
            unsigned long long writeIndex = atomicAdd(d_numUsefulCnt, 1);
            if (writeIndex < capacity) {
                float3 cntPnt = granData->contactPointGeometryB[i];
                bodyID_t ownerID = ownerB;
                double3 CoM;
                float4 oriQ;
                oriQ.w = granData->oriQw[ownerID];
                oriQ.x = granData->oriQx[ownerID];
                oriQ.y = granData->oriQy[ownerID];
                oriQ.z = granData->oriQz[ownerID];
                if (need_torque) {
                    applyOriQToVector3<float, oriQ_t>(torque.x, torque.y, torque.z, oriQ.w, -oriQ.x, -oriQ.y, -oriQ.z);
                    torque = cross(cntPnt, torque);
                    if (!torque_in_local) {
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
                if (d_contact_owner)
                    d_contact_owner[writeIndex] = ownerID;
                if (need_torque)
                    d_torques[writeIndex] = torque;
            }
        }
    }
}

void getContactForcesConcerningOwners(float3* d_points,
                                      float3* d_forces,
                                      float3* d_torques,
                                      bodyID_t* d_contact_owner,
                                      size_t* d_numUsefulCnt,
                                      bodyID_t* d_ownerIDs,
                                      size_t IDListSize,
                                      DEMSimParams* simParams,
                                      DEMDataDT* granData,
                                      size_t numCnt,
                                      size_t capacity,
                                      bool need_torque,
                                      bool torque_in_local,
                                      cudaStream_t& this_stream) {
    size_t blocks_needed = (numCnt + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    getContactForcesConcerningOwners_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
        d_points, d_forces, d_torques, d_contact_owner, reinterpret_cast<unsigned long long*>(d_numUsefulCnt),
        d_ownerIDs, IDListSize, simParams, granData, numCnt, capacity, need_torque, torque_in_local);
}

////////////////////////////////////////////////////////////////////////////////
// Patch-based voting kernels for mesh contact correction
////////////////////////////////////////////////////////////////////////////////

// Kernel to compute weighted normals (normal * area) for voting.
// Keys are read directly from geomToPatchMap on the fly, so only weightedNormals need to be written here.
__global__ void prepareWeightedNormalsForVoting_impl(DEMDataDT* granData,
                                                     float3* weightedNormals,
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

// Kernel to normalize voted normals and scatter them based on unique keys.
// Uses uniqueKeys (geomToPatchMap) to locate the patch slot, removing the need for total area arrays.
__global__ void normalizeAndScatterVotedNormalsFromUniqueKeys_impl(float3* votedWeightedNormals,
                                                                   contactPairs_t* uniqueKeys,
                                                                   float3* output,
                                                                   contactPairs_t startOffsetPatch,
                                                                   contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t patchIdx = uniqueKeys[idx];
        contactPairs_t localIdx = patchIdx - startOffsetPatch;

        float3 votedNormal = votedWeightedNormals[idx];
        float len2 = votedNormal.x * votedNormal.x + votedNormal.y * votedNormal.y + votedNormal.z * votedNormal.z;
        // normalize when length is non-zero; otherwise leave zero vector
        output[localIdx] = (len2 > 0.f) ? normalize(votedNormal) : make_float3(0, 0, 0);
    }
}

void normalizeAndScatterVotedNormalsFromUniqueKeys(float3* votedWeightedNormals,
                                                   contactPairs_t* uniqueKeys,
                                                   float3* output,
                                                   contactPairs_t startOffsetPatch,
                                                   contactPairs_t count,
                                                   cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        normalizeAndScatterVotedNormalsFromUniqueKeys_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0,
                                                             this_stream>>>(votedWeightedNormals, uniqueKeys, output,
                                                                            startOffsetPatch, count);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Penetration depth computation kernels for mesh contact correction
////////////////////////////////////////////////////////////////////////////////

// Kernel to compute per-primitive patch accumulators (projected area, max projected penetration, weighted CP sum).
__global__ void computePatchContactAccumulators_impl(DEMDataDT* granData,
                                                     float3* votedNormals,
                                                     const contactPairs_t* keys,
                                                     PatchContactAccum* accumulators,
                                                     contactPairs_t startOffsetPrimitive,
                                                     contactPairs_t startOffsetPatch,
                                                     contactPairs_t count,
                                                     notStupidBool_t compute_relvel) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t myContactID = startOffsetPrimitive + idx;

        contactPairs_t patchIdx = keys[idx];
        contactPairs_t localPatchIdx = patchIdx - startOffsetPatch;
        float3 votedNormal = votedNormals[localPatchIdx];

        float3 originalNormal = granData->contactForces[myContactID];
        float3 penetrationStorage = granData->contactPointGeometryA[myContactID];
        double originalPenetration = float3StorageToDouble(penetrationStorage);
        originalPenetration = (originalPenetration > 0.0) ? originalPenetration : 0.0;

        double area = float3StorageToDouble(granData->contactPointGeometryB[myContactID]);

        float dotProduct = dot(originalNormal, votedNormal);
        double cospos = (dotProduct > 0.f) ? (double)dotProduct : 0.0;

        double projectedPenetration = originalPenetration * cospos;
        double projectedArea = area * cospos;
        double weight = projectedPenetration * projectedArea;

        double3 contactPoint = to_double3(granData->contactTorque_convToForce[myContactID]);
        double3 weightedCP = make_double3(contactPoint.x * weight, contactPoint.y * weight, contactPoint.z * weight);

        double3 weightedRelVel = make_double3(0.0, 0.0, 0.0);
        if (compute_relvel && granData->triVelCenter != nullptr) {
            const contact_t ctype = granData->contactTypePrimitive[myContactID];
            const geoType_t typeA = decodeTypeA(ctype);
            const geoType_t typeB = decodeTypeB(ctype);
            if (typeA == GEO_T_TRIANGLE && typeB == GEO_T_TRIANGLE) {
                const bodyID_t triA = granData->idPrimitiveA[myContactID];
                const bodyID_t triB = granData->idPrimitiveB[myContactID];
                const float3 vA = granData->triVelCenter[triA];
                const float3 vB = granData->triVelCenter[triB];
                const double rvx = static_cast<double>(vA.x) - static_cast<double>(vB.x);
                const double rvy = static_cast<double>(vA.y) - static_cast<double>(vB.y);
                const double rvz = static_cast<double>(vA.z) - static_cast<double>(vB.z);
                weightedRelVel = make_double3(rvx * weight, rvy * weight, rvz * weight);
            }
        }

        PatchContactAccum acc;
        acc.sumProjArea = projectedArea;
        acc.maxProjPen = projectedPenetration;
        acc.sumWeight = weight;
        acc.sumWeightedCP = weightedCP;
        acc.sumWeightedRelVel = weightedRelVel;
        accumulators[idx] = acc;
    }
}

void computePatchContactAccumulators(DEMDataDT* granData,
                                     float3* votedNormals,
                                     const contactPairs_t* keys,
                                     PatchContactAccum* accumulators,
                                     contactPairs_t startOffsetPrimitive,
                                     contactPairs_t startOffsetPatch,
                                     contactPairs_t count,
                                     notStupidBool_t compute_relvel,
                                     cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        computePatchContactAccumulators_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, votedNormals, keys, accumulators, startOffsetPrimitive, startOffsetPatch, count, compute_relvel);
    }
}

// Kernel to scatter reduced patch accumulators to the final arrays expected by patch-based correction.
__global__ void scatterPatchContactAccumulators_impl(const PatchContactAccum* accumulators,
                                                     const contactPairs_t* uniqueKeys,
                                                     double* totalProjectedAreas,
                                                     double* maxProjectedPenetrations,
                                                     double3* votedContactPoints,
                                                     float3* votedRelVel,
                                                     contactPairs_t startOffsetPatch,
                                                     contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t patchIdx = uniqueKeys[idx];
        contactPairs_t localIdx = patchIdx - startOffsetPatch;

        PatchContactAccum acc = accumulators[idx];
        totalProjectedAreas[localIdx] = acc.sumProjArea;
        maxProjectedPenetrations[localIdx] = acc.maxProjPen;

        if (acc.sumWeight > 0.0) {
            double invWeight = 1.0 / acc.sumWeight;
            votedContactPoints[localIdx] = make_double3(
                acc.sumWeightedCP.x * invWeight, acc.sumWeightedCP.y * invWeight, acc.sumWeightedCP.z * invWeight);
            if (votedRelVel != nullptr) {
                votedRelVel[localIdx] = make_float3(static_cast<float>(acc.sumWeightedRelVel.x * invWeight),
                                                    static_cast<float>(acc.sumWeightedRelVel.y * invWeight),
                                                    static_cast<float>(acc.sumWeightedRelVel.z * invWeight));
            }
        } else {
            votedContactPoints[localIdx] = make_double3(0.0, 0.0, 0.0);
            if (votedRelVel != nullptr) {
                votedRelVel[localIdx] = make_float3(0.f, 0.f, 0.f);
            }
        }
    }
}

void scatterPatchContactAccumulators(const PatchContactAccum* accumulators,
                                     const contactPairs_t* uniqueKeys,
                                     double* totalProjectedAreas,
                                     double* maxProjectedPenetrations,
                                     double3* votedContactPoints,
                                     float3* votedRelVel,
                                     contactPairs_t startOffsetPatch,
                                     contactPairs_t count,
                                     cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        scatterPatchContactAccumulators_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            accumulators, uniqueKeys, totalProjectedAreas, maxProjectedPenetrations, votedContactPoints, votedRelVel,
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
                                                                   float3* zeroAreaRelVel,
                                                                   contactPairs_t* keys,
                                                                   contactPairs_t startOffsetPrimitive,
                                                                   contactPairs_t startOffsetPatch,
                                                                   contactPairs_t countPrimitive,
                                                                   notStupidBool_t compute_relvel) {
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

            if (compute_relvel && zeroAreaRelVel != nullptr && granData->triVelCenter != nullptr) {
                const contact_t ctype = granData->contactTypePrimitive[myContactID];
                const geoType_t typeA = decodeTypeA(ctype);
                const geoType_t typeB = decodeTypeB(ctype);
                if (typeA == GEO_T_TRIANGLE && typeB == GEO_T_TRIANGLE) {
                    const bodyID_t triA = granData->idPrimitiveA[myContactID];
                    const bodyID_t triB = granData->idPrimitiveB[myContactID];
                    const float3 vA = granData->triVelCenter[triA];
                    const float3 vB = granData->triVelCenter[triB];
                    zeroAreaRelVel[localPatchIdx] = make_float3(vA.x - vB.x, vA.y - vB.y, vA.z - vB.z);
                } else {
                    zeroAreaRelVel[localPatchIdx] = make_float3(0.f, 0.f, 0.f);
                }
            }
        }
    }
}

void findMaxPenetrationPrimitiveForZeroAreaPatches(DEMDataDT* granData,
                                                   double* maxPenetrations,
                                                   float3* zeroAreaNormals,
                                                   double* zeroAreaPenetrations,
                                                   double3* zeroAreaContactPoints,
                                                   float3* zeroAreaRelVel,
                                                   contactPairs_t* keys,
                                                   contactPairs_t startOffsetPrimitive,
                                                   contactPairs_t startOffsetPatch,
                                                   contactPairs_t countPrimitive,
                                                   notStupidBool_t compute_relvel,
                                                   cudaStream_t& this_stream) {
    size_t blocks_needed = (countPrimitive + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        findMaxPenetrationPrimitiveForZeroAreaPatches_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0,
                                                             this_stream>>>(
            granData, maxPenetrations, zeroAreaNormals, zeroAreaPenetrations, zeroAreaContactPoints, zeroAreaRelVel,
            keys, startOffsetPrimitive, startOffsetPatch, countPrimitive, compute_relvel);
    }
}

// Kernel to check if any primitive in each patch satisfies SAT (for tri-tri contacts)
// Uses simple idempotent writes to set patchHasSAT[patchIdx] = 1 if any primitive has contactSATSatisfied = 1
// Since we only transition from 0 to 1, and the array is pre-initialized to 0, multiple threads writing 1 is safe
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
    }
}

// Kernel to finalize patch results by combining normal voting results with zero-area case handling
// For patches with totalArea > 0 AND patchHasSAT = 1: use voted normal and weighted penetration
// For patches with totalArea == 0 OR patchHasSAT = 0: use max-penetration primitive's normal and penetration (Step 8
// fallback)
__global__ void finalizePatchResults_impl(double* totalProjectedAreas,
                                          float3* votedNormals,
                                          double* votedPenetrations,
                                          double3* votedContactPoints,
                                          float3* votedRelVel,
                                          float3* zeroAreaNormals,
                                          double* zeroAreaPenetrations,
                                          double3* zeroAreaContactPoints,
                                          float3* zeroAreaRelVel,
                                          notStupidBool_t* patchHasSAT,
                                          double* finalAreas,
                                          float3* finalNormals,
                                          double* finalPenetrations,
                                          double3* finalContactPoints,
                                          float3* finalRelVel,
                                          contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        double projectedArea = totalProjectedAreas[idx];
        // Default to 1 (SAT satisfied) for non-triangle-triangle contacts where patchHasSAT is null
        notStupidBool_t hasSAT = (patchHasSAT != nullptr) ? patchHasSAT[idx] : 1;

        // Use voted results only if projectedArea > 0 AND at least one primitive satisfies SAT
        if (projectedArea > 0.0 && hasSAT) {
            // Normal case: use voted results
            finalAreas[idx] = projectedArea;
            finalNormals[idx] = votedNormals[idx];
            finalPenetrations[idx] = votedPenetrations[idx];
            finalContactPoints[idx] = votedContactPoints[idx];
            if (finalRelVel != nullptr) {
                finalRelVel[idx] = (votedRelVel != nullptr) ? votedRelVel[idx] : make_float3(0.f, 0.f, 0.f);
            }
        } else {
            // Zero-area case OR no SAT-satisfying primitives: use max-penetration primitive's results (Step 8 fallback)
            // Set finalArea to 0 for these cases
            finalAreas[idx] = 0.0;
            finalNormals[idx] = zeroAreaNormals[idx];
            finalPenetrations[idx] = zeroAreaPenetrations[idx];
            finalContactPoints[idx] = zeroAreaContactPoints[idx];
            if (finalRelVel != nullptr) {
                finalRelVel[idx] = (zeroAreaRelVel != nullptr) ? zeroAreaRelVel[idx] : make_float3(0.f, 0.f, 0.f);
            }
        }
    }
}

void finalizePatchResults(double* totalProjectedAreas,
                          float3* votedNormals,
                          double* votedPenetrations,
                          double3* votedContactPoints,
                          float3* votedRelVel,
                          float3* zeroAreaNormals,
                          double* zeroAreaPenetrations,
                          double3* zeroAreaContactPoints,
                          float3* zeroAreaRelVel,
                          notStupidBool_t* patchHasSAT,
                          double* finalAreas,
                          float3* finalNormals,
                          double* finalPenetrations,
                          double3* finalContactPoints,
                          float3* finalRelVel,
                          contactPairs_t count,
                          cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        finalizePatchResults_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            totalProjectedAreas, votedNormals, votedPenetrations, votedContactPoints, votedRelVel, zeroAreaNormals,
            zeroAreaPenetrations, zeroAreaContactPoints, zeroAreaRelVel, patchHasSAT, finalAreas, finalNormals,
            finalPenetrations, finalContactPoints, finalRelVel, count);
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
