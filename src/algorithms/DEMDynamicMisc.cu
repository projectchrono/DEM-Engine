//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <algorithms/DEMStaticDeviceSubroutines.h>
#include <algorithms/DEMStaticDeviceUtilities.cuh>

#include <core/utils/GpuError.h>
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
            applyOriQToVector3<float, deme::oriQ_t>(torque.x, torque.y, torque.z, oriQ.w, -oriQ.x, -oriQ.y, -oriQ.z);
            // Force times point...
            torque = cross(cntPnt, torque);
            if (!torque_in_local) {  // back to global if needed
                applyOriQToVector3<float, deme::oriQ_t>(torque.x, torque.y, torque.z, oriQ.w, oriQ.x, oriQ.y, oriQ.z);
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
__global__ void normalizeAndScatterVotedNormals_impl(contactPairs_t* originalKeys,
                                                     contactPairs_t* uniqueKeys,
                                                     float3* votedWeightedNormals,
                                                     double* totalAreas,
                                                     float3* output,
                                                     size_t* numUniqueKeys,
                                                     contactPairs_t startOffset,
                                                     contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Get the key for this contact (already extracted in preparation step)
        contactPairs_t myKey = originalKeys[idx];

        // Find the corresponding unique key using binary search
        size_t numUnique = *numUniqueKeys;
        ssize_t keyIdx = -1;
        cuda_binary_search<contactPairs_t, ssize_t>(uniqueKeys, myKey, 0, numUnique - 1, keyIdx);

        float3 votedNormal = make_float3(0.0f, 0.0f, 0.0f);
        if (keyIdx >= 0 && keyIdx < (ssize_t)numUnique) {
            double totalArea = totalAreas[keyIdx];
            if (totalArea > 0.0) {
                // Normalize by dividing by total area (use reciprocal multiplication for efficiency)
                double invTotalArea = 1.0 / totalArea;
                votedNormal.x = votedWeightedNormals[keyIdx].x * invTotalArea;
                votedNormal.y = votedWeightedNormals[keyIdx].y * invTotalArea;
                votedNormal.z = votedWeightedNormals[keyIdx].z * invTotalArea;
            }
            // else: votedNormal remains (0,0,0)
        }

        // Write to output at the correct position
        output[startOffset + idx] = votedNormal;
    }
}

void normalizeAndScatterVotedNormals(contactPairs_t* originalKeys,
                                     contactPairs_t* uniqueKeys,
                                     float3* votedWeightedNormals,
                                     double* totalAreas,
                                     float3* output,
                                     size_t* numUniqueKeys,
                                     contactPairs_t startOffset,
                                     contactPairs_t count,
                                     cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        normalizeAndScatterVotedNormals_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            originalKeys, uniqueKeys, votedWeightedNormals, totalAreas, output, numUniqueKeys, startOffset, count);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
}

}  // namespace deme
