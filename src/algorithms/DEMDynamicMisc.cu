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
        bodyID_t geoA = granData->idGeometryA[i];
        bodyID_t ownerA = granData->ownerClumpBody[geoA];
        bodyID_t geoB = granData->idGeometryB[i];
        contact_t typeB = granData->contactType[i];
        bodyID_t ownerB = DEME_GET_GEO_OWNER_ID(geoB, typeB);

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

}  // namespace deme
