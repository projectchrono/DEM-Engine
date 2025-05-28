//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <cub/cub.cuh>
// #include <thrust/sort.h>
#include <kernel/DEMHelperKernels.cuh>

#include <algorithms/DEMStaticDeviceSubroutines.h>
#include <DEM/HostSideHelpers.hpp>

#include <algorithms/DEMCubWrappers.cu>

#include <core/utils/GpuError.h>

namespace deme {

void collectContactForcesThruCub(std::shared_ptr<jitify::Program>& collect_force_kernels,
                                 DualStruct<DEMDataDT>& granData,
                                 const size_t nContactPairs,
                                 const size_t nClumps,
                                 bool contactPairArr_isFresh,
                                 cudaStream_t& this_stream,
                                 DEMSolverScratchData& scratchPad,
                                 SolverTimers& timers) {
    // Preparation: allocate enough temp array memory and chop it to pieces, for the usage of cub operations. Note that
    // if contactPairArr_isFresh is false, then this allocation should not alter the size and content of the temp array
    // space, so the information in it can be used in the next iteration.
    size_t cachedArraySizeOwner = (size_t)2 * nContactPairs * sizeof(bodyID_t);
    // Use temp vector to store the flattened owner IDs.
    // Note this one is not temp vector as it could be used between time steps.
    bodyID_t* idAOwner = (bodyID_t*)scratchPad.allocateVector("idAOwner", cachedArraySizeOwner);
    bodyID_t* idBOwner = (bodyID_t*)(idAOwner + nContactPairs);
    // size_t cachedArraySizeMass = (size_t)2 * nContactPairs * sizeof(float);
    // size_t cachedArraySizeMOI = (size_t)2 * nContactPairs * sizeof(float3);
    // float* massAOwner = (float*)scratchPad.allocateCachedMass(cachedArraySizeMass);
    // float* massBOwner = (float*)(massAOwner + nContactPairs);
    // float3* moiAOwner = (float3*)scratchPad.allocateCachedMOI(cachedArraySizeMOI);
    // float3* moiBOwner = (float3*)(moiAOwner + nContactPairs);

    // size_t blocks_needed_for_twice_contacts = (2 * nContactPairs + DEME_NUM_BODIES_PER_BLOCK - 1) /
    // DEME_NUM_BODIES_PER_BLOCK;
    size_t blocks_needed_for_contacts = (nContactPairs + DEME_NUM_BODIES_PER_BLOCK - 1) / DEME_NUM_BODIES_PER_BLOCK;
    if (contactPairArr_isFresh) {
        // First step, prepare the owner ID array (nContactPairs * bodyID_t) for usage in final reduction by key (do it
        // for both A and B)
        // Note for A, it is always a sphere or a triangle
        collect_force_kernels->kernel("cashInOwnerIndexA")
            .instantiate()
            .configure(dim3(blocks_needed_for_contacts), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, this_stream)
            .launch(idAOwner, granData->idGeometryA, granData->ownerClumpBody, granData->contactType, nContactPairs);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));

        // But for B, it can be sphere, triangle or some analytical geometries
        collect_force_kernels->kernel("cashInOwnerIndexB")
            .instantiate()
            .configure(dim3(blocks_needed_for_contacts), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, this_stream)
            .launch(idBOwner, granData->idGeometryB, granData->ownerClumpBody, granData->ownerMesh,
                    granData->contactType, nContactPairs);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
        // displayDeviceArray<bodyID_t>(idAOwner, nContactPairs);
        // displayDeviceArray<bodyID_t>(idBOwner, nContactPairs);
    }

    // ==============================================
    // 2nd, combine mass and force to get (contact pair-wise) acceleration, which will be reduced...
    // Note here allocated is temp vector, since unlike cached vectors, they cannot be reused in the next iteration
    size_t tempArraySizeAcc = (size_t)2 * nContactPairs * sizeof(float3);
    size_t tempArraySizeAcc_sorted = (size_t)2 * nContactPairs * sizeof(float3);
    size_t tempArraySizeOwnerAcc = (size_t)nClumps * sizeof(float3);
    size_t tempArraySizeOwner = (size_t)nClumps * sizeof(bodyID_t);
    float3* acc_A = (float3*)scratchPad.allocateTempVector("acc_A", tempArraySizeAcc);
    float3* acc_B = (float3*)(acc_A + nContactPairs);
    float3* acc_A_sorted = (float3*)scratchPad.allocateTempVector("acc_A_sorted", tempArraySizeAcc_sorted);
    // float3* acc_B_sorted = (float3*)(acc_A_sorted  + nContactPairs);
    bodyID_t* idAOwner_sorted = (bodyID_t*)scratchPad.allocateTempVector("idAOwner_sorted", cachedArraySizeOwner);
    // bodyID_t* idBOwner_sorted = (bodyID_t*)(idAOwner_sorted + nContactPairs);
    float3* accOwner = (float3*)scratchPad.allocateTempVector(
        "accOwner", tempArraySizeOwnerAcc);  // can store both linear and angular acceleration
    bodyID_t* uniqueOwner = (bodyID_t*)scratchPad.allocateTempVector("uniqueOwner", tempArraySizeOwner);
    // Collect accelerations for body A (modifier used to be h * h / l when we stored acc as h^2*acc)
    // NOTE!! If you pass floating point number to kernels, the number needs to be something like 1.f, not 1.0.
    // Somtimes 1.0 got converted to 0.f with the kernel call.
    collect_force_kernels->kernel("forceToAcc")
        .instantiate()
        .configure(dim3(blocks_needed_for_contacts), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(acc_A, granData->contactForces, idAOwner, 1.f, nContactPairs, &granData);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    // and don't forget body B
    collect_force_kernels->kernel("forceToAcc")
        .instantiate()
        .configure(dim3(blocks_needed_for_contacts), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(acc_B, granData->contactForces, idBOwner, -1.f, nContactPairs, &granData);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    // displayDeviceFloat3(acc_A, 2 * nContactPairs);
    // displayDeviceFloat3(granData->contactForces, nContactPairs);

    // Reducing the acceleration (2 * nContactPairs for both body A and B)
    // Note: to do this, idAOwner needs to be sorted along with acc_A. So we sort first.
    cubDEMSortByKeys<bodyID_t, float3>(idAOwner, idAOwner_sorted, acc_A, acc_A_sorted, nContactPairs * 2, this_stream,
                                       scratchPad);
    // Then we reduce by key
    // This variable stores the cub output of how many cub runs it executed for collecting forces
    scratchPad.allocateDualStruct("forceCollectionRuns");
    size_t* dpForceCollectionRuns = scratchPad.getDualStructDevice("forceCollectionRuns");
    size_t* hpForceCollectionRuns = scratchPad.getDualStructHost("forceCollectionRuns");
    CubFloat3Add float3_add_op;
    cubDEMReduceByKeys<bodyID_t, float3, CubFloat3Add>(idAOwner_sorted, uniqueOwner, acc_A_sorted, accOwner,
                                                       dpForceCollectionRuns, float3_add_op, nContactPairs * 2,
                                                       this_stream, scratchPad);
    scratchPad.syncDualStructDeviceToHost("forceCollectionRuns");
    // Then we stash acceleration
    size_t blocks_needed_for_stashing =
        (*hpForceCollectionRuns + DEME_NUM_BODIES_PER_BLOCK - 1) / DEME_NUM_BODIES_PER_BLOCK;
    collect_force_kernels->kernel("stashElem")
        .instantiate()
        .configure(dim3(blocks_needed_for_stashing), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(granData->aX, granData->aY, granData->aZ, uniqueOwner, accOwner, *hpForceCollectionRuns);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    // displayDeviceArray<float>(granData->aX, nClumps);
    // displayDeviceArray<float>(granData->aY, nClumps);
    // displayDeviceArray<float>(granData->aZ, nClumps);

    // =====================================================
    // Then take care of angular accelerations
    float3* alpha_A = (float3*)(acc_A);  // Memory spaces for accelerations can be reused
    float3* alpha_B = (float3*)(acc_B);
    float3* alpha_A_sorted = (float3*)(acc_A_sorted);
    // float3* alpha_B_sorted = (float3*)(acc_B_sorted);
    // collect angular accelerations for body A (modifier used to be h * h when we stored acc as h^2*acc)
    collect_force_kernels->kernel("forceToAngAcc")
        .instantiate()
        .configure(dim3(blocks_needed_for_contacts), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(alpha_A, granData->contactPointGeometryA, granData->oriQw, granData->oriQx, granData->oriQy,
                granData->oriQz, granData->contactForces, granData->contactTorque_convToForce, idAOwner, 1.f,
                nContactPairs, &granData);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    // and don't forget body B
    collect_force_kernels->kernel("forceToAngAcc")
        .instantiate()
        .configure(dim3(blocks_needed_for_contacts), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(alpha_B, granData->contactPointGeometryB, granData->oriQw, granData->oriQx, granData->oriQy,
                granData->oriQz, granData->contactForces, granData->contactTorque_convToForce, idBOwner, -1.f,
                nContactPairs, &granData);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    // Reducing the angular acceleration (2 * nContactPairs for both body A and B)
    // Note: to do this, idAOwner needs to be sorted along with alpha_A. So we sort first.
    cubDEMSortByKeys<bodyID_t, float3>(idAOwner, idAOwner_sorted, alpha_A, alpha_A_sorted, nContactPairs * 2,
                                       this_stream, scratchPad);
    // Then we reduce
    cubDEMReduceByKeys<bodyID_t, float3, CubFloat3Add>(idAOwner_sorted, uniqueOwner, alpha_A_sorted, accOwner,
                                                       dpForceCollectionRuns, float3_add_op, nContactPairs * 2,
                                                       this_stream, scratchPad);
    scratchPad.syncDualStructDeviceToHost("forceCollectionRuns");
    // Then we stash angular acceleration
    blocks_needed_for_stashing = (*hpForceCollectionRuns + DEME_NUM_BODIES_PER_BLOCK - 1) / DEME_NUM_BODIES_PER_BLOCK;
    collect_force_kernels->kernel("stashElem")
        .instantiate()
        .configure(dim3(blocks_needed_for_stashing), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(granData->alphaX, granData->alphaY, granData->alphaZ, uniqueOwner, accOwner, *hpForceCollectionRuns);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));

    scratchPad.finishUsingTempVector("acc_A");
    scratchPad.finishUsingTempVector("acc_A_sorted");
    scratchPad.finishUsingTempVector("idAOwner_sorted");
    scratchPad.finishUsingTempVector("accOwner");
    scratchPad.finishUsingTempVector("uniqueOwner");
    scratchPad.finishUsingDualStruct("forceCollectionRuns");
}

}  // namespace deme
