//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cub/cub.cuh>
// #include <thrust/sort.h>
#include <core/utils/JitHelper.h>
#include <helper_math.cuh>

#include <algorithms/DEMCubHelperFunctions.h>
#include <granular/HostSideHelpers.cpp>

#include <core/utils/GpuError.h>

namespace sgps {

struct CubFloat3Add {
    CUB_RUNTIME_FUNCTION __forceinline__ __device__ __host__ float3 operator()(const float3& a, const float3& b) const {
        return ::make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
};

void cubCollectForces(std::shared_ptr<jitify::Program>& collect_force,
                      clumpBodyInertiaOffset_t* inertiaPropOffsets,
                      bodyID_t* idA,
                      bodyID_t* idB,
                      float3* contactForces,
                      float3* contactPointA,
                      float3* contactPointB,
                      float* clump_h2aX,
                      float* clump_h2aY,
                      float* clump_h2aZ,
                      float* clump_h2AlphaX,
                      float* clump_h2AlphaY,
                      float* clump_h2AlphaZ,
                      bodyID_t* ownerClumpBody,
                      float* massClumpBody,
                      float* mmiXX,
                      float* mmiYY,
                      float* mmiZZ,
                      const double h,
                      const size_t nContactPairs,
                      const size_t nClumps,
                      const double l,
                      bool contactPairArr_isFresh,
                      cudaStream_t& this_stream,
                      DEMSolverStateDataDT& scratchPad,
                      clumpBodyInertiaOffset_t nDistinctClumpBodyTopologies) {
    // Preparation: allocate enough temp array memory and chop it to pieces, for the usage of cub operations. Note that
    // if contactPairArr_isFresh is false, then this allocation should not alter the size and content of the temp array
    // space, so the information in it can be used in the next iteration.
    size_t cachedArraySizeOwner = (size_t)2 * nContactPairs * sizeof(bodyID_t);
    // size_t cachedArraySizeMass = (size_t)2 * nContactPairs * sizeof(float);
    // size_t cachedArraySizeMOI = (size_t)2 * nContactPairs * sizeof(float3);
    bodyID_t* idAOwner = (bodyID_t*)scratchPad.allocateCachedOwner(cachedArraySizeOwner);
    bodyID_t* idBOwner = (bodyID_t*)(idAOwner + nContactPairs);
    // float* massAOwner = (float*)scratchPad.allocateCachedMass(cachedArraySizeMass);
    // float* massBOwner = (float*)(massAOwner + nContactPairs);
    // float3* moiAOwner = (float3*)scratchPad.allocateCachedMOI(cachedArraySizeMOI);
    // float3* moiBOwner = (float3*)(moiAOwner + nContactPairs);

    // size_t blocks_needed_for_twice_contacts = (2 * nContactPairs + SGPS_DEM_NUM_BODIES_PER_BLOCK - 1) /
    // SGPS_DEM_NUM_BODIES_PER_BLOCK;
    size_t blocks_needed_for_contacts =
        (nContactPairs + SGPS_DEM_NUM_BODIES_PER_BLOCK - 1) / SGPS_DEM_NUM_BODIES_PER_BLOCK;
    if (contactPairArr_isFresh) {
        // First step, prepare the owner ID array (nContactPairs * bodyID_t) for usage in final reduction by key (do it
        // for both A and B)
        collect_force->kernel("cashInOwnerIndex")
            .instantiate()
            .configure(dim3(blocks_needed_for_contacts), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, this_stream)
            .launch(idAOwner, idA, ownerClumpBody, nContactPairs);
        GPU_CALL(cudaStreamSynchronize(this_stream));

        collect_force->kernel("cashInOwnerIndex")
            .instantiate()
            .configure(dim3(blocks_needed_for_contacts), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, this_stream)
            .launch(idBOwner, idB, ownerClumpBody, nContactPairs);
        GPU_CALL(cudaStreamSynchronize(this_stream));
        // displayArray<bodyID_t>(idAOwner, nContactPairs>3?3:nContactPairs);
        // displayArray<bodyID_t>(idBOwner, nContactPairs>3?3:nContactPairs);

        // // Secondly, prepare the owner mass and moi arrays (nContactPairs * float/float3) for usage in final
        // reduction
        // // by key (do it for both A and B)
        // collect_force->kernel("cashInMassMoiIndex")
        //     .instantiate()
        //     .configure(dim3(blocks_needed_for_twice_contacts), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK),
        //                sizeof(float) * TEST_SHARED_SIZE * 4, this_stream)
        //     .launch(massAOwner, moiAOwner, inertiaPropOffsets, idAOwner, 2 * nContactPairs, massClumpBody, mmiXX,
        //     mmiYY,
        //             mmiZZ, nDistinctClumpBodyTopologies);
        // GPU_CALL(cudaStreamSynchronize(this_stream));
    }

    // ==============================================
    // 2nd, combine mass and force to get (contact pair-wise) acceleration, which will be reduced...
    // Note here allocated is temp vector, since unlike cached vectors, they cannot be reused in the next iteration
    size_t tempArraySizeAcc = (size_t)2 * nContactPairs * sizeof(float3);
    size_t tempArraySizeAcc_sorted = (size_t)2 * nContactPairs * sizeof(float3);
    size_t tempArraySizeOwnerAcc = (size_t)nClumps * sizeof(float3);
    size_t tempArraySizeOwner = (size_t)nClumps * sizeof(bodyID_t);
    float3* h2a_A = (float3*)scratchPad.allocateTempVector1(tempArraySizeAcc);
    float3* h2a_B = (float3*)(h2a_A + nContactPairs);
    float3* h2a_A_sorted = (float3*)scratchPad.allocateTempVector2(tempArraySizeAcc_sorted);
    // float3* h2a_B_sorted = (float3*)(h2a_A_sorted  + nContactPairs);
    bodyID_t* idAOwner_sorted = (bodyID_t*)scratchPad.allocateTempVector3(cachedArraySizeOwner);
    // bodyID_t* idBOwner_sorted = (bodyID_t*)(idAOwner_sorted + nContactPairs);
    float3* accOwner = (float3*)scratchPad.allocateTempVector4(
        tempArraySizeOwnerAcc);  // can store both linear and angular acceleration
    bodyID_t* uniqueOwner = (bodyID_t*)scratchPad.allocateTempVector5(tempArraySizeOwner);
    // collect accelerations for body A
    collect_force->kernel("forceToAcc")
        .instantiate()
        .configure(dim3(blocks_needed_for_contacts), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(h2a_A, contactForces, idAOwner, h * h / l, nContactPairs, inertiaPropOffsets, massClumpBody);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    // and don't forget body B
    collect_force->kernel("forceToAcc")
        .instantiate()
        .configure(dim3(blocks_needed_for_contacts), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(h2a_B, contactForces, idBOwner, -1. * h * h / l, nContactPairs, inertiaPropOffsets, massClumpBody);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    CubFloat3Add float3_add_op;
    // Reducing the acceleration (2 * nContactPairs for both body A and B)
    // Note: to do this, idAOwner needs to be sorted along with h2a_A. So we sort first.
    size_t cub_scratch_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, cub_scratch_bytes, idAOwner, idAOwner_sorted, h2a_A, h2a_A_sorted,
                                    nContactPairs * 2, 0, sizeof(bodyID_t) * SGPS_BITS_PER_BYTE, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceRadixSort::SortPairs(d_scratch_space, cub_scratch_bytes, idAOwner, idAOwner_sorted, h2a_A, h2a_A_sorted,
                                    nContactPairs * 2, 0, sizeof(bodyID_t) * SGPS_BITS_PER_BYTE, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    // Then we reduce by key
    cub::DeviceReduce::ReduceByKey(NULL, cub_scratch_bytes, idAOwner_sorted, uniqueOwner, h2a_A_sorted, accOwner,
                                   scratchPad.getForceCollectionRunsPointer(), float3_add_op, nContactPairs * 2,
                                   this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceReduce::ReduceByKey(d_scratch_space, cub_scratch_bytes, idAOwner_sorted, uniqueOwner, h2a_A_sorted,
                                   accOwner, scratchPad.getForceCollectionRunsPointer(), float3_add_op,
                                   nContactPairs * 2, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    // stash acceleration
    size_t blocks_needed_for_stashing =
        (scratchPad.getForceCollectionRuns() + SGPS_DEM_NUM_BODIES_PER_BLOCK - 1) / SGPS_DEM_NUM_BODIES_PER_BLOCK;
    collect_force->kernel("stashElem")
        .instantiate()
        .configure(dim3(blocks_needed_for_stashing), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(clump_h2aX, clump_h2aY, clump_h2aZ, uniqueOwner, accOwner, scratchPad.getForceCollectionRuns());
    GPU_CALL(cudaStreamSynchronize(this_stream));

    // =====================================================
    // Then take care of angular accelerations
    float3* h2Alpha_A = (float3*)(h2a_A);  // Memory spaces for accelerations can be reused
    float3* h2Alpha_B = (float3*)(h2a_B);
    float3* h2Alpha_A_sorted = (float3*)(h2a_A_sorted);
    // float3* h2Alpha_B_sorted = (float3*)(h2a_B_sorted);
    // collect angular accelerations for body A
    collect_force->kernel("forceToAngAcc")
        .instantiate()
        .configure(dim3(blocks_needed_for_contacts), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(h2Alpha_A, contactPointA, contactForces, idAOwner, h * h, nContactPairs, inertiaPropOffsets, mmiXX,
                mmiYY, mmiZZ);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    // and don't forget body B
    collect_force->kernel("forceToAngAcc")
        .instantiate()
        .configure(dim3(blocks_needed_for_contacts), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(h2Alpha_B, contactPointB, contactForces, idBOwner, -1. * h * h, nContactPairs, inertiaPropOffsets,
                mmiXX, mmiYY, mmiZZ);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    // Reducing the angular acceleration (2 * nContactPairs for both body A and B)
    // Note: to do this, idAOwner needs to be sorted along with h2Alpha_A. So we sort first.
    cub::DeviceRadixSort::SortPairs(NULL, cub_scratch_bytes, idAOwner, idAOwner_sorted, h2Alpha_A, h2Alpha_A_sorted,
                                    nContactPairs * 2, 0, sizeof(bodyID_t) * SGPS_BITS_PER_BYTE, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceRadixSort::SortPairs(d_scratch_space, cub_scratch_bytes, idAOwner, idAOwner_sorted, h2Alpha_A,
                                    h2Alpha_A_sorted, nContactPairs * 2, 0, sizeof(bodyID_t) * SGPS_BITS_PER_BYTE,
                                    this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    // Then we reduce
    cub::DeviceReduce::ReduceByKey(NULL, cub_scratch_bytes, idAOwner_sorted, uniqueOwner, h2Alpha_A_sorted, accOwner,
                                   scratchPad.getForceCollectionRunsPointer(), float3_add_op, nContactPairs * 2,
                                   this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceReduce::ReduceByKey(d_scratch_space, cub_scratch_bytes, idAOwner_sorted, uniqueOwner, h2Alpha_A_sorted,
                                   accOwner, scratchPad.getForceCollectionRunsPointer(), float3_add_op,
                                   nContactPairs * 2, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    // stash angular acceleration
    blocks_needed_for_stashing =
        (scratchPad.getForceCollectionRuns() + SGPS_DEM_NUM_BODIES_PER_BLOCK - 1) / SGPS_DEM_NUM_BODIES_PER_BLOCK;
    collect_force->kernel("stashElem")
        .instantiate()
        .configure(dim3(blocks_needed_for_stashing), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(clump_h2AlphaX, clump_h2AlphaY, clump_h2AlphaZ, uniqueOwner, accOwner,
                scratchPad.getForceCollectionRuns());
    GPU_CALL(cudaStreamSynchronize(this_stream));
}

}  // namespace sgps
