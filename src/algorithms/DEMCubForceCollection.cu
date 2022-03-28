//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cub/cub.cuh>
#include <core/utils/JitHelper.h>
#include <helper_math.cuh>

#include <algorithms/DEMCubHelperFunctions.h>
#include <granular/HostSideHelpers.cpp>

#include <core/utils/GpuError.h>

namespace sgps {

struct CubAdd {
    template <typename T>
    CUB_RUNTIME_FUNCTION __forceinline__ T operator()(const T& a, const T& b) const {
        return a + b;
    }
};

void cubCollectForces(clumpBodyInertiaOffset_t* inertiaPropOffsets,
                      bodyID_t* idA,
                      bodyID_t* idB,
                      float3* contactForces,
                      float* clump_h2aX,
                      float* clump_h2aY,
                      float* clump_h2aZ,
                      bodyID_t* ownerClumpBody,
                      float* massClumpBody,
                      double h,
                      size_t nContactPairs,
                      size_t nClumps,
                      double l,
                      bool contactPairArr_isFresh,
                      GpuManager::StreamInfo& streamInfo,
                      DEMSolverStateData& scratchPad,
                      clumpBodyInertiaOffset_t nDistinctClumpBodyTopologies) {
    // Preparation: allocate enough temp array memory and chop it to pieces, for the usage of cub operations. Note that
    // if contactPairArr_isFresh is false, then this allocation should not alter the size and content of the temp array
    // space.
    size_t cachedArraySize = 2 * (nContactPairs * sizeof(bodyID_t) + nContactPairs * sizeof(float));
    bodyID_t* idAOwner = (bodyID_t*)scratchPad.allocateCachedVector(cachedArraySize);
    bodyID_t* idBOwner = (bodyID_t*)(idAOwner + nContactPairs * sizeof(bodyID_t));
    float* massAOwner = (float*)(idBOwner + nContactPairs * sizeof(bodyID_t));
    float* massBOwner = (float*)(massAOwner + nContactPairs * sizeof(float));

    size_t blocks_needed_for_contacts = (nContactPairs + NUM_BODIES_PER_BLOCK - 1) / NUM_BODIES_PER_BLOCK;
    auto collect_force =
        JitHelper::buildProgram("DEMCollectForceKernels", JitHelper::KERNEL_DIR / "DEMCollectForceKernels.cu",
                                std::vector<JitHelper::Header>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});
    if (contactPairArr_isFresh) {
        // First step, prepare the owner ID array (nContactPairs * bodyID_t) for usage in final reduction by key (do it
        // for both A and B)
        collect_force.kernel("cashInOwnerIndex")
            .instantiate()
            .configure(dim3(blocks_needed_for_contacts), dim3(NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
            .launch(idAOwner, idA, ownerClumpBody, nContactPairs);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        collect_force.kernel("cashInOwnerIndex")
            .instantiate()
            .configure(dim3(blocks_needed_for_contacts), dim3(NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
            .launch(idBOwner, idB, ownerClumpBody, nContactPairs);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // Secondly, prepare the owner mass array (nContactPairs * float) for usage in final reduction by key (do it for
        // both A and B)
        // TODO: massClumpBody is template and should be jitified
        collect_force.kernel("cashInMassIndex")
            .instantiate()
            .configure(dim3(blocks_needed_for_contacts), dim3(NUM_BODIES_PER_BLOCK), sizeof(float) * TEST_SHARED_SIZE,
                       streamInfo.stream)
            .launch(massAOwner, inertiaPropOffsets, idAOwner, l, h, nContactPairs, massClumpBody,
                    nDistinctClumpBodyTopologies);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        collect_force.kernel("cashInMassIndex")
            .instantiate()
            .configure(dim3(blocks_needed_for_contacts), dim3(NUM_BODIES_PER_BLOCK), sizeof(float) * TEST_SHARED_SIZE,
                       streamInfo.stream)
            .launch(massBOwner, inertiaPropOffsets, idBOwner, l, h, nContactPairs, massClumpBody,
                    nDistinctClumpBodyTopologies);

        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        // displayArray<bodyID_t>(idAOwner, nContactPairs);
    }

    // Thirdly, combine mass and force to get (contact pair-wise) acceleration, which will be reduced...
    // Note here allocated is temp vector, since unlike cached vectors, they cannot be reused in the next iteration
    size_t tempArraySize = 2 * nContactPairs * sizeof(float3) + nClumps * sizeof(float3) + nClumps * sizeof(bodyID_t);
    float3* h2a_A = (float3*)scratchPad.allocateTempVector(tempArraySize);
    float3* h2a_B = (float3*)(h2a_A + nContactPairs * sizeof(float3));
    float3* forceOwner = (float3*)(h2a_B + nContactPairs * sizeof(float3));
    bodyID_t* uniqueOwner = (bodyID_t*)(forceOwner + nClumps * sizeof(float3));
    // collect accelerations for body A
    collect_force.kernel("elemDivide")
        .instantiate()
        .configure(dim3(blocks_needed_for_contacts), dim3(NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
        .launch(h2a_A, contactForces, massAOwner, 1.f, nContactPairs);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    // and don't forget body B
    collect_force.kernel("elemDivide")
        .instantiate()
        .configure(dim3(blocks_needed_for_contacts), dim3(NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
        .launch(h2a_B, contactForces, massBOwner, -1.f, nContactPairs);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

    // Finally, do the reduction by key
    CubAdd reduction_op;
    size_t temp_storage_bytes = 0;

    cub::DeviceReduce::ReduceByKey(NULL, temp_storage_bytes, idAOwner, uniqueOwner, h2a_A, forceOwner,
                                   scratchPad.getForceCollectionRunsPointer(), reduction_op, nContactPairs,
                                   streamInfo.stream, false);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(temp_storage_bytes);
    cub::DeviceReduce::ReduceByKey(d_scratch_space, temp_storage_bytes, idAOwner, uniqueOwner, h2a_A, forceOwner,
                                   scratchPad.getForceCollectionRunsPointer(), reduction_op, nContactPairs,
                                   streamInfo.stream, false);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

    size_t blocks_needed_for_stashing =
        (scratchPad.getForceCollectionRuns() + NUM_BODIES_PER_BLOCK - 1) / NUM_BODIES_PER_BLOCK;
    collect_force.kernel("stashElem")
        .instantiate()
        .configure(dim3(blocks_needed_for_stashing), dim3(NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
        .launch(clump_h2aX, clump_h2aY, clump_h2aZ, uniqueOwner, forceOwner, scratchPad.getForceCollectionRuns());
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

    // don't forget do that for B too
    cub::DeviceReduce::ReduceByKey(NULL, temp_storage_bytes, idBOwner, uniqueOwner, h2a_B, forceOwner,
                                   scratchPad.getForceCollectionRunsPointer(), reduction_op, nContactPairs,
                                   streamInfo.stream, false);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    d_scratch_space = (void*)scratchPad.allocateScratchSpace(temp_storage_bytes);
    cub::DeviceReduce::ReduceByKey(d_scratch_space, temp_storage_bytes, idBOwner, uniqueOwner, h2a_B, forceOwner,
                                   scratchPad.getForceCollectionRunsPointer(), reduction_op, nContactPairs,
                                   streamInfo.stream, false);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

    blocks_needed_for_stashing =
        (scratchPad.getForceCollectionRuns() + NUM_BODIES_PER_BLOCK - 1) / NUM_BODIES_PER_BLOCK;
    collect_force.kernel("stashElem")
        .instantiate()
        .configure(dim3(blocks_needed_for_stashing), dim3(NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
        .launch(clump_h2aX, clump_h2aY, clump_h2aZ, uniqueOwner, forceOwner, scratchPad.getForceCollectionRuns());
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
}

}  // namespace sgps
