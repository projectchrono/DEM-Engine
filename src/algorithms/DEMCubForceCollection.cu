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
                      double h,
                      size_t nContactPairs,
                      size_t nClumps,
                      double l,
                      bool contactPairArr_isFresh,
                      cudaStream_t& this_stream,
                      DEMSolverStateData& scratchPad,
                      clumpBodyInertiaOffset_t nDistinctClumpBodyTopologies) {
    // Preparation: allocate enough temp array memory and chop it to pieces, for the usage of cub operations. Note that
    // if contactPairArr_isFresh is false, then this allocation should not alter the size and content of the temp array
    // space, so the information in it can be used in the next iteration.
    size_t cachedArraySize =
        2 * (nContactPairs * sizeof(bodyID_t) + nContactPairs * sizeof(float) + nContactPairs * sizeof(float3));
    bodyID_t* idAOwner = (bodyID_t*)scratchPad.allocateCachedVector(cachedArraySize);
    bodyID_t* idBOwner = (bodyID_t*)(idAOwner + nContactPairs);
    float* massAOwner = (float*)(idBOwner + nContactPairs);
    float* massBOwner = (float*)(massAOwner + nContactPairs);
    float3* moiAOwner = (float3*)(massBOwner + nContactPairs);
    float3* moiBOwner = (float3*)(moiAOwner + nContactPairs);

    size_t blocks_needed_for_all_contacts = (2 * nContactPairs + NUM_BODIES_PER_BLOCK - 1) / NUM_BODIES_PER_BLOCK;
    size_t blocks_needed_for_half_contacts = (nContactPairs + NUM_BODIES_PER_BLOCK - 1) / NUM_BODIES_PER_BLOCK;
    auto collect_force =
        JitHelper::buildProgram("DEMCollectForceKernels", JitHelper::KERNEL_DIR / "DEMCollectForceKernels.cu",
                                std::vector<JitHelper::Header>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});
    if (contactPairArr_isFresh) {
        // First step, prepare the owner ID array (nContactPairs * bodyID_t) for usage in final reduction by key (do it
        // for both A and B)
        collect_force.kernel("cashInOwnerIndex")
            .instantiate()
            .configure(dim3(blocks_needed_for_half_contacts), dim3(NUM_BODIES_PER_BLOCK), 0, this_stream)
            .launch(idAOwner, idA, ownerClumpBody, nContactPairs);
        GPU_CALL(cudaStreamSynchronize(this_stream));

        collect_force.kernel("cashInOwnerIndex")
            .instantiate()
            .configure(dim3(blocks_needed_for_half_contacts), dim3(NUM_BODIES_PER_BLOCK), 0, this_stream)
            .launch(idBOwner, idB, ownerClumpBody, nContactPairs);
        GPU_CALL(cudaStreamSynchronize(this_stream));

        // Secondly, prepare the owner mass and moi arrays (nContactPairs * float/float3) for usage in final reduction
        // by key (do it for both A and B)
        // TODO: massClumpBody is template and should be jitified
        collect_force.kernel("cashInMassMoiIndex")
            .instantiate()
            .configure(dim3(blocks_needed_for_all_contacts), dim3(NUM_BODIES_PER_BLOCK),
                       sizeof(float) * TEST_SHARED_SIZE * 4, this_stream)
            .launch(massAOwner, moiAOwner, inertiaPropOffsets, idAOwner, 2 * nContactPairs, massClumpBody, mmiXX, mmiYY,
                    mmiZZ, nDistinctClumpBodyTopologies);
        GPU_CALL(cudaStreamSynchronize(this_stream));
        // displayArray<bodyID_t>(idAOwner, nContactPairs);
    }

    // Thirdly, combine mass and force to get (contact pair-wise) acceleration, which will be reduced...
    // Note here allocated is temp vector, since unlike cached vectors, they cannot be reused in the next iteration
    size_t tempArraySize = 2 * (nContactPairs * sizeof(float3) + nContactPairs * sizeof(float3)) +
                           nClumps * sizeof(float3) + nClumps * sizeof(bodyID_t);
    float3* h2a_A = (float3*)scratchPad.allocateTempVector(tempArraySize);
    float3* h2a_B = (float3*)(h2a_A + nContactPairs);
    float3* h2Alpha_A = (float3*)(h2a_B + nContactPairs);
    float3* h2Alpha_B = (float3*)(h2Alpha_A + nContactPairs);
    float3* accOwner = (float3*)(h2Alpha_B + nContactPairs);  // can store both linear and angular acceleration
    bodyID_t* uniqueOwner = (bodyID_t*)(accOwner + nClumps);
    // collect accelerations for body A
    collect_force.kernel("elemDivide")
        .instantiate()
        .configure(dim3(blocks_needed_for_half_contacts), dim3(NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(h2a_A, contactForces, massAOwner, h * h / l, nContactPairs);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    // and don't forget body B
    collect_force.kernel("elemDivide")
        .instantiate()
        .configure(dim3(blocks_needed_for_half_contacts), dim3(NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(h2a_B, contactForces, massBOwner, -1. * h * h / l, nContactPairs);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    // collect angular accelerations for body A
    collect_force.kernel("elemCrossDivide")
        .instantiate()
        .configure(dim3(blocks_needed_for_half_contacts), dim3(NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(h2Alpha_A, contactPointA, contactForces, moiAOwner, h * h, nContactPairs);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    // and don't forget body B
    collect_force.kernel("elemCrossDivide")
        .instantiate()
        .configure(dim3(blocks_needed_for_half_contacts), dim3(NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(h2Alpha_B, contactPointB, contactForces, moiBOwner, -1. * h * h, nContactPairs);
    GPU_CALL(cudaStreamSynchronize(this_stream));

    // Finally, do the reduction by key
    CubAdd reduction_op;
    size_t cub_scratch_bytes = 0;
    // reducing the acceleration
    cub::DeviceReduce::ReduceByKey(NULL, cub_scratch_bytes, idAOwner, uniqueOwner, h2a_A, accOwner,
                                   scratchPad.getForceCollectionRunsPointer(), reduction_op, 2 * nContactPairs,
                                   this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceReduce::ReduceByKey(d_scratch_space, cub_scratch_bytes, idAOwner, uniqueOwner, h2a_A, accOwner,
                                   scratchPad.getForceCollectionRunsPointer(), reduction_op, 2 * nContactPairs,
                                   this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));

    size_t blocks_needed_for_stashing =
        (scratchPad.getForceCollectionRuns() + NUM_BODIES_PER_BLOCK - 1) / NUM_BODIES_PER_BLOCK;
    collect_force.kernel("stashElem")
        .instantiate()
        .configure(dim3(blocks_needed_for_stashing), dim3(NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(clump_h2aX, clump_h2aY, clump_h2aZ, uniqueOwner, accOwner, scratchPad.getForceCollectionRuns());
    GPU_CALL(cudaStreamSynchronize(this_stream));

    // reducing the angular acceleration
    cub::DeviceReduce::ReduceByKey(NULL, cub_scratch_bytes, idAOwner, uniqueOwner, h2Alpha_A, accOwner,
                                   scratchPad.getForceCollectionRunsPointer(), reduction_op, 2 * nContactPairs,
                                   this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceReduce::ReduceByKey(d_scratch_space, cub_scratch_bytes, idAOwner, uniqueOwner, h2Alpha_A, accOwner,
                                   scratchPad.getForceCollectionRunsPointer(), reduction_op, 2 * nContactPairs,
                                   this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));

    blocks_needed_for_stashing =
        (scratchPad.getForceCollectionRuns() + NUM_BODIES_PER_BLOCK - 1) / NUM_BODIES_PER_BLOCK;
    collect_force.kernel("stashElem")
        .instantiate()
        .configure(dim3(blocks_needed_for_stashing), dim3(NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(clump_h2AlphaX, clump_h2AlphaY, clump_h2AlphaZ, uniqueOwner, accOwner,
                scratchPad.getForceCollectionRuns());
    GPU_CALL(cudaStreamSynchronize(this_stream));
}

}  // namespace sgps
