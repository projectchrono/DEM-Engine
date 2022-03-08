//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cstring>
#include <iostream>
#include <thread>

// #include <cub/cub.cuh>

#include <core/ApiVersion.h>
#include <core/utils/Macros.h>
#include <core/utils/chpf/particle_writer.hpp>
#include <granular/GranularDefines.h>
#include <granular/PhysicsSystem.h>
#include <core/utils/JitHelper.h>
#include <granular/HostSideHelpers.cpp>

namespace sgps {

__host__ void DEMKinematicThread::contactDetection() {

/*
    size_t blocks_needed_for_bodies = (simParams->nSpheresGM + NUM_BODIES_PER_BLOCK - 1) / NUM_BODIES_PER_BLOCK;
    auto bin_occupation =
        JitHelper::buildProgram("DEMBinSphereKernels", JitHelper::KERNEL_DIR / "DEMBinSphereKernels.cu",
                                std::vector<JitHelper::Header>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

    bin_occupation.kernel("getNumberOfBinsEachSphereTouches")
        .instantiate()
        .configure(dim3(blocks_needed_for_bodies), dim3(NUM_BODIES_PER_BLOCK), sizeof(float) * TEST_SHARED_SIZE * 4,
                   streamInfo.stream)
        .launch(simParams, granData, granTemplates);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

    // cubPrefixScan<binsSphereTouches_t>(granData->numBinsSphereTouches, simParams->nSpheresGM + 1);
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(NULL, temp_storage_bytes, granData->numBinsSphereTouches, granData->numBinsSphereTouches, simParams->nSpheresGM + 1);
    GPU_CALL(cudaDeviceSynchronize());
    GPU_CALL(cudaPeekAtLastError());

    // give CUB needed temporary storage on the device
    // void* d_scratch_space = (void*)stateOfSolver_resources.pDeviceMemoryScratchSpace(temp_storage_bytes);
    // cub::DeviceScan::ExclusiveSum(d_scratch_space, temp_storage_bytes, in_ptr, out_ptr, nSDs);
    // GPU_CALL(cudaDeviceSynchronize());
    // GPU_CALL(cudaPeekAtLastError());

    // displayArray<binsSphereTouches_t>(granData->numBinsSphereTouches, simParams->nSpheresGM + 1);
    // Resize those work arrays to be the size of the number of all sphere--bin pairs
    binIDsEachSphereTouches.resize(granData->numBinsSphereTouches[simParams->nSpheresGM]);
    sphereIDsEachBinTouches.resize(granData->numBinsSphereTouches[simParams->nSpheresGM]);
    granData->binIDsEachSphereTouches = binIDsEachSphereTouches.data();
    granData->sphereIDsEachBinTouches = sphereIDsEachBinTouches.data();

    bin_occupation.kernel("populateBinSphereTouchingPairs")
        .instantiate()
        .configure(dim3(blocks_needed_for_bodies), dim3(NUM_BODIES_PER_BLOCK), sizeof(float) * TEST_SHARED_SIZE * 4,
                   streamInfo.stream)
        .launch(simParams, granData, granTemplates);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    // std::cout << "Unsorted bin IDs: ";
    // displayArray<binID_t>(granData->binIDsEachSphereTouches, granData->numBinsSphereTouches[simParams->nSpheresGM]);
    // std::cout << "Corresponding sphere IDs: ";
    // displayArray<bodyID_t>(granData->sphereIDsEachBinTouches, granData->numBinsSphereTouches[simParams->nSpheresGM]);

    hostSortByKey<binID_t, bodyID_t>(granData->binIDsEachSphereTouches, granData->sphereIDsEachBinTouches,
                                     granData->numBinsSphereTouches[simParams->nSpheresGM]);
    // std::cout << "Sorted bin IDs: ";
    // displayArray<binID_t>(granData->binIDsEachSphereTouches, granData->numBinsSphereTouches[simParams->nSpheresGM]);
    // std::cout << "Corresponding sphere IDs: ";
    // displayArray<bodyID_t>(granData->sphereIDsEachBinTouches, granData->numBinsSphereTouches[simParams->nSpheresGM]);

    // TODO: use cub to do this. Probably one-two punch: first the number of jumps, then jump locations
    // Search for bins that have at least 2 spheres living in.
    // TODO: Is it good to ensure that the bins are non-empty here? If we do it here, we can increase the occupation
    // rate in the contact detection kernels, since it reduces the number of idle threads there (well, maybe this is not
    // even that much, since there will be bins that suffice but has like 2 spheres in it, so they will finish rather
    // quickly); but is it easy to use CUB to ensure each jump has at least 2 relevant elements?
    hostScanForJumpsNum<binID_t>(granData->binIDsEachSphereTouches,
                                 granData->numBinsSphereTouches[simParams->nSpheresGM], 2, simParams->nActiveBins);

    // OK, 2 choices here: either use this array activeBinIDs to register active bin IDs (we need this info to rule out
    // double-count in CD), or screw the idea of activeBins, just give every bin a place in these following 2 arrays,
    // and quickly retire empty ones in kernels
    activeBinIDs.resize(simParams->nActiveBins);
    sphereIDsLookUpTable.resize(simParams->nActiveBins);
    numSpheresBinTouches.resize(simParams->nActiveBins);
    granData->activeBinIDs = activeBinIDs.data();
    granData->sphereIDsLookUpTable = sphereIDsLookUpTable.data();
    granData->numSpheresBinTouches = numSpheresBinTouches.data();
    hostScanForJumps<binID_t, binsSphereTouches_t, spheresBinTouches_t>(
        granData->binIDsEachSphereTouches, granData->activeBinIDs, granData->sphereIDsLookUpTable,
        granData->numSpheresBinTouches, granData->numBinsSphereTouches[simParams->nSpheresGM], 2);
    // std::cout << "activeBinIDs: ";
    // displayArray<binID_t>(granData->activeBinIDs, simParams->nActiveBins);
    // std::cout << "numSpheresBinTouches: ";
    // displayArray<spheresBinTouches_t>(granData->numSpheresBinTouches, simParams->nActiveBins);
    // std::cout << "binIDsEachSphereTouches: ";
    // displayArray<binID_t>(granData->binIDsEachSphereTouches, granData->numBinsSphereTouches[simParams->nSpheresGM]);
    // std::cout << "sphereIDsLookUpTable: ";
    // displayArray<binsSphereTouches_t>(granData->sphereIDsLookUpTable, simParams->nActiveBins);
    // std::cout << "sphereIDsEachBinTouches: ";
    // displayArray<bodyID_t>(granData->sphereIDsEachBinTouches, granData->numBinsSphereTouches[simParams->nSpheresGM]);

    // Now find the contact pairs. One-two punch: first find num of contacts in each bin, then pre-scan, then find the
    // actual pair names, (then finally we remove the redundant pairs, through cub??)
    // 1 extra element is given to numContactsInEachBin for an easier prefix scan. Therefore, its last element registers
    // the total number of contact pairs.
    numContactsInEachBin.resize(simParams->nActiveBins + 1);
    granData->numContactsInEachBin = numContactsInEachBin.data();
    size_t blocks_needed_for_bins = (simParams->nActiveBins + NUM_BINS_PER_BLOCK - 1) / NUM_BINS_PER_BLOCK;
    if (blocks_needed_for_bins > 0) {
        auto contact_detection =
            JitHelper::buildProgram("DEMContactKernels", JitHelper::KERNEL_DIR / "DEMContactKernels.cu",
                                    std::vector<JitHelper::Header>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

        contact_detection.kernel("getNumberOfContactsEachBin")
            .instantiate()
            .configure(dim3(blocks_needed_for_bins), dim3(NUM_BINS_PER_BLOCK), sizeof(float) * TEST_SHARED_SIZE * 4,
                       streamInfo.stream)
            .launch(simParams, granData, granTemplates);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        hostPrefixScan<contactPairs_t>(granData->numContactsInEachBin, simParams->nActiveBins + 1);
        // displayArray<contactPairs_t>(granData->numContactsInEachBin, simParams->nActiveBins + 1);
        idGeometryA.resize(granData->numContactsInEachBin[simParams->nActiveBins]);
        idGeometryB.resize(granData->numContactsInEachBin[simParams->nActiveBins]);
        granData->idGeometryA = idGeometryA.data();
        granData->idGeometryB = idGeometryB.data();

        contact_detection.kernel("populateContactPairsEachBin")
            .instantiate()
            .configure(dim3(blocks_needed_for_bins), dim3(NUM_BINS_PER_BLOCK), sizeof(float) * TEST_SHARED_SIZE * 4,
                       streamInfo.stream)
            .launch(simParams, granData, granTemplates);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        // displayArray<bodyID_t>(granData->idGeometryA, granData->numContactsInEachBin[simParams->nActiveBins]);
        // displayArray<bodyID_t>(granData->idGeometryB, granData->numContactsInEachBin[simParams->nActiveBins]);

        simParams->nContactPairs = granData->numContactsInEachBin[simParams->nActiveBins];
    } else {
        simParams->nContactPairs = 0;
    }

*/
}


}
