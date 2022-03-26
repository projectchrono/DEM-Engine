//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cstring>
#include <iostream>
#include <thread>

#include <core/ApiVersion.h>
#include <core/utils/Macros.h>
#include <core/utils/chpf/particle_writer.hpp>
#include <granular/GranularDefines.h>
#include <granular/PhysicsSystem.h>
#include <granular/HostSideHelpers.cpp>
#include <core/utils/JitHelper.h>

#include <algorithms/DEMCubHelperFunctions.h>

namespace sgps {

inline void DEMKinematicThread::contactDetection() {
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

    // hostPrefixScan<binsSphereTouches_t>(granData->numBinsSphereTouches, simParams->nSpheresGM + 1);
    cubPrefixScan(granData->numBinsSphereTouches, granData->numBinsSphereTouchesScan, simParams->nSpheresGM, streamInfo,
                  stateOfSolver_resources);
    stateOfSolver_resources.setTotalBinSphereTouchPairs(
        (size_t)granData->numBinsSphereTouchesScan[simParams->nSpheresGM - 1] +
        (size_t)granData->numBinsSphereTouches[simParams->nSpheresGM - 1]);
    // displayArray<binsSphereTouches_t>(granData->numBinsSphereTouches, simParams->nSpheresGM);
    // displayArray<binsSphereTouchesScan_t>(granData->numBinsSphereTouchesScan, simParams->nSpheresGM);
    // Resize those work arrays to be the size of the number of all sphere--bin pairs
    // After resize we need to reassign pointers in case they changed lengths
    if (stateOfSolver_resources.getTotalBinSphereTouchPairs() > binIDsEachSphereTouches.size()) {
        binIDsEachSphereTouches.resize(stateOfSolver_resources.getTotalBinSphereTouchPairs());
        sphereIDsEachBinTouches.resize(stateOfSolver_resources.getTotalBinSphereTouchPairs());
        granData->binIDsEachSphereTouches = binIDsEachSphereTouches.data();
        granData->sphereIDsEachBinTouches = sphereIDsEachBinTouches.data();
    }

    bin_occupation.kernel("populateBinSphereTouchingPairs")
        .instantiate()
        .configure(dim3(blocks_needed_for_bodies), dim3(NUM_BODIES_PER_BLOCK), sizeof(float) * TEST_SHARED_SIZE * 4,
                   streamInfo.stream)
        .launch(simParams, granData, granTemplates);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    // std::cout << "Unsorted bin IDs: ";
    // displayArray<binID_t>(granData->binIDsEachSphereTouches, stateOfSolver_resources.getTotalBinSphereTouchPairs());
    // std::cout << "Corresponding sphere IDs: ";
    // displayArray<bodyID_t>(granData->sphereIDsEachBinTouches, stateOfSolver_resources.getTotalBinSphereTouchPairs());

    hostSortByKey<binID_t, bodyID_t>(granData->binIDsEachSphereTouches, granData->sphereIDsEachBinTouches,
                                     stateOfSolver_resources.getTotalBinSphereTouchPairs());
    // cubSortByKeys(granData->binIDsEachSphereTouches,
    // granData->sphereIDsEachBinTouches,stateOfSolver_resources.getTotalBinSphereTouchPairs(),streamInfo,stateOfSolver_resources);
    // std::cout << "Sorted bin IDs: ";
    // displayArray<binID_t>(granData->binIDsEachSphereTouches, stateOfSolver_resources.getTotalBinSphereTouchPairs());
    // std::cout << "Corresponding sphere IDs: ";
    // displayArray<bodyID_t>(granData->sphereIDsEachBinTouches, stateOfSolver_resources.getTotalBinSphereTouchPairs());

    // TODO: use cub to do this. Probably one-two punch: first the number of jumps, then jump locations
    // Search for bins that have at least 2 spheres living in.
    // TODO: Is it good to ensure that the bins are non-empty here? If we do it here, we can increase the occupation
    // rate in the contact detection kernels, since it reduces the number of idle threads there (well, maybe this is not
    // even that much, since there will be bins that suffice but has like 2 spheres in it, so they will finish rather
    // quickly); but is it easy to use CUB to ensure each jump has at least 2 relevant elements?
    hostScanForJumpsNum<binID_t>(granData->binIDsEachSphereTouches,
                                 stateOfSolver_resources.getTotalBinSphereTouchPairs(), 2, simParams->nActiveBins);

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
        granData->numSpheresBinTouches, stateOfSolver_resources.getTotalBinSphereTouchPairs(), 2);
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
}

inline void DEMKinematicThread::unpackMyBuffer() {
    GPU_CALL(cudaMemcpy(granData->voxelID, granData->voxelID_buffer, simParams->nClumpBodies * sizeof(voxelID_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->locX, granData->locX_buffer, simParams->nClumpBodies * sizeof(subVoxelPos_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->locY, granData->locY_buffer, simParams->nClumpBodies * sizeof(subVoxelPos_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->locZ, granData->locZ_buffer, simParams->nClumpBodies * sizeof(subVoxelPos_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->oriQ0, granData->oriQ0_buffer, simParams->nClumpBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->oriQ1, granData->oriQ1_buffer, simParams->nClumpBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->oriQ2, granData->oriQ2_buffer, simParams->nClumpBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->oriQ3, granData->oriQ3_buffer, simParams->nClumpBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));
}

inline void DEMKinematicThread::sendToTheirBuffer() {
    GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_nContactPairs, &(simParams->nContactPairs), sizeof(size_t),
                        cudaMemcpyDeviceToDevice));
    // Resize dT owned buffers before usage
    if (simParams->nContactPairs > pDTOwnedVector_idGeometryA->size()) {
        pDTOwnedVector_idGeometryA->resize(simParams->nContactPairs);
        pDTOwnedVector_idGeometryB->resize(simParams->nContactPairs);
        granData->pDTOwnedBuffer_idGeometryA = pDTOwnedVector_idGeometryA->data();
        granData->pDTOwnedBuffer_idGeometryB = pDTOwnedVector_idGeometryB->data();
    }
    GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_idGeometryA, granData->idGeometryA,
                        simParams->nContactPairs * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_idGeometryB, granData->idGeometryB,
                        simParams->nContactPairs * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
}

void DEMKinematicThread::workerThread() {
    // Set the device for this thread
    cudaSetDevice(streamInfo.device);
    cudaStreamCreate(&streamInfo.stream);

    while (!pSchedSupport->kinematicShouldJoin) {
        {
            std::unique_lock<std::mutex> lock(pSchedSupport->kinematicStartLock);
            while (!pSchedSupport->kinematicStarted) {
                pSchedSupport->cv_KinematicStartLock.wait(lock);
            }
            // Ensure that we wait for start signal on next iteration
            pSchedSupport->kinematicStarted = false;
            if (pSchedSupport->kinematicShouldJoin) {
                break;
            }
        }
        // run a while loop producing stuff in each iteration;
        // once produced, it should be made available to the dynamic via memcpy
        while (!pSchedSupport->dynamicDone) {
            // before producing something, a new work order should be in place. Wait on
            // it
            if (!pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh) {
                pSchedSupport->schedulingStats.nTimesKinematicHeldBack++;
                std::unique_lock<std::mutex> lock(pSchedSupport->kinematicCanProceed);

                while (!pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh) {
                    // loop to avoid spurious wakeups
                    pSchedSupport->cv_KinematicCanProceed.wait(lock);
                }

                // getting here means that new "work order" data has been provided
                {
                    // acquire lock and get the work order
                    std::lock_guard<std::mutex> lock(pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
                    unpackMyBuffer();
                }
            }

            // figure out the amount of shared mem
            // cudaDeviceGetAttribute.cudaDevAttrMaxSharedMemoryPerBlock

            // produce something here; fake stuff for now
            // cudaStream_t currentStream;
            // cudaStreamCreate(&currentStream);pSchedSupport->dynamicShouldWait()

            // Two versions here: the host version is just for debugging purposes
            contactDetection();
            // hostContactDetection();

            /* for the reference
            for (int j = 0; j < N_MANUFACTURED_ITEMS; j++) {
                // kinematicTestKernel<<<1, 1, 0, kinematicStream.stream>>>();

                // use cudaLaunchKernel
                // cudaLaunchKernel((void*)&kinematicTestKernel, dim3(1), dim3(1), NULL, 0, stream_id);
                // example argument list:
                //  args = { &arg1, &arg2, ... &argN };
                // cudaLaunchKernel((void*)&kinematicTestKernelWithArgs, dim3(1), dim3(1), &args, 0, stream_id);
                kinematicTestKernel<<<1, 1>>>();
                cudaDeviceSynchronize();
                pSchedSupport->dynamicShouldWait()
                int indx = j % N_INPUT_ITEMS;
                product[j] += this->costlyProductionStep(j) + inputData[indx];
            }
            */

            // make it clear that the data for most recent work order has
            // been used, in case there is interest in updating it
            pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = false;

            {
                // acquire lock and supply the dynamic with fresh produce
                std::lock_guard<std::mutex> lock(pSchedSupport->dynamicOwnedBuffer_AccessCoordination);
                sendToTheirBuffer();
            }
            pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = true;
            pSchedSupport->schedulingStats.nDynamicUpdates++;

            // signal the dynamic that it has fresh produce
            pSchedSupport->cv_DynamicCanProceed.notify_all();
        }

        // in case the dynamic is hanging in there...
        pSchedSupport->cv_DynamicCanProceed.notify_all();

        // When getting here, kT has finished one user call (although perhaps not at the end of the user script).
        userCallDone = true;
    }
}

void DEMKinematicThread::startThread() {
    std::lock_guard<std::mutex> lock(pSchedSupport->kinematicStartLock);
    pSchedSupport->kinematicStarted = true;
    pSchedSupport->cv_KinematicStartLock.notify_one();
}

bool DEMKinematicThread::isUserCallDone() {
    // return true if done, false if not
    return userCallDone;
}

void DEMKinematicThread::resetUserCallStat() {
    userCallDone = false;
    // Reset kT stats variables, making ready for next user call
    pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = false;
}

// Put sim data array pointers in place
void DEMKinematicThread::packDataPointers() {
    granData->voxelID = voxelID.data();
    granData->locX = locX.data();
    granData->locY = locY.data();
    granData->locZ = locZ.data();
    granData->oriQ0 = oriQ0.data();
    granData->oriQ1 = oriQ1.data();
    granData->oriQ2 = oriQ2.data();
    granData->oriQ3 = oriQ3.data();
    granData->idGeometryA = idGeometryA.data();
    granData->idGeometryB = idGeometryB.data();

    // for kT, those state vectors are fed by dT, so each has a buffer
    granData->voxelID_buffer = voxelID_buffer.data();
    granData->locX_buffer = locX_buffer.data();
    granData->locY_buffer = locY_buffer.data();
    granData->locZ_buffer = locZ_buffer.data();
    granData->oriQ0_buffer = oriQ0_buffer.data();
    granData->oriQ1_buffer = oriQ1_buffer.data();
    granData->oriQ2_buffer = oriQ2_buffer.data();
    granData->oriQ3_buffer = oriQ3_buffer.data();

    // The offset info that indexes into the template arrays
    granData->ownerClumpBody = ownerClumpBody.data();
    granData->clumpComponentOffset = clumpComponentOffset.data();

    // kT's own work arrays
    granData->numBinsSphereTouches = numBinsSphereTouches.data();
    granData->numBinsSphereTouchesScan = numBinsSphereTouchesScan.data();
    granData->binIDsEachSphereTouches = binIDsEachSphereTouches.data();
    granData->sphereIDsEachBinTouches = sphereIDsEachBinTouches.data();
    granData->activeBinIDs = activeBinIDs.data();
    granData->sphereIDsLookUpTable = sphereIDsLookUpTable.data();
    granData->numSpheresBinTouches = numSpheresBinTouches.data();
    granData->numContactsInEachBin = numContactsInEachBin.data();

    // Template array pointers, which will be removed after JIT is fully functional
    granTemplates->radiiSphere = radiiSphere.data();
    granTemplates->relPosSphereX = relPosSphereX.data();
    granTemplates->relPosSphereY = relPosSphereY.data();
    granTemplates->relPosSphereZ = relPosSphereZ.data();
    // granTemplates->inflatedRadiiVoxelRatio = inflatedRadiiVoxelRatio.data();
}
void DEMKinematicThread::packTransferPointers(DEMDynamicThread* dT) {
    // Set the pointers to dT owned buffers
    granData->pDTOwnedBuffer_nContactPairs = &(dT->granData->nContactPairs_buffer);
    granData->pDTOwnedBuffer_idGeometryA = dT->granData->idGeometryA_buffer;
    granData->pDTOwnedBuffer_idGeometryB = dT->granData->idGeometryB_buffer;
    // We need to resize dT owned buffer arrays from kT side (because the contact number is not known beforehand), so
    // the pointers to geometry id buffers need to be registered as well. It is a bummer, but I don't know if there is a
    // better solution.
    pDTOwnedVector_idGeometryA = &(dT->idGeometryA_buffer);
    pDTOwnedVector_idGeometryB = &(dT->idGeometryB_buffer);
}

void DEMKinematicThread::setSimParams(unsigned char nvXp2,
                                      unsigned char nvYp2,
                                      unsigned char nvZp2,
                                      float l,
                                      double voxelSize,
                                      double binSize,
                                      float3 LBFPoint,
                                      float3 G,
                                      double ts_size,
                                      float expand_factor) {
    simParams->nvXp2 = nvXp2;
    simParams->nvYp2 = nvYp2;
    simParams->nvZp2 = nvZp2;
    simParams->l = l;
    simParams->voxelSize = voxelSize;
    simParams->binSize = binSize;
    simParams->LBFX = LBFPoint.x;
    simParams->LBFY = LBFPoint.y;
    simParams->LBFZ = LBFPoint.z;
    simParams->Gx = G.x;
    simParams->Gy = G.y;
    simParams->Gz = G.z;
    simParams->h = ts_size;
    simParams->beta = expand_factor;
    // Figure out how many bins there are in each direction
    simParams->nbX = (binID_t)(voxelSize * (double)((size_t)1 << nvXp2) / binSize) + 1;
    simParams->nbY = (binID_t)(voxelSize * (double)((size_t)1 << nvYp2) / binSize) + 1;
    simParams->nbZ = (binID_t)(voxelSize * (double)((size_t)1 << nvZp2) / binSize) + 1;
}

void DEMKinematicThread::allocateManagedArrays(size_t nClumpBodies,
                                               size_t nSpheresGM,
                                               unsigned int nClumpTopo,
                                               unsigned int nClumpComponents,
                                               unsigned int nMatTuples) {
    // Sizes of these arrays
    simParams->nSpheresGM = nSpheresGM;
    simParams->nClumpBodies = nClumpBodies;
    simParams->nDistinctClumpBodyTopologies = nClumpTopo;
    simParams->nDistinctClumpComponents = nClumpComponents;
    simParams->nMatTuples = nMatTuples;

    // Resize to the number of clumps
    TRACKED_VECTOR_RESIZE(voxelID, nClumpBodies, "voxelID", 0);
    TRACKED_VECTOR_RESIZE(locX, nClumpBodies, "locX", 0);
    TRACKED_VECTOR_RESIZE(locY, nClumpBodies, "locY", 0);
    TRACKED_VECTOR_RESIZE(locZ, nClumpBodies, "locZ", 0);
    TRACKED_VECTOR_RESIZE(oriQ0, nClumpBodies, "oriQ0", 1);
    TRACKED_VECTOR_RESIZE(oriQ1, nClumpBodies, "oriQ1", 0);
    TRACKED_VECTOR_RESIZE(oriQ2, nClumpBodies, "oriQ2", 0);
    TRACKED_VECTOR_RESIZE(oriQ3, nClumpBodies, "oriQ3", 0);

    // Transfer buffer arrays
    TRACKED_VECTOR_RESIZE(voxelID_buffer, nClumpBodies, "voxelID_buffer", 0);
    TRACKED_VECTOR_RESIZE(locX_buffer, nClumpBodies, "locX_buffer", 0);
    TRACKED_VECTOR_RESIZE(locY_buffer, nClumpBodies, "locY_buffer", 0);
    TRACKED_VECTOR_RESIZE(locZ_buffer, nClumpBodies, "locZ_buffer", 0);
    TRACKED_VECTOR_RESIZE(oriQ0_buffer, nClumpBodies, "oriQ0_buffer", 0);
    TRACKED_VECTOR_RESIZE(oriQ1_buffer, nClumpBodies, "oriQ1_buffer", 0);
    TRACKED_VECTOR_RESIZE(oriQ2_buffer, nClumpBodies, "oriQ2_buffer", 0);
    TRACKED_VECTOR_RESIZE(oriQ3_buffer, nClumpBodies, "oriQ3_buffer", 0);

    // Resize to the number of spheres
    TRACKED_VECTOR_RESIZE(ownerClumpBody, nSpheresGM, "ownerClumpBody", 0);
    TRACKED_VECTOR_RESIZE(clumpComponentOffset, nSpheresGM, "clumpComponentOffset", 0);
    TRACKED_VECTOR_RESIZE(numBinsSphereTouches, nSpheresGM, "numBinsSphereTouches", 0);
    TRACKED_VECTOR_RESIZE(numBinsSphereTouchesScan, nSpheresGM, "numBinsSphereTouchesScan", 0);
    // The following several arrays will have variable sizes, so here we only used an estimate.
    // TODO: Find a good estimate.
    TRACKED_VECTOR_RESIZE(binIDsEachSphereTouches, nSpheresGM, "binIDsEachSphereTouches", 0);
    TRACKED_VECTOR_RESIZE(sphereIDsEachBinTouches, nSpheresGM, "sphereIDsEachBinTouches", 0);
    TRACKED_VECTOR_RESIZE(activeBinIDs, nSpheresGM, "activeBinIDs", 0);
    TRACKED_VECTOR_RESIZE(sphereIDsLookUpTable, nSpheresGM, "sphereIDsLookUpTable", 0);
    TRACKED_VECTOR_RESIZE(numSpheresBinTouches, nSpheresGM, "numSpheresBinTouches", 0);
    TRACKED_VECTOR_RESIZE(numContactsInEachBin, nSpheresGM, "numContactsInEachBin", 0);

    // Resize to the length of the clump templates
    TRACKED_VECTOR_RESIZE(radiiSphere, nClumpComponents, "radiiSphere", 0);
    TRACKED_VECTOR_RESIZE(relPosSphereX, nClumpComponents, "relPosSphereX", 0);
    TRACKED_VECTOR_RESIZE(relPosSphereY, nClumpComponents, "relPosSphereY", 0);
    TRACKED_VECTOR_RESIZE(relPosSphereZ, nClumpComponents, "relPosSphereZ", 0);
    // TRACKED_VECTOR_RESIZE(inflatedRadiiVoxelRatio, nClumpComponents, "inflatedRadiiVoxelRatio", 0);

    // Arrays for kT produced contact info
    // The following several arrays will have variable sizes, so here we only used an estimate. My estimate of total
    // contact pairs is 4n, and I think the max is 6n (although I can't prove it). Note the estimate should be large
    // enough to decrease the number of reallocations in the simulation, but not too large that eats too much memory.
    TRACKED_VECTOR_RESIZE(idGeometryA, nClumpBodies * 4, "idGeometryA", 0);
    TRACKED_VECTOR_RESIZE(idGeometryB, nClumpBodies * 4, "idGeometryB", 0);
}

void DEMKinematicThread::populateManagedArrays(const std::vector<unsigned int>& input_clump_types,
                                               const std::vector<float3>& input_clump_xyz,
                                               const std::vector<float3>& input_clump_vel,
                                               const std::vector<float>& clumps_mass_types,
                                               const std::vector<std::vector<float>>& clumps_sp_radii_types,
                                               const std::vector<std::vector<float3>>& clumps_sp_location_types) {
    // Use some temporary hacks to get the info in the managed mem
    // All the input vectors should have the same length, nClumpTopo
    unsigned int k = 0;
    std::vector<unsigned int> prescans;

    prescans.push_back(0);
    for (auto elem : clumps_sp_radii_types) {
        for (auto radius : elem) {
            radiiSphere.at(k) = radius;
            // inflatedRadiiVoxelRatio.at(k) = (unsigned int)(radius * simParams->beta / simParams->voxelSize) + 1;
            k++;
        }
        prescans.push_back(k);
    }
    prescans.pop_back();
    k = 0;

    for (auto elem : clumps_sp_location_types) {
        for (auto loc : elem) {
            relPosSphereX.at(k) = loc.x;
            relPosSphereY.at(k) = loc.y;
            relPosSphereZ.at(k) = loc.z;
            k++;
        }
        // std::cout << "sphere location types: " << elem.x << ", " << elem.y << ", " << elem.z << std::endl;
    }
    k = 0;

    for (size_t i = 0; i < input_clump_types.size(); i++) {
        auto type_of_this_clump = input_clump_types.at(i);
        float3 LBF;
        LBF.x = simParams->LBFX;
        LBF.y = simParams->LBFY;
        LBF.z = simParams->LBFZ;
        auto this_CoM_coord = input_clump_xyz.at(i) - LBF;
        auto this_clump_no_sp_radii = clumps_sp_radii_types.at(type_of_this_clump);
        auto this_clump_no_sp_relPos = clumps_sp_location_types.at(type_of_this_clump);

        for (size_t j = 0; j < this_clump_no_sp_radii.size(); j++) {
            clumpComponentOffset.at(k) = prescans.at(type_of_this_clump) + j;
            ownerClumpBody.at(k) = i;
            k++;
        }
    }
}

void DEMKinematicThread::primeDynamic() {
    // transfer produce to dynamic buffer
    // cudaMemcpy(pDTOwnedBuffer_voxelID, voxelID.data(), N_INPUT_ITEMS * sizeof(voxelID_t),
    //            cudaMemcpyDeviceToDevice);
    pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = true;
    pSchedSupport->schedulingStats.nDynamicUpdates++;
}

}  // namespace sgps
