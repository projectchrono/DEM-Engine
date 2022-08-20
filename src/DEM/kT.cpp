//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cstring>
#include <iostream>
#include <thread>

#include <core/ApiVersion.h>
#include <core/utils/JitHelper.h>
#include <DEM/kT.h>
#include <DEM/dT.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/DEMDefines.h>

#include <algorithms/DEMCubBasedSubroutines.h>

namespace sgps {

inline void DEMKinematicThread::transferArraysResize(size_t nContactPairs) {
    // TODO: This memory usage is not tracked... How can I track the size changes on my friend's end??
    dT->idGeometryA_buffer.resize(nContactPairs);
    dT->idGeometryB_buffer.resize(nContactPairs);
    dT->contactType_buffer.resize(nContactPairs);
    SGPS_DEM_ADVISE_DEVICE(dT->idGeometryA_buffer, dT->streamInfo.device);
    SGPS_DEM_ADVISE_DEVICE(dT->idGeometryB_buffer, dT->streamInfo.device);
    SGPS_DEM_ADVISE_DEVICE(dT->contactType_buffer, dT->streamInfo.device);
    granData->pDTOwnedBuffer_idGeometryA = dT->idGeometryA_buffer.data();
    granData->pDTOwnedBuffer_idGeometryB = dT->idGeometryB_buffer.data();
    granData->pDTOwnedBuffer_contactType = dT->contactType_buffer.data();

    if (!solverFlags.isHistoryless) {
        dT->contactMapping_buffer.resize(nContactPairs);
        SGPS_DEM_ADVISE_DEVICE(dT->contactMapping_buffer, dT->streamInfo.device);
        granData->pDTOwnedBuffer_contactMapping = dT->contactMapping_buffer.data();
    }
}

inline void DEMKinematicThread::unpackMyBuffer() {
    GPU_CALL(cudaMemcpy(granData->voxelID, granData->voxelID_buffer, simParams->nOwnerBodies * sizeof(voxelID_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->locX, granData->locX_buffer, simParams->nOwnerBodies * sizeof(subVoxelPos_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->locY, granData->locY_buffer, simParams->nOwnerBodies * sizeof(subVoxelPos_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->locZ, granData->locZ_buffer, simParams->nOwnerBodies * sizeof(subVoxelPos_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->oriQ0, granData->oriQ0_buffer, simParams->nOwnerBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->oriQ1, granData->oriQ1_buffer, simParams->nOwnerBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->oriQ2, granData->oriQ2_buffer, simParams->nOwnerBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->oriQ3, granData->oriQ3_buffer, simParams->nOwnerBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));

    // Family number is a typical changable quantity on-the-fly. If this flag is on, kT received changes from dT.
    if (solverFlags.canFamilyChange) {
        GPU_CALL(cudaMemcpy(granData->familyID, granData->familyID_buffer, simParams->nOwnerBodies * sizeof(family_t),
                            cudaMemcpyDeviceToDevice));
    }
}

inline void DEMKinematicThread::sendToTheirBuffer() {
    GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_nContactPairs, stateOfSolver_resources.pNumContacts, sizeof(size_t),
                        cudaMemcpyDeviceToDevice));
    // Resize dT owned buffers before usage
    if (*stateOfSolver_resources.pNumContacts > dT->idGeometryA_buffer.size()) {
        transferArraysResize(*stateOfSolver_resources.pNumContacts);
    }
    GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_idGeometryA, granData->idGeometryA,
                        (*stateOfSolver_resources.pNumContacts) * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_idGeometryB, granData->idGeometryB,
                        (*stateOfSolver_resources.pNumContacts) * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_contactType, granData->contactType,
                        (*stateOfSolver_resources.pNumContacts) * sizeof(contact_t), cudaMemcpyDeviceToDevice));
    if (!solverFlags.isHistoryless) {
        GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_contactMapping, granData->contactMapping,
                            (*stateOfSolver_resources.pNumContacts) * sizeof(contactPairs_t),
                            cudaMemcpyDeviceToDevice));
    }
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
            // Now dT is running
            workerRunning = true;
            // Ensure that we wait for start signal on next iteration
            pSchedSupport->kinematicStarted = false;
            // The following is executed when kT and dT are being destroyed
            if (pSchedSupport->kinematicShouldJoin) {
                break;
            }
        }

        // Run a while loop producing stuff in each iteration; once produced, it should be made available to the dynamic
        // via memcpy
        while (!pSchedSupport->dynamicDone) {
            // Before producing something, a new work order should be in place. Wait on it.
            if (!pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh) {
                timers.GetTimer("Wait for dT update").start();
                pSchedSupport->schedulingStats.nTimesKinematicHeldBack++;
                std::unique_lock<std::mutex> lock(pSchedSupport->kinematicCanProceed);

                // kT never got locked in here indefinitely because, dT will always send a cv_KinematicCanProceed signal
                // AFTER setting dynamicDone to true, if dT is about to finish
                while (!pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh) {
                    // Loop to avoid spurious wakeups
                    pSchedSupport->cv_KinematicCanProceed.wait(lock);
                }
                timers.GetTimer("Wait for dT update").stop();

                // In the case where this weak-up call is at the destructor (dT has been executing without notifying the
                // end of user calls, aka running DoDynamics), we don't have to do CD one more time, just break
                if (kTShouldReset) {
                    break;
                }
            }

            timers.GetTimer("Unpack updates from dT").start();
            // Getting here means that new `work order' data has been provided
            {
                // Acquire lock and get the work order
                std::lock_guard<std::mutex> lock(pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
                unpackMyBuffer();
            }
            timers.GetTimer("Unpack updates from dT").stop();

            // Make it clear that the data for most recent work order has been used, in case there is interest in
            // updating it
            pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = false;

            // figure out the amount of shared mem
            // cudaDeviceGetAttribute.cudaDevAttrMaxSharedMemoryPerBlock

            // kT's main task
            contactDetection(bin_occupation_kernels, contact_detection_kernels, history_kernels, granData, simParams,
                             solverFlags, verbosity, idGeometryA, idGeometryB, contactType, previous_idGeometryA,
                             previous_idGeometryB, previous_contactType, contactMapping, streamInfo.stream,
                             stateOfSolver_resources, timers);

            timers.GetTimer("Send to dT buffer").start();
            {
                // Acquire lock and supply the dynamic with fresh produce
                std::lock_guard<std::mutex> lock(pSchedSupport->dynamicOwnedBuffer_AccessCoordination);
                sendToTheirBuffer();
            }
            pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = true;
            pSchedSupport->schedulingStats.nDynamicUpdates++;
            timers.GetTimer("Send to dT buffer").stop();

            // Signal the dynamic that it has fresh produce
            pSchedSupport->cv_DynamicCanProceed.notify_all();
        }

        // In case the dynamic is hanging in there...
        pSchedSupport->cv_DynamicCanProceed.notify_all();

        // When getting here, kT has finished one user call (although perhaps not at the end of the user script)
        pPagerToMain->userCallDone = true;
        pPagerToMain->cv_mainCanProceed.notify_all();

        // Now kT finished running
        workerRunning = false;
    }
}

void DEMKinematicThread::getTiming(std::vector<std::string>& names, std::vector<double>& vals) {
    names = timer_names;
    for (const auto& name : timer_names) {
        vals.push_back(timers.GetTimer(name).GetTimeSeconds());
    }
}

void DEMKinematicThread::changeFamily(unsigned int ID_from, unsigned int ID_to) {
    family_t ID_from_impl = familyUserImplMap.at(ID_from);
    family_t ID_to_impl = familyUserImplMap.at(ID_to);
    std::replace_if(
        familyID.begin(), familyID.end(), [ID_from_impl](family_t& i) { return i == ID_from_impl; }, ID_to_impl);
}

void DEMKinematicThread::changeOwnerSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors) {
    // Set the gpu for this thread
    // cudaSetDevice(streamInfo.device);
    // cudaStream_t new_stream;
    // cudaStreamCreate(&new_stream);

    // First get IDs and factors to device side
    size_t IDSize = IDs.size() * sizeof(bodyID_t);
    bodyID_t* dIDs = (bodyID_t*)stateOfSolver_resources.allocateTempVector(1, IDSize);
    GPU_CALL(cudaMemcpy(dIDs, IDs.data(), IDSize, cudaMemcpyHostToDevice));
    size_t factorSize = factors.size() * sizeof(float);
    float* dFactors = (float*)stateOfSolver_resources.allocateTempVector(2, factorSize);
    GPU_CALL(cudaMemcpy(dFactors, factors.data(), factorSize, cudaMemcpyHostToDevice));

    size_t idBoolSize = (size_t)simParams->nOwnerBodies * sizeof(notStupidBool_t);
    size_t ownerFactorSize = (size_t)simParams->nOwnerBodies * sizeof(float);
    // Bool table for whether this owner should change
    notStupidBool_t* idBool = (notStupidBool_t*)stateOfSolver_resources.allocateTempVector(3, idBoolSize);
    GPU_CALL(cudaMemset(idBool, 0, idBoolSize));
    float* ownerFactors = (float*)stateOfSolver_resources.allocateTempVector(4, ownerFactorSize);
    size_t blocks_needed_for_marking =
        (IDs.size() + SGPS_DEM_MAX_THREADS_PER_BLOCK - 1) / SGPS_DEM_MAX_THREADS_PER_BLOCK;

    // Mark on the bool array those owners that need a change
    misc_kernels->kernel("markOwnerToChange")
        .instantiate()
        .configure(dim3(blocks_needed_for_marking), dim3(SGPS_DEM_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
        .launch(idBool, ownerFactors, dIDs, dFactors, IDs.size());
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

    // Change the size of the sphere components in question
    size_t blocks_needed_for_changing =
        (simParams->nSpheresGM + SGPS_DEM_MAX_THREADS_PER_BLOCK - 1) / SGPS_DEM_MAX_THREADS_PER_BLOCK;
    misc_kernels->kernel("kTModifyComponents")
        .instantiate()
        .configure(dim3(blocks_needed_for_changing), dim3(SGPS_DEM_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
        .launch(granData, idBool, ownerFactors, simParams->nSpheresGM);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

    // cudaStreamDestroy(new_stream);
}

void DEMKinematicThread::startThread() {
    std::lock_guard<std::mutex> lock(pSchedSupport->kinematicStartLock);
    pSchedSupport->kinematicStarted = true;
    pSchedSupport->cv_KinematicStartLock.notify_one();
}

void DEMKinematicThread::breakWaitingStatus() {
    // dynamicDone == true and cv_KinematicCanProceed should ensure kT breaks to the outer loop
    pSchedSupport->dynamicDone = true;
    // We distrubed kinematicOwned_Cons2ProdBuffer_isFresh and kTShouldReset here, but it matters not, as when
    // breakWaitingStatus is called, they will always be reset to default soon
    pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = true;
    kTShouldReset = true;

    std::lock_guard<std::mutex> lock(pSchedSupport->kinematicCanProceed);
    pSchedSupport->cv_KinematicCanProceed.notify_one();
}

void DEMKinematicThread::resetUserCallStat() {
    // Reset kT stats variables, making ready for next user call
    pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = false;
    kTShouldReset = false;
}

size_t DEMKinematicThread::estimateMemUsage() const {
    return m_approx_bytes_used;
}

// Put sim data array pointers in place
void DEMKinematicThread::packDataPointers() {
    granData->familyID = familyID.data();
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
    granData->contactType = contactType.data();
    granData->previous_idGeometryA = previous_idGeometryA.data();
    granData->previous_idGeometryB = previous_idGeometryB.data();
    granData->previous_contactType = previous_contactType.data();
    granData->contactMapping = contactMapping.data();

    // for kT, those state vectors are fed by dT, so each has a buffer
    granData->voxelID_buffer = voxelID_buffer.data();
    granData->locX_buffer = locX_buffer.data();
    granData->locY_buffer = locY_buffer.data();
    granData->locZ_buffer = locZ_buffer.data();
    granData->oriQ0_buffer = oriQ0_buffer.data();
    granData->oriQ1_buffer = oriQ1_buffer.data();
    granData->oriQ2_buffer = oriQ2_buffer.data();
    granData->oriQ3_buffer = oriQ3_buffer.data();
    granData->familyID_buffer = familyID_buffer.data();

    // The offset info that indexes into the template arrays
    granData->ownerClumpBody = ownerClumpBody.data();
    granData->clumpComponentOffset = clumpComponentOffset.data();
    granData->clumpComponentOffsetExt = clumpComponentOffsetExt.data();

    // Template array pointers
    granData->radiiSphere = radiiSphere.data();
    granData->relPosSphereX = relPosSphereX.data();
    granData->relPosSphereY = relPosSphereY.data();
    granData->relPosSphereZ = relPosSphereZ.data();
}

void DEMKinematicThread::packTransferPointers(DEMDynamicThread* dT) {
    // Set the pointers to dT owned buffers
    granData->pDTOwnedBuffer_nContactPairs = &(dT->granData->nContactPairs_buffer);
    granData->pDTOwnedBuffer_idGeometryA = dT->idGeometryA_buffer.data();
    granData->pDTOwnedBuffer_idGeometryB = dT->idGeometryB_buffer.data();
    granData->pDTOwnedBuffer_contactType = dT->contactType_buffer.data();
    granData->pDTOwnedBuffer_contactMapping = dT->contactMapping_buffer.data();
}

void DEMKinematicThread::setSimParams(unsigned char nvXp2,
                                      unsigned char nvYp2,
                                      unsigned char nvZp2,
                                      float l,
                                      double voxelSize,
                                      double binSize,
                                      binID_t nbX,
                                      binID_t nbY,
                                      binID_t nbZ,
                                      float3 LBFPoint,
                                      float3 G,
                                      double ts_size,
                                      float expand_factor,
                                      float approx_max_vel,
                                      float expand_safety_param) {
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
    simParams->approxMaxVel = approx_max_vel;
    simParams->expSafetyParam = expand_safety_param;
    simParams->nbX = nbX;
    simParams->nbY = nbY;
    simParams->nbZ = nbZ;
}

void DEMKinematicThread::allocateManagedArrays(size_t nOwnerBodies,
                                               size_t nOwnerClumps,
                                               unsigned int nExtObj,
                                               size_t nTriEntities,
                                               size_t nSpheresGM,
                                               size_t nTriGM,
                                               unsigned int nAnalGM,
                                               unsigned int nMassProperties,
                                               unsigned int nClumpTopo,
                                               unsigned int nClumpComponents,
                                               unsigned int nJitifiableClumpComponents,
                                               unsigned int nMatTuples) {
    // Sizes of these arrays
    simParams->nSpheresGM = nSpheresGM;
    simParams->nTriGM = nTriGM;
    simParams->nAnalGM = nAnalGM;
    simParams->nOwnerBodies = nOwnerBodies;
    simParams->nOwnerClumps = nOwnerClumps;
    simParams->nExtObj = nExtObj;
    simParams->nTriEntities = nTriEntities;
    simParams->nDistinctMassProperties = nMassProperties;
    simParams->nDistinctClumpBodyTopologies = nClumpTopo;
    simParams->nJitifiableClumpComponents = nJitifiableClumpComponents;
    simParams->nDistinctClumpComponents = nClumpComponents;
    simParams->nMatTuples = nMatTuples;

    // Resize to the number of clumps
    SGPS_DEM_TRACKED_RESIZE(familyID, nOwnerBodies, "familyID", 0);
    SGPS_DEM_TRACKED_RESIZE(voxelID, nOwnerBodies, "voxelID", 0);
    SGPS_DEM_TRACKED_RESIZE(locX, nOwnerBodies, "locX", 0);
    SGPS_DEM_TRACKED_RESIZE(locY, nOwnerBodies, "locY", 0);
    SGPS_DEM_TRACKED_RESIZE(locZ, nOwnerBodies, "locZ", 0);
    SGPS_DEM_TRACKED_RESIZE(oriQ0, nOwnerBodies, "oriQ0", 1);
    SGPS_DEM_TRACKED_RESIZE(oriQ1, nOwnerBodies, "oriQ1", 0);
    SGPS_DEM_TRACKED_RESIZE(oriQ2, nOwnerBodies, "oriQ2", 0);
    SGPS_DEM_TRACKED_RESIZE(oriQ3, nOwnerBodies, "oriQ3", 0);

    // Transfer buffer arrays
    SGPS_DEM_TRACKED_RESIZE(voxelID_buffer, nOwnerBodies, "voxelID_buffer", 0);
    SGPS_DEM_TRACKED_RESIZE(locX_buffer, nOwnerBodies, "locX_buffer", 0);
    SGPS_DEM_TRACKED_RESIZE(locY_buffer, nOwnerBodies, "locY_buffer", 0);
    SGPS_DEM_TRACKED_RESIZE(locZ_buffer, nOwnerBodies, "locZ_buffer", 0);
    SGPS_DEM_TRACKED_RESIZE(oriQ0_buffer, nOwnerBodies, "oriQ0_buffer", 0);
    SGPS_DEM_TRACKED_RESIZE(oriQ1_buffer, nOwnerBodies, "oriQ1_buffer", 0);
    SGPS_DEM_TRACKED_RESIZE(oriQ2_buffer, nOwnerBodies, "oriQ2_buffer", 0);
    SGPS_DEM_TRACKED_RESIZE(oriQ3_buffer, nOwnerBodies, "oriQ3_buffer", 0);

    SGPS_DEM_ADVISE_DEVICE(voxelID_buffer, dT->streamInfo.device);
    SGPS_DEM_ADVISE_DEVICE(locX_buffer, dT->streamInfo.device);
    SGPS_DEM_ADVISE_DEVICE(locY_buffer, dT->streamInfo.device);
    SGPS_DEM_ADVISE_DEVICE(locZ_buffer, dT->streamInfo.device);
    SGPS_DEM_ADVISE_DEVICE(oriQ0_buffer, dT->streamInfo.device);
    SGPS_DEM_ADVISE_DEVICE(oriQ1_buffer, dT->streamInfo.device);
    SGPS_DEM_ADVISE_DEVICE(oriQ2_buffer, dT->streamInfo.device);
    SGPS_DEM_ADVISE_DEVICE(oriQ3_buffer, dT->streamInfo.device);
    if (solverFlags.canFamilyChange) {
        SGPS_DEM_TRACKED_RESIZE(familyID_buffer, nOwnerBodies, "familyID_buffer", 0);
        SGPS_DEM_ADVISE_DEVICE(familyID_buffer, dT->streamInfo.device);
    }

    // Resize to the number of spheres
    SGPS_DEM_TRACKED_RESIZE(ownerClumpBody, nSpheresGM + nTriGM, "ownerClumpBody", 0);

    if (solverFlags.useClumpJitify) {
        SGPS_DEM_TRACKED_RESIZE(clumpComponentOffset, nSpheresGM, "clumpComponentOffset", 0);
        // This extended component offset array can hold offset numbers even for big clumps (whereas
        // clumpComponentOffset is typically uint_8, so it may not). If a sphere's component offset index falls in this
        // range then it is not jitified, and the kernel needs to look for it in the global memory.
        SGPS_DEM_TRACKED_RESIZE(clumpComponentOffsetExt, nSpheresGM, "clumpComponentOffsetExt", 0);
        // Resize to the length of the clump templates
        SGPS_DEM_TRACKED_RESIZE(radiiSphere, nClumpComponents, "radiiSphere", 0);
        SGPS_DEM_TRACKED_RESIZE(relPosSphereX, nClumpComponents, "relPosSphereX", 0);
        SGPS_DEM_TRACKED_RESIZE(relPosSphereY, nClumpComponents, "relPosSphereY", 0);
        SGPS_DEM_TRACKED_RESIZE(relPosSphereZ, nClumpComponents, "relPosSphereZ", 0);
    } else {
        SGPS_DEM_TRACKED_RESIZE(radiiSphere, nSpheresGM, "radiiSphere", 0);
        SGPS_DEM_TRACKED_RESIZE(relPosSphereX, nSpheresGM, "relPosSphereX", 0);
        SGPS_DEM_TRACKED_RESIZE(relPosSphereY, nSpheresGM, "relPosSphereY", 0);
        SGPS_DEM_TRACKED_RESIZE(relPosSphereZ, nSpheresGM, "relPosSphereZ", 0);
    }

    // Arrays for kT produced contact info
    // The following several arrays will have variable sizes, so here we only used an estimate. My estimate of total
    // contact pairs is 2n, and I think the max is 6n (although I can't prove it). Note the estimate should be large
    // enough to decrease the number of reallocations in the simulation, but not too large that eats too much memory.
    SGPS_DEM_TRACKED_RESIZE(idGeometryA, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "idGeometryA", 0);
    SGPS_DEM_TRACKED_RESIZE(idGeometryB, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "idGeometryB", 0);
    SGPS_DEM_TRACKED_RESIZE(contactType, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "contactType", DEM_NOT_A_CONTACT);
    if (!solverFlags.isHistoryless) {
        SGPS_DEM_TRACKED_RESIZE(previous_idGeometryA, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER,
                                "previous_idGeometryA", 0);
        SGPS_DEM_TRACKED_RESIZE(previous_idGeometryB, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER,
                                "previous_idGeometryB", 0);
        SGPS_DEM_TRACKED_RESIZE(previous_contactType, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER,
                                "previous_contactType", DEM_NOT_A_CONTACT);
        SGPS_DEM_TRACKED_RESIZE(contactMapping, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "contactMapping",
                                DEM_NULL_MAPPING_PARTNER);
    }
}

void DEMKinematicThread::registerPolicies(const std::unordered_map<unsigned int, family_t>& family_user_impl_map,
                                          const std::unordered_map<family_t, unsigned int>& family_impl_user_map) {
    // kT may need this family number map, store it
    familyUserImplMap = family_user_impl_map;
    familyImplUserMap = family_impl_user_map;
}

void DEMKinematicThread::populateEntityArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                              const std::vector<unsigned int>& input_ext_obj_family,
                                              const std::vector<unsigned int>& input_mesh_obj_family,
                                              const std::unordered_map<unsigned int, family_t>& family_user_impl_map,
                                              const std::unordered_map<family_t, unsigned int>& family_impl_user_map,
                                              const std::vector<float>& clumps_mass_types,
                                              const std::vector<std::vector<float>>& clumps_sp_radii_types,
                                              const std::vector<std::vector<float3>>& clumps_sp_location_types,
                                              size_t nExistOwners,
                                              size_t nExistSpheres) {
    // All the input vectors should have the same length, nClumpTopo
    size_t k = 0;
    std::vector<unsigned int> prescans_comp;

    if (solverFlags.useClumpJitify) {
        prescans_comp.push_back(0);
        for (auto elem : clumps_sp_radii_types) {
            for (auto radius : elem) {
                radiiSphere.at(k) = radius;
                k++;
            }
            prescans_comp.push_back(k);
        }
        prescans_comp.pop_back();
        k = 0;

        for (auto elem : clumps_sp_location_types) {
            for (auto loc : elem) {
                relPosSphereX.at(k) = loc.x;
                relPosSphereY.at(k) = loc.y;
                relPosSphereZ.at(k) = loc.z;
                k++;
            }
        }
    }

    k = 0;
    // float3 LBF;
    // LBF.x = simParams->LBFX;
    // LBF.y = simParams->LBFY;
    // LBF.z = simParams->LBFZ;
    // Now load clump init info
    std::vector<unsigned int> input_clump_types;
    {
        std::vector<unsigned int> input_clump_family;
        // Flatten the input clump batches (because by design we transfer flatten clump info to GPU)
        for (const auto& a_batch : input_clump_batches) {
            // Decode type number and flatten
            std::vector<unsigned int> type_marks(a_batch->GetNumClumps());
            for (size_t i = 0; i < a_batch->GetNumClumps(); i++) {
                type_marks.at(i) = a_batch->types.at(i)->mark;
            }
            input_clump_types.insert(input_clump_types.end(), type_marks.begin(), type_marks.end());
            input_clump_family.insert(input_clump_family.end(), a_batch->families.begin(), a_batch->families.end());
        }

        for (size_t i = 0; i < input_clump_types.size(); i++) {
            auto type_of_this_clump = input_clump_types.at(i);

            // auto this_CoM_coord = input_clump_xyz.at(i) - LBF; // kT don't have to init owner xyz
            auto this_clump_no_sp_radii = clumps_sp_radii_types.at(type_of_this_clump);
            auto this_clump_no_sp_relPos = clumps_sp_location_types.at(type_of_this_clump);

            for (size_t j = 0; j < this_clump_no_sp_radii.size(); j++) {
                ownerClumpBody.at(nExistSpheres + k) = nExistOwners + i;

                // Depending on whether we jitify or flatten
                if (solverFlags.useClumpJitify) {
                    // This component offset, is it too large that can't live in the jitified array?
                    unsigned int this_comp_offset = prescans_comp.at(type_of_this_clump) + j;
                    clumpComponentOffsetExt.at(nExistSpheres + k) = this_comp_offset;
                    if (this_comp_offset < simParams->nJitifiableClumpComponents) {
                        clumpComponentOffset.at(nExistSpheres + k) = this_comp_offset;
                    } else {
                        // If not, an indicator will be put there
                        clumpComponentOffset.at(nExistSpheres + k) = DEM_RESERVED_CLUMP_COMPONENT_OFFSET;
                    }
                } else {
                    radiiSphere.at(nExistSpheres + k) = this_clump_no_sp_radii.at(j);
                    const float3 relPos = this_clump_no_sp_relPos.at(j);
                    relPosSphereX.at(nExistSpheres + k) = relPos.x;
                    relPosSphereY.at(nExistSpheres + k) = relPos.y;
                    relPosSphereZ.at(nExistSpheres + k) = relPos.z;
                }

                k++;
            }

            family_t this_family_num = familyUserImplMap.at(input_clump_family.at(i));
            familyID.at(nExistOwners + i) = this_family_num;
        }
    }

    size_t offset_for_ext_obj = input_clump_types.size();
    for (size_t i = 0; i < input_ext_obj_family.size(); i++) {
        family_t this_family_num = familyUserImplMap.at(input_ext_obj_family.at(i));
        familyID.at(i + offset_for_ext_obj) = this_family_num;
    }
}

void DEMKinematicThread::initManagedArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                           const std::vector<unsigned int>& input_ext_obj_family,
                                           const std::vector<unsigned int>& input_mesh_obj_family,
                                           const std::unordered_map<unsigned int, family_t>& family_user_impl_map,
                                           const std::unordered_map<family_t, unsigned int>& family_impl_user_map,
                                           const std::vector<float>& clumps_mass_types,
                                           const std::vector<std::vector<float>>& clumps_sp_radii_types,
                                           const std::vector<std::vector<float3>>& clumps_sp_location_types) {
    // Get the info into the managed memory from the host side. Can this process be more efficient? Maybe, but it's
    // initialization anyway.

    registerPolicies(family_user_impl_map, family_impl_user_map);

    populateEntityArrays(input_clump_batches, input_ext_obj_family, input_mesh_obj_family, family_user_impl_map,
                         family_impl_user_map, clumps_mass_types, clumps_sp_radii_types, clumps_sp_location_types, 0,
                         0);
}

void DEMKinematicThread::updateClumpMeshArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                               const std::vector<unsigned int>& input_ext_obj_family,
                                               const std::vector<unsigned int>& input_mesh_obj_family,
                                               const std::unordered_map<unsigned int, family_t>& family_user_impl_map,
                                               const std::unordered_map<family_t, unsigned int>& family_impl_user_map,
                                               const std::vector<float>& clumps_mass_types,
                                               const std::vector<std::vector<float>>& clumps_sp_radii_types,
                                               const std::vector<std::vector<float3>>& clumps_sp_location_types,
                                               size_t nExistingOwners,
                                               size_t nExistingClumps,
                                               size_t nExistingSpheres,
                                               size_t nExistingTriMesh,
                                               size_t nExistingFacets) {
    populateEntityArrays(input_clump_batches, input_ext_obj_family, input_mesh_obj_family, family_user_impl_map,
                         family_impl_user_map, clumps_mass_types, clumps_sp_radii_types, clumps_sp_location_types,
                         nExistingOwners, nExistingSpheres);
}

void DEMKinematicThread::jitifyKernels(const std::unordered_map<std::string, std::string>& Subs) {
    // First one is bin_occupation_kernels kernels, which figure out the bin--sphere touch pairs
    {
        bin_occupation_kernels = std::make_shared<jitify::Program>(
            std::move(JitHelper::buildProgram("DEMBinSphereKernels", JitHelper::KERNEL_DIR / "DEMBinSphereKernels.cu",
                                              Subs, {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
    // Then CD kernels
    if (solverFlags.useOneBinPerThread) {
        contact_detection_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMContactKernels", JitHelper::KERNEL_DIR / "DEMContactKernels.cu", Subs,
            {"-I" + (JitHelper::KERNEL_DIR / "..").string(), "-I/opt/apps/cuda/x86_64/11.6.0/default/include"})));
    } else {
        contact_detection_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMContactKernels_Blockwise", JitHelper::KERNEL_DIR / "DEMContactKernels_Blockwise.cu", Subs,
            {"-I" + (JitHelper::KERNEL_DIR / "..").string(), "-I/opt/apps/cuda/x86_64/11.6.0/default/include"})));
    }
    // Then contact history mapping kernels
    {
        history_kernels = std::make_shared<jitify::Program>(std::move(
            JitHelper::buildProgram("DEMHistoryMappingKernels", JitHelper::KERNEL_DIR / "DEMHistoryMappingKernels.cu",
                                    Subs, {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
    // Then misc kernels
    {
        misc_kernels = std::make_shared<jitify::Program>(
            std::move(JitHelper::buildProgram("DEMMiscKernels", JitHelper::KERNEL_DIR / "DEMMiscKernels.cu", Subs,
                                              {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
}

}  // namespace sgps
