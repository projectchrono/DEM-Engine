//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <cstring>
#include <iostream>
#include <thread>

#include <core/ApiVersion.h>
#include <core/utils/JitHelper.h>
#include <DEM/kT.h>
#include "dT.h"
#include "HostSideHelpers.hpp"
#include "Defines.h"
#include "algorithms/DEMStaticDeviceSubroutines.h"
#include "kernel/DEMHelperKernels.cuh"

namespace deme {

inline void DEMKinematicThread::transferArraysResize(size_t nContactPairs) {
    // These buffers are on dT
    DEME_GPU_CALL(cudaSetDevice(dT->streamInfo.device));
    dT->buffer_size = nContactPairs;
    DEME_DEVICE_ARRAY_RESIZE(dT->idGeometryA_buffer, nContactPairs);
    DEME_DEVICE_ARRAY_RESIZE(dT->idGeometryB_buffer, nContactPairs);
    DEME_DEVICE_ARRAY_RESIZE(dT->contactType_buffer, nContactPairs);
    granData->pDTOwnedBuffer_idGeometryA = dT->idGeometryA_buffer.data();
    granData->pDTOwnedBuffer_idGeometryB = dT->idGeometryB_buffer.data();
    granData->pDTOwnedBuffer_contactType = dT->contactType_buffer.data();

    if (!solverFlags.isHistoryless) {
        DEME_DEVICE_ARRAY_RESIZE(dT->contactMapping_buffer, nContactPairs);
        granData->pDTOwnedBuffer_contactMapping = dT->contactMapping_buffer.data();
    }
    // Unset the device change we just made
    DEME_GPU_CALL(cudaSetDevice(streamInfo.device));

    // But we don't have to toDevice granData or dT->granData, and this is because all _buffer arrays don't
    // particupate kernel computations, so even if their values are fresh only on host, it's fine
}

void DEMKinematicThread::calibrateParams() {
    double prev_time, curr_time;
    // If it is true, then it's the AccumTimer telling us it is the right time to decide how to change bin size
    if (CDAccumTimer.QueryOn(prev_time, curr_time, stateParams.binChangeObserveSteps)) {
        // Auto-adjust bin size
        if (solverFlags.autoBinSize) {
            int speed_dir = sign_func(stateParams.binCurrentChangeRate);
            // Note the speed can be 0, yet we find performance variance. Then this is purely noise. We still wish the
            // bin size to change in the next iteration, so we assign a direction randomly.
            if (speed_dir == 0)
                speed_dir = (randomZeroOrOne() == 0) ? -1 : 1;
            float speed_update;
            if (curr_time < prev_time) {
                // If there is improvement, then we accelerate the current change direction
                speed_update = speed_dir * stateParams.binChangeRateAcc * stateParams.binTopChangeRate;
            } else {
                // If no improvement, revert the direction
                speed_update = -speed_dir * stateParams.binChangeRateAcc * stateParams.binTopChangeRate;
            }
            // But, if the bin size is going to get too big or too small, a penalty is enforced
            if (stateParams.maxSphFoundInBin > stateParams.binChangeUpperSafety * simParams->errOutBinSphNum ||
                stateParams.maxTriFoundInBin > stateParams.binChangeUpperSafety * simParams->errOutBinTriNum) {
                // Then the size must start to decrease
                speed_update = -1.0 * stateParams.binChangeRateAcc * stateParams.binTopChangeRate;
            }
            if (stateParams.numBins >
                stateParams.binChangeLowerSafety * (double)(std::numeric_limits<binID_t>::max())) {
                // Then size must start to increase
                speed_update = 1.0 * stateParams.binChangeRateAcc * stateParams.binTopChangeRate;
            }

            // Acc is done. Now apply it to bin size change speed
            stateParams.binCurrentChangeRate += speed_update;
            // But, the speed must fall in range
            stateParams.binCurrentChangeRate = clampBetween(
                stateParams.binCurrentChangeRate, -stateParams.binTopChangeRate, stateParams.binTopChangeRate);

            // Change bin size
            if (stateParams.binCurrentChangeRate > 0) {
                simParams->binSize *= (1. + stateParams.binCurrentChangeRate);
            } else {
                simParams->binSize /= (1. - stateParams.binCurrentChangeRate);
            }
            // Register the new bin size
            stateParams.numBins =
                hostCalcBinNum(simParams->nbX, simParams->nbY, simParams->nbZ, simParams->voxelSize, simParams->binSize,
                               simParams->nvXp2, simParams->nvYp2, simParams->nvZp2);

            DEME_DEBUG_PRINTF("Bin size is now: %.7g", simParams->binSize);
            DEME_DEBUG_PRINTF("Total num of bins is now: %zu", stateParams.numBins);
        }
        DEME_DEBUG_PRINTF("kT runtime per step: %.7gs", CDAccumTimer.GetPrevTime());
    }
    // binSize is now calculated, we need to migrate that to device
    // simParams.syncMemberToDevice<double>(offsetof(DEMSimParams, binSize));
    simParams.toDevice();
}

inline void DEMKinematicThread::unpackMyBuffer() {
    DEME_GPU_CALL(cudaMemcpy(granData->voxelID, voxelID_buffer.data(), simParams->nOwnerBodies * sizeof(voxelID_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->locX, locX_buffer.data(), simParams->nOwnerBodies * sizeof(subVoxelPos_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->locY, locY_buffer.data(), simParams->nOwnerBodies * sizeof(subVoxelPos_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->locZ, locZ_buffer.data(), simParams->nOwnerBodies * sizeof(subVoxelPos_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->oriQw, oriQ0_buffer.data(), simParams->nOwnerBodies * sizeof(oriQ_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->oriQx, oriQ1_buffer.data(), simParams->nOwnerBodies * sizeof(oriQ_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->oriQy, oriQ2_buffer.data(), simParams->nOwnerBodies * sizeof(oriQ_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->oriQz, oriQ3_buffer.data(), simParams->nOwnerBodies * sizeof(oriQ_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->marginSize, absVel_buffer.data(), simParams->nOwnerBodies * sizeof(float),
                             cudaMemcpyDeviceToDevice));

    DEME_GPU_CALL(cudaMemcpy(&(stateParams.ts), &(stateParams.ts_buffer), sizeof(double), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(&(stateParams.maxDrift), &(stateParams.maxDrift_buffer), sizeof(unsigned int),
                             cudaMemcpyDeviceToDevice));

    // Whatever drift value dT says, kT listens; unless kinematicMaxFutureDrift is negative in which case the user
    // explicitly said not caring the future drift.
    stateParams.maxDrift.toHost();
    pSchedSupport->kinematicMaxFutureDrift = (pSchedSupport->kinematicMaxFutureDrift.load() < 0.)
                                                 ? pSchedSupport->kinematicMaxFutureDrift.load()
                                                 : *(stateParams.maxDrift);

    // Need to reduce to check if max velocity is exceeded (right now, array marginSize is still storing absv...)
    cubMaxReduce<float>(granData->marginSize, &(stateParams.maxVel), simParams->nOwnerBodies, streamInfo.stream,
                        solverScratchSpace);
    // Get the reduced maxVel value
    stateParams.maxVel.toHost();
    if (*(stateParams.maxVel) > simParams->errOutVel || (!std::isfinite(*(stateParams.maxVel)))) {
        DEME_ERROR(
            "System max velocity is %.7g, exceeded max allowance (%.7g).\nIf this velocity is not abnormal and you "
            "want to increase this allowance, use SetErrorOutVelocity before initializing simulation.\nOtherwise, the "
            "simulation may have diverged and relaxing the physics may help, such as decreasing the step size and "
            "modifying material properties.\nIf this happens at the start of simulation, check if there are initial "
            "penetrations, a.k.a. elements initialized inside walls.",
            *(stateParams.maxVel), simParams->errOutVel);
    } else if (*(stateParams.maxVel) >
               simParams->approxMaxVel) {  // If maxVel is larger than the user estimation, that is an anomaly
        DEME_STEP_ANOMALY("Simulation entity velocity reached %.6g, over the user-estimated %.6g",
                          *(stateParams.maxVel), simParams->approxMaxVel);
        anomalies.over_max_vel = true;
    }

    // kT will need to derive the thickness of the CD margin, based on dT's info on system vel.
    if (!solverFlags.isExpandFactorFixed) {
        // This kernel will turn absv to marginSize, and if a vel is over max, it will clamp it.
        // Converting to size_t is SUPER important... CUDA kernel call basically does not have type conversion.
        size_t blocks_needed = (simParams->nOwnerBodies + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
        misc_kernels->kernel("computeMarginFromAbsv")
            .instantiate()
            .configure(dim3(blocks_needed), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
            .launch(&simParams, &granData, &(stateParams.ts), &(stateParams.maxDrift),
                    (size_t)(simParams->nOwnerBodies));
        DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    } else {  // If isExpandFactorFixed, then just fill in that constant array.
        size_t blocks_needed = (simParams->nOwnerBodies + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
        misc_kernels->kernel("fillMarginValues")
            .instantiate()
            .configure(dim3(blocks_needed), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
            .launch(&simParams, &granData, (size_t)(simParams->nOwnerBodies));
        DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    }

    DEME_DEBUG_PRINTF("kT received a velocity update: %.6g", *(stateParams.maxVel));
    DEME_DEBUG_PRINTF("kT received roughly %zu bytes in clump states transfer",
                      (size_t)simParams->nOwnerBodies * sizeof(voxelID_t) +
                          simParams->nOwnerBodies * sizeof(subVoxelPos_t) * 3 +
                          simParams->nOwnerBodies * sizeof(oriQ_t) * 4 + simParams->nOwnerBodies * sizeof(float));
    // DEME_DEBUG_PRINTF("A margin of thickness %.6g is added", simParams->beta);

    // Family number is a typical changable quantity on-the-fly. If this flag is on, kT received changes from dT.
    if (solverFlags.canFamilyChangeOnDevice) {
        DEME_GPU_CALL(cudaMemcpy(granData->familyID, familyID_buffer.data(), simParams->nOwnerBodies * sizeof(family_t),
                                 cudaMemcpyDeviceToDevice));
    }

    // If dT received a mesh deformation request from user, then it is now passed to kT
    if (solverFlags.willMeshDeform) {
        DEME_DEBUG_PRINTF("kT is unpacking a triangle soup update...");
        DEME_GPU_CALL(cudaMemcpy(granData->relPosNode1, relPosNode1_buffer.data(), simParams->nTriGM * sizeof(float3),
                                 cudaMemcpyDeviceToDevice));
        DEME_GPU_CALL(cudaMemcpy(granData->relPosNode2, relPosNode2_buffer.data(), simParams->nTriGM * sizeof(float3),
                                 cudaMemcpyDeviceToDevice));
        DEME_GPU_CALL(cudaMemcpy(granData->relPosNode3, relPosNode3_buffer.data(), simParams->nTriGM * sizeof(float3),
                                 cudaMemcpyDeviceToDevice));
        // dT won't be sending if kT is loading, so it is safe
        solverFlags.willMeshDeform = false;
        DEME_DEBUG_PRINTF("kT received roughly %zu in this triangle soup update",
                          simParams->nTriGM * sizeof(float3) * 3);
    }
}

inline void DEMKinematicThread::sendToTheirBuffer() {
    DEME_GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_nContactPairs, &(solverScratchSpace.numContacts), sizeof(size_t),
                             cudaMemcpyDeviceToDevice));
    // Resize dT owned buffers before usage
    if (*solverScratchSpace.numContacts > dT->buffer_size) {
        transferArraysResize(*solverScratchSpace.numContacts);
    }

    DEME_GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_idGeometryA, granData->idGeometryA,
                             (*solverScratchSpace.numContacts) * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_idGeometryB, granData->idGeometryB,
                             (*solverScratchSpace.numContacts) * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_contactType, granData->contactType,
                             (*solverScratchSpace.numContacts) * sizeof(contact_t), cudaMemcpyDeviceToDevice));
    // DEME_MIGRATE_TO_DEVICE(dT->idGeometryA_buffer, dT->streamInfo.device, streamInfo.stream);
    // DEME_MIGRATE_TO_DEVICE(dT->idGeometryB_buffer, dT->streamInfo.device, streamInfo.stream);
    // DEME_MIGRATE_TO_DEVICE(dT->contactType_buffer, dT->streamInfo.device, streamInfo.stream);
    if (!solverFlags.isHistoryless) {
        DEME_GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_contactMapping, granData->contactMapping,
                                 (*solverScratchSpace.numContacts) * sizeof(contactPairs_t), cudaMemcpyDeviceToDevice));
        // DEME_MIGRATE_TO_DEVICE(dT->contactMapping_buffer, dT->streamInfo.device, streamInfo.stream);
    }
    // DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
}

void DEMKinematicThread::workerThread() {
    // Set the device for this thread
    DEME_GPU_CALL(cudaSetDevice(streamInfo.device));

    // Allocate arrays whose length does not depend on user inputs
    initAllocation();

    while (!pSchedSupport->kinematicShouldJoin) {
        {
            std::unique_lock<std::mutex> lock(pSchedSupport->kinematicStartLock);
            while (!pSchedSupport->kinematicStarted) {
                pSchedSupport->cv_KinematicStartLock.wait(lock);
            }
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
                // pSchedSupport->schedulingStats.nKinematicReceives++;
            }
            timers.GetTimer("Unpack updates from dT").stop();

            // Make it clear that the data for most recent work order has been used, in case there is interest in
            // updating it
            pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = false;

            // figure out the amount of shared mem
            // cudaDeviceGetAttribute.cudaDevAttrMaxSharedMemoryPerBlock

            // kT's main task, contact detection.
            // For auto-adjusting bin size, this part of code is encapsuled in an accumulative timer.
            CDAccumTimer.Begin();
            contactDetection(bin_sphere_kernels, bin_triangle_kernels, sphere_contact_kernels, sphTri_contact_kernels,
                             history_kernels, granData, simParams, solverFlags, verbosity, idGeometryA, idGeometryB,
                             contactType, previous_idGeometryA, previous_idGeometryB, previous_contactType,
                             contactPersistency, contactMapping, streamInfo.stream, solverScratchSpace, timers,
                             stateParams);
            CDAccumTimer.End();

            timers.GetTimer("Send to dT buffer").start();
            {
                // kT will reflect on how good the choice of parameters is
                calibrateParams();
                // Acquire lock and supply the dynamic with fresh produce
                std::lock_guard<std::mutex> lock(pSchedSupport->dynamicOwnedBuffer_AccessCoordination);
                sendToTheirBuffer();
            }
            pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = true;
            pSchedSupport->schedulingStats.nDynamicUpdates++;
            timers.GetTimer("Send to dT buffer").stop();

            // Signal the dynamic that it has fresh produce
            pSchedSupport->cv_DynamicCanProceed.notify_all();

            // std::cout << "kT host mem usage: " << pretty_format_bytes(estimateHostMemUsage()) << std::endl;
            // std::cout << "kT device mem usage: " << pretty_format_bytes(estimateDeviceMemUsage()) << std::endl;
            // solverScratchSpace.printVectorUsage();
        }

        // In case the dynamic is hanging in there...
        pSchedSupport->cv_DynamicCanProceed.notify_all();

        // When getting here, kT has finished one user call (although perhaps not at the end of the user script)
        {
            std::lock_guard<std::mutex> lock(pPagerToMain->mainCanProceed);
            pPagerToMain->userCallDone = true;
            pPagerToMain->cv_mainCanProceed.notify_all();
        }
    }
}

void DEMKinematicThread::getTiming(std::vector<std::string>& names, std::vector<double>& vals) {
    names = timer_names;
    for (const auto& name : timer_names) {
        vals.push_back(timers.GetTimer(name).GetTimeSeconds());
    }
}

void DEMKinematicThread::changeFamily(unsigned int ID_from, unsigned int ID_to) {
    family_t ID_from_impl = ID_from;
    family_t ID_to_impl = ID_to;

    migrateFamilyToHost();
    std::replace_if(
        familyID.getHostVector().begin(), familyID.getHostVector().end(),
        [ID_from_impl](family_t& i) { return i == ID_from_impl; }, ID_to_impl);
    familyID.toDevice();
}

void DEMKinematicThread::changeOwnerSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors) {
    // Set the gpu for this thread
    cudaSetDevice(streamInfo.device);
    // cudaStream_t new_stream;
    // cudaStreamCreate(&new_stream);

    // First get IDs and factors to device side
    size_t IDSize = IDs.size() * sizeof(bodyID_t);
    bodyID_t* dIDs = (bodyID_t*)solverScratchSpace.allocateTempVector("dIDs", IDSize);
    DEME_GPU_CALL(cudaMemcpy(dIDs, IDs.data(), IDSize, cudaMemcpyHostToDevice));
    size_t factorSize = factors.size() * sizeof(float);
    float* dFactors = (float*)solverScratchSpace.allocateTempVector("dFactors", factorSize);
    DEME_GPU_CALL(cudaMemcpy(dFactors, factors.data(), factorSize, cudaMemcpyHostToDevice));

    size_t idBoolSize = (size_t)simParams->nOwnerBodies * sizeof(notStupidBool_t);
    size_t ownerFactorSize = (size_t)simParams->nOwnerBodies * sizeof(float);
    // Bool table for whether this owner should change
    notStupidBool_t* idBool = (notStupidBool_t*)solverScratchSpace.allocateTempVector("idBool", idBoolSize);
    DEME_GPU_CALL(cudaMemset(idBool, 0, idBoolSize));
    float* ownerFactors = (float*)solverScratchSpace.allocateTempVector("ownerFactors", ownerFactorSize);
    size_t blocks_needed_for_marking = (IDs.size() + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;

    // Mark on the bool array those owners that need a change
    misc_kernels->kernel("markOwnerToChange")
        .instantiate()
        .configure(dim3(blocks_needed_for_marking), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
        .launch(idBool, ownerFactors, dIDs, dFactors, (size_t)IDs.size());
    DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

    // Change the size of the sphere components in question
    size_t blocks_needed_for_changing =
        (simParams->nSpheresGM + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    misc_kernels->kernel("modifyComponents")
        .instantiate("deme::DEMDataKT")
        .configure(dim3(blocks_needed_for_changing), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
        .launch(&granData, idBool, ownerFactors, (size_t)simParams->nSpheresGM);
    DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

    solverScratchSpace.finishUsingTempVector("dIDs");
    solverScratchSpace.finishUsingTempVector("dFactors");
    solverScratchSpace.finishUsingTempVector("idBool");
    solverScratchSpace.finishUsingTempVector("ownerFactors");
    // cudaStreamDestroy(new_stream);

    // Update them back to host
    relPosSphereX.toHost();
    relPosSphereY.toHost();
    relPosSphereZ.toHost();
    radiiSphere.toHost();
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
    // My ingredient production date is... unknown now
    pSchedSupport->kinematicIngredProdDateStamp = -1;

    // We also reset the CD timer (for adjusting bin size)
    CDAccumTimer.Clear();
    // Reset bin size change speed
    stateParams.binCurrentChangeRate = 0.;
}

size_t DEMKinematicThread::estimateDeviceMemUsage() const {
    return m_approxDeviceBytesUsed;
}

size_t DEMKinematicThread::estimateHostMemUsage() const {
    return m_approxHostBytesUsed;
}

// Put sim data array pointers in place
void DEMKinematicThread::packDataPointers() {
    familyID.bindDevicePointer(&(granData->familyID));
    voxelID.bindDevicePointer(&(granData->voxelID));
    locX.bindDevicePointer(&(granData->locX));
    locY.bindDevicePointer(&(granData->locY));
    locZ.bindDevicePointer(&(granData->locZ));
    oriQw.bindDevicePointer(&(granData->oriQw));
    oriQx.bindDevicePointer(&(granData->oriQx));
    oriQy.bindDevicePointer(&(granData->oriQy));
    oriQz.bindDevicePointer(&(granData->oriQz));
    marginSize.bindDevicePointer(&(granData->marginSize));
    idGeometryA.bindDevicePointer(&(granData->idGeometryA));
    idGeometryB.bindDevicePointer(&(granData->idGeometryB));
    contactType.bindDevicePointer(&(granData->contactType));
    contactPersistency.bindDevicePointer(&(granData->contactPersistency));
    previous_idGeometryA.bindDevicePointer(&(granData->previous_idGeometryA));
    previous_idGeometryB.bindDevicePointer(&(granData->previous_idGeometryB));
    previous_contactType.bindDevicePointer(&(granData->previous_contactType));
    contactMapping.bindDevicePointer(&(granData->contactMapping));
    familyMaskMatrix.bindDevicePointer(&(granData->familyMasks));
    familyExtraMarginSize.bindDevicePointer(&(granData->familyExtraMarginSize));

    // The offset info that indexes into the template arrays
    ownerClumpBody.bindDevicePointer(&(granData->ownerClumpBody));
    clumpComponentOffset.bindDevicePointer(&(granData->clumpComponentOffset));
    clumpComponentOffsetExt.bindDevicePointer(&(granData->clumpComponentOffsetExt));

    // Mesh-related
    ownerMesh.bindDevicePointer(&(granData->ownerMesh));
    relPosNode1.bindDevicePointer(&(granData->relPosNode1));
    relPosNode2.bindDevicePointer(&(granData->relPosNode2));
    relPosNode3.bindDevicePointer(&(granData->relPosNode3));

    // Template array pointers
    radiiSphere.bindDevicePointer(&(granData->radiiSphere));
    relPosSphereX.bindDevicePointer(&(granData->relPosSphereX));
    relPosSphereY.bindDevicePointer(&(granData->relPosSphereY));
    relPosSphereZ.bindDevicePointer(&(granData->relPosSphereZ));
}

void DEMKinematicThread::migrateDataToDevice() {
    familyID.toDeviceAsync(streamInfo.stream);
    voxelID.toDeviceAsync(streamInfo.stream);
    locX.toDeviceAsync(streamInfo.stream);
    locY.toDeviceAsync(streamInfo.stream);
    locZ.toDeviceAsync(streamInfo.stream);
    oriQw.toDeviceAsync(streamInfo.stream);
    oriQx.toDeviceAsync(streamInfo.stream);
    oriQy.toDeviceAsync(streamInfo.stream);
    oriQz.toDeviceAsync(streamInfo.stream);
    marginSize.toDeviceAsync(streamInfo.stream);
    idGeometryA.toDeviceAsync(streamInfo.stream);
    idGeometryB.toDeviceAsync(streamInfo.stream);
    contactType.toDeviceAsync(streamInfo.stream);
    contactPersistency.toDeviceAsync(streamInfo.stream);
    previous_idGeometryA.toDeviceAsync(streamInfo.stream);
    previous_idGeometryB.toDeviceAsync(streamInfo.stream);
    previous_contactType.toDeviceAsync(streamInfo.stream);
    contactMapping.toDeviceAsync(streamInfo.stream);
    familyMaskMatrix.toDeviceAsync(streamInfo.stream);
    familyExtraMarginSize.toDeviceAsync(streamInfo.stream);

    ownerClumpBody.toDeviceAsync(streamInfo.stream);
    clumpComponentOffset.toDeviceAsync(streamInfo.stream);
    clumpComponentOffsetExt.toDeviceAsync(streamInfo.stream);

    ownerMesh.toDeviceAsync(streamInfo.stream);
    relPosNode1.toDeviceAsync(streamInfo.stream);
    relPosNode2.toDeviceAsync(streamInfo.stream);
    relPosNode3.toDeviceAsync(streamInfo.stream);

    radiiSphere.toDeviceAsync(streamInfo.stream);
    relPosSphereX.toDeviceAsync(streamInfo.stream);
    relPosSphereY.toDeviceAsync(streamInfo.stream);
    relPosSphereZ.toDeviceAsync(streamInfo.stream);

    // Might not be necessary... but it's a big call anyway, let's sync
    syncMemoryTransfer();
}

void DEMKinematicThread::migrateFamilyToHost() {
    if (solverFlags.canFamilyChangeOnDevice) {
        familyID.toHost();
    }
}

void DEMKinematicThread::migrateDeviceModifiableInfoToHost() {
    migrateFamilyToHost();
}

void DEMKinematicThread::packTransferPointers(DEMDynamicThread*& dT) {
    // Set the pointers to dT owned buffers
    granData->pDTOwnedBuffer_nContactPairs = &(dT->nContactPairs_buffer);
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
                                      float3 user_box_min,
                                      float3 user_box_max,
                                      float3 G,
                                      double ts_size,
                                      float expand_factor,
                                      float approx_max_vel,
                                      float expand_safety_param,
                                      float expand_safety_adder,
                                      const std::set<std::string>& contact_wildcards,
                                      const std::set<std::string>& owner_wildcards,
                                      const std::set<std::string>& geo_wildcards) {
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
    simParams->beta = expand_factor;  // If beta is auto-adapting, this assignment has no effect
    simParams->approxMaxVel = approx_max_vel;
    simParams->expSafetyMulti = expand_safety_param;
    simParams->expSafetyAdder = expand_safety_adder;
    simParams->nbX = nbX;
    simParams->nbY = nbY;
    simParams->nbZ = nbZ;
    simParams->userBoxMin = user_box_min;
    simParams->userBoxMax = user_box_max;

    simParams->nContactWildcards = contact_wildcards.size();
    simParams->nOwnerWildcards = owner_wildcards.size();
    simParams->nGeoWildcards = geo_wildcards.size();
}

void DEMKinematicThread::allocateGPUArrays(size_t nOwnerBodies,
                                           size_t nOwnerClumps,
                                           unsigned int nExtObj,
                                           size_t nTriMeshes,
                                           size_t nSpheresGM,
                                           size_t nTriGM,
                                           unsigned int nAnalGM,
                                           size_t nExtraContacts,
                                           unsigned int nMassProperties,
                                           unsigned int nClumpTopo,
                                           unsigned int nClumpComponents,
                                           unsigned int nJitifiableClumpComponents,
                                           unsigned int nMatTuples) {
    DEME_GPU_CALL(cudaSetDevice(streamInfo.device));

    // Sizes of these arrays
    simParams->nSpheresGM = nSpheresGM;
    simParams->nTriGM = nTriGM;
    simParams->nAnalGM = nAnalGM;
    simParams->nOwnerBodies = nOwnerBodies;
    simParams->nOwnerClumps = nOwnerClumps;
    simParams->nExtObj = nExtObj;
    simParams->nTriMeshes = nTriMeshes;
    simParams->nDistinctMassProperties = nMassProperties;
    simParams->nDistinctClumpBodyTopologies = nClumpTopo;
    simParams->nJitifiableClumpComponents = nJitifiableClumpComponents;
    simParams->nDistinctClumpComponents = nClumpComponents;
    simParams->nMatTuples = nMatTuples;

    // Resize the family mask `matrix' (in fact it is flattened)
    DEME_DUAL_ARRAY_RESIZE(familyMaskMatrix, (NUM_AVAL_FAMILIES + 1) * NUM_AVAL_FAMILIES / 2, DONT_PREVENT_CONTACT);

    // Resize to the number of clumps
    DEME_DUAL_ARRAY_RESIZE(familyID, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(voxelID, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(locX, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(locY, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(locZ, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(oriQw, nOwnerBodies, 1);
    DEME_DUAL_ARRAY_RESIZE(oriQx, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(oriQy, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(oriQz, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(marginSize, nOwnerBodies, 0);

    // Transfer buffer arrays
    // It is cudaMalloc-ed memory, not on host, because we want explicit locality control of buffers
    {
        // These buffers should be on dT, to save dT access time
        DEME_GPU_CALL(cudaSetDevice(dT->streamInfo.device));
        DEME_DEVICE_ARRAY_RESIZE(voxelID_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(locX_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(locY_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(locZ_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(oriQ0_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(oriQ1_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(oriQ2_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(oriQ3_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(absVel_buffer, nOwnerBodies);
        // DEME_ADVISE_DEVICE(voxelID_buffer, dT->streamInfo.device);
        // DEME_ADVISE_DEVICE(locX_buffer, dT->streamInfo.device);
        // DEME_ADVISE_DEVICE(locY_buffer, dT->streamInfo.device);
        // DEME_ADVISE_DEVICE(locZ_buffer, dT->streamInfo.device);
        // DEME_ADVISE_DEVICE(oriQ0_buffer, dT->streamInfo.device);
        // DEME_ADVISE_DEVICE(oriQ1_buffer, dT->streamInfo.device);
        // DEME_ADVISE_DEVICE(oriQ2_buffer, dT->streamInfo.device);
        // DEME_ADVISE_DEVICE(oriQ3_buffer, dT->streamInfo.device);

        if (solverFlags.canFamilyChangeOnDevice) {
            // DEME_ADVISE_DEVICE(familyID_buffer, dT->streamInfo.device);
            DEME_DEVICE_ARRAY_RESIZE(familyID_buffer, nOwnerBodies);
        }

        DEME_DEVICE_ARRAY_RESIZE(relPosNode1_buffer, nTriGM);
        DEME_DEVICE_ARRAY_RESIZE(relPosNode2_buffer, nTriGM);
        DEME_DEVICE_ARRAY_RESIZE(relPosNode3_buffer, nTriGM);

        // Unset the device change we just did
        DEME_GPU_CALL(cudaSetDevice(streamInfo.device));
    }

    // Resize to the number of spheres (or plus num of triangle facets)
    DEME_DUAL_ARRAY_RESIZE(ownerClumpBody, nSpheresGM, 0);

    // Resize to the number of triangle facets
    DEME_DUAL_ARRAY_RESIZE(ownerMesh, nTriGM, 0);
    DEME_DUAL_ARRAY_RESIZE(relPosNode1, nTriGM, make_float3(0));
    DEME_DUAL_ARRAY_RESIZE(relPosNode2, nTriGM, make_float3(0));
    DEME_DUAL_ARRAY_RESIZE(relPosNode3, nTriGM, make_float3(0));

    if (solverFlags.useClumpJitify) {
        DEME_DUAL_ARRAY_RESIZE(clumpComponentOffset, nSpheresGM, 0);
        // This extended component offset array can hold offset numbers even for big clumps (whereas
        // clumpComponentOffset is typically uint_8, so it may not). If a sphere's component offset index falls in this
        // range then it is not jitified, and the kernel needs to look for it in the global memory.
        DEME_DUAL_ARRAY_RESIZE(clumpComponentOffsetExt, nSpheresGM, 0);
        // Resize to the length of the clump templates
        DEME_DUAL_ARRAY_RESIZE(radiiSphere, nClumpComponents, 0);
        DEME_DUAL_ARRAY_RESIZE(relPosSphereX, nClumpComponents, 0);
        DEME_DUAL_ARRAY_RESIZE(relPosSphereY, nClumpComponents, 0);
        DEME_DUAL_ARRAY_RESIZE(relPosSphereZ, nClumpComponents, 0);
    } else {
        DEME_DUAL_ARRAY_RESIZE(radiiSphere, nSpheresGM, 0);
        DEME_DUAL_ARRAY_RESIZE(relPosSphereX, nSpheresGM, 0);
        DEME_DUAL_ARRAY_RESIZE(relPosSphereY, nSpheresGM, 0);
        DEME_DUAL_ARRAY_RESIZE(relPosSphereZ, nSpheresGM, 0);
    }

    // Arrays for kT produced contact info
    // The following several arrays will have variable sizes, so here we only used an estimate. My estimate of total
    // contact pairs is 2n, and I think the max is 6n (although I can't prove it). Note the estimate should be large
    // enough to decrease the number of reallocations in the simulation, but not too large that eats too much memory.
    {
        size_t cnt_arr_size = DEME_MAX(*solverScratchSpace.numPrevContacts, nSpheresGM * DEME_INIT_CNT_MULTIPLIER);
        DEME_DUAL_ARRAY_RESIZE(idGeometryA, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(idGeometryB, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(contactType, cnt_arr_size, NOT_A_CONTACT);
        if (!solverFlags.isHistoryless) {
            DEME_DUAL_ARRAY_RESIZE(contactPersistency, cnt_arr_size, CONTACT_NOT_PERSISTENT);
            DEME_DUAL_ARRAY_RESIZE(previous_idGeometryA, cnt_arr_size, 0);
            DEME_DUAL_ARRAY_RESIZE(previous_idGeometryB, cnt_arr_size, 0);
            DEME_DUAL_ARRAY_RESIZE(previous_contactType, cnt_arr_size, NOT_A_CONTACT);
            DEME_DUAL_ARRAY_RESIZE(contactMapping, cnt_arr_size, NULL_MAPPING_PARTNER);
        }
    }
}

void DEMKinematicThread::registerPolicies(const std::vector<notStupidBool_t>& family_mask_matrix) {
    // Store family mask
    for (size_t i = 0; i < family_mask_matrix.size(); i++)
        familyMaskMatrix[i] = family_mask_matrix.at(i);
}

void DEMKinematicThread::populateEntityArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                              const std::vector<unsigned int>& input_ext_obj_family,
                                              const std::vector<unsigned int>& input_mesh_obj_family,
                                              const std::vector<unsigned int>& input_mesh_facet_owner,
                                              const std::vector<DEMTriangle>& input_mesh_facets,
                                              const ClumpTemplateFlatten& clump_templates,
                                              size_t nExistOwners,
                                              size_t nExistSpheres,
                                              size_t nExistingFacets) {
    // All the input vectors should have the same length, nClumpTopo
    size_t k = 0;
    std::vector<unsigned int> prescans_comp;

    if (solverFlags.useClumpJitify) {
        prescans_comp.push_back(0);
        for (auto elem : clump_templates.spRadii) {
            for (auto radius : elem) {
                radiiSphere[k] = radius;
                k++;
            }
            prescans_comp.push_back(k);
        }
        prescans_comp.pop_back();
        k = 0;

        for (auto elem : clump_templates.spRelPos) {
            for (auto loc : elem) {
                relPosSphereX[k] = loc.x;
                relPosSphereY[k] = loc.y;
                relPosSphereZ[k] = loc.z;
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
            auto this_clump_no_sp_radii = clump_templates.spRadii.at(type_of_this_clump);
            auto this_clump_no_sp_relPos = clump_templates.spRelPos.at(type_of_this_clump);

            for (size_t j = 0; j < this_clump_no_sp_radii.size(); j++) {
                ownerClumpBody[nExistSpheres + k] = nExistOwners + i;

                // Depending on whether we jitify or flatten
                if (solverFlags.useClumpJitify) {
                    // This component offset, is it too large that can't live in the jitified array?
                    unsigned int this_comp_offset = prescans_comp.at(type_of_this_clump) + j;
                    clumpComponentOffsetExt[nExistSpheres + k] = this_comp_offset;
                    if (this_comp_offset < simParams->nJitifiableClumpComponents) {
                        clumpComponentOffset[nExistSpheres + k] = this_comp_offset;
                    } else {
                        // If not, an indicator will be put there
                        clumpComponentOffset[nExistSpheres + k] = RESERVED_CLUMP_COMPONENT_OFFSET;
                    }
                } else {
                    radiiSphere[nExistSpheres + k] = this_clump_no_sp_radii.at(j);
                    const float3 relPos = this_clump_no_sp_relPos.at(j);
                    relPosSphereX[nExistSpheres + k] = relPos.x;
                    relPosSphereY[nExistSpheres + k] = relPos.y;
                    relPosSphereZ[nExistSpheres + k] = relPos.z;
                }

                k++;
            }

            family_t this_family_num = input_clump_family.at(i);
            familyID[nExistOwners + i] = this_family_num;
        }
    }

    // Analytical objs
    size_t owner_offset_for_ext_obj = nExistOwners + input_clump_types.size();
    for (size_t i = 0; i < input_ext_obj_family.size(); i++) {
        family_t this_family_num = input_ext_obj_family.at(i);
        familyID[i + owner_offset_for_ext_obj] = this_family_num;
    }

    // Mesh objs
    size_t owner_offset_for_mesh_obj = owner_offset_for_ext_obj + input_ext_obj_family.size();
    // k for indexing the triangle facets
    k = 0;
    for (size_t i = 0; i < input_mesh_obj_family.size(); i++) {
        // Per-facet info
        size_t this_facet_owner = input_mesh_facet_owner.at(k);
        for (; k < input_mesh_facet_owner.size(); k++) {
            // input_mesh_facet_owner run length is the num of facets in this mesh entity
            if (input_mesh_facet_owner.at(k) != this_facet_owner)
                break;
            ownerMesh[nExistingFacets + k] = owner_offset_for_mesh_obj + this_facet_owner;
            DEMTriangle this_tri = input_mesh_facets.at(k);
            relPosNode1[nExistingFacets + k] = this_tri.p1;
            relPosNode2[nExistingFacets + k] = this_tri.p2;
            relPosNode3[nExistingFacets + k] = this_tri.p3;
        }

        family_t this_family_num = input_mesh_obj_family.at(i);
        familyID[i + owner_offset_for_mesh_obj] = this_family_num;
        // DEME_DEBUG_PRINTF("kT just loaded a mesh in family %u", +(this_family_num));
        // DEME_DEBUG_PRINTF("Number of triangle facets loaded thus far: %zu", k);
    }
}

void DEMKinematicThread::initGPUArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                       const std::vector<unsigned int>& input_ext_obj_family,
                                       const std::vector<unsigned int>& input_mesh_obj_family,
                                       const std::vector<unsigned int>& input_mesh_facet_owner,
                                       const std::vector<DEMTriangle>& input_mesh_facets,
                                       const std::vector<notStupidBool_t>& family_mask_matrix,
                                       const ClumpTemplateFlatten& clump_templates) {
    // Get the info into the GPU memory from the host side. Can this process be more efficient? Maybe, but it's
    // initialization anyway.

    registerPolicies(family_mask_matrix);

    populateEntityArrays(input_clump_batches, input_ext_obj_family, input_mesh_obj_family, input_mesh_facet_owner,
                         input_mesh_facets, clump_templates, 0, 0, 0);
}

void DEMKinematicThread::updateClumpMeshArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                               const std::vector<unsigned int>& input_ext_obj_family,
                                               const std::vector<unsigned int>& input_mesh_obj_family,
                                               const std::vector<unsigned int>& input_mesh_facet_owner,
                                               const std::vector<DEMTriangle>& input_mesh_facets,
                                               const std::vector<notStupidBool_t>& family_mask_matrix,
                                               const ClumpTemplateFlatten& clump_templates,
                                               size_t nExistingOwners,
                                               size_t nExistingClumps,
                                               size_t nExistingSpheres,
                                               size_t nExistingTriMesh,
                                               size_t nExistingFacets,
                                               unsigned int nExistingObj,
                                               unsigned int nExistingAnalGM) {
    populateEntityArrays(input_clump_batches, input_ext_obj_family, input_mesh_obj_family, input_mesh_facet_owner,
                         input_mesh_facets, clump_templates, nExistingOwners, nExistingSpheres, nExistingFacets);
}

void DEMKinematicThread::updatePrevContactArrays(DualStruct<DEMDataDT>& dT_data, size_t nContacts) {
    // Store the incoming info in kT's arrays
    // Note kT never had the responsibility to migrate contact info to host, even at UpdateClumps, as even in this case
    // its host-side update comes from dT
    overwritePrevContactArrays(granData, dT_data, previous_idGeometryA, previous_idGeometryB, previous_contactType,
                               simParams, contactPersistency, solverScratchSpace, streamInfo.stream, nContacts);
    DEME_DEBUG_PRINTF("Number of contacts after a user-manual contact load: %zu", nContacts);
    DEME_DEBUG_PRINTF("Number of spheres after a user-manual contact load: %zu", (size_t)simParams->nSpheresGM);
}

void DEMKinematicThread::jitifyKernels(const std::unordered_map<std::string, std::string>& Subs,
                                       const std::vector<std::string>& JitifyOptions) {
    // First one is bin_sphere_kernels kernels, which figure out the bin--sphere touch pairs
    {
        bin_sphere_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMBinSphereKernels", JitHelper::KERNEL_DIR / "DEMBinSphereKernels.cu", Subs, JitifyOptions)));
    }
    // Then CD kernels
    {
        sphere_contact_kernels = std::make_shared<jitify::Program>(std::move(
            JitHelper::buildProgram("DEMContactKernels_SphereSphere",
                                    JitHelper::KERNEL_DIR / "DEMContactKernels_SphereSphere.cu", Subs, JitifyOptions)));
    }
    // Then triangle--bin intersection-related kernels
    {
        bin_triangle_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMBinTriangleKernels", JitHelper::KERNEL_DIR / "DEMBinTriangleKernels.cu", Subs, JitifyOptions)));
    }
    // Then sphere--triangle contact detection-related kernels
    {
        sphTri_contact_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMContactKernels_SphereTriangle", JitHelper::KERNEL_DIR / "DEMContactKernels_SphereTriangle.cu", Subs,
            JitifyOptions)));
    }
    // Then contact history mapping kernels
    {
        history_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMHistoryMappingKernels", JitHelper::KERNEL_DIR / "DEMHistoryMappingKernels.cu", Subs, JitifyOptions)));
    }
    // Then misc kernels
    {
        misc_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMMiscKernels", JitHelper::KERNEL_DIR / "DEMMiscKernels.cu", Subs, JitifyOptions)));
    }
}

void DEMKinematicThread::initAllocation() {
    DEME_DUAL_ARRAY_RESIZE(familyExtraMarginSize, NUM_AVAL_FAMILIES, 0);
}

void DEMKinematicThread::deallocateEverything() {
    // Device and dual array will have their destructor called once kT is gone, so this is not needed
}

void DEMKinematicThread::setOwnerFamily(bodyID_t ownerID, family_t fam, bodyID_t n) {
    familyID.setVal(std::vector<family_t>(n, fam), ownerID);
}

void DEMKinematicThread::setTriNodeRelPos(size_t start, const std::vector<DEMTriangle>& triangles) {
    for (size_t i = 0; i < triangles.size(); i++) {
        relPosNode1[start + i] = triangles[i].p1;
        relPosNode2[start + i] = triangles[i].p2;
        relPosNode3[start + i] = triangles[i].p3;
    }
    relPosNode1.toDeviceAsync(streamInfo.stream, start, triangles.size());
    relPosNode2.toDeviceAsync(streamInfo.stream, start, triangles.size());
    relPosNode3.toDeviceAsync(streamInfo.stream, start, triangles.size());
    syncMemoryTransfer();
}

// It's true that this method is never used in either kT or dT
void DEMKinematicThread::updateTriNodeRelPos(size_t start, const std::vector<DEMTriangle>& updates) {
    for (size_t i = 0; i < updates.size(); i++) {
        relPosNode1[start + i] += updates[i].p1;
        relPosNode2[start + i] += updates[i].p2;
        relPosNode3[start + i] += updates[i].p3;
    }
    relPosNode1.toDeviceAsync(streamInfo.stream, start, updates.size());
    relPosNode2.toDeviceAsync(streamInfo.stream, start, updates.size());
    relPosNode3.toDeviceAsync(streamInfo.stream, start, updates.size());
    syncMemoryTransfer();
}

}  // namespace deme
