//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef DEME_USE_CHPF
    #include <chpf.hpp>
#endif
#include <core/ApiVersion.h>
#include <core/utils/JitHelper.h>
#include <DEM/dT.h>
#include <DEM/kT.h>
#include <DEM/utils/HostSideHelpers.hpp>
#include <DEM/Defines.h>

#include <algorithms/DEMStaticDeviceSubroutines.h>
#include <kernel/DEMHelperKernels.cuh>

#ifdef DEME_ENABLE_NVTX
    #include <nvtx3/nvtx3.hpp>
    #include <nvtx3/nvToolsExtCudaRt.h>
    #define DEME_NVTX_CONCAT_IMPL(x, y) x##y
    #define DEME_NVTX_CONCAT(x, y) DEME_NVTX_CONCAT_IMPL(x, y)
    #define DEME_NVTX_RANGE(name)                                                   \
        auto DEME_NVTX_CONCAT(__deme_nvtx_range_, __LINE__) = nvtx3::scoped_range { \
            name                                                                    \
        }
    #define DEME_NVTX_NAME_STREAM(stream, label) nvtxNameCudaStreamA((stream), (label))
#else
    #define DEME_NVTX_RANGE(name) \
        do {                      \
        } while (0)
    #define DEME_NVTX_NAME_STREAM(stream, label) \
        do {                                     \
        } while (0)
#endif

namespace deme {

// Put sim data array pointers in place
void DEMDynamicThread::packDataPointers() {
    inertiaPropOffsets.bindDevicePointer(&(granData->inertiaPropOffsets));
    familyID.bindDevicePointer(&(granData->familyID));
    voxelID.bindDevicePointer(&(granData->voxelID));
    ownerTypes.bindDevicePointer(&(granData->ownerTypes));
    locX.bindDevicePointer(&(granData->locX));
    locY.bindDevicePointer(&(granData->locY));
    locZ.bindDevicePointer(&(granData->locZ));
    aX.bindDevicePointer(&(granData->aX));
    aY.bindDevicePointer(&(granData->aY));
    aZ.bindDevicePointer(&(granData->aZ));
    vX.bindDevicePointer(&(granData->vX));
    vY.bindDevicePointer(&(granData->vY));
    vZ.bindDevicePointer(&(granData->vZ));
    oriQw.bindDevicePointer(&(granData->oriQw));
    oriQx.bindDevicePointer(&(granData->oriQx));
    oriQy.bindDevicePointer(&(granData->oriQy));
    oriQz.bindDevicePointer(&(granData->oriQz));
    omgBarX.bindDevicePointer(&(granData->omgBarX));
    omgBarY.bindDevicePointer(&(granData->omgBarY));
    omgBarZ.bindDevicePointer(&(granData->omgBarZ));
    alphaX.bindDevicePointer(&(granData->alphaX));
    alphaY.bindDevicePointer(&(granData->alphaY));
    alphaZ.bindDevicePointer(&(granData->alphaZ));
    accSpecified.bindDevicePointer(&(granData->accSpecified));
    angAccSpecified.bindDevicePointer(&(granData->angAccSpecified));
    idPrimitiveA.bindDevicePointer(&(granData->idPrimitiveA));
    idPrimitiveB.bindDevicePointer(&(granData->idPrimitiveB));
    contactTypePrimitive.bindDevicePointer(&(granData->contactTypePrimitive));
    geomToPatchMap.bindDevicePointer(&(granData->geomToPatchMap));

    // NEW: Bind separate patch ID and mapping array pointers
    idPatchA.bindDevicePointer(&(granData->idPatchA));
    idPatchB.bindDevicePointer(&(granData->idPatchB));
    contactTypePatch.bindDevicePointer(&(granData->contactTypePatch));

    familyMaskMatrix.bindDevicePointer(&(granData->familyMasks));
    familyExtraMarginSize.bindDevicePointer(&(granData->familyExtraMarginSize));

    contactForces.bindDevicePointer(&(granData->contactForces));
    contactTorque_convToForce.bindDevicePointer(&(granData->contactTorque_convToForce));
    contactPointGeometryA.bindDevicePointer(&(granData->contactPointGeometryA));
    contactPointGeometryB.bindDevicePointer(&(granData->contactPointGeometryB));
    contactSATSatisfied.bindDevicePointer(&(granData->contactSATSatisfied));
    // granData->contactHistory = contactHistory.data();
    // granData->contactDuration = contactDuration.data();

    for (unsigned int i = 0; i < simParams->nContactWildcards; i++) {
        contactWildcards[i]->bindDevicePointer(&(granData->contactWildcards[i]));
    }
    for (unsigned int i = 0; i < simParams->nOwnerWildcards; i++) {
        ownerWildcards[i]->bindDevicePointer(&(granData->ownerWildcards[i]));
    }
    for (unsigned int i = 0; i < simParams->nGeoWildcards; i++) {
        sphereWildcards[i]->bindDevicePointer(&(granData->sphereWildcards[i]));
        analWildcards[i]->bindDevicePointer(&(granData->analWildcards[i]));
        patchWildcards[i]->bindDevicePointer(&(granData->patchWildcards[i]));
    }

    // The offset info that indexes into the template arrays
    ownerClumpBody.bindDevicePointer(&(granData->ownerClumpBody));
    clumpComponentOffset.bindDevicePointer(&(granData->clumpComponentOffset));
    clumpComponentOffsetExt.bindDevicePointer(&(granData->clumpComponentOffsetExt));
    sphereMaterialOffset.bindDevicePointer(&(granData->sphereMaterialOffset));
    volumeOwnerBody.bindDevicePointer(&(granData->volumeOwnerBody));

    // Mesh and analytical-related
    ownerTriMesh.bindDevicePointer(&(granData->ownerTriMesh));
    ownerPatchMesh.bindDevicePointer(&(granData->ownerPatchMesh));
    triPatchID.bindDevicePointer(&(granData->triPatchID));
    ownerAnalBody.bindDevicePointer(&(granData->ownerAnalBody));
    relPosNode1.bindDevicePointer(&(granData->relPosNode1));
    relPosNode2.bindDevicePointer(&(granData->relPosNode2));
    relPosNode3.bindDevicePointer(&(granData->relPosNode3));
    relPosPatch.bindDevicePointer(&(granData->relPosPatch));
    patchMaterialOffset.bindDevicePointer(&(granData->patchMaterialOffset));

    // Template array pointers
    radiiSphere.bindDevicePointer(&(granData->radiiSphere));
    relPosSphereX.bindDevicePointer(&(granData->relPosSphereX));
    relPosSphereY.bindDevicePointer(&(granData->relPosSphereY));
    relPosSphereZ.bindDevicePointer(&(granData->relPosSphereZ));
    massOwnerBody.bindDevicePointer(&(granData->massOwnerBody));
    mmiXX.bindDevicePointer(&(granData->mmiXX));
    mmiYY.bindDevicePointer(&(granData->mmiYY));
    mmiZZ.bindDevicePointer(&(granData->mmiZZ));
}

void DEMDynamicThread::recordAndSyncEvent() {
    DEME_GPU_CALL(cudaEventRecord(streamSyncEvent, streamInfo.stream));
    DEME_GPU_CALL(cudaEventSynchronize(streamSyncEvent));
}

void DEMDynamicThread::recordEventOnly() {
    DEME_GPU_CALL(cudaEventRecord(streamSyncEvent, streamInfo.stream));
}

void DEMDynamicThread::recordProgressEvent(int64_t stamp) {
    drainProgressEvents();
    if (progressEventCount == kProgressEventDepth) {
        // Fall back to a single sync when we are completely back-pressured.
        const int idx = progressEventHead;
        DEME_GPU_CALL(cudaEventSynchronize(progressEvents[idx]));
        pSchedSupport->completedStampOfDynamic.store(progressEventStamps[idx], std::memory_order_release);
        progressEventHead = (progressEventHead + 1) % kProgressEventDepth;
        progressEventCount--;
    }
    const int idx = (progressEventHead + progressEventCount) % kProgressEventDepth;
    progressEventStamps[idx] = stamp;
    DEME_GPU_CALL(cudaEventRecord(progressEvents[idx], streamInfo.stream));
    progressEventCount++;
}

void DEMDynamicThread::drainProgressEvents() {
    while (progressEventCount > 0) {
        const int idx = progressEventHead;
        cudaError_t err = cudaEventQuery(progressEvents[idx]);
        if (err == cudaSuccess) {
            pSchedSupport->completedStampOfDynamic.store(progressEventStamps[idx], std::memory_order_release);
            progressEventHead = (progressEventHead + 1) % kProgressEventDepth;
            progressEventCount--;
            continue;
        }
        if (err != cudaErrorNotReady) {
            DEME_GPU_CALL(err);
        }
        break;
    }
}

void DEMDynamicThread::throttleInFlightProgress() {
    drainProgressEvents();
    while (progressEventCount >= kMaxInFlightProgress) {
        const int idx = progressEventHead;
        DEME_GPU_CALL(cudaEventSynchronize(progressEvents[idx]));
        pSchedSupport->completedStampOfDynamic.store(progressEventStamps[idx], std::memory_order_release);
        progressEventHead = (progressEventHead + 1) % kProgressEventDepth;
        progressEventCount--;
        drainProgressEvents();
    }
}

void DEMDynamicThread::syncRecordedEvent() {
    // Wait on the most recently recorded barrier event (typically recorded after integration).
    DEME_GPU_CALL(cudaEventSynchronize(streamSyncEvent));
}

void DEMDynamicThread::startCycleStopwatch() {
    if (!cycle_stopwatch_started) {
        cycle_stopwatch_start = std::chrono::steady_clock::now();
        cycle_stopwatch_started = true;
    }
}

double DEMDynamicThread::getCycleElapsedSeconds() const {
    if (!cycle_stopwatch_started) {
        return 0.0;
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - cycle_stopwatch_start).count();
}

void DEMDynamicThread::migrateDataToDevice() {
    inertiaPropOffsets.toDeviceAsync(streamInfo.stream);
    familyID.toDeviceAsync(streamInfo.stream);
    voxelID.toDeviceAsync(streamInfo.stream);
    ownerTypes.toDeviceAsync(streamInfo.stream);
    locX.toDeviceAsync(streamInfo.stream);
    locY.toDeviceAsync(streamInfo.stream);
    locZ.toDeviceAsync(streamInfo.stream);
    aX.toDeviceAsync(streamInfo.stream);
    aY.toDeviceAsync(streamInfo.stream);
    aZ.toDeviceAsync(streamInfo.stream);
    vX.toDeviceAsync(streamInfo.stream);
    vY.toDeviceAsync(streamInfo.stream);
    vZ.toDeviceAsync(streamInfo.stream);
    oriQw.toDeviceAsync(streamInfo.stream);
    oriQx.toDeviceAsync(streamInfo.stream);
    oriQy.toDeviceAsync(streamInfo.stream);
    oriQz.toDeviceAsync(streamInfo.stream);
    omgBarX.toDeviceAsync(streamInfo.stream);
    omgBarY.toDeviceAsync(streamInfo.stream);
    omgBarZ.toDeviceAsync(streamInfo.stream);
    alphaX.toDeviceAsync(streamInfo.stream);
    alphaY.toDeviceAsync(streamInfo.stream);
    alphaZ.toDeviceAsync(streamInfo.stream);
    accSpecified.toDeviceAsync(streamInfo.stream);
    angAccSpecified.toDeviceAsync(streamInfo.stream);

    // Primitive contact info
    idPrimitiveA.toDeviceAsync(streamInfo.stream);
    idPrimitiveB.toDeviceAsync(streamInfo.stream);
    contactTypePrimitive.toDeviceAsync(streamInfo.stream);
    geomToPatchMap.toDeviceAsync(streamInfo.stream);

    // Separate patch contact info
    contactTypePatch.toDeviceAsync(streamInfo.stream);
    idPatchA.toDeviceAsync(streamInfo.stream);
    idPatchB.toDeviceAsync(streamInfo.stream);

    familyMaskMatrix.toDeviceAsync(streamInfo.stream);
    familyExtraMarginSize.toDeviceAsync(streamInfo.stream);

    contactForces.toDeviceAsync(streamInfo.stream);
    contactTorque_convToForce.toDeviceAsync(streamInfo.stream);
    contactPointGeometryA.toDeviceAsync(streamInfo.stream);
    contactPointGeometryB.toDeviceAsync(streamInfo.stream);

    for (unsigned int i = 0; i < simParams->nContactWildcards; i++) {
        contactWildcards[i]->toDeviceAsync(streamInfo.stream);
    }
    for (unsigned int i = 0; i < simParams->nOwnerWildcards; i++) {
        ownerWildcards[i]->toDeviceAsync(streamInfo.stream);
    }
    for (unsigned int i = 0; i < simParams->nGeoWildcards; i++) {
        sphereWildcards[i]->toDeviceAsync(streamInfo.stream);
        analWildcards[i]->toDeviceAsync(streamInfo.stream);
        patchWildcards[i]->toDeviceAsync(streamInfo.stream);
    }

    ownerClumpBody.toDeviceAsync(streamInfo.stream);
    clumpComponentOffset.toDeviceAsync(streamInfo.stream);
    clumpComponentOffsetExt.toDeviceAsync(streamInfo.stream);
    sphereMaterialOffset.toDeviceAsync(streamInfo.stream);
    volumeOwnerBody.toDeviceAsync(streamInfo.stream);

    ownerTriMesh.toDeviceAsync(streamInfo.stream);
    ownerPatchMesh.toDeviceAsync(streamInfo.stream);
    triPatchID.toDeviceAsync(streamInfo.stream);
    ownerAnalBody.toDeviceAsync(streamInfo.stream);
    relPosNode1.toDeviceAsync(streamInfo.stream);
    relPosNode2.toDeviceAsync(streamInfo.stream);
    relPosNode3.toDeviceAsync(streamInfo.stream);
    relPosPatch.toDeviceAsync(streamInfo.stream);
    patchMaterialOffset.toDeviceAsync(streamInfo.stream);

    radiiSphere.toDeviceAsync(streamInfo.stream);
    relPosSphereX.toDeviceAsync(streamInfo.stream);
    relPosSphereY.toDeviceAsync(streamInfo.stream);
    relPosSphereZ.toDeviceAsync(streamInfo.stream);
    massOwnerBody.toDeviceAsync(streamInfo.stream);
    mmiXX.toDeviceAsync(streamInfo.stream);
    mmiYY.toDeviceAsync(streamInfo.stream);
    mmiZZ.toDeviceAsync(streamInfo.stream);

    // Might not be necessary... but it's a big call anyway, let's sync
    syncMemoryTransfer();
}

void DEMDynamicThread::migrateDeviceModifiableInfoToHost() {
    migrateClumpPosInfoToHost();
    migrateClumpHighOrderInfoToHost();
    migrateFamilyToHost();
    migrateContactInfoToHost();
    migrateOwnerWildcardToHost();
    migrateSphGeoWildcardToHost();
    migratePatchGeoWildcardToHost();
    migrateAnalGeoWildcardToHost();
}

void DEMDynamicThread::migrateClumpHighOrderInfoToHost() {
    vX.toHost();
    vY.toHost();
    vZ.toHost();
    aX.toHost();
    aY.toHost();
    aZ.toHost();
    omgBarX.toHost();
    omgBarY.toHost();
    omgBarZ.toHost();
    alphaX.toHost();
    alphaY.toHost();
    alphaZ.toHost();
}

void DEMDynamicThread::migrateClumpPosInfoToHost() {
    voxelID.toHost();
    locX.toHost();
    locY.toHost();
    locZ.toHost();
    oriQw.toHost();
    oriQx.toHost();
    oriQy.toHost();
    oriQz.toHost();
}

void DEMDynamicThread::migrateContactInfoToHost() {
    // Primitive contact info
    idPrimitiveA.toHost();
    idPrimitiveB.toHost();
    contactTypePrimitive.toHost();
    geomToPatchMap.toHost();

    // Separate patch contact info
    contactTypePatch.toHost();
    idPatchA.toHost();
    idPatchB.toHost();

    // Contact results
    contactForces.toHost();
    contactTorque_convToForce.toHost();
    contactPointGeometryA.toHost();
    contactPointGeometryB.toHost();
    for (unsigned int i = 0; i < simParams->nContactWildcards; i++) {
        contactWildcards[i]->toHost();
    }
}

void DEMDynamicThread::migrateFamilyToHost() {
    if (solverFlags.canFamilyChangeOnDevice) {
        familyID.toHost();
    }
}

void DEMDynamicThread::migrateOwnerWildcardToHost() {
    for (unsigned int i = 0; i < simParams->nOwnerWildcards; i++) {
        ownerWildcards[i]->toHost();
    }
}
void DEMDynamicThread::migrateSphGeoWildcardToHost() {
    for (unsigned int i = 0; i < simParams->nGeoWildcards; i++) {
        sphereWildcards[i]->toHost();
    }
}
void DEMDynamicThread::migratePatchGeoWildcardToHost() {
    for (unsigned int i = 0; i < simParams->nGeoWildcards; i++) {
        patchWildcards[i]->toHost();
    }
}
void DEMDynamicThread::migrateAnalGeoWildcardToHost() {
    for (unsigned int i = 0; i < simParams->nGeoWildcards; i++) {
        analWildcards[i]->toHost();
    }
}

bodyID_t DEMDynamicThread::getGeoOwnerID(const bodyID_t& geo, const geoType_t& type) const {
    // These arrays can't change on device
    switch (type) {
        case (GEO_T_SPHERE):
            return ownerClumpBody[geo];
        case (GEO_T_TRIANGLE):
            return ownerTriMesh[geo];
        case (GEO_T_ANALYTICAL):
            return ownerAnalBody[geo];
        default:
            return NULL_BODYID;
    }
}

bodyID_t DEMDynamicThread::getPatchOwnerID(const bodyID_t& patchID, const geoType_t& type) const {
    switch (type) {
        case (GEO_T_TRIANGLE):
            return ownerPatchMesh[patchID];
        case (GEO_T_SPHERE):
            return ownerClumpBody[patchID];
        case (GEO_T_ANALYTICAL):
            return ownerAnalBody[patchID];
        default:
            return NULL_BODYID;
    }
}

// packTransferPointers
void DEMDynamicThread::packTransferPointers(DEMKinematicThread*& kT) {
    // These are the pointers for sending data to dT
    granData->pKTOwnedBuffer_absVel = kT->absVel_buffer.data();
    granData->pKTOwnedBuffer_absAngVel = kT->absAngVel_buffer.data();
    granData->pKTOwnedBuffer_voxelID = kT->voxelID_buffer.data();
    granData->pKTOwnedBuffer_locX = kT->locX_buffer.data();
    granData->pKTOwnedBuffer_locY = kT->locY_buffer.data();
    granData->pKTOwnedBuffer_locZ = kT->locZ_buffer.data();
    granData->pKTOwnedBuffer_oriQ0 = kT->oriQ0_buffer.data();
    granData->pKTOwnedBuffer_oriQ1 = kT->oriQ1_buffer.data();
    granData->pKTOwnedBuffer_oriQ2 = kT->oriQ2_buffer.data();
    granData->pKTOwnedBuffer_oriQ3 = kT->oriQ3_buffer.data();
    granData->pKTOwnedBuffer_familyID = kT->familyID_buffer.data();
    granData->pKTOwnedBuffer_relPosNode1 = kT->relPosNode1_buffer.data();
    granData->pKTOwnedBuffer_relPosNode2 = kT->relPosNode2_buffer.data();
    granData->pKTOwnedBuffer_relPosNode3 = kT->relPosNode3_buffer.data();

    // Single-number data are now not packaged in granData...
    granData->pKTOwnedBuffer_ts = &(kT->stateParams.ts_buffer);
    granData->pKTOwnedBuffer_maxDrift = &(kT->stateParams.maxDrift_buffer);
    granData->pKTOwnedBuffer_maxTriTriPenetration = &(kT->stateParams.maxTriTriPenetration_buffer);
}

void DEMDynamicThread::changeFamily(unsigned int ID_from, unsigned int ID_to) {
    family_t ID_from_impl = ID_from;
    family_t ID_to_impl = ID_to;

    migrateFamilyToHost();
    std::replace_if(
        familyID.getHostVector().begin(), familyID.getHostVector().end(),
        [ID_from_impl](family_t& i) { return i == ID_from_impl; }, ID_to_impl);
    familyID.toDevice();
}

void DEMDynamicThread::setSimParams(unsigned char nvXp2,
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
                                    double max_tritri_penetration,
                                    float expand_safety_param,
                                    float expand_safety_adder,
                                    bool use_angvel_margin,
                                    const std::set<std::string>& contact_wildcards,
                                    const std::set<std::string>& owner_wildcards,
                                    const std::set<std::string>& geo_wildcards) {
    simParams->nvXp2 = nvXp2;
    simParams->nvYp2 = nvYp2;
    simParams->nvZp2 = nvZp2;
    simParams->l = l;
    simParams->voxelSize = voxelSize;
    simParams->LBFX = LBFPoint.x;
    simParams->LBFY = LBFPoint.y;
    simParams->LBFZ = LBFPoint.z;
    simParams->Gx = G.x;
    simParams->Gy = G.y;
    simParams->Gz = G.z;
    simParams->nbX = nbX;
    simParams->nbY = nbY;
    simParams->nbZ = nbZ;
    simParams->userBoxMin = user_box_min;
    simParams->userBoxMax = user_box_max;

    simParams->dyn.binSize = binSize;
    simParams->dyn.inv_binSize = 1. / binSize;
    simParams->dyn.h = ts_size;
    simParams->dyn.beta = expand_factor;  // If beta is auto-adapting, this assignment has no effect
    simParams->dyn.approxMaxVel = approx_max_vel;
    simParams->dyn.expSafetyMulti = expand_safety_param;
    simParams->dyn.expSafetyAdder = expand_safety_adder;
    simParams->capTriTriPenetration = max_tritri_penetration;
    simParams->useAngVelMargin = use_angvel_margin ? 1 : 0;

    simParams->nContactWildcards = contact_wildcards.size();
    simParams->nOwnerWildcards = owner_wildcards.size();
    simParams->nGeoWildcards = geo_wildcards.size();

    m_contact_wildcard_names = contact_wildcards;
    m_owner_wildcard_names = owner_wildcards;
    m_geo_wildcard_names = geo_wildcards;
}

void DEMDynamicThread::changeOwnerSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors) {
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

    // Mark on the bool array those owners that need a change
    markOwnerToChange(idBool, ownerFactors, dIDs, dFactors, (size_t)IDs.size(), streamInfo.stream);

    // Change the size of the sphere components in question
    modifyComponents<DEMDataDT>(&granData, idBool, ownerFactors, (size_t)simParams->nSpheresGM, streamInfo.stream);

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

void DEMDynamicThread::allocateGPUArrays(size_t nOwnerBodies,
                                         size_t nOwnerClumps,
                                         unsigned int nExtObj,
                                         size_t nTriMeshes,
                                         size_t nSpheresGM,
                                         size_t nTriGM,
                                         size_t nMeshPatches,
                                         unsigned int nAnalGM,
                                         size_t nExtraContacts,
                                         unsigned int nMassProperties,
                                         unsigned int nClumpTopo,
                                         unsigned int nClumpComponents,
                                         unsigned int nJitifiableClumpComponents,
                                         unsigned int nMatTuples) {
    // dT buffer arrays should be on dT and this is to ensure that
    DEME_GPU_CALL(cudaSetDevice(streamInfo.device));

    // Sizes of these arrays
    simParams->nSpheresGM = nSpheresGM;
    simParams->nTriGM = nTriGM;
    simParams->nMeshPatches = nMeshPatches;
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
    DEME_DUAL_ARRAY_RESIZE(vX, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(vY, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(vZ, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(omgBarX, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(omgBarY, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(omgBarZ, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(aX, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(aY, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(aZ, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(alphaX, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(alphaY, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(alphaZ, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(accSpecified, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(angAccSpecified, nOwnerBodies, 0);

    // Resize the family mask `matrix' (in fact it is flattened)
    DEME_DUAL_ARRAY_RESIZE(familyMaskMatrix, (NUM_AVAL_FAMILIES + 1) * NUM_AVAL_FAMILIES / 2, DONT_PREVENT_CONTACT);

    // Resize to the number of geometries
    DEME_DUAL_ARRAY_RESIZE(ownerClumpBody, nSpheresGM, 0);
    DEME_DUAL_ARRAY_RESIZE(sphereMaterialOffset, nSpheresGM, 0);
    // For clump component offset, it's only needed if clump components are jitified
    if (solverFlags.useClumpJitify) {
        DEME_DUAL_ARRAY_RESIZE(clumpComponentOffset, nSpheresGM, 0);
        // This extended component offset array can hold offset numbers even for big clumps (whereas
        // clumpComponentOffset is typically uint_8, so it may not). If a sphere's component offset index falls in this
        // range then it is not jitified, and the kernel needs to look for it in the global memory.
        DEME_DUAL_ARRAY_RESIZE(clumpComponentOffsetExt, nSpheresGM, 0);
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

    // Resize to the number of triangle facets
    DEME_DUAL_ARRAY_RESIZE(ownerTriMesh, nTriGM, 0);
    DEME_DUAL_ARRAY_RESIZE(relPosNode1, nTriGM, make_float3(0));
    DEME_DUAL_ARRAY_RESIZE(relPosNode2, nTriGM, make_float3(0));
    DEME_DUAL_ARRAY_RESIZE(relPosNode3, nTriGM, make_float3(0));
    DEME_DUAL_ARRAY_RESIZE(triPatchID, nTriGM, 0);

    // Resize to the number of mesh patches
    DEME_DUAL_ARRAY_RESIZE(ownerPatchMesh, nMeshPatches, 0);
    DEME_DUAL_ARRAY_RESIZE(patchMaterialOffset, nMeshPatches, 0);
    DEME_DUAL_ARRAY_RESIZE(relPosPatch, nMeshPatches, make_float3(0));
    // maxTriTriPenetration usually keeps the max tri--tri penetration during the on-going simulation. But after
    // initialization, when it stores no meaningful values, dT will send a work order to kT, so maxTriTriPenetration's
    // value has to be initialized.
    DEME_GPU_CALL(cudaMemset(maxTriTriPenetration.getDevicePointer(), 0, sizeof(double)));

    // Resize to the number of analytical geometries
    DEME_DUAL_ARRAY_RESIZE(ownerAnalBody, nAnalGM, 0);

    // Resize to number of owners
    DEME_DUAL_ARRAY_RESIZE(ownerTypes, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(inertiaPropOffsets, nOwnerBodies, 0);
    // If we jitify mass properties, then
    if (solverFlags.useMassJitify) {
        DEME_DUAL_ARRAY_RESIZE(massOwnerBody, nMassProperties, 0);
        DEME_DUAL_ARRAY_RESIZE(mmiXX, nMassProperties, 0);
        DEME_DUAL_ARRAY_RESIZE(mmiYY, nMassProperties, 0);
        DEME_DUAL_ARRAY_RESIZE(mmiZZ, nMassProperties, 0);
    } else {
        DEME_DUAL_ARRAY_RESIZE(massOwnerBody, nOwnerBodies, 0);
        DEME_DUAL_ARRAY_RESIZE(mmiXX, nOwnerBodies, 0);
        DEME_DUAL_ARRAY_RESIZE(mmiYY, nOwnerBodies, 0);
        DEME_DUAL_ARRAY_RESIZE(mmiZZ, nOwnerBodies, 0);
    }
    // Volume info is jitified
    DEME_DUAL_ARRAY_RESIZE(volumeOwnerBody, nMassProperties, 0);

    // Arrays for contact info
    // The lengths of contact event-based arrays are just estimates.
    {
        // In any case, in this initialization process we should not make contact arrays smaller than it used to be, or
        // we may lose data. Also, if this is a new-boot, we allocate this array for at least INITIAL_CONTACT_ARRAY_SIZE
        // elements.
        //// TODO: Resizing contact arrays at initialization is a must and almost like a liability at this point. If you
        /// forget one of them, then if the sim entity number is small, you are likely to get segfault when you use them
        /// because some of them may never experienced resizing. This is not a good design.
        size_t cnt_arr_size =
            DEME_MAX(*solverScratchSpace.numPrimitiveContacts + nExtraContacts, INITIAL_CONTACT_ARRAY_SIZE);
        DEME_DUAL_ARRAY_RESIZE(idPrimitiveA, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(idPrimitiveB, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(contactTypePrimitive, cnt_arr_size, NOT_A_CONTACT);
        DEME_DUAL_ARRAY_RESIZE(geomToPatchMap, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(contactSATSatisfied, cnt_arr_size, 0);

        DEME_DUAL_ARRAY_RESIZE(idPatchA, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(idPatchB, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(contactTypePatch, cnt_arr_size, NOT_A_CONTACT);

        // If there are meshes, then sph--mesh case always use force storage, no getting around; if no mesh, then if no
        // need to store forces, we can choose to not resize these arrays.
        if (!(solverFlags.useNoContactRecord && simParams->nTriGM == 0)) {
            DEME_DUAL_ARRAY_RESIZE(contactForces, cnt_arr_size, make_float3(0));
            DEME_DUAL_ARRAY_RESIZE(contactTorque_convToForce, cnt_arr_size, make_float3(0));
            DEME_DUAL_ARRAY_RESIZE(contactPointGeometryA, cnt_arr_size, make_float3(0));
            DEME_DUAL_ARRAY_RESIZE(contactPointGeometryB, cnt_arr_size, make_float3(0));
        }
        // Allocate memory for each wildcard array
        contactWildcards.resize(simParams->nContactWildcards);
        ownerWildcards.resize(simParams->nOwnerWildcards);
        sphereWildcards.resize(simParams->nGeoWildcards);
        analWildcards.resize(simParams->nGeoWildcards);
        patchWildcards.resize(simParams->nGeoWildcards);
        for (unsigned int i = 0; i < simParams->nContactWildcards; i++) {
            contactWildcards[i] =
                std::make_unique<DualArray<float>>(cnt_arr_size, 0, &m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
        }
        for (unsigned int i = 0; i < simParams->nOwnerWildcards; i++) {
            ownerWildcards[i] =
                std::make_unique<DualArray<float>>(nOwnerBodies, 0, &m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
        }
        for (unsigned int i = 0; i < simParams->nGeoWildcards; i++) {
            sphereWildcards[i] =
                std::make_unique<DualArray<float>>(nSpheresGM, 0, &m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
            analWildcards[i] =
                std::make_unique<DualArray<float>>(nAnalGM, 0, &m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
            patchWildcards[i] =
                std::make_unique<DualArray<float>>(nMeshPatches, 0, &m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
        }
    }
    // existingContactTypes has a fixed size depending on how many contact types are defined
    DEME_DUAL_ARRAY_RESIZE(existingContactTypes, NUM_SUPPORTED_CONTACT_TYPES + 1, NOT_A_CONTACT);
    DEME_DUAL_ARRAY_RESIZE(typeStartOffsetsPrimitive, NUM_SUPPORTED_CONTACT_TYPES + 1, 0);
    DEME_DUAL_ARRAY_RESIZE(typeStartOffsetsPatch, NUM_SUPPORTED_CONTACT_TYPES + 1, 0);

    // You know what, let's not init dT's buffers, since kT will change it when needed anyway. Besides, changing it here
    // will cause problems in the case of a re-init-ed simulation with more clumps added to system, since we may
    // accidentally clamp those arrays.
}

void DEMDynamicThread::registerPolicies(const std::unordered_map<unsigned int, std::string>& template_number_name_map,
                                        const ClumpTemplateFlatten& clump_templates,
                                        const std::vector<float>& ext_obj_mass_types,
                                        const std::vector<float3>& ext_obj_moi_types,
                                        const std::vector<float>& mesh_obj_mass_types,
                                        const std::vector<float3>& mesh_obj_moi_types,
                                        const std::vector<std::shared_ptr<DEMMaterial>>& loaded_materials,
                                        const std::vector<notStupidBool_t>& family_mask_matrix,
                                        const std::set<unsigned int>& no_output_families) {
    // No modification for the arrays in this function. They can only be completely re-constructed.

    // Load in mass and MOI template info
    size_t k = 0;

    for (unsigned int i = 0; i < clump_templates.mass.size(); i++) {
        if (solverFlags.useMassJitify) {
            massOwnerBody[k] = clump_templates.mass.at(i);
            float3 this_moi = clump_templates.MOI.at(i);
            mmiXX[k] = this_moi.x;
            mmiYY[k] = this_moi.y;
            mmiZZ[k] = this_moi.z;
        }
        // Volume info is always registered, and even if the user does not use mass/MOI jitify, volume info may be
        // needed in void ratio computation
        volumeOwnerBody[k] = clump_templates.volume.at(i);
        k++;
    }
    for (unsigned int i = 0; i < ext_obj_mass_types.size(); i++) {
        if (solverFlags.useMassJitify) {
            massOwnerBody[k] = ext_obj_mass_types.at(i);
            float3 this_moi = ext_obj_moi_types.at(i);
            mmiXX[k] = this_moi.x;
            mmiYY[k] = this_moi.y;
            mmiZZ[k] = this_moi.z;
        }
        // Currently analytical object volume is not used
        k++;
    }
    for (unsigned int i = 0; i < mesh_obj_mass_types.size(); i++) {
        if (solverFlags.useMassJitify) {
            massOwnerBody[k] = mesh_obj_mass_types.at(i);
            float3 this_moi = mesh_obj_moi_types.at(i);
            mmiXX[k] = this_moi.x;
            mmiYY[k] = this_moi.y;
            mmiZZ[k] = this_moi.z;
        }
        // Currently mesh volume is not used
        k++;
    }

    // Store family mask
    for (size_t i = 0; i < family_mask_matrix.size(); i++)
        familyMaskMatrix[i] = family_mask_matrix.at(i);

    // Store clump naming map
    templateNumNameMap = template_number_name_map;

    // Take notes of the families that should not be outputted
    familiesNoOutput.clear();
    for (unsigned int x : no_output_families) {
        familiesNoOutput.insert(static_cast<family_t>(x));
    }
    DEME_DEBUG_PRINTF("Impl-level families that will not be outputted:");
    DEME_DEBUG_EXEC(for (family_t x : familiesNoOutput) { printf("%d ", static_cast<int>(x)); } printf("\n"););
}

void DEMDynamicThread::populateEntityArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                            const std::vector<float3>& input_ext_obj_xyz,
                                            const std::vector<float4>& input_ext_obj_rot,
                                            const std::vector<unsigned int>& input_ext_obj_family,
                                            const std::vector<std::shared_ptr<DEMMesh>>& input_mesh_objs,
                                            const std::vector<float3>& input_mesh_obj_xyz,
                                            const std::vector<float4>& input_mesh_obj_rot,
                                            const std::vector<unsigned int>& input_mesh_obj_family,
                                            const std::vector<unsigned int>& mesh_facet_owner,
                                            const std::vector<bodyID_t>& mesh_facet_patch,
                                            const std::vector<DEMTriangle>& mesh_facets,
                                            const std::vector<bodyID_t>& mesh_patch_owner,
                                            const std::vector<materialsOffset_t>& mesh_patch_materials,
                                            const ClumpTemplateFlatten& clump_templates,
                                            const std::vector<float>& ext_obj_mass_types,
                                            const std::vector<float3>& ext_obj_moi_types,
                                            const std::vector<unsigned int>& ext_obj_comp_num,
                                            const std::vector<float>& mesh_obj_mass_types,
                                            const std::vector<float3>& mesh_obj_moi_types,
                                            const std::vector<inertiaOffset_t>& mesh_obj_mass_offsets,
                                            size_t nExistOwners,
                                            size_t nExistSpheres,
                                            size_t nExistingFacets,
                                            size_t nExistingMeshPatches) {
    // Load in clump components info (but only if instructed to use jitified clump templates). This step will be
    // repeated even if we are just adding some more clumps to system, not a complete re-initialization.
    size_t k = 0;
    std::vector<unsigned int> prescans_comp;
    if (solverFlags.useClumpJitify) {
        prescans_comp.push_back(0);
        for (const auto& elem : clump_templates.spRadii) {
            for (const auto& radius : elem) {
                radiiSphere[k] = radius;
                k++;
            }
            prescans_comp.push_back(k);
        }
        prescans_comp.pop_back();
        k = 0;

        for (const auto& elem : clump_templates.spRelPos) {
            for (const auto& loc : elem) {
                relPosSphereX[k] = loc.x;
                relPosSphereY[k] = loc.y;
                relPosSphereZ[k] = loc.z;
                k++;
            }
        }
    }

    // Left-bottom-front point of the `world'
    float3 LBF;
    LBF.x = simParams->LBFX;
    LBF.y = simParams->LBFY;
    LBF.z = simParams->LBFZ;
    k = 0;

    size_t nTotalClumpsThisCall = 0;
    {
        // Use i to record the current index of clump being processed
        size_t i = 0;
        // We give warning only once
        bool pop_family_msg = false;
        bool in_domain_msg = false;
        float3 sus_point;
        // Keep tab of the number of sphere components processed in this initialization call, especially if there are
        // multiple batches loaded for this initialization call
        size_t n_processed_sp_comp = 0;
        // This number serves as an offset for loading existing contact pairs/history. Contact array should have been
        // enlarged for loading these user-manually added contact pairs. Those pairs go after existing contact pairs.
        size_t cnt_arr_offset = *solverScratchSpace.numContacts;
        for (const auto& a_batch : input_clump_batches) {
            // Decode type number and flatten
            std::vector<unsigned int> type_marks(a_batch->GetNumClumps());
            for (size_t j = 0; j < a_batch->GetNumClumps(); j++) {
                type_marks.at(j) = a_batch->types.at(j)->mark;
            }
            // Now a ref to xyz
            const std::vector<float3>& input_clump_xyz = a_batch->xyz;
            // Now a ref to vel
            const std::vector<float3>& input_clump_vel = a_batch->vel;
            // Now a ref to quaternion
            const std::vector<float4>& input_clump_oriQ = a_batch->oriQ;
            // Now a ref to angular velocity
            const std::vector<float3>& input_clump_angVel = a_batch->angVel;
            // For family numbers, we check if the user has explicitly set them. If not, send a warning.
            if (!(a_batch->family_isSpecified)) {
                pop_family_msg = true;
            }
            const std::vector<unsigned int>& input_clump_family = a_batch->families;

            for (size_t j = 0; j < a_batch->GetNumClumps(); j++) {
                // If got here, this is a clump
                ownerTypes[nExistOwners + i] = OWNER_T_CLUMP;

                auto type_of_this_clump = type_marks.at(j);
                inertiaPropOffsets[nExistOwners + i] = type_of_this_clump;
                if (!solverFlags.useMassJitify) {
                    massOwnerBody[nExistOwners + i] = clump_templates.mass.at(type_of_this_clump);
                    const float3 this_moi = clump_templates.MOI.at(type_of_this_clump);
                    mmiXX[nExistOwners + i] = this_moi.x;
                    mmiYY[nExistOwners + i] = this_moi.y;
                    mmiZZ[nExistOwners + i] = this_moi.z;
                }

                // For clumps, special courtesy from us to check if it falls in user's box
                float3 this_clump_xyz = input_clump_xyz.at(j);
                if (!isBetween(this_clump_xyz, simParams->userBoxMin, simParams->userBoxMax)) {
                    sus_point = this_clump_xyz;
                    in_domain_msg = true;
                }
                float3 this_CoM_coord = this_clump_xyz - LBF;

                auto this_clump_no_sp_radii = clump_templates.spRadii.at(type_of_this_clump);
                auto this_clump_no_sp_relPos = clump_templates.spRelPos.at(type_of_this_clump);
                auto this_clump_no_sp_mat_ids = clump_templates.matIDs.at(type_of_this_clump);

                for (size_t jj = 0; jj < this_clump_no_sp_radii.size(); jj++) {
                    sphereMaterialOffset[nExistSpheres + k] = this_clump_no_sp_mat_ids.at(jj);
                    ownerClumpBody[nExistSpheres + k] = nExistOwners + i;

                    // Depending on whether we jitify or flatten
                    if (solverFlags.useClumpJitify) {
                        // This component offset, is it too large that can't live in the jitified array?
                        unsigned int this_comp_offset = prescans_comp.at(type_of_this_clump) + jj;
                        clumpComponentOffsetExt[nExistSpheres + k] = this_comp_offset;
                        if (this_comp_offset < simParams->nJitifiableClumpComponents) {
                            clumpComponentOffset[nExistSpheres + k] = this_comp_offset;
                        } else {
                            // If not, an indicator will be put there
                            clumpComponentOffset[nExistSpheres + k] = RESERVED_CLUMP_COMPONENT_OFFSET;
                        }
                    } else {
                        radiiSphere[nExistSpheres + k] = this_clump_no_sp_radii.at(jj);
                        const float3 relPos = this_clump_no_sp_relPos.at(jj);
                        relPosSphereX[nExistSpheres + k] = relPos.x;
                        relPosSphereY[nExistSpheres + k] = relPos.y;
                        relPosSphereZ[nExistSpheres + k] = relPos.z;
                    }

                    k++;
                    // std::cout << "Sphere Rel Pos offset: " << this_clump_no_sp_loc_offsets.at(j) << std::endl;
                }

                positionToVoxelID<voxelID_t, subVoxelPos_t, double>(
                    voxelID[nExistOwners + i], locX[nExistOwners + i], locY[nExistOwners + i], locZ[nExistOwners + i],
                    (double)this_CoM_coord.x, (double)this_CoM_coord.y, (double)this_CoM_coord.z, simParams->nvXp2,
                    simParams->nvYp2, simParams->voxelSize, simParams->l);

                // Set initial oriQ
                auto oriQ_of_this_clump = input_clump_oriQ.at(j);
                oriQw[nExistOwners + i] = oriQ_of_this_clump.w;
                oriQx[nExistOwners + i] = oriQ_of_this_clump.x;
                oriQy[nExistOwners + i] = oriQ_of_this_clump.y;
                oriQz[nExistOwners + i] = oriQ_of_this_clump.z;

                // Set initial velocity
                auto vel_of_this_clump = input_clump_vel.at(j);
                vX[nExistOwners + i] = vel_of_this_clump.x;
                vY[nExistOwners + i] = vel_of_this_clump.y;
                vZ[nExistOwners + i] = vel_of_this_clump.z;

                // Set initial angular velocity
                auto angVel_of_this_clump = input_clump_angVel.at(j);
                omgBarX[nExistOwners + i] = angVel_of_this_clump.x;
                omgBarY[nExistOwners + i] = angVel_of_this_clump.y;
                omgBarZ[nExistOwners + i] = angVel_of_this_clump.z;

                // Set family code
                family_t this_family_num = input_clump_family.at(j);
                familyID[nExistOwners + i] = this_family_num;

                i++;
            }
            // If this batch has wildcards, we load it in
            {
                unsigned int w_num = 0;
                // Owner wildcard first
                for (const auto& w_name : m_owner_wildcard_names) {
                    if (a_batch->owner_wildcards.find(w_name) == a_batch->owner_wildcards.end()) {
                        // No such wildcard loaded
                        DEME_WARNING(
                            "Owner wildcard %s is needed by force model, yet not specified for a batch of "
                            "clumps.\nTheir initial values are defauled to 0.",
                            w_name.c_str());
                    } else {
                        for (size_t jj = 0; jj < a_batch->GetNumClumps(); jj++) {
                            (*ownerWildcards[w_num])[nExistOwners + nTotalClumpsThisCall + jj] =
                                a_batch->owner_wildcards[w_name].at(jj);
                        }
                    }
                    w_num++;
                }
                // Then geo wildcards
                w_num = 0;
                for (const auto& w_name : m_geo_wildcard_names) {
                    if (a_batch->geo_wildcards.find(w_name) == a_batch->geo_wildcards.end()) {
                        // No such wildcard loaded
                        DEME_WARNING(
                            "Geometry wildcard %s is needed by force model, yet not specified for a batch of "
                            "clumps.\nTheir initial values are defauled to 0.",
                            w_name.c_str());
                    } else {
                        for (size_t jj = 0; jj < a_batch->GetNumSpheres(); jj++) {
                            (*sphereWildcards[w_num])[nExistSpheres + n_processed_sp_comp + jj] =
                                a_batch->geo_wildcards[w_name].at(jj);
                        }
                    }
                    w_num++;
                }
            }

            DEME_DEBUG_PRINTF("Loaded a batch of %zu clumps.", a_batch->GetNumClumps());
            DEME_DEBUG_PRINTF("This batch has %zu spheres.", a_batch->GetNumSpheres());

            // Write the extra contact pairs to memory
            for (size_t jj = 0; jj < a_batch->GetNumContacts(); jj++) {
                const auto& idPair = a_batch->contact_pairs.at(jj);
                // idPair.first + n_processed_sp_comp can take into account the sphere components that have been loaded
                // in previous batches, makes this loading process scalable.
                idPatchA[cnt_arr_offset] = idPair.first + n_processed_sp_comp + nExistSpheres;
                idPatchB[cnt_arr_offset] = idPair.second + n_processed_sp_comp + nExistSpheres;
                contactTypePatch[cnt_arr_offset] = SPHERE_SPHERE_CONTACT;  // Only sph--sph cnt for now
                unsigned int w_num = 0;
                for (const auto& w_name : m_contact_wildcard_names) {
                    (*contactWildcards[w_num])[cnt_arr_offset] = a_batch->contact_wildcards.at(w_name).at(jj);
                    w_num++;
                }
                cnt_arr_offset++;
            }

            // Make ready for the next batch...
            n_processed_sp_comp = k;
            nTotalClumpsThisCall = i;
        }

        DEME_DEBUG_PRINTF("Total number of transferred clumps this time: %zu", i);
        DEME_DEBUG_PRINTF("Total number of existing owners in simulation: %zu", nExistOwners);
        DEME_DEBUG_PRINTF("Total number of owners in simulation after this init call: %zu",
                          (size_t)simParams->nOwnerBodies);

        // If user loaded contact pairs, we need to inform kT on the first time step...
        if (cnt_arr_offset > *solverScratchSpace.numContacts) {
            *solverScratchSpace.numContacts = cnt_arr_offset;
            new_contacts_loaded = true;
            DEME_DEBUG_PRINTF("Total number of contact pairs this sim starts with: %zu",
                              *solverScratchSpace.numContacts);
        }

        if (pop_family_msg) {
            DEME_WARNING("Some clumps do not have their family numbers specified, so defaulted to %u",
                         DEFAULT_CLUMP_FAMILY_NUM);
        }
        if (in_domain_msg) {
            DEME_WARNING(
                "At least one clump is initialized with a position out of the box domain you specified.\nIt is found "
                "at %.5g, %.5g, %.5g (this message only shows one such example).\nThis simulation is unlikely to go as "
                "planned.",
                sus_point.x, sus_point.y, sus_point.z);
        }
    }

    // Load in initial positions and mass properties for the owners of those external objects
    // They go after clump owners
    k = 0;
    size_t owner_offset_for_ext_obj = nExistOwners + nTotalClumpsThisCall;
    unsigned int offset_for_ext_obj_mass_template = simParams->nDistinctClumpBodyTopologies;
    for (size_t i = 0; i < input_ext_obj_xyz.size(); i++) {
        // If got here, it is an analytical obj
        ownerTypes[i + owner_offset_for_ext_obj] = OWNER_T_ANALYTICAL;
        // For each analytical geometry component of this obj, it needs to know its owner number
        for (size_t j = 0; j < ext_obj_comp_num.at(i); j++) {
            ownerAnalBody[k] = i + owner_offset_for_ext_obj;
            k++;
        }

        // Analytical object mass properties are useful in force collection, but not useful in force calculation:
        // analytical component masses are jitified into kernels directly.
        inertiaPropOffsets[i + owner_offset_for_ext_obj] = i + offset_for_ext_obj_mass_template;
        if (!solverFlags.useMassJitify) {
            massOwnerBody[i + owner_offset_for_ext_obj] = ext_obj_mass_types.at(i);
            const float3 this_moi = ext_obj_moi_types.at(i);
            mmiXX[i + owner_offset_for_ext_obj] = this_moi.x;
            mmiYY[i + owner_offset_for_ext_obj] = this_moi.y;
            mmiZZ[i + owner_offset_for_ext_obj] = this_moi.z;
        }
        auto this_CoM_coord = input_ext_obj_xyz.at(i) - LBF;
        // std::cout << this_CoM_coord.x << "," << this_CoM_coord.y << "," << this_CoM_coord.z << std::endl;
        positionToVoxelID<voxelID_t, subVoxelPos_t, double>(
            voxelID[i + owner_offset_for_ext_obj], locX[i + owner_offset_for_ext_obj],
            locY[i + owner_offset_for_ext_obj], locZ[i + owner_offset_for_ext_obj], (double)this_CoM_coord.x,
            (double)this_CoM_coord.y, (double)this_CoM_coord.z, simParams->nvXp2, simParams->nvYp2,
            simParams->voxelSize, simParams->l);
        // Set mesh owner's oriQ
        auto oriQ_of_this = input_ext_obj_rot.at(i);
        oriQw[i + owner_offset_for_ext_obj] = oriQ_of_this.w;
        oriQx[i + owner_offset_for_ext_obj] = oriQ_of_this.x;
        oriQy[i + owner_offset_for_ext_obj] = oriQ_of_this.y;
        oriQz[i + owner_offset_for_ext_obj] = oriQ_of_this.z;

        //// For setting initial vel ang angvel, DEME's expected usage is now setting them using trackers after
        /// initialization. / For clumps, their init vel can be set via initializers because of historical reasons.

        family_t this_family_num = input_ext_obj_family.at(i);
        familyID[i + owner_offset_for_ext_obj] = this_family_num;
    }

    // Load in initial positions and mass properties for the owners of the meshed objects
    // They go after analytical object owners
    size_t owner_offset_for_mesh_obj = owner_offset_for_ext_obj + input_ext_obj_xyz.size();
    unsigned int offset_for_mesh_obj_mass_template = offset_for_ext_obj_mass_template + input_ext_obj_xyz.size();
    // k for indexing the triangle facets
    k = 0;
    // p for indexing patches (flattened across all meshes)
    size_t p = 0;
    for (size_t i = 0; i < input_mesh_objs.size(); i++) {
        // If got here, it is a mesh
        ownerTypes[i + owner_offset_for_mesh_obj] = OWNER_T_MESH;

        // Store inherent geo wildcards
        {
            unsigned int w_num = 0;
            for (const auto& w_name : m_geo_wildcard_names) {
                if (input_mesh_objs.at(i)->geo_wildcards.find(w_name) == input_mesh_objs.at(i)->geo_wildcards.end()) {
                    // No such wildcard loaded
                    DEME_WARNING(
                        "Geometry wildcard %s is needed by force model, yet not specified for a mesh.\nTheir "
                        "initial values are defauled to 0.",
                        w_name.c_str());
                } else {
                    for (size_t jj = 0; jj < input_mesh_objs.at(i)->GetNumPatches(); jj++) {
                        (*patchWildcards[w_num])[nExistingMeshPatches + p + jj] =
                            input_mesh_objs.at(i)->geo_wildcards[w_name].at(jj);
                    }
                }
                w_num++;
            }
        }

        // Store this mesh in dT's cache
        input_mesh_objs.at(i)->owner = i + owner_offset_for_mesh_obj;
        input_mesh_objs.at(i)->cache_offset = m_meshes.size();
        m_meshes.push_back(input_mesh_objs.at(i));

        inertiaPropOffsets[i + owner_offset_for_mesh_obj] =
            offset_for_mesh_obj_mass_template + mesh_obj_mass_offsets.at(i);
        if (!solverFlags.useMassJitify) {
            massOwnerBody[i + owner_offset_for_mesh_obj] = mesh_obj_mass_types.at(i);
            const float3 this_moi = mesh_obj_moi_types.at(i);
            mmiXX[i + owner_offset_for_mesh_obj] = this_moi.x;
            mmiYY[i + owner_offset_for_mesh_obj] = this_moi.y;
            mmiZZ[i + owner_offset_for_mesh_obj] = this_moi.z;
        }
        auto this_CoM_coord = input_mesh_obj_xyz.at(i) - LBF;
        positionToVoxelID<voxelID_t, subVoxelPos_t, double>(
            voxelID[i + owner_offset_for_mesh_obj], locX[i + owner_offset_for_mesh_obj],
            locY[i + owner_offset_for_mesh_obj], locZ[i + owner_offset_for_mesh_obj], (double)this_CoM_coord.x,
            (double)this_CoM_coord.y, (double)this_CoM_coord.z, simParams->nvXp2, simParams->nvYp2,
            simParams->voxelSize, simParams->l);

        // Set mesh owner's oriQ
        auto oriQ_of_this = input_mesh_obj_rot.at(i);
        oriQw[i + owner_offset_for_mesh_obj] = oriQ_of_this.w;
        oriQx[i + owner_offset_for_mesh_obj] = oriQ_of_this.x;
        oriQy[i + owner_offset_for_mesh_obj] = oriQ_of_this.y;
        oriQz[i + owner_offset_for_mesh_obj] = oriQ_of_this.z;

        //// For setting initial vel ang angvel, DEME's expected usage is now setting them using trackers after
        /// initialization. / For clumps, their init vel can be set via initializers because of historical reasons.

        // Populate patch info for this mesh

        // Populate patch locations for this mesh
        // If explicitly set, use those; otherwise compute them
        std::vector<float3> this_mesh_patch_locations;
        if (input_mesh_objs.at(i)->patch_locations_explicitly_set) {
            this_mesh_patch_locations = input_mesh_objs.at(i)->m_patch_locations;
        } else {
            this_mesh_patch_locations = input_mesh_objs.at(i)->ComputePatchLocations();
        }

        // mesh_patch_owner run length is the num of patches in this mesh entity
        //// TODO: This flatten-then-init approach is historical and too ugly.
        size_t this_patch_owner = mesh_patch_owner.at(p);
        size_t p_start = p;  // Record where patch ID of this run starts
        for (; p < mesh_patch_owner.size(); p++) {
            if (mesh_patch_owner.at(p) != this_patch_owner)
                break;
            ownerPatchMesh[nExistingMeshPatches + p] = owner_offset_for_mesh_obj + this_patch_owner;
            patchMaterialOffset[nExistingMeshPatches + p] = mesh_patch_materials.at(p);
            relPosPatch[nExistingMeshPatches + p] = this_mesh_patch_locations[p - p_start];
        }

        // Per-facet info
        //// TODO: This flatten-then-init approach is historical and too ugly.
        size_t this_facet_owner = mesh_facet_owner.at(k);
        for (; k < mesh_facet_owner.size(); k++) {
            // mesh_facet_owner run length is the num of facets in this mesh entity
            if (mesh_facet_owner.at(k) != this_facet_owner)
                break;
            ownerTriMesh[nExistingFacets + k] = owner_offset_for_mesh_obj + this_facet_owner;
            // Tri's patch belonging needs to take into account those patches that are previously added
            triPatchID[nExistingFacets + k] = nExistingMeshPatches + mesh_facet_patch.at(k);
            DEMTriangle this_tri = mesh_facets.at(k);
            relPosNode1[nExistingFacets + k] = this_tri.p1;
            relPosNode2[nExistingFacets + k] = this_tri.p2;
            relPosNode3[nExistingFacets + k] = this_tri.p3;
        }

        family_t this_family_num = input_mesh_obj_family.at(i);
        familyID[i + owner_offset_for_mesh_obj] = this_family_num;

        // Cached initial values for wildcards of this mesh is not needed anymore
        m_meshes.back()->ClearWildcards();

        // DEME_DEBUG_PRINTF("dT just loaded a mesh in family %u", +(this_family_num));
        // DEME_DEBUG_PRINTF("This mesh is owner %zu", (i + owner_offset_for_mesh_obj));
    }
    DEME_DEBUG_PRINTF("Number of meshes loaded this time: %zu", input_mesh_objs.size());
    DEME_DEBUG_PRINTF("Number of mesh patches loaded this time: %zu", p);
    DEME_DEBUG_PRINTF("Number of triangle facets loaded this time: %zu", k);
}

void DEMDynamicThread::buildTrackedObjs(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                        const std::vector<unsigned int>& ext_obj_comp_num,
                                        const std::vector<std::shared_ptr<DEMMesh>>& input_mesh_objs,
                                        std::vector<std::shared_ptr<DEMTrackedObj>>& tracked_objs,
                                        size_t nExistOwners,
                                        size_t nExistSpheres,
                                        size_t nExistingPatches,
                                        unsigned int nExistingAnalGM) {
    // We take notes on how many clumps each batch has, it will be useful when we assemble the tracker information
    std::vector<size_t> prescans_batch_size, prescans_batch_sphere_size;
    prescans_batch_size.push_back(0);
    prescans_batch_sphere_size.push_back(0);
    for (const auto& a_batch : input_clump_batches) {
        prescans_batch_size.push_back(prescans_batch_size.back() + a_batch->GetNumClumps());
        prescans_batch_sphere_size.push_back(prescans_batch_sphere_size.back() + a_batch->GetNumSpheres());
    }
    // Also take notes of num of analytical geometries of each analytical body
    std::vector<size_t> prescans_ext_obj_size;
    prescans_ext_obj_size.push_back(0);
    for (const auto& geo_num : ext_obj_comp_num) {
        prescans_ext_obj_size.push_back(prescans_ext_obj_size.back() + geo_num);
    }
    // Also take notes of num of patches of each mesh obj
    std::vector<size_t> prescans_mesh_size;
    prescans_mesh_size.push_back(0);
    for (const auto& a_mesh : input_mesh_objs) {
        prescans_mesh_size.push_back(prescans_mesh_size.back() + a_mesh->GetNumPatches());
    }

    // Provide feedback to the tracked objects, tell them the owner numbers they are looking for
    // Little computation is needed, as long as we know the structure of our owner array: nOwnerClumps go first, then
    // nExtObj, then nTriMeshes
    // Also note, we just have to process those haven't been processed
    for (unsigned int i = nTrackersProcessed; i < tracked_objs.size(); i++) {
        auto& tracked_obj = tracked_objs.at(i);
        switch (tracked_obj->obj_type) {
            case (OWNER_TYPE::CLUMP):
                tracked_obj->ownerID = nExistOwners + prescans_batch_size.at(tracked_obj->load_order);
                tracked_obj->nSpanOwners = prescans_batch_size.at(tracked_obj->load_order + 1) -
                                           prescans_batch_size.at(tracked_obj->load_order);
                tracked_obj->geoID = nExistSpheres + prescans_batch_sphere_size.at(tracked_obj->load_order);
                tracked_obj->nGeos = prescans_batch_sphere_size.at(tracked_obj->load_order + 1) -
                                     prescans_batch_sphere_size.at(tracked_obj->load_order);
                break;
            case (OWNER_TYPE::ANALYTICAL):
                // prescans_batch_size.back() is the total num of loaded clumps this time
                tracked_obj->ownerID = nExistOwners + tracked_obj->load_order + prescans_batch_size.back();
                tracked_obj->nSpanOwners = 1;
                tracked_obj->geoID = nExistingAnalGM + prescans_ext_obj_size.at(tracked_obj->load_order);
                tracked_obj->nGeos = prescans_ext_obj_size.at(tracked_obj->load_order + 1) -
                                     prescans_ext_obj_size.at(tracked_obj->load_order);
                break;
            case (OWNER_TYPE::MESH):
                tracked_obj->ownerID =
                    nExistOwners + ext_obj_comp_num.size() + prescans_batch_size.back() + tracked_obj->load_order;
                tracked_obj->nSpanOwners = 1;
                tracked_obj->geoID = nExistingPatches + prescans_mesh_size.at(tracked_obj->load_order);
                // For mesh, nGeos is the number of patches
                tracked_obj->nGeos =
                    prescans_mesh_size.at(tracked_obj->load_order + 1) - prescans_mesh_size.at(tracked_obj->load_order);
                break;
            default:
                DEME_ERROR(std::string("A DEM tracked object has an unknown type."));
        }
    }
    nTrackersProcessed = tracked_objs.size();
    DEME_DEBUG_PRINTF("Total number of trackers on the record: %u", nTrackersProcessed);
}

void DEMDynamicThread::initGPUArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                     const std::vector<float3>& input_ext_obj_xyz,
                                     const std::vector<float4>& input_ext_obj_rot,
                                     const std::vector<unsigned int>& input_ext_obj_family,
                                     const std::vector<std::shared_ptr<DEMMesh>>& input_mesh_objs,
                                     const std::vector<float3>& input_mesh_obj_xyz,
                                     const std::vector<float4>& input_mesh_obj_rot,
                                     const std::vector<unsigned int>& input_mesh_obj_family,
                                     const std::vector<unsigned int>& mesh_facet_owner,
                                     const std::vector<bodyID_t>& mesh_facet_patch,
                                     const std::vector<DEMTriangle>& mesh_facets,
                                     const std::vector<bodyID_t>& mesh_patch_owner,
                                     const std::vector<materialsOffset_t>& mesh_patch_materials,
                                     const std::unordered_map<unsigned int, std::string>& template_number_name_map,
                                     const ClumpTemplateFlatten& clump_templates,
                                     const std::vector<float>& ext_obj_mass_types,
                                     const std::vector<float3>& ext_obj_moi_types,
                                     const std::vector<unsigned int>& ext_obj_comp_num,
                                     const std::vector<float>& mesh_obj_mass_types,
                                     const std::vector<float3>& mesh_obj_moi_types,
                                     const std::vector<float>& mesh_obj_mass_jit_types,
                                     const std::vector<float3>& mesh_obj_moi_jit_types,
                                     const std::vector<inertiaOffset_t>& mesh_obj_mass_offsets,
                                     const std::vector<std::shared_ptr<DEMMaterial>>& loaded_materials,
                                     const std::vector<notStupidBool_t>& family_mask_matrix,
                                     const std::set<unsigned int>& no_output_families,
                                     std::vector<std::shared_ptr<DEMTrackedObj>>& tracked_objs) {
    // Get the info into the GPU memory from the host side. Can this process be more efficient? Maybe, but it's
    // initialization anyway.

    registerPolicies(template_number_name_map, clump_templates, ext_obj_mass_types, ext_obj_moi_types,
                     mesh_obj_mass_jit_types, mesh_obj_moi_jit_types, loaded_materials, family_mask_matrix,
                     no_output_families);

    // For initialization, owner array offset is 0
    populateEntityArrays(input_clump_batches, input_ext_obj_xyz, input_ext_obj_rot, input_ext_obj_family,
                         input_mesh_objs, input_mesh_obj_xyz, input_mesh_obj_rot, input_mesh_obj_family,
                         mesh_facet_owner, mesh_facet_patch, mesh_facets, mesh_patch_owner, mesh_patch_materials,
                         clump_templates, ext_obj_mass_types, ext_obj_moi_types, ext_obj_comp_num, mesh_obj_mass_types,
                         mesh_obj_moi_types, mesh_obj_mass_offsets, 0, 0, 0, 0);

    buildTrackedObjs(input_clump_batches, ext_obj_comp_num, input_mesh_objs, tracked_objs, 0, 0, 0, 0);
}

void DEMDynamicThread::updateClumpMeshArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                             const std::vector<float3>& input_ext_obj_xyz,
                                             const std::vector<float4>& input_ext_obj_rot,
                                             const std::vector<unsigned int>& input_ext_obj_family,
                                             const std::vector<std::shared_ptr<DEMMesh>>& input_mesh_objs,
                                             const std::vector<float3>& input_mesh_obj_xyz,
                                             const std::vector<float4>& input_mesh_obj_rot,
                                             const std::vector<unsigned int>& input_mesh_obj_family,
                                             const std::vector<unsigned int>& mesh_facet_owner,
                                             const std::vector<bodyID_t>& mesh_facet_patch,
                                             const std::vector<DEMTriangle>& mesh_facets,
                                             const std::vector<bodyID_t>& mesh_patch_owner,
                                             const std::vector<materialsOffset_t>& mesh_patch_materials,
                                             const ClumpTemplateFlatten& clump_templates,
                                             const std::vector<float>& ext_obj_mass_types,
                                             const std::vector<float3>& ext_obj_moi_types,
                                             const std::vector<unsigned int>& ext_obj_comp_num,
                                             const std::vector<float>& mesh_obj_mass_types,
                                             const std::vector<float3>& mesh_obj_moi_types,
                                             const std::vector<float>& mesh_obj_mass_jit_types,
                                             const std::vector<float3>& mesh_obj_moi_jit_types,
                                             const std::vector<inertiaOffset_t>& mesh_obj_mass_offsets,
                                             const std::vector<std::shared_ptr<DEMMaterial>>& loaded_materials,
                                             const std::vector<notStupidBool_t>& family_mask_matrix,
                                             const std::set<unsigned int>& no_output_families,
                                             std::vector<std::shared_ptr<DEMTrackedObj>>& tracked_objs,
                                             size_t nExistingOwners,
                                             size_t nExistingClumps,
                                             size_t nExistingSpheres,
                                             size_t nExistingTriMesh,
                                             size_t nExistingFacets,
                                             size_t nExistingPatches,
                                             unsigned int nExistingObj,
                                             unsigned int nExistingAnalGM) {
    // No policy changes here
    (void)mesh_obj_mass_jit_types;
    (void)mesh_obj_moi_jit_types;

    // Analytical objects-related arrays should be empty
    populateEntityArrays(input_clump_batches, input_ext_obj_xyz, input_ext_obj_rot, input_ext_obj_family,
                         input_mesh_objs, input_mesh_obj_xyz, input_mesh_obj_rot, input_mesh_obj_family,
                         mesh_facet_owner, mesh_facet_patch, mesh_facets, mesh_patch_owner, mesh_patch_materials,
                         clump_templates, ext_obj_mass_types, ext_obj_moi_types, ext_obj_comp_num, mesh_obj_mass_types,
                         mesh_obj_moi_types, mesh_obj_mass_offsets, nExistingOwners, nExistingSpheres, nExistingFacets,
                         nExistingPatches);

    // Make changes to tracked objects (potentially add more)
    buildTrackedObjs(input_clump_batches, ext_obj_comp_num, input_mesh_objs, tracked_objs, nExistingOwners,
                     nExistingSpheres, nExistingPatches, nExistingAnalGM);
}

#ifdef DEME_USE_CHPF
void DEMDynamicThread::writeSpheresAsChpf(std::ofstream& ptFile) {
    chpf::Writer pw;
    // pw.write(ptFile, chpf::Compressor::Type::USE_DEFAULT, mass);
    migrateFamilyToHost();
    migrateClumpPosInfoToHost();
    migrateClumpHighOrderInfoToHost();

    // simParams host version should not be different from device version, so no need to update
    std::vector<float> posX(simParams->nSpheresGM);
    std::vector<float> posY(simParams->nSpheresGM);
    std::vector<float> posZ(simParams->nSpheresGM);
    std::vector<float> spRadii(simParams->nSpheresGM);
    std::vector<unsigned int> families;
    if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
        families.resize(simParams->nSpheresGM);
    }
    size_t num_output_spheres = 0;

    for (size_t i = 0; i < simParams->nSpheresGM; i++) {
        auto this_owner = ownerClumpBody[i];
        family_t this_family = familyID[this_owner];
        // If this (impl-level) family is in the no-output list, skip it
        if (familiesNoOutput.find(this_family) != familiesNoOutput.end()) {
            continue;
        }

        float3 CoM;
        float X, Y, Z;
        voxelID_t voxel = voxelID[this_owner];
        subVoxelPos_t subVoxX = locX[this_owner];
        subVoxelPos_t subVoxY = locY[this_owner];
        subVoxelPos_t subVoxZ = locZ[this_owner];
        voxelIDToPosition<float, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel, subVoxX, subVoxY, subVoxZ, simParams->nvXp2,
                                                           simParams->nvYp2, simParams->voxelSize, simParams->l);
        CoM.x = X + simParams->LBFX;
        CoM.y = Y + simParams->LBFY;
        CoM.z = Z + simParams->LBFZ;

        size_t compOffset = (solverFlags.useClumpJitify) ? clumpComponentOffsetExt[i] : i;
        float this_sp_deviation_x = relPosSphereX[compOffset];
        float this_sp_deviation_y = relPosSphereY[compOffset];
        float this_sp_deviation_z = relPosSphereZ[compOffset];
        float this_sp_rot_0 = oriQw[this_owner];
        float this_sp_rot_1 = oriQx[this_owner];
        float this_sp_rot_2 = oriQy[this_owner];
        float this_sp_rot_3 = oriQz[this_owner];
        applyOriQToVector3<float, float>(this_sp_deviation_x, this_sp_deviation_y, this_sp_deviation_z, this_sp_rot_0,
                                         this_sp_rot_1, this_sp_rot_2, this_sp_rot_3);
        posX.at(num_output_spheres) = CoM.x + this_sp_deviation_x;
        posY.at(num_output_spheres) = CoM.y + this_sp_deviation_y;
        posZ.at(num_output_spheres) = CoM.z + this_sp_deviation_z;
        // std::cout << "Sphere Pos: " << posX.at(i) << ", " << posY.at(i) << ", " << posZ.at(i) << std::endl;

        spRadii.at(num_output_spheres) = radiiSphere[compOffset];

        // Family number
        if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
            families.at(num_output_spheres) = this_family;
        }

        num_output_spheres++;
    }
    // Write basics
    posX.resize(num_output_spheres);
    posY.resize(num_output_spheres);
    posZ.resize(num_output_spheres);
    spRadii.resize(num_output_spheres);
    // TODO: Set {} to the list of column names
    pw.write(ptFile, chpf::Compressor::Type::USE_DEFAULT,
             {OUTPUT_FILE_X_COL_NAME, OUTPUT_FILE_Y_COL_NAME, OUTPUT_FILE_Z_COL_NAME, OUTPUT_FILE_R_COL_NAME}, posX,
             posY, posZ, spRadii);
    // Write family numbers
    if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
        families.resize(num_output_spheres);
        //// TODO: How to do that?
        // pw.write(ptFile, chpf::Compressor::Type::USE_DEFAULT, {}, families);
    }
}
#endif

void DEMDynamicThread::writeSpheresAsCsv(std::ofstream& ptFile) {
    std::ostringstream outstrstream;

    migrateFamilyToHost();
    migrateClumpPosInfoToHost();
    migrateClumpHighOrderInfoToHost();
    migrateOwnerWildcardToHost();
    migrateSphGeoWildcardToHost();

    outstrstream << OUTPUT_FILE_X_COL_NAME + "," + OUTPUT_FILE_Y_COL_NAME + "," + OUTPUT_FILE_Z_COL_NAME + "," +
                        OUTPUT_FILE_R_COL_NAME;

    if (solverFlags.outputFlags & OUTPUT_CONTENT::ABSV) {
        outstrstream << ",absv";
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::VEL) {
        outstrstream << "," + OUTPUT_FILE_VEL_X_COL_NAME + "," + OUTPUT_FILE_VEL_Y_COL_NAME + "," +
                            OUTPUT_FILE_VEL_Z_COL_NAME;
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ANG_VEL) {
        outstrstream << "," + OUTPUT_FILE_ANGVEL_X_COL_NAME + "," + OUTPUT_FILE_ANGVEL_Y_COL_NAME + "," +
                            OUTPUT_FILE_ANGVEL_Z_COL_NAME;
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ABS_ACC) {
        outstrstream << ",abs_acc";
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ACC) {
        outstrstream << ",a_x,a_y,a_z";
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ANG_ACC) {
        outstrstream << ",alpha_x,alpha_y,alpha_z";
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
        outstrstream << ",family";
    }
    // if (solverFlags.outputFlags & OUTPUT_CONTENT::MAT) {
    //     outstrstream << ",material";
    // }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::OWNER_WILDCARD) {
        for (const auto& name : m_owner_wildcard_names) {
            outstrstream << "," + name;
        }
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::GEO_WILDCARD) {
        for (const auto& name : m_geo_wildcard_names) {
            outstrstream << "," + name;
        }
    }

    outstrstream << "\n";

    // simParams host version should not be different from device version, so no need to update
    for (size_t i = 0; i < simParams->nSpheresGM; i++) {
        auto this_owner = ownerClumpBody[i];
        family_t this_family = familyID[this_owner];
        // If this (impl-level) family is in the no-output list, skip it
        if (familiesNoOutput.find(this_family) != familiesNoOutput.end()) {
            continue;
        }

        float3 CoM;
        float3 pos;
        float radius;
        float X, Y, Z;
        voxelID_t voxel = voxelID[this_owner];
        subVoxelPos_t subVoxX = locX[this_owner];
        subVoxelPos_t subVoxY = locY[this_owner];
        subVoxelPos_t subVoxZ = locZ[this_owner];
        voxelIDToPosition<float, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel, subVoxX, subVoxY, subVoxZ, simParams->nvXp2,
                                                           simParams->nvYp2, simParams->voxelSize, simParams->l);
        CoM.x = X + simParams->LBFX;
        CoM.y = Y + simParams->LBFY;
        CoM.z = Z + simParams->LBFZ;

        size_t compOffset = (solverFlags.useClumpJitify) ? clumpComponentOffsetExt[i] : i;
        float3 this_sp_deviation;
        this_sp_deviation.x = relPosSphereX[compOffset];
        this_sp_deviation.y = relPosSphereY[compOffset];
        this_sp_deviation.z = relPosSphereZ[compOffset];
        float this_sp_rot_0 = oriQw[this_owner];
        float this_sp_rot_1 = oriQx[this_owner];
        float this_sp_rot_2 = oriQy[this_owner];
        float this_sp_rot_3 = oriQz[this_owner];
        applyOriQToVector3<float, float>(this_sp_deviation.x, this_sp_deviation.y, this_sp_deviation.z, this_sp_rot_0,
                                         this_sp_rot_1, this_sp_rot_2, this_sp_rot_3);
        pos = CoM + this_sp_deviation;
        outstrstream << pos.x << "," << pos.y << "," << pos.z;

        radius = radiiSphere[compOffset];
        outstrstream << "," << radius;

        // Only linear velocity
        float3 vxyz, acc;
        vxyz.x = vX[this_owner];
        vxyz.y = vY[this_owner];
        vxyz.z = vZ[this_owner];
        acc.x = aX[this_owner];
        acc.y = aY[this_owner];
        acc.z = aZ[this_owner];
        if (solverFlags.outputFlags & OUTPUT_CONTENT::ABSV) {
            outstrstream << "," << length(vxyz);
        }
        if (solverFlags.outputFlags & OUTPUT_CONTENT::VEL) {
            outstrstream << "," << vxyz.x << "," << vxyz.y << "," << vxyz.z;
        }
        if (solverFlags.outputFlags & OUTPUT_CONTENT::ANG_VEL) {
            float3 ang_v;
            ang_v.x = omgBarX[this_owner];
            ang_v.y = omgBarY[this_owner];
            ang_v.z = omgBarZ[this_owner];
            outstrstream << "," << ang_v.x << "," << ang_v.y << "," << ang_v.z;
        }

        if (solverFlags.outputFlags & OUTPUT_CONTENT::ABS_ACC) {
            outstrstream << "," << length(acc);
        }
        if (solverFlags.outputFlags & OUTPUT_CONTENT::ACC) {
            outstrstream << "," << acc.x << "," << acc.y << "," << acc.z;
        }
        if (solverFlags.outputFlags & OUTPUT_CONTENT::ANG_ACC) {
            float3 ang_acc;
            ang_acc.x = alphaX[this_owner];
            ang_acc.y = alphaY[this_owner];
            ang_acc.z = alphaZ[this_owner];
            outstrstream << "," << ang_acc.x << "," << ang_acc.y << "," << ang_acc.z;
        }

        // Family number needs to be user number
        if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
            outstrstream << "," << +(this_family);
        }

        // Wildcards
        if (solverFlags.outputFlags & OUTPUT_CONTENT::OWNER_WILDCARD) {
            // The order shouldn't be an issue... the same set is being processed here and in equip_owner_wildcards, see
            // Model.h
            for (unsigned int j = 0; j < m_owner_wildcard_names.size(); j++) {
                outstrstream << "," << (*ownerWildcards[j])[i];
            }
        }
        if (solverFlags.outputFlags & OUTPUT_CONTENT::GEO_WILDCARD) {
            for (unsigned int j = 0; j < m_geo_wildcard_names.size(); j++) {
                outstrstream << "," << (*sphereWildcards[j])[i];
            }
        }

        outstrstream << "\n";
    }

    ptFile << outstrstream.str();
}

#ifdef DEME_USE_CHPF
void DEMDynamicThread::writeClumpsAsChpf(std::ofstream& ptFile, unsigned int accuracy) {
    //// TODO: Note using accuracy
    chpf::Writer pw;
    migrateFamilyToHost();
    migrateClumpPosInfoToHost();
    migrateClumpHighOrderInfoToHost();

    // simParams host version should not be different from device version, so no need to update
    std::vector<float> posX(simParams->nOwnerBodies);
    std::vector<float> posY(simParams->nOwnerBodies);
    std::vector<float> posZ(simParams->nOwnerBodies);
    std::vector<float> Qw(simParams->nOwnerBodies);
    std::vector<float> Qx(simParams->nOwnerBodies);
    std::vector<float> Qy(simParams->nOwnerBodies);
    std::vector<float> Qz(simParams->nOwnerBodies);
    std::vector<std::string> clump_type(simParams->nOwnerBodies);
    std::vector<unsigned int> families;
    if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
        families.resize(simParams->nOwnerBodies);
    }
    size_t num_output_clumps = 0;

    for (size_t i = 0; i < simParams->nOwnerBodies; i++) {
        auto this_owner = ownerClumpBody[i];
        family_t this_family = familyID[this_owner];
        // If this (impl-level) family is in the no-output list, skip it
        if (familiesNoOutput.find(this_family) != familiesNoOutput.end()) {
            continue;
        }

        float3 CoM;
        float X, Y, Z;
        voxelID_t voxel = voxelID[i];
        subVoxelPos_t subVoxX = locX[i];
        subVoxelPos_t subVoxY = locY[i];
        subVoxelPos_t subVoxZ = locZ[i];
        voxelIDToPosition<float, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel, subVoxX, subVoxY, subVoxZ, simParams->nvXp2,
                                                           simParams->nvYp2, simParams->voxelSize, simParams->l);
        CoM.x = X + simParams->LBFX;
        CoM.y = Y + simParams->LBFY;
        CoM.z = Z + simParams->LBFZ;
        posX.at(num_output_clumps) = CoM.x;
        posY.at(num_output_clumps) = CoM.y;
        posZ.at(num_output_clumps) = CoM.z;

        // Then quaternions
        Qw.at(num_output_clumps) = oriQw[i];
        Qx.at(num_output_clumps) = oriQx[i];
        Qy.at(num_output_clumps) = oriQy[i];
        Qz.at(num_output_clumps) = oriQz[i];

        // Then type of clump
        unsigned int clump_mark = inertiaPropOffsets[i];
        clump_type.at(num_output_clumps) = templateNumNameMap.at(clump_mark);

        // Family number
        if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
            families.at(num_output_clumps) = this_family;
        }

        num_output_clumps++;
    }
    // Write basics
    posX.resize(num_output_clumps);
    posY.resize(num_output_clumps);
    posZ.resize(num_output_clumps);
    Qw.resize(num_output_clumps);
    Qx.resize(num_output_clumps);
    Qy.resize(num_output_clumps);
    Qz.resize(num_output_clumps);
    clump_type.resize(num_output_clumps);
    pw.write(ptFile, chpf::Compressor::Type::USE_DEFAULT,
             {OUTPUT_FILE_X_COL_NAME, OUTPUT_FILE_Y_COL_NAME, OUTPUT_FILE_Z_COL_NAME, OUTPUT_FILE_QW_COL_NAME,
              OUTPUT_FILE_QX_COL_NAME, OUTPUT_FILE_QY_COL_NAME, OUTPUT_FILE_QZ_COL_NAME, OUTPUT_FILE_CLUMP_TYPE_NAME},
             posX, posY, posZ, Qw, Qx, Qy, Qz, clump_type);
    // Write family numbers
    if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
        families.resize(num_output_clumps);
        //// TODO: How to do that?
        // pw.write(ptFile, chpf::Compressor::Type::USE_DEFAULT, {}, families);
    }
}
#endif

void DEMDynamicThread::writeClumpsAsCsv(std::ofstream& ptFile, unsigned int accuracy) {
    std::ostringstream outstrstream;
    outstrstream.precision(accuracy);

    migrateFamilyToHost();
    migrateClumpPosInfoToHost();
    migrateClumpHighOrderInfoToHost();
    migrateOwnerWildcardToHost();

    // xyz and quaternion are always there
    outstrstream << OUTPUT_FILE_X_COL_NAME + "," + OUTPUT_FILE_Y_COL_NAME + "," + OUTPUT_FILE_Z_COL_NAME +
                        ",Qw,Qx,Qy,Qz," + OUTPUT_FILE_CLUMP_TYPE_NAME;
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ABSV) {
        outstrstream << ",absv";
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::VEL) {
        outstrstream << "," + OUTPUT_FILE_VEL_X_COL_NAME + "," + OUTPUT_FILE_VEL_Y_COL_NAME + "," +
                            OUTPUT_FILE_VEL_Z_COL_NAME;
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ANG_VEL) {
        outstrstream << "," + OUTPUT_FILE_ANGVEL_X_COL_NAME + "," + OUTPUT_FILE_ANGVEL_Y_COL_NAME + "," +
                            OUTPUT_FILE_ANGVEL_Z_COL_NAME;
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ABS_ACC) {
        outstrstream << ",abs_acc";
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ACC) {
        outstrstream << ",a_x,a_y,a_z";
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ANG_ACC) {
        outstrstream << ",alpha_x,alpha_y,alpha_z";
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
        outstrstream << ",family";
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::OWNER_WILDCARD) {
        for (const auto& name : m_owner_wildcard_names) {
            outstrstream << "," + name;
        }
    }
    outstrstream << "\n";

    // simParams host version should not be different from device version, so no need to update
    for (size_t i = 0; i < simParams->nOwnerBodies; i++) {
        // i is this owner's number. And if it is not a clump, we can move on.
        if (ownerTypes[i] != OWNER_T_CLUMP)
            continue;

        family_t this_family = familyID[i];
        // If this (impl-level) family is in the no-output list, skip it
        if (familiesNoOutput.find(this_family) != familiesNoOutput.end()) {
            continue;
        }

        float3 CoM;
        float X, Y, Z;
        voxelID_t voxel = voxelID[i];
        subVoxelPos_t subVoxX = locX[i];
        subVoxelPos_t subVoxY = locY[i];
        subVoxelPos_t subVoxZ = locZ[i];
        voxelIDToPosition<float, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel, subVoxX, subVoxY, subVoxZ, simParams->nvXp2,
                                                           simParams->nvYp2, simParams->voxelSize, simParams->l);
        CoM.x = X + simParams->LBFX;
        CoM.y = Y + simParams->LBFY;
        CoM.z = Z + simParams->LBFZ;
        // Output position
        outstrstream << CoM.x << "," << CoM.y << "," << CoM.z;

        // Then quaternions
        outstrstream << "," << oriQw[i] << "," << oriQx[i] << "," << oriQy[i] << "," << oriQz[i];

        // Then type of clump
        unsigned int clump_mark = inertiaPropOffsets[i];
        outstrstream << "," << templateNumNameMap.at(clump_mark);

        // Only linear velocity
        float3 vxyz, ang_v, acc, ang_acc;
        vxyz.x = vX[i];
        vxyz.y = vY[i];
        vxyz.z = vZ[i];
        acc.x = aX[i];
        acc.y = aY[i];
        acc.z = aZ[i];
        if (solverFlags.outputFlags & OUTPUT_CONTENT::ABSV) {
            outstrstream << "," << length(vxyz);
        }
        if (solverFlags.outputFlags & OUTPUT_CONTENT::VEL) {
            outstrstream << "," << vxyz.x << "," << vxyz.y << "," << vxyz.z;
        }
        if (solverFlags.outputFlags & OUTPUT_CONTENT::ANG_VEL) {
            ang_v.x = omgBarX[i];
            ang_v.y = omgBarY[i];
            ang_v.z = omgBarZ[i];
            outstrstream << "," << ang_v.x << "," << ang_v.y << "," << ang_v.z;
        }
        if (solverFlags.outputFlags & OUTPUT_CONTENT::ABS_ACC) {
            outstrstream << "," << length(acc);
        }
        if (solverFlags.outputFlags & OUTPUT_CONTENT::ACC) {
            outstrstream << "," << acc.x << "," << acc.y << "," << acc.z;
        }
        if (solverFlags.outputFlags & OUTPUT_CONTENT::ANG_ACC) {
            ang_acc.x = alphaX[i];
            ang_acc.y = alphaY[i];
            ang_acc.z = alphaZ[i];
            outstrstream << "," << ang_acc.x << "," << ang_acc.y << "," << ang_acc.z;
        }

        // Family number needs to be user number
        if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
            outstrstream << "," << +(this_family);
        }

        // Wildcards
        if (solverFlags.outputFlags & OUTPUT_CONTENT::OWNER_WILDCARD) {
            // The order shouldn't be an issue... the same set is being processed here and in equip_owner_wildcards, see
            // Model.h
            for (unsigned int j = 0; j < m_owner_wildcard_names.size(); j++) {
                outstrstream << "," << (*ownerWildcards[j])[i];
            }
        }

        outstrstream << "\n";
    }

    ptFile << outstrstream.str();
}

std::shared_ptr<ContactInfoContainer> DEMDynamicThread::generateContactInfo(float force_thres) {
    // Migrate contact info to host
    migrateFamilyToHost();
    migrateClumpPosInfoToHost();
    migrateContactInfoToHost();

    size_t total_contacts = *(solverScratchSpace.numContacts);
    // Wildcards supports only floats now
    std::vector<std::pair<std::string, std::string>> existing_wildcards(m_contact_wildcard_names.size());
    size_t name_i = 0;
    for (const auto& name : m_contact_wildcard_names) {
        existing_wildcards[name_i++] = {name, "float"};
    }
    ContactInfoContainer contactInfo(solverFlags.cntOutFlags, existing_wildcards);
    contactInfo.ResizeAll(total_contacts);

    size_t useful_cnt = 0;
    for (size_t i = 0; i < total_contacts; i++) {
        // Geos that are involved in this contact
        auto geoA = idPatchA[i];
        auto geoB = idPatchB[i];
        auto type = contactTypePatch[i];
        // We don't output fake contacts; but right now, no contact will be marked fake by kT, so no need to check that
        // if (type == NOT_A_CONTACT)
        //     continue;

        float3 forcexyz = contactForces[i];
        float3 torque = contactTorque_convToForce[i];
        // If this force+torque is too small, then it's not an active contact
        if (length(forcexyz + torque) < force_thres) {
            continue;
        }

        bodyID_t ownerA = getPatchOwnerID(geoA, decodeTypeA(type));
        bodyID_t ownerB = getPatchOwnerID(geoB, decodeTypeB(type));

        // Type is mapped to SS, SM and such....
        contactInfo.Get<std::string>("ContactType")[useful_cnt] = contact_type_out_name_map.at(type);

        // Add family, always
        {
            family_t familyA = familyID[ownerA];
            family_t familyB = familyID[ownerB];
            contactInfo.Get<family_t>("AOwnerFamily")[useful_cnt] = familyA;
            contactInfo.Get<family_t>("BOwnerFamily")[useful_cnt] = familyB;
        }

        // (Internal) ownerID and/or geometry ID
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::OWNER) {
            contactInfo.Get<bodyID_t>("AOwner")[useful_cnt] = ownerA;
            contactInfo.Get<bodyID_t>("BOwner")[useful_cnt] = ownerB;
        }
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::GEO_ID) {
            contactInfo.Get<bodyID_t>("AGeo")[useful_cnt] = geoA;
            contactInfo.Get<bodyID_t>("BGeo")[useful_cnt] = geoB;
        }

        // Force is already in global...
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::FORCE) {
            contactInfo.Get<float3>("Force")[useful_cnt] = forcexyz;
        }

        // Contact point is in local frame. To make it global, first map that vector to axis-aligned global frame, then
        // add the location of body A CoM
        float4 oriQA;
        float3 CoM, cntPntA, cntPntALocal;
        {
            oriQA.w = oriQw[ownerA];
            oriQA.x = oriQx[ownerA];
            oriQA.y = oriQy[ownerA];
            oriQA.z = oriQz[ownerA];
            voxelID_t voxel = voxelID[ownerA];
            subVoxelPos_t subVoxX = locX[ownerA];
            subVoxelPos_t subVoxY = locY[ownerA];
            subVoxelPos_t subVoxZ = locZ[ownerA];
            voxelIDToPosition<float, voxelID_t, subVoxelPos_t>(CoM.x, CoM.y, CoM.z, voxel, subVoxX, subVoxY, subVoxZ,
                                                               simParams->nvXp2, simParams->nvYp2, simParams->voxelSize,
                                                               simParams->l);
            CoM.x += simParams->LBFX;
            CoM.y += simParams->LBFY;
            CoM.z += simParams->LBFZ;
            cntPntA = contactPointGeometryA[i];
            cntPntALocal = cntPntA;
            applyOriQToVector3(cntPntA.x, cntPntA.y, cntPntA.z, oriQA.w, oriQA.x, oriQA.y, oriQA.z);
            cntPntA += CoM;
        }
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::CNT_POINT) {
            // oriQ is updated already... whereas the contact point is effectively last step's... That's unfortunate.
            // Should we do somthing ahout it?
            contactInfo.Get<float3>("Point")[useful_cnt] = cntPntA;
        }

        // To get contact normal: it's just contact point - sphereA center, that gives you the outward normal for body A
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::NORMAL) {
            size_t compOffset = (solverFlags.useClumpJitify) ? clumpComponentOffsetExt[geoA] : geoA;
            float3 this_sp_deviation;
            this_sp_deviation.x = relPosSphereX[compOffset];
            this_sp_deviation.y = relPosSphereY[compOffset];
            this_sp_deviation.z = relPosSphereZ[compOffset];
            applyOriQToVector3<float, float>(this_sp_deviation.x, this_sp_deviation.y, this_sp_deviation.z, oriQA.w,
                                             oriQA.x, oriQA.y, oriQA.z);
            float3 pos = CoM + this_sp_deviation;
            float3 normal = normalize(cntPntA - pos);
            contactInfo.Get<float3>("Normal")[useful_cnt] = normal;
        }

        // Torque is in global already...
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::TORQUE) {
            // Must derive torque in local...
            {
                applyOriQToVector3(torque.x, torque.y, torque.z, oriQA.w, -oriQA.x, -oriQA.y, -oriQA.z);
                // Force times point...
                torque = cross(cntPntALocal, torque);
                // back to global
                applyOriQToVector3(torque.x, torque.y, torque.z, oriQA.w, oriQA.x, oriQA.y, oriQA.z);
            }
            contactInfo.Get<float3>("Torque")[useful_cnt] = torque;
        }

        // Contact wildcards
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::CNT_WILDCARD) {
            // The order shouldn't be an issue... the same set is being processed here and in equip_contact_wildcards,
            // see Model.h
            size_t name_i = 0;
            for (const auto& name : m_contact_wildcard_names) {
                contactInfo.Get<float>(name)[useful_cnt] = (*contactWildcards[name_i++])[i];
            }
        }

        useful_cnt++;
    }
    contactInfo.ResizeAll(useful_cnt);
    return std::make_shared<ContactInfoContainer>(std::move(contactInfo));
}

void DEMDynamicThread::writeContactsAsCsv(std::ofstream& ptFile, float force_thres) {
    std::ostringstream outstrstream;

    std::shared_ptr<ContactInfoContainer> contactInfo = generateContactInfo(force_thres);

    outstrstream << OUTPUT_FILE_CNT_TYPE_NAME;
    if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::OWNER) {
        outstrstream << "," + OUTPUT_FILE_OWNER_1_NAME + "," + OUTPUT_FILE_OWNER_2_NAME;
    }
    if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::GEO_ID) {
        outstrstream << "," + OUTPUT_FILE_GEO_ID_1_NAME + "," + OUTPUT_FILE_GEO_ID_2_NAME;
    }
    if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::FORCE) {
        outstrstream << "," + OUTPUT_FILE_FORCE_X_NAME + "," + OUTPUT_FILE_FORCE_Y_NAME + "," +
                            OUTPUT_FILE_FORCE_Z_NAME;
    }
    if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::CNT_POINT) {
        outstrstream << "," + OUTPUT_FILE_X_COL_NAME + "," + OUTPUT_FILE_Y_COL_NAME + "," + OUTPUT_FILE_Z_COL_NAME;
    }
    // if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::COMPONENT) {
    //     outstrstream << ","+OUTPUT_FILE_COMP_1_NAME+","+OUTPUT_FILE_COMP_2_NAME;
    // }
    // if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::NICKNAME) {
    //     outstrstream << ","+OUTPUT_FILE_OWNER_NICKNAME_1_NAME+","+OUTPUT_FILE_OWNER_NICKNAME_2_NAME;
    // }
    if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::NORMAL) {
        outstrstream << "," + OUTPUT_FILE_NORMAL_X_NAME + "," + OUTPUT_FILE_NORMAL_Y_NAME + "," +
                            OUTPUT_FILE_NORMAL_Z_NAME;
    }
    if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::TORQUE) {
        outstrstream << "," + OUTPUT_FILE_TORQUE_X_NAME + "," + OUTPUT_FILE_TORQUE_Y_NAME + "," +
                            OUTPUT_FILE_TORQUE_Z_NAME;
    }
    if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::CNT_WILDCARD) {
        // Write all wildcard names as header
        for (const auto& w_name : m_contact_wildcard_names) {
            outstrstream << "," + w_name;
        }
    }
    outstrstream << "\n";

    for (size_t i = 0; i < contactInfo->Size(); i++) {
        outstrstream << contactInfo->Get<std::string>("ContactType")[i];

        // (Internal) ownerID and/or geometry ID
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::OWNER) {
            outstrstream << "," << contactInfo->Get<bodyID_t>("AOwner")[i] << ","
                         << contactInfo->Get<bodyID_t>("BOwner")[i];
        }
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::GEO_ID) {
            outstrstream << "," << contactInfo->Get<bodyID_t>("AGeo")[i] << ","
                         << contactInfo->Get<bodyID_t>("BGeo")[i];
        }

        // Force is already in global...
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::FORCE) {
            outstrstream << "," << contactInfo->Get<float3>("Force")[i].x << ","
                         << contactInfo->Get<float3>("Force")[i].y << "," << contactInfo->Get<float3>("Force")[i].z;
        }

        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::CNT_POINT) {
            // oriQ is updated already... whereas the contact point is effectively last step's... That's unfortunate.
            // Should we do somthing ahout it?
            outstrstream << "," << contactInfo->Get<float3>("Point")[i].x << ","
                         << contactInfo->Get<float3>("Point")[i].y << "," << contactInfo->Get<float3>("Point")[i].z;
        }

        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::NORMAL) {
            outstrstream << "," << contactInfo->Get<float3>("Normal")[i].x << ","
                         << contactInfo->Get<float3>("Normal")[i].y << "," << contactInfo->Get<float3>("Normal")[i].z;
        }

        // Torque is in global already...
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::TORQUE) {
            outstrstream << "," << contactInfo->Get<float3>("Torque")[i].x << ","
                         << contactInfo->Get<float3>("Torque")[i].y << "," << contactInfo->Get<float3>("Torque")[i].z;
        }

        // Contact wildcards
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::CNT_WILDCARD) {
            // The order shouldn't be an issue... the same set is being processed here and in equip_contact_wildcards,
            // see Model.h
            for (const auto& name : m_contact_wildcard_names) {
                outstrstream << "," << contactInfo->Get<float>(name)[i];
            }
        }

        outstrstream << "\n";
    }

    ptFile << outstrstream.str();
}

void DEMDynamicThread::writeMeshesAsVtk(std::ofstream& ptFile) {
    std::ostringstream ostream;
    migrateFamilyToHost();

    std::vector<size_t> vertexOffset(m_meshes.size() + 1, 0);
    size_t total_f = 0;
    size_t total_v = 0;
    unsigned int mesh_num = 0;

    // May want to jump the families that the user disabled output for
    std::vector<notStupidBool_t> thisMeshSkip(m_meshes.size(), 0);
    for (const auto& mmesh : m_meshes) {
        bodyID_t mowner = mmesh->owner;
        family_t this_family = familyID[mowner];
        // If this (impl-level) family is in the no-output list, skip it
        if (familiesNoOutput.find(this_family) != familiesNoOutput.end()) {
            thisMeshSkip[mesh_num] = 1;
        }
        mesh_num++;
    }

    ostream << "# vtk DataFile Version 2.0\n";
    ostream << "VTK from DEM simulation\n";
    ostream << "ASCII\n";
    ostream << "\n\n";

    ostream << "DATASET UNSTRUCTURED_GRID\n";

    // Prescan the V and F: to write all meshes to one file, we need vertex number offset info
    mesh_num = 0;
    for (const auto& mmesh : m_meshes) {
        if (!thisMeshSkip[mesh_num]) {
            vertexOffset[mesh_num + 1] = mmesh->GetCoordsVertices().size();
            total_v += mmesh->GetCoordsVertices().size();
            total_f += mmesh->GetIndicesVertexes().size();
        }
        mesh_num++;
    }
    for (unsigned int i = 1; i < m_meshes.size(); i++)
        vertexOffset[i] = vertexOffset[i] + vertexOffset[i - 1];

    // Writing m_vertices
    ostream << "POINTS " << total_v << " float" << std::endl;
    mesh_num = 0;
    for (const auto& mmesh : m_meshes) {
        if (!thisMeshSkip[mesh_num]) {
            bodyID_t mowner = mmesh->owner;
            float3 ownerPos = this->getOwnerPos(mowner)[0];
            float4 ownerOriQ = this->getOwnerOriQ(mowner)[0];
            for (const auto& v : mmesh->GetCoordsVertices()) {
                float3 point = v;
                applyFrameTransformLocalToGlobal(point, ownerPos, ownerOriQ);
                ostream << point.x << " " << point.y << " " << point.z << std::endl;
            }
        }
        mesh_num++;
    }

    // Writing faces
    ostream << "\n\n";
    ostream << "CELLS " << total_f << " " << 4 * total_f << std::endl;
    mesh_num = 0;
    for (const auto& mmesh : m_meshes) {
        if (!thisMeshSkip[mesh_num]) {
            for (const auto& f : mmesh->GetIndicesVertexes()) {
                ostream << "3 " << (size_t)f.x + vertexOffset[mesh_num] << " " << (size_t)f.y + vertexOffset[mesh_num]
                        << " " << (size_t)f.z + vertexOffset[mesh_num] << std::endl;
            }
        }
        mesh_num++;
    }

    // Writing face types. Type 5 is generally triangles
    ostream << "\n\n";
    ostream << "CELL_TYPES " << total_f << std::endl;
    mesh_num = 0;
    for (const auto& mmesh : m_meshes) {
        if (!thisMeshSkip[mesh_num]) {
            auto nfaces = mmesh->GetIndicesVertexes().size();
            for (size_t j = 0; j < nfaces; j++)
                ostream << "5 " << std::endl;
        }
        mesh_num++;
    }

    ptFile << ostream.str();
}

void DEMDynamicThread::writeMeshesAsStl(std::ofstream& ptFile) {
    std::ostringstream ostream;
    migrateFamilyToHost();

    std::vector<notStupidBool_t> thisMeshSkip(m_meshes.size(), 0);
    unsigned int mesh_num = 0;
    for (const auto& mmesh : m_meshes) {
        bodyID_t mowner = mmesh->owner;
        family_t this_family = familyID[mowner];
        if (familiesNoOutput.find(this_family) != familiesNoOutput.end()) {
            thisMeshSkip[mesh_num] = 1;
        }
        mesh_num++;
    }

    ostream << "solid DEMSimulation" << std::endl;
    mesh_num = 0;
    for (const auto& mmesh : m_meshes) {
        if (!thisMeshSkip[mesh_num]) {
            bodyID_t mowner = mmesh->owner;
            float3 ownerPos = this->getOwnerPos(mowner)[0];
            float4 ownerOriQ = this->getOwnerOriQ(mowner)[0];
            const auto& vertices = mmesh->GetCoordsVertices();
            const auto& faces = mmesh->GetIndicesVertexes();

            for (const auto& f : faces) {
                float3 v0 = vertices[f.x];
                float3 v1 = vertices[f.y];
                float3 v2 = vertices[f.z];

                applyFrameTransformLocalToGlobal(v0, ownerPos, ownerOriQ);
                applyFrameTransformLocalToGlobal(v1, ownerPos, ownerOriQ);
                applyFrameTransformLocalToGlobal(v2, ownerPos, ownerOriQ);

                float3 normal = face_normal(v0, v1, v2);
                ostream << "  facet normal " << normal.x << " " << normal.y << " " << normal.z << std::endl;
                ostream << "    outer loop" << std::endl;
                ostream << "      vertex " << v0.x << " " << v0.y << " " << v0.z << std::endl;
                ostream << "      vertex " << v1.x << " " << v1.y << " " << v1.z << std::endl;
                ostream << "      vertex " << v2.x << " " << v2.y << " " << v2.z << std::endl;
                ostream << "    endloop" << std::endl;
                ostream << "  endfacet" << std::endl;
            }
        }
        mesh_num++;
    }
    ostream << "endsolid DEMSimulation" << std::endl;
    ptFile << ostream.str();
}

void DEMDynamicThread::writeMeshesAsPly(std::ofstream& ptFile) {
    std::ostringstream ostream;
    migrateFamilyToHost();

    std::vector<size_t> vertexOffset(m_meshes.size() + 1, 0);
    size_t total_f = 0;
    size_t total_v = 0;
    unsigned int mesh_num = 0;

    std::vector<notStupidBool_t> thisMeshSkip(m_meshes.size(), 0);
    for (const auto& mmesh : m_meshes) {
        bodyID_t mowner = mmesh->owner;
        family_t this_family = familyID[mowner];
        if (familiesNoOutput.find(this_family) != familiesNoOutput.end()) {
            thisMeshSkip[mesh_num] = 1;
        } else {
            vertexOffset[mesh_num + 1] = mmesh->GetCoordsVertices().size();
            total_v += mmesh->GetCoordsVertices().size();
            total_f += mmesh->GetIndicesVertexes().size();
        }
        mesh_num++;
    }

    for (unsigned int i = 1; i < m_meshes.size(); i++) {
        vertexOffset[i] = vertexOffset[i] + vertexOffset[i - 1];
    }

    ostream << "ply" << std::endl;
    ostream << "format ascii 1.0" << std::endl;
    ostream << "comment DEM simulation mesh export" << std::endl;
    ostream << "element vertex " << total_v << std::endl;
    ostream << "property float x" << std::endl;
    ostream << "property float y" << std::endl;
    ostream << "property float z" << std::endl;
    ostream << "element face " << total_f << std::endl;
    ostream << "property list uchar int vertex_indices" << std::endl;
    ostream << "end_header" << std::endl;

    mesh_num = 0;
    for (const auto& mmesh : m_meshes) {
        if (!thisMeshSkip[mesh_num]) {
            bodyID_t mowner = mmesh->owner;
            float3 ownerPos = this->getOwnerPos(mowner)[0];
            float4 ownerOriQ = this->getOwnerOriQ(mowner)[0];
            for (const auto& v : mmesh->GetCoordsVertices()) {
                float3 point = v;
                applyFrameTransformLocalToGlobal(point, ownerPos, ownerOriQ);
                ostream << point.x << " " << point.y << " " << point.z << std::endl;
            }
        }
        mesh_num++;
    }

    ostream << std::endl;
    mesh_num = 0;
    for (const auto& mmesh : m_meshes) {
        if (!thisMeshSkip[mesh_num]) {
            for (const auto& f : mmesh->GetIndicesVertexes()) {
                ostream << "3 " << (size_t)f.x + vertexOffset[mesh_num] << " "
                        << (size_t)f.y + vertexOffset[mesh_num] << " "
                        << (size_t)f.z + vertexOffset[mesh_num] << std::endl;
            }
        }
        mesh_num++;
    }

    ptFile << ostream.str();
}

inline void DEMDynamicThread::contactPrimitivesArraysResize(size_t nContactPairs) {
    DEME_DUAL_ARRAY_RESIZE(idPrimitiveA, nContactPairs, 0);
    DEME_DUAL_ARRAY_RESIZE(idPrimitiveB, nContactPairs, 0);
    DEME_DUAL_ARRAY_RESIZE(contactTypePrimitive, nContactPairs, NOT_A_CONTACT);

    // NEW: Resize geomToPatchMap to match geometry array size
    DEME_DUAL_ARRAY_RESIZE(geomToPatchMap, nContactPairs, 0);

    // If there are meshes, then sph--mesh case always use force storage, no getting around; if no mesh, then if no need
    // to store forces, we can choose to not resize these arrays.
    if (!(solverFlags.useNoContactRecord && simParams->nTriGM == 0)) {
        DEME_DUAL_ARRAY_RESIZE(contactForces, nContactPairs, make_float3(0));
        DEME_DUAL_ARRAY_RESIZE(contactTorque_convToForce, nContactPairs, make_float3(0));
        DEME_DUAL_ARRAY_RESIZE(contactPointGeometryA, nContactPairs, make_float3(0));
        DEME_DUAL_ARRAY_RESIZE(contactPointGeometryB, nContactPairs, make_float3(0));
        // NEW: Resize SAT satisfaction array for tracking tri-tri physical contact
        DEME_DUAL_ARRAY_RESIZE(contactSATSatisfied, nContactPairs, 0);
    }

    // Re-packing pointers now is automatic

    // Sync pointers to device can be delayed... we'll only need to do that before kernel calls

    // Also note that dT does not have to worry about contact persistence, because kT handles that
}

inline void DEMDynamicThread::contactPatchArrayResize(size_t nPatchPairs) {
    // NEW: Resize separate patch ID arrays (sized to patch pairs, the shorter array)
    DEME_DUAL_ARRAY_RESIZE(idPatchA, nPatchPairs, 0);
    DEME_DUAL_ARRAY_RESIZE(idPatchB, nPatchPairs, 0);
    DEME_DUAL_ARRAY_RESIZE(contactTypePatch, nPatchPairs, NOT_A_CONTACT);

    // Re-packing pointers to device now is automatic
    // Sync pointers to device can be delayed... we'll only need to do that before kernel calls
}

inline void DEMDynamicThread::unpackMyBuffer() {
    DEME_NVTX_RANGE("dT::unpack");
    if (kT) {
        const bool same_dev = (streamInfo.device == kT->streamInfo.device);
        // If sharing a device, ensure kT finished populating the kT->dT transfer buffers before we consume them.
        if (same_dev && kT->kT_to_dT_BufferReadyEvent) {
            DEME_GPU_CALL(cudaStreamWaitEvent(streamInfo.stream, kT->kT_to_dT_BufferReadyEvent, 0));
        }
    }

    // Make a note on the contact number of the previous time step
    *solverScratchSpace.numPrevContacts = *solverScratchSpace.numContacts;
    *solverScratchSpace.numPrevPrimitiveContacts = *solverScratchSpace.numPrimitiveContacts;
    // kT's batch of produce is made with this max drift in mind
    pSchedSupport->dynamicMaxFutureDrift = (pSchedSupport->kinematicMaxFutureDrift).load();
    // DEME_DEBUG_PRINTF("dynamicMaxFutureDrift is %u", (pSchedSupport->dynamicMaxFutureDrift).load());

    contactMappingUsesBuffer = false;
    DEME_GPU_CALL(cudaMemcpy(&(solverScratchSpace.numPrimitiveContacts), &nPrimitiveContactPairs_buffer, sizeof(size_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(&(solverScratchSpace.numContacts), &nPatchContactPairs_buffer, sizeof(size_t),
                             cudaMemcpyDeviceToDevice));
    solverScratchSpace.numContacts.toHost();
    solverScratchSpace.numPrimitiveContacts.toHost();
    // Need to resize those contact event-based arrays before usage
    if (*solverScratchSpace.numPrimitiveContacts > idPrimitiveA.size()) {
        contactPrimitivesArraysResize(*solverScratchSpace.numPrimitiveContacts);
    }
    if (*solverScratchSpace.numContacts > idPatchA.size()) {
        contactPatchArrayResize(*solverScratchSpace.numContacts);
    }

    const size_t nPrimitive = *solverScratchSpace.numPrimitiveContacts;
    const size_t nPatch = *solverScratchSpace.numContacts;
    const int read_idx = kt_write_buf;
    const int dev = streamInfo.device;
    static const bool allow_swap = []() {
        const char* env = std::getenv("DEME_DT_UNPACK_SWAP");
        if (!env || !*env) {
            return true;
        }
        return !(env[0] == '0' && env[1] == '\0');
    }();
    static const bool allow_direct_mapping = []() {
        const char* env = std::getenv("DEME_DT_UNPACK_DIRECT_MAPPING");
        if (!env || !*env) {
            return true;
        }
        return !(env[0] == '0' && env[1] == '\0');
    }();
    bool swapped = false;
#ifndef DEME_USE_MANAGED_ARRAYS
    if (kT && allow_swap && streamInfo.device == kT->streamInfo.device) {
        swapped = swap_device_buffer(idPrimitiveA, idPrimitiveA_buffer[read_idx]);
        swapped = swap_device_buffer(idPrimitiveB, idPrimitiveB_buffer[read_idx]) && swapped;
        swapped = swap_device_buffer(contactTypePrimitive, contactTypePrimitive_buffer[read_idx]) && swapped;
        swapped = swap_device_buffer(geomToPatchMap, geomToPatchMap_buffer[read_idx]) && swapped;
        swapped = swap_device_buffer(idPatchA, idPatchA_buffer[read_idx]) && swapped;
        swapped = swap_device_buffer(idPatchB, idPatchB_buffer[read_idx]) && swapped;
        swapped = swap_device_buffer(contactTypePatch, contactTypePatch_buffer[read_idx]) && swapped;
    }
#endif
    xfer::XferList xu;
    if (!swapped) {
        xu.add(granData->idPrimitiveA, idPrimitiveA_buffer[read_idx].data(), nPrimitive * sizeof(bodyID_t));
        xu.add(granData->idPrimitiveB, idPrimitiveB_buffer[read_idx].data(), nPrimitive * sizeof(bodyID_t));
        xu.add(granData->contactTypePrimitive, contactTypePrimitive_buffer[read_idx].data(),
               nPrimitive * sizeof(contact_t));
        xu.add(granData->geomToPatchMap, geomToPatchMap_buffer[read_idx].data(), nPrimitive * sizeof(contactPairs_t));

        // Unpack separate patch ID arrays
        xu.add(granData->idPatchA, idPatchA_buffer[read_idx].data(), nPatch * sizeof(bodyID_t));
        xu.add(granData->idPatchB, idPatchB_buffer[read_idx].data(), nPatch * sizeof(bodyID_t));
        xu.add(granData->contactTypePatch, contactTypePatch_buffer[read_idx].data(), nPatch * sizeof(contact_t));
    }

    if (!solverFlags.isHistoryless) {
        if (kT && allow_direct_mapping && streamInfo.device == kT->streamInfo.device) {
            granData->contactMapping = contactMapping_buffer[read_idx].data();
            contactMappingUsesBuffer = true;
        } else {
            size_t mapping_bytes = nPatch * sizeof(contactPairs_t);
            granData->contactMapping =
                (contactPairs_t*)solverScratchSpace.allocateTempVector("contactMapping", mapping_bytes);
            xu.add(granData->contactMapping, contactMapping_buffer[read_idx].data(), mapping_bytes);
            contactMappingUsesBuffer = false;
        }
    } else {
        granData->contactMapping = nullptr;
        contactMappingUsesBuffer = false;
    }
    xu.run(dev, dev, streamInfo.stream);
    // Flip buffer for next kT production
    kt_write_buf = 1 - read_idx;
    if (kT) {
        kT->granData->pDTOwnedBuffer_idPrimitiveA = idPrimitiveA_buffer[kt_write_buf].data();
        kT->granData->pDTOwnedBuffer_idPrimitiveB = idPrimitiveB_buffer[kt_write_buf].data();
        kT->granData->pDTOwnedBuffer_contactType = contactTypePrimitive_buffer[kt_write_buf].data();
        kT->granData->pDTOwnedBuffer_geomToPatchMap = geomToPatchMap_buffer[kt_write_buf].data();
        kT->granData->pDTOwnedBuffer_idPatchA = idPatchA_buffer[kt_write_buf].data();
        kT->granData->pDTOwnedBuffer_idPatchB = idPatchB_buffer[kt_write_buf].data();
        kT->granData->pDTOwnedBuffer_contactTypePatch = contactTypePatch_buffer[kt_write_buf].data();
        if (!solverFlags.isHistoryless) {
            kT->granData->pDTOwnedBuffer_contactMapping = contactMapping_buffer[kt_write_buf].data();
        }
    }
    // Prepare for kernel calls immediately after
    granData.toDeviceAsync(streamInfo.stream);
}

bool DEMDynamicThread::tryConsumeKinematicProduce(bool allow_blocking, bool mark_receive, bool use_logical_stamp) {
    (void)allow_blocking;
    if (!kT) {
        return false;
    }
    if (!pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh.load(std::memory_order_acquire)) {
        return false;
    }
    const int64_t logical_recv = use_logical_stamp
                                     ? (pSchedSupport->currentStampOfDynamic.load(std::memory_order_relaxed) + 1)
                                     : pSchedSupport->currentStampOfDynamic.load(std::memory_order_relaxed);
    if (use_logical_stamp) {
        recv_stamp_override = logical_recv;
    }
    timers.GetTimer("Unpack updates from kT").start();
    unpack_impl();
    timers.GetTimer("Unpack updates from kT").stop();
    recv_stamp_override = -1;
    if (mark_receive) {
        auto& reg = futureDriftRegulator;
        reg.receive_pending = true;
        drainProgressEvents();
        reg.pending_recv_stamp = static_cast<uint64_t>(logical_recv);
    }
    return true;
}

inline void DEMDynamicThread::sendToTheirBuffer() {
    const int srcDev = streamInfo.device;      // dT GPU
    const int dstDev = kT->streamInfo.device;  // kT GPU
    const size_t nOwners = (size_t)simParams->nOwnerBodies;
    const bool same_dev = (srcDev == dstDev);
    const cudaStream_t xfer_stream = same_dev ? streamInfo.stream : 0;

    xfer::XferList xt;
    xt.add(granData->pKTOwnedBuffer_voxelID, granData->voxelID, nOwners * sizeof(voxelID_t));
    xt.add(granData->pKTOwnedBuffer_locX, granData->locX, nOwners * sizeof(subVoxelPos_t));
    xt.add(granData->pKTOwnedBuffer_locY, granData->locY, nOwners * sizeof(subVoxelPos_t));
    xt.add(granData->pKTOwnedBuffer_locZ, granData->locZ, nOwners * sizeof(subVoxelPos_t));
    xt.add(granData->pKTOwnedBuffer_oriQ0, granData->oriQw, nOwners * sizeof(oriQ_t));
    xt.add(granData->pKTOwnedBuffer_oriQ1, granData->oriQx, nOwners * sizeof(oriQ_t));
    xt.add(granData->pKTOwnedBuffer_oriQ2, granData->oriQy, nOwners * sizeof(oriQ_t));
    xt.add(granData->pKTOwnedBuffer_oriQ3, granData->oriQz, nOwners * sizeof(oriQ_t));
    xt.add(granData->pKTOwnedBuffer_absVel, pCycleVel, nOwners * sizeof(float));
    xt.add(granData->pKTOwnedBuffer_absAngVel, pCycleAngVel, nOwners * sizeof(float));
    xt.run(dstDev, srcDev, xfer_stream);

    // Optionals
    xfer::XferList xk;
    if (solverFlags.canFamilyChangeOnDevice) {
        xk.add(granData->pKTOwnedBuffer_familyID, granData->familyID,
               (size_t)simParams->nOwnerBodies * sizeof(family_t));
    }
    if (solverFlags.willMeshDeform) {
        xk.add(granData->pKTOwnedBuffer_relPosNode1, granData->relPosNode1, (size_t)simParams->nTriGM * sizeof(float3));
        xk.add(granData->pKTOwnedBuffer_relPosNode2, granData->relPosNode2, (size_t)simParams->nTriGM * sizeof(float3));
        xk.add(granData->pKTOwnedBuffer_relPosNode3, granData->relPosNode3, (size_t)simParams->nTriGM * sizeof(float3));
    }
    xk.run(dstDev, srcDev, xfer_stream);

    // Send simulation metrics for kT's reference.
    if (same_dev) {
        DEME_GPU_CALL(cudaMemcpyAsync(granData->pKTOwnedBuffer_ts, &(simParams->dyn.h), sizeof(float),
                                      cudaMemcpyHostToDevice, streamInfo.stream));
    } else {
        DEME_GPU_CALL(
            cudaMemcpy(granData->pKTOwnedBuffer_ts, &(simParams->dyn.h), sizeof(float), cudaMemcpyHostToDevice));
    }
    // Note that perhapsIdealFutureDrift is non-negative, and it will be used to determine the margin size; however, if
    // scheduleHelper is instructed to have negative future drift then perhapsIdealFutureDrift no longer affects them.
    if (same_dev) {
        DEME_GPU_CALL(cudaMemcpyAsync(granData->pKTOwnedBuffer_maxDrift, perhapsIdealFutureDrift.getHostPointer(),
                                      sizeof(unsigned int), cudaMemcpyHostToDevice, streamInfo.stream));
    } else {
        DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_maxDrift, perhapsIdealFutureDrift.getHostPointer(),
                                 sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    xfer::XferList xm;
    xm.add(granData->pKTOwnedBuffer_maxTriTriPenetration, maxTriTriPenetration.getDevicePointer(), sizeof(double));
    xm.run(dstDev, srcDev, xfer_stream);

    if (solverFlags.willMeshDeform) {
        solverFlags.willMeshDeform = false;
        kT->solverFlags.willMeshDeform = true;
    }
    // This subroutine also includes recording the time stamp of this batch ingredient dT sent to kT.
    // Use the logical step stamp (not the completion stamp) so scheduling is not biased by progress-event drainage.
    pSchedSupport->kinematicIngredProdDateStamp = (pSchedSupport->currentStampOfDynamic).load();

    // Signal kT that dT->kT buffers are populated (same-device path); kT waits on this event before unpacking.
    if (same_dev && dT_to_kT_BufferReadyEvent) {
        DEME_GPU_CALL(cudaEventRecord(dT_to_kT_BufferReadyEvent, streamInfo.stream));
    }
}

inline void DEMDynamicThread::migrateEnduringContacts() {
    // Use granData->contactMapping's information (now directly the transfer buffer) to map old and new contacts

    // All contact wildcards are the same type, so we can just allocate one temp array for all of them
    float* newWildcards[DEME_MAX_WILDCARD_NUM];
    size_t wildcard_arr_bytes = (*solverScratchSpace.numContacts) * sizeof(float) * simParams->nContactWildcards;
    newWildcards[0] = (float*)solverScratchSpace.allocateTempVector("newWildcards", wildcard_arr_bytes);
    for (unsigned int i = 1; i < simParams->nContactWildcards; i++) {
        newWildcards[i] = newWildcards[i - 1] + (*solverScratchSpace.numContacts);
    }

    // This is used for checking if there are contact history got lost in the transition by surprise. But no need to
    // check if the user did not ask for it.
    size_t sentry_bytes = (*solverScratchSpace.numPrevContacts) * sizeof(notStupidBool_t);
    notStupidBool_t* contactSentry =
        (notStupidBool_t*)solverScratchSpace.allocateTempVector("contactSentry", sentry_bytes);

    // A sentry array is here to see if there exist a contact that dT thinks it's alive but kT doesn't map it to the new
    // history array. This is just a quick and rough check: we only look at the last contact wildcard to see if it is
    // non-0, whatever it represents.
    if (DEME_GET_VERBOSITY() >= VERBOSITY_METRIC) {
        if (*solverScratchSpace.numPrevContacts > 0) {
            markAliveContacts(granData->contactWildcards[simParams->nContactWildcards - 1], contactSentry,
                              *solverScratchSpace.numPrevContacts, streamInfo.stream);
        }
    }

    // Rearrange contact histories based on kT instruction
    rearrangeContactWildcards(&granData, newWildcards[0], contactSentry, simParams->nContactWildcards,
                              *solverScratchSpace.numContacts, streamInfo.stream);

    // Take a look, does the sentry indicate that there is an `alive' contact got lost?
    if (DEME_GET_VERBOSITY() >= VERBOSITY_METRIC) {
        if (*solverScratchSpace.numPrevContacts > 0 && simParams->nContactWildcards > 0) {
            // Temp DualStruct defaults to size_t type
            solverScratchSpace.allocateDualStruct("lostContact");
            size_t* lostContact = solverScratchSpace.getDualStructDevice("lostContact");
            cubSumReduce<notStupidBool_t, size_t>(contactSentry, lostContact, *solverScratchSpace.numPrevContacts,
                                                  streamInfo.stream, solverScratchSpace);
            solverScratchSpace.syncDualStructDeviceToHost("lostContact");
            lostContact = solverScratchSpace.getDualStructHost("lostContact");
            if (*lostContact && solverFlags.isAsync) {
                // This prints when verbosity higher than METRIC
                DEME_STATUS(
                    "ALIVE_CONTACT_NOT_DETECTED",
                    "%zu contacts were active at time %.9g on dT, but they are not detected on kT, therefore being "
                    "removed unexpectedly!",
                    *lostContact, simParams->dyn.timeElapsed);
                DEME_DEBUG_PRINTF("New number of contacts: %zu", *solverScratchSpace.numContacts);
                DEME_DEBUG_PRINTF("Old number of contacts: %zu", *solverScratchSpace.numPrevContacts);
                DEME_DEBUG_PRINTF("New contact A:");
                DEME_DEBUG_EXEC(displayDeviceArray<bodyID_t>(granData->idPrimitiveA, *solverScratchSpace.numContacts));
                DEME_DEBUG_PRINTF("New contact B:");
                DEME_DEBUG_EXEC(displayDeviceArray<bodyID_t>(granData->idPrimitiveB, *solverScratchSpace.numContacts));
                DEME_DEBUG_PRINTF("Old version of the last contact wildcard:");
                DEME_DEBUG_EXEC(displayDeviceArray<float>(granData->contactWildcards[simParams->nContactWildcards - 1],
                                                          *solverScratchSpace.numPrevContacts));
                DEME_DEBUG_PRINTF("Old--new mapping:");
                DEME_DEBUG_EXEC(
                    displayDeviceArray<contactPairs_t>(granData->contactMapping, *solverScratchSpace.numContacts));
                DEME_DEBUG_PRINTF("Sentry:");
                DEME_DEBUG_EXEC(
                    displayDeviceArray<notStupidBool_t>(contactSentry, *solverScratchSpace.numPrevContacts));
            }
            solverScratchSpace.finishUsingDualStruct("lostContact");
        }
    }

    // Copy new history back to history array (after resizing the `main' history array)
    if (*solverScratchSpace.numContacts > contactWildcards[0]->size()) {
        for (unsigned int i = 0; i < simParams->nContactWildcards; i++) {
            // Packing data pointer is not needed after binding
            DEME_DUAL_ARRAY_RESIZE((*contactWildcards[i]), *solverScratchSpace.numContacts, 0);
        }
    }
    for (unsigned int i = 0; i < simParams->nContactWildcards; i++) {
        DEME_GPU_CALL(cudaMemcpyAsync(granData->contactWildcards[i], newWildcards[i],
                                      (*solverScratchSpace.numContacts) * sizeof(float), cudaMemcpyDeviceToDevice,
                                      streamInfo.stream));
    }

    solverScratchSpace.finishUsingTempVector("newWildcards");
    solverScratchSpace.finishUsingTempVector("contactSentry");

    // granData may have changed in some of the earlier steps
    granData.toDeviceAsync(streamInfo.stream);
}

// The argument is two maps: contact type -> (start offset, count), contact type -> list of [(program bundle name,
// kernel name)]
inline void DEMDynamicThread::dispatchPrimitiveForceKernels(
    const ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>>& typeStartCountMap,
    const ContactTypeMap<std::vector<std::pair<std::shared_ptr<JitHelper::CachedProgram>, std::string>>>&
        typeKernelMap) {
    // For each contact type that exists, call its corresponding kernel(s)
    for (size_t i = 0; i < m_numExistingTypes; i++) {
        contact_t contact_type = existingContactTypes[i];
        const auto& start_count = typeStartCountMap.at(contact_type);
        // Offset and count being contactPairs_t is very important, as CUDA kernel arguments cannot safely implicitly
        // convert type (from size_t to unsigned int, for example)
        contactPairs_t startOffset = start_count.first;
        contactPairs_t count = start_count.second;

        // For this contact type, get its list of (program bundle name, kernel name)
        if (typeKernelMap.count(contact_type) == 0) {
            // displayDeviceArray<bodyID_t>(granData->idPrimitiveA, *solverScratchSpace.numPrimitiveContacts);
            // displayDeviceArray<bodyID_t>(granData->idPrimitiveB, *solverScratchSpace.numPrimitiveContacts);
            // displayDeviceArray<contact_t>(granData->contactTypePrimitive, *solverScratchSpace.numPrimitiveContacts);
            // for (size_t j = 0; j < m_numExistingTypes; j++) {
            //     DEME_PRINTF("existingContactTypes[%zu] = %d\n", j, existingContactTypes[j]);
            // }
            DEME_ERROR("Contact type %d has no associated force kernel in to execute!", contact_type);
        }
        const auto& kernelList = typeKernelMap.at(contact_type);
        for (const auto& [progName, kernelName] : kernelList) {
            size_t blocks = (count + DT_FORCE_CALC_NTHREADS_PER_BLOCK - 1) / DT_FORCE_CALC_NTHREADS_PER_BLOCK;
            if (blocks > 0) {
                progName->kernel(kernelName)
                    .instantiate()
                    .configure(dim3(blocks), dim3(DT_FORCE_CALC_NTHREADS_PER_BLOCK), 0, streamInfo.stream)
                    .launch(&simParams, &granData, startOffset, count);
            }
        }
    }
}

inline void DEMDynamicThread::dispatchPatchBasedForceCorrections(
    const ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>>& typeStartCountPrimitiveMap,
    const ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>>& typeStartCountPatchMap,
    const ContactTypeMap<std::vector<std::pair<std::shared_ptr<JitHelper::CachedProgram>, std::string>>>&
        typeKernelMap) {
    // Reset max tri-tri penetration for this timestep on device (kT may need this info)
    DEME_GPU_CALL(cudaMemset(maxTriTriPenetration.getDevicePointer(), 0, sizeof(double)));

    // For each contact type that exists, check if it is patch(mesh)-related type...
    for (size_t i = 0; i < m_numExistingTypes; i++) {
        contact_t contact_type = existingContactTypes[i];
        if (contact_type == SPHERE_TRIANGLE_CONTACT || contact_type == TRIANGLE_TRIANGLE_CONTACT ||
            contact_type == TRIANGLE_ANALYTICAL_CONTACT) {
            const auto& start_count_primitive = typeStartCountPrimitiveMap.at(contact_type);
            const auto& start_count_patch = typeStartCountPatchMap.at(contact_type);
            contactPairs_t startOffsetPrimitive = start_count_primitive.first;
            contactPairs_t countPrimitive = start_count_primitive.second;
            contactPairs_t startOffsetPatch = start_count_patch.first;
            contactPairs_t countPatch = start_count_patch.second;

            // Vote for the contact direction; voting power depends on the contact area
            // This reduce-by-key operation reduces primitive-recorded force pairs into patch/convex part-based
            // force pairs. All elements that share the same geomToPatchMap value vote together.
            if (countPrimitive > 0) {
                // Allocate temporary arrays for the voting process
                float3* weightedNormals =
                    (float3*)solverScratchSpace.allocateTempVector("weightedNormals", countPrimitive * sizeof(float3));
                double* areas =
                    (double*)solverScratchSpace.allocateTempVector("areas", countPrimitive * sizeof(double));
                // Keys extracted from geomToPatchMap - these map primitives to patch pairs
                contactPairs_t* keys = (contactPairs_t*)solverScratchSpace.allocateTempVector(
                    "votingKeys", countPrimitive * sizeof(contactPairs_t));

                // Allocate arrays for reduce-by-key results (uniqueKeys uses contactPairs_t, not patchIDPair_t)
                contactPairs_t* uniqueKeys = (contactPairs_t*)solverScratchSpace.allocateTempVector(
                    "uniqueKeys", countPrimitive * sizeof(contactPairs_t));
                float3* votedWeightedNormals = (float3*)solverScratchSpace.allocateTempVector(
                    "votedWeightedNormals", countPrimitive * sizeof(float3));
                solverScratchSpace.allocateDualStruct("numUniqueKeys");
                size_t* numUniqueKeys = solverScratchSpace.getDualStructDevice("numUniqueKeys");

                // Step 1: Prepare weighted normals, areas, and keys
                // The kernel extracts keys from geomToPatchMap, computes weighted normals, and stores areas
                prepareWeightedNormalsForVoting(&granData, weightedNormals, areas, keys, startOffsetPrimitive,
                                                countPrimitive, streamInfo.stream);

                // Step 2: Reduce-by-key for weighted normals (sum)
                // The keys are geomToPatchMap values (contactPairs_t), which group primitives by patch pair
                cubSumReduceByKey<contactPairs_t, float3>(keys, uniqueKeys, weightedNormals, votedWeightedNormals,
                                                          numUniqueKeys, countPrimitive, streamInfo.stream,
                                                          solverScratchSpace);
                solverScratchSpace.finishUsingTempVector("weightedNormals");
                // For extra safety
                solverScratchSpace.syncDualStructDeviceToHost("numUniqueKeys");
                size_t numUniqueKeysHost = *(solverScratchSpace.getDualStructHost("numUniqueKeys"));
                // std::cout << "Keys:" << std::endl;
                // displayDeviceArray<contactPairs_t>(keys, countPrimitive);
                // std::cout << "Unique Keys:" << std::endl;
                // displayDeviceArray<contactPairs_t>(uniqueKeys, numUniqueKeysHost);
                if (numUniqueKeysHost != countPatch) {
                    DEME_ERROR(
                        "Patch-based contact voting produced %zu unique patch pairs, but expected %zu pairs for "
                        "contact type %d!",
                        numUniqueKeysHost, countPatch, contact_type);
                }

                // Step 3: Normalize the voted normals by total area and scatter back to a temp array.
                float3* votedNormals =
                    (float3*)solverScratchSpace.allocateTempVector("votedNormals", countPatch * sizeof(float3));
                normalizeAndScatterVotedNormals(votedWeightedNormals, votedNormals, countPatch, streamInfo.stream);
                solverScratchSpace.finishUsingTempVector("votedWeightedNormals");
                // displayDeviceFloat3(votedNormals, countPatch);

                // Step 4: Compute projected penetration and area for each primitive contact
                // Both the penetration and area are projected onto the voted normal
                // If the projected penetration becomes negative, both are set to 0
                // Reuse keys array for the reduce-by-key operation
                double* projectedPenetrations = (double*)solverScratchSpace.allocateTempVector(
                    "projectedPenetrations", countPrimitive * sizeof(double));
                double* projectedAreas =
                    (double*)solverScratchSpace.allocateTempVector("projectedAreas", countPrimitive * sizeof(double));
                computeWeightedUsefulPenetration(&granData, votedNormals, keys, areas, projectedPenetrations,
                                                 projectedAreas, startOffsetPrimitive, startOffsetPatch, countPrimitive,
                                                 streamInfo.stream);
                solverScratchSpace.finishUsingTempVector("areas");

                // Step 5: Reduce-by-key to get total projected area per patch pair (sum)
                double* totalProjectedAreas =
                    (double*)solverScratchSpace.allocateTempVector("totalProjectedAreas", countPatch * sizeof(double));
                cubSumReduceByKey<contactPairs_t, double>(keys, uniqueKeys, projectedAreas, totalProjectedAreas,
                                                          numUniqueKeys, countPrimitive, streamInfo.stream,
                                                          solverScratchSpace);

                // Step 6: Reduce-by-key to get max projected penetration per patch pair (max).
                // This result, maxProjectedPenetrations, is the max of projected penetration, aka the max pen in the
                // physical overlap case, and it's not the same as maxPenetrations in step 9 which is a fallback
                // primitive-derived penetration.
                double* maxProjectedPenetrations = (double*)solverScratchSpace.allocateTempVector(
                    "maxProjectedPenetrations", countPatch * sizeof(double));
                cubMaxReduceByKey<contactPairs_t, double>(keys, uniqueKeys, projectedPenetrations,
                                                          maxProjectedPenetrations, numUniqueKeys, countPrimitive,
                                                          streamInfo.stream, solverScratchSpace);

                // Step 7: Compute weighted contact points for each primitive (normal case)
                // The weight is: projected_penetration * projected_area
                // Reuse keys, uniqueKeys, and numUniqueKeys that are still allocated
                double3* weightedContactPoints = (double3*)solverScratchSpace.allocateTempVector(
                    "weightedContactPoints", countPrimitive * sizeof(double3));
                double* contactWeights =
                    (double*)solverScratchSpace.allocateTempVector("contactWeights", countPrimitive * sizeof(double));
                computeWeightedContactPoints(&granData, weightedContactPoints, contactWeights, projectedPenetrations,
                                             projectedAreas, startOffsetPrimitive, countPrimitive, streamInfo.stream);
                solverScratchSpace.finishUsingTempVector("projectedPenetrations");
                solverScratchSpace.finishUsingTempVector("projectedAreas");
                // Reduce-by-key to get total weighted contact points per patch pair
                double3* totalWeightedContactPoints = (double3*)solverScratchSpace.allocateTempVector(
                    "totalWeightedContactPoints", countPatch * sizeof(double3));
                double* totalContactWeights =
                    (double*)solverScratchSpace.allocateTempVector("totalContactWeights", countPatch * sizeof(double));
                cubSumReduceByKey<contactPairs_t, double3>(keys, uniqueKeys, weightedContactPoints,
                                                           totalWeightedContactPoints, numUniqueKeys, countPrimitive,
                                                           streamInfo.stream, solverScratchSpace);
                cubSumReduceByKey<contactPairs_t, double>(keys, uniqueKeys, contactWeights, totalContactWeights,
                                                          numUniqueKeys, countPrimitive, streamInfo.stream,
                                                          solverScratchSpace);
                solverScratchSpace.finishUsingTempVector("weightedContactPoints");
                solverScratchSpace.finishUsingTempVector("contactWeights");
                // Compute voted contact points per patch pair by dividing by total weight
                double3* votedContactPoints =
                    (double3*)solverScratchSpace.allocateTempVector("votedContactPoints", countPatch * sizeof(double3));
                computeFinalContactPointsPerPatch(totalWeightedContactPoints, totalContactWeights, votedContactPoints,
                                                  countPatch, streamInfo.stream);
                solverScratchSpace.finishUsingTempVector("totalWeightedContactPoints");
                solverScratchSpace.finishUsingTempVector("totalContactWeights");

                // Step 8: Handle zero-area patches (all primitive areas are 0)
                // For these patches, we need to find the max penetration primitive and use its normal/penetration

                // 8a: Extract primitive penetrations for max-reduce
                double* primitivePenetrations = (double*)solverScratchSpace.allocateTempVector(
                    "primitivePenetrations", countPrimitive * sizeof(double));
                extractPrimitivePenetrations(&granData, primitivePenetrations, startOffsetPrimitive, countPrimitive,
                                             streamInfo.stream);

                // 8b: Max-negative-reduce-by-key to get max negative penetration per patch
                // This finds the largest negative value (smallest absolute value among negatives)
                // Positive values are treated as very negative to indicate invalid/non-physical state
                double* maxPenetrations =
                    (double*)solverScratchSpace.allocateTempVector("maxPenetrations", countPatch * sizeof(double));
                cubMaxNegativeReduceByKey<contactPairs_t, double>(keys, uniqueKeys, primitivePenetrations,
                                                                  maxPenetrations, numUniqueKeys, countPrimitive,
                                                                  streamInfo.stream, solverScratchSpace);
                solverScratchSpace.finishUsingTempVector("primitivePenetrations");

                // 8c: Find max-penetration primitives for zero-area patches and extract their normals, penetrations,
                // and contact points
                float3* zeroAreaNormals =
                    (float3*)solverScratchSpace.allocateTempVector("zeroAreaNormals", countPatch * sizeof(float3));
                double* zeroAreaPenetrations =
                    (double*)solverScratchSpace.allocateTempVector("zeroAreaPenetrations", countPatch * sizeof(double));
                double3* zeroAreaContactPoints = (double3*)solverScratchSpace.allocateTempVector(
                    "zeroAreaContactPoints", countPatch * sizeof(double3));
                findMaxPenetrationPrimitiveForZeroAreaPatches(
                    &granData, maxPenetrations, zeroAreaNormals, zeroAreaPenetrations, zeroAreaContactPoints, keys,
                    startOffsetPrimitive, startOffsetPatch, countPrimitive, streamInfo.stream);
                solverScratchSpace.finishUsingTempVector("maxPenetrations");

                // Step 8d: Check if each patch has any SAT-satisfying primitive (for tri-tri contacts)
                // If no primitive satisfies SAT, the patch contact is non-physical and should use Step 9 fallback
                notStupidBool_t* patchHasSAT = nullptr;
                if (contact_type == TRIANGLE_TRIANGLE_CONTACT) {
                    patchHasSAT = (notStupidBool_t*)solverScratchSpace.allocateTempVector(
                        "patchHasSAT", countPatch * sizeof(notStupidBool_t));
                    checkPatchHasSATSatisfyingPrimitive(&granData, patchHasSAT, keys, startOffsetPrimitive,
                                                        startOffsetPatch, countPrimitive, countPatch,
                                                        streamInfo.stream);
                }

                // Clean up keys arrays now that we're done with reductions
                solverScratchSpace.finishUsingTempVector("votingKeys");
                solverScratchSpace.finishUsingTempVector("uniqueKeys");
                solverScratchSpace.finishUsingDualStruct("numUniqueKeys");

                // Step 9: Finalize patch results by combining voting with zero-area handling.
                // If patch-based projected area is 0 (or this patch pair consists of no SAT pair), meaning no physical
                // contact, we use the fallback estimations (zeroArea*) of CP, penetration and areas.
                double* finalAreas =
                    (double*)solverScratchSpace.allocateTempVector("finalAreas", countPatch * sizeof(double));
                float3* finalNormals =
                    (float3*)solverScratchSpace.allocateTempVector("finalNormals", countPatch * sizeof(float3));
                // Resize permanent finalPenetrations array for this patch contact batch.
                // Note: I made it a permanent array in case that in the future, we want to transfer this entire array
                // to kT for better margin derivation.
                DEME_DEVICE_ARRAY_RESIZE(finalPenetrations, countPatch);

                double3* finalContactPoints =
                    (double3*)solverScratchSpace.allocateTempVector("finalContactPoints", countPatch * sizeof(double3));
                finalizePatchResults(totalProjectedAreas, votedNormals, maxProjectedPenetrations, votedContactPoints,
                                     zeroAreaNormals, zeroAreaPenetrations, zeroAreaContactPoints, patchHasSAT,
                                     finalAreas, finalNormals, finalPenetrations.data(), finalContactPoints, countPatch,
                                     streamInfo.stream);
                solverScratchSpace.finishUsingTempVector("totalProjectedAreas");
                solverScratchSpace.finishUsingTempVector("votedNormals");
                solverScratchSpace.finishUsingTempVector("maxProjectedPenetrations");
                solverScratchSpace.finishUsingTempVector("zeroAreaNormals");
                solverScratchSpace.finishUsingTempVector("zeroAreaPenetrations");
                solverScratchSpace.finishUsingTempVector("votedContactPoints");
                solverScratchSpace.finishUsingTempVector("zeroAreaContactPoints");
                solverScratchSpace.finishUsingTempVector("patchHasSAT");

                // Now we have:
                // - finalAreas: final contact area per patch pair (countPatch elements)
                // - finalNormals: final normal direction per patch pair (countPatch elements)
                // - finalPenetrations: final penetration depth per patch pair (countPatch elements)
                // - finalContactPoints: final contact point per patch pair (countPatch elements)
                // These can be used for subsequent force calculations
                // std::cout << "Patch-based contact penetration, area, normal, contact point for contact type "
                //           << (int)contact_type << ":" << std::endl;
                // displayDeviceArray<double>(finalPenetrations, countPatch);
                // displayDeviceArray<double>(finalAreas, countPatch);
                // displayDeviceFloat3(finalNormals, countPatch);
                // displayDeviceFloat3<double3>(finalContactPoints, countPatch);

                // Call specialized patch-based force correction kernels here
                if (contactTypePatchKernelMap.count(contact_type) > 0) {
                    const auto& kernelList = contactTypePatchKernelMap.at(contact_type);
                    for (const auto& [progName, kernelName] : kernelList) {
                        size_t blocks =
                            (countPatch + DT_FORCE_CALC_NTHREADS_PER_BLOCK - 1) / DT_FORCE_CALC_NTHREADS_PER_BLOCK;
                        if (blocks > 0) {
                            progName->kernel(kernelName)
                                .instantiate()
                                .configure(dim3(blocks), dim3(DT_FORCE_CALC_NTHREADS_PER_BLOCK), 0, streamInfo.stream)
                                .launch(&simParams, &granData, finalAreas, finalNormals, finalPenetrations.data(),
                                        finalContactPoints, startOffsetPatch, countPatch);
                        }
                    }
                }
                DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

                // If this is a tri-tri contact, compute max penetration for kT
                // The max value stays on device until sendToTheirBuffer transfers it
                if (contact_type == TRIANGLE_TRIANGLE_CONTACT && countPatch > 0) {
                    // Compute max penetration and store it on device
                    // Note: penetration values should always be non-negative in physical contacts
                    cubMaxReduce<double>(finalPenetrations.data(), &maxTriTriPenetration, countPatch, streamInfo.stream,
                                         solverScratchSpace);
                    // No toHost() here - keep on device since host never needs it
                    // maxTriTriPenetration.toHost();
                    // std::cout << "Max tri-tri penetration after patch-based correction: " << *maxTriTriPenetration
                    //           << std::endl;
                }

                // Final clean up
                solverScratchSpace.finishUsingTempVector("finalAreas");
                solverScratchSpace.finishUsingTempVector("finalNormals");
                // Note: finalPenetrations is now a permanent array, not freed here
                solverScratchSpace.finishUsingTempVector("finalContactPoints");
            }
        }
    }
    // std::cout << "===========================" << std::endl;
}

void DEMDynamicThread::calculateForces() {
    DEME_NVTX_RANGE("dT::calculateForces");
    // Reset force (acceleration) arrays for this time step
    size_t nContactPairs = *solverScratchSpace.numContacts;

    timers.StartGpuTimer("Clear force array", streamInfo.stream);
    {
        DEME_NVTX_RANGE("dT::prepareAccArrays");
        prepareAccArrays(&simParams, &granData, simParams->nOwnerBodies, streamInfo.stream);

        // prepareForceArrays is no longer needed
        // if (!solverFlags.useNoContactRecord) {
        //     // Pay attention that the force result-related arrays have nPrimitiveContactPairs elements, not
        //     // nContactPairs
        //     prepareForceArrays(&simParams, &granData, nPrimitiveContactPairs, streamInfo.stream);
        // }
    }
    timers.StopGpuTimer("Clear force array", streamInfo.stream);

    // If no contact then we don't have to calculate forces. Note there might still be forces, coming from prescription
    // or other sources.
    if (nContactPairs > 0) {
        timers.StartGpuTimer("Calculate contact forces", streamInfo.stream);
        DEME_NVTX_RANGE("dT::contactForces");

        // Call specialized kernels for each contact type that exists
        dispatchPrimitiveForceKernels(typeStartCountPrimitiveMap, contactTypePrimitiveKernelMap);
        // Note: dispatchPrimitiveForceKernels calculates forces induced by the most basic primitives, aka spheres,
        // triangles... However, for contacts to be truely physical, sometimes such contact pairs within a patch (which
        // marks a convex component of a owner) need to vote to decide the true contact. This is where the second step
        // comes in.
        dispatchPatchBasedForceCorrections(typeStartCountPrimitiveMap, typeStartCountPatchMap,
                                           contactTypePatchKernelMap);

        // displayDeviceFloat3(granData->contactForces, nContactPairs);
        // displayDeviceArray<contact_t>(granData->contactTypePatch, nContactPairs);
        // displayDeviceArray<bodyID_t>(granData->idPatchA, nContactPairs);
        // displayDeviceArray<bodyID_t>(granData->idPatchB, nContactPairs);
        // std::cout << "===========================" << std::endl;
        timers.StopGpuTimer("Calculate contact forces", streamInfo.stream);

        if (!solverFlags.useForceCollectInPlace) {
            DEME_NVTX_RANGE("dT::collectForces");
            timers.StartGpuTimer("Optional force reduction", streamInfo.stream);
            // Reflect those body-wise forces on their owner clumps
            size_t blocks_needed_for_contacts =
                (nContactPairs + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
            // This does both acc and ang acc
            collect_force_kernels->kernel("forceToAcc")
                .instantiate()
                .configure(dim3(blocks_needed_for_contacts), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
                .launch(&granData, nContactPairs);
            // displayDeviceArray<float>(granData->aZ, simParams->nOwnerBodies);
            // displayDeviceFloat3(granData->contactForces, nContactPairs);
            // std::cout << nContactPairs << std::endl;
            timers.StopGpuTimer("Optional force reduction", streamInfo.stream);
        }
    }
}

inline void DEMDynamicThread::integrateOwnerMotions() {
    DEME_NVTX_RANGE("dT::integrateOwners");
    timers.StartGpuTimer("Integration", streamInfo.stream);
    size_t blocks_needed_for_clumps =
        (simParams->nOwnerBodies + DEME_NUM_BODIES_PER_BLOCK - 1) / DEME_NUM_BODIES_PER_BLOCK;
    integrator_kernels->kernel("integrateOwners")
        .instantiate()
        .configure(dim3(blocks_needed_for_clumps), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
        .launch(&simParams, &granData, (double)simParams->dyn.timeElapsed);
    timers.StopGpuTimer("Integration", streamInfo.stream);
    // Integration results must be visible before subsequent host-side coordination.
    recordEventOnly();
    const int64_t stamp = pSchedSupport->currentStampOfDynamic.load(std::memory_order_relaxed) + 1;
    recordProgressEvent(stamp);
}

inline void DEMDynamicThread::routineChecks() {
    DEME_NVTX_RANGE("dT::applyFamilyChanges");
    if (solverFlags.canFamilyChangeOnDevice) {
        size_t blocks_needed_for_clumps =
            (simParams->nOwnerBodies + DEME_NUM_MODERATORS_PER_BLOCK - 1) / DEME_NUM_MODERATORS_PER_BLOCK;
        mod_kernels->kernel("applyFamilyChanges")
            .instantiate()
            .configure(dim3(blocks_needed_for_clumps), dim3(DEME_NUM_MODERATORS_PER_BLOCK), 0, streamInfo.stream)
            .launch(&simParams, &granData, simParams->nOwnerBodies);
    }
}

inline void DEMDynamicThread::determineSysVel() {
    // Get linear velocity
    pCycleVel = approxVelFunc->dT_GetDeviceValues();
    // Get angular velocity magnitude
    pCycleAngVel = approxAngVelFunc->dT_GetDeviceValues();
}

inline void DEMDynamicThread::unpack_impl() {
    drainProgressEvents();
    // Single-producer/single-consumer: kT fills the buffer, dT unpacks it once the fresh flag flips.
    unpackMyBuffer();
    contactPairArr_isFresh = true;
    // pSchedSupport->schedulingStats.nDynamicReceives++;

    // dT got the produce, now mark its buffer to be no longer fresh.
    pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh.store(false, std::memory_order_release);
    // Used for inspecting on average how stale kT's produce is (in dT steps).
    // Reference to the stamp of the ingredient batch that produced this update (exclude the 1-step pipeline).
    const int64_t recv_stamp =
        (recv_stamp_override >= 0) ? recv_stamp_override : (pSchedSupport->completedStampOfDynamic).load();
    const int64_t send_stamp = (pSchedSupport->kinematicIngredProdDateStamp).load();
    int64_t lag_steps = recv_stamp - send_stamp;
    if (lag_steps > 0) {
        lag_steps -= 1;
    }
    if (lag_steps < 0) {
        lag_steps = 0;
    }
    pSchedSupport->schedulingStats.accumKinematicLagSteps += static_cast<uint64_t>(lag_steps);
    // dT needs to know how fresh the contact pair info is, and that is determined by when kT received this batch of
    // ingredients.
    pSchedSupport->stampLastDynamicUpdateProdDate = send_stamp;

    // If this is a history-based run, then when contacts are received, we need to migrate the contact
    // history info, to match the structure of the new contact array
    if (!solverFlags.isHistoryless) {
        migrateEnduringContacts();
        if (!contactMappingUsesBuffer) {
            solverScratchSpace.finishUsingTempVector("contactMapping");
        }
    }

    // On dT side, we also calculate how many (the offsets in contact arrays) contacts they are for each type.
    // But note here we are working on primitive-based contact types, not patch-based contact types yet.
    solverScratchSpace.allocateDualStruct("numExistingTypes");
    contactPairs_t* typeCounts = (contactPairs_t*)solverScratchSpace.allocateTempVector(
        "typeCounts", (NUM_SUPPORTED_CONTACT_TYPES + 1) * sizeof(contactPairs_t));
    // Recall existingContactTypes is pre-allocated for maximum possible types
    cubRunLengthEncode<contact_t, contactPairs_t>(
        granData->contactTypePrimitive, existingContactTypes.device(), typeCounts,
        solverScratchSpace.getDualStructDevice("numExistingTypes"), *solverScratchSpace.numPrimitiveContacts,
        streamInfo.stream, solverScratchSpace);
    solverScratchSpace.syncDualStructDeviceToHost("numExistingTypes");
    m_numExistingTypes = *solverScratchSpace.getDualStructHost("numExistingTypes");
    cubPrefixScan<contactPairs_t, contactPairs_t>(typeCounts, typeStartOffsetsPrimitive.device(), m_numExistingTypes,
                                                  streamInfo.stream, solverScratchSpace);
    existingContactTypes.toHost();
    typeStartOffsetsPrimitive.toHost();
    typeStartCountPrimitiveMap.SetAll({0, 0});
    for (size_t i = 0; i < m_numExistingTypes; i++) {
        DEME_DEBUG_PRINTF("Contact type %d starts at offset %u", existingContactTypes[i], typeStartOffsetsPrimitive[i]);
        typeStartCountPrimitiveMap[existingContactTypes[i]] =
            std::make_pair(typeStartOffsetsPrimitive[i],
                           (i + 1 < m_numExistingTypes ? typeStartOffsetsPrimitive[i + 1]
                                                       : (contactPairs_t)*solverScratchSpace.numPrimitiveContacts) -
                               typeStartOffsetsPrimitive[i]);
    }
    // Debug output of the map
    // for (const auto& entry : typeStartCountPrimitiveMap) {
    //     printf("Contact type %d starts at offset %u and has count %u\n", entry.first, entry.second.first,
    //     entry.second.second);
    // }

    // Now for patch-based contacts, we do the same thing. Note the unique types herein will be the same as thosein.
    cubRunLengthEncode<contact_t, contactPairs_t>(granData->contactTypePatch, existingContactTypes.device(), typeCounts,
                                                  solverScratchSpace.getDualStructDevice("numExistingTypes"),
                                                  *solverScratchSpace.numContacts, streamInfo.stream,
                                                  solverScratchSpace);
    cubPrefixScan<contactPairs_t, contactPairs_t>(typeCounts, typeStartOffsetsPatch.device(), m_numExistingTypes,
                                                  streamInfo.stream, solverScratchSpace);
    typeStartOffsetsPatch.toHost();
    typeStartCountPatchMap.SetAll({0, 0});
    for (size_t i = 0; i < m_numExistingTypes; i++) {
        typeStartCountPatchMap[existingContactTypes[i]] = std::make_pair(
            typeStartOffsetsPatch[i], (i + 1 < m_numExistingTypes ? typeStartOffsetsPatch[i + 1]
                                                                  : (contactPairs_t)*solverScratchSpace.numContacts) -
                                          typeStartOffsetsPatch[i]);
    }

    solverScratchSpace.finishUsingTempVector("typeCounts");
    solverScratchSpace.finishUsingDualStruct("numExistingTypes");
}

inline void DEMDynamicThread::ifProduceFreshThenUseIt(bool allow_blocking) {
    tryConsumeKinematicProduce(allow_blocking, false, true);
}

inline void DEMDynamicThread::calibrateParams() {
    auto& r = futureDriftRegulator;
    const unsigned MAX = solverFlags.upperBoundFutureDrift;
    if (!r.receive_pending)
        return;
    const bool aut = solverFlags.autoUpdateFreq;
    const double ur = std::clamp((double)solverFlags.futureDriftSendUpperBoundRatio, 0.0, 1.0);
    const double lr = std::clamp((double)solverFlags.futureDriftSendLowerBoundRatio, 0.0, 1.0);
    const uint64_t recv = r.pending_recv_stamp;
    double tnow = r.pending_total_time;
    r.receive_pending = false;
    if (tnow <= 0.0)
        tnow = getCycleElapsedSeconds();
    int64_t send_i = pSchedSupport->stampLastDynamicUpdateProdDate.load();
    if (send_i < 0)
        send_i = (int64_t)recv;
    const uint64_t send = (uint64_t)send_i;
    const unsigned lag_steps = (recv > send + 1) ? (unsigned)(recv - send - 1) : 0u;
    r.last_observed_kinematic_lag_steps = lag_steps;
    uint64_t prev = r.has_last_step_sample ? r.last_step_sample : recv;
    if (prev > recv)
        prev = recv;
    const uint64_t steps = recv - prev;
    double dt = 0.0;
    if (r.has_last_step_sample) {
        const double dbg = std::max(0.0, r.debug_cum_time - r.last_debug_cum_time);
        dt = tnow - r.last_total_time - dbg;
        if (dt < 0.0)
            dt = 0.0;
    }
    r.has_last_step_sample = true;
    r.last_step_sample = recv;
    r.last_total_time = tnow;
    r.last_debug_cum_time = r.debug_cum_time;
    const unsigned drift_total = (steps > 0) ? (unsigned)std::min<uint64_t>(steps - 1, MAX) : 0u;
    if (r.lag_ema_initialized || lag_steps > 0) {
        ema_asym(r.lag_ema, r.lag_ema_initialized, (double)lag_steps, 0.35, 0.10, 0.0);
    }
    const double lag_pred = std::max(r.lag_ema, (double)lag_steps);
    int lag_i = (int)std::floor(lag_pred + 0.5);
    if (lag_i < 0)
        lag_i = 0;
    if ((unsigned)lag_i > MAX)
        lag_i = (int)MAX;
    const unsigned lag_u = (unsigned)lag_i;
    const bool meas = (aut && steps > 0 && dt > 0.0 && drift_total > 0u);
    const double cost = meas ? dt / (double)steps : 0.0;
    const unsigned dmin = std::max(1u, lag_u);
    const double drift_floor = std::max(5.0, (double)dmin);
    double drift_ref = drift_floor;
    if (meas) {
        const unsigned obs = clamp_drift_u(drift_total, MAX);
        const double c_now = std::max(1e-9, cost);
        const double c_ref = r.cost_scale_initialized ? r.cost_scale_ema : c_now;
        const bool outlier = r.cost_scale_initialized && (c_now > c_ref * 10.0);
        const double c_for_scale = r.cost_scale_initialized ? std::min(c_now, c_ref * 5.0) : c_now;
        ema_asym(r.cost_scale_ema, r.cost_scale_initialized, c_for_scale, 0.10, 0.02, 1e-9);
        ema_asym(r.drift_scale_ema, r.drift_scale_initialized, (double)obs, 0.15, 0.15, 1.0);
        ring_push(r, cost, obs);
        drift_ref = drift_ref_quantile(r, drift_floor);
        if (!outlier) {
            r.drift_rls.update(obs, drift_ref, cost);
            const double scale = r.cost_scale_ema;
            if (rls_is_bad(r.drift_rls, obs, drift_ref, scale)) {
                r.drift_rls.reset();
                r.drift_rls_samples = 0;
                r.window_size = 0;
                r.window_pos = 0;
                r.drift_scale_initialized = true;
                r.drift_scale_ema = std::max(1.0, (double)dmin);
                drift_ref = drift_ref_quantile(r, drift_floor);
            } else {
                r.drift_rls_samples++;
            }
        }
    } else {
        drift_ref = drift_ref_quantile(r, drift_floor);
    }
    const uint64_t n = r.drift_rls_samples;
    const unsigned cmd_cur = clamp_drift_u(std::max(1u, *perhapsIdealFutureDrift), MAX);
    const double safety = (double)solverFlags.futureDriftEffDriftSafetyFactor;
    const unsigned true_cmd = clamp_drift_u((unsigned)std::max(1.0, std::floor(((double)cmd_cur / safety) + 0.5)), MAX);
    unsigned dcur = (r.last_proposed > 0) ? clamp_drift_u(r.last_proposed, MAX) : true_cmd;
    if (dcur < dmin)
        dcur = dmin;
    constexpr uint64_t PROBE_N = 24, ACT_N = 40;
    constexpr double MOVE_FR = 0.1, ANC_FR = 0.05, IMP_FR = 0.01, CAP_R = 1.5, LAG_M = 2.0;
    constexpr unsigned STEP = 2;
    unsigned dtgt = dcur;
    if (!aut)
        dtgt = std::max(true_cmd, dmin);
    else if (r.drift_rls.initialized && n > 0) {
        const unsigned pstep = std::max(1u, std::min(STEP, dcur / 10u));
        const unsigned dup = clamp_drift_u(std::min(MAX, dcur + pstep), MAX);
        const unsigned ddn = clamp_drift_u(std::max(dmin, (dcur > pstep) ? (dcur - pstep) : dmin), MAX);
        if (n < PROBE_N) {
            const unsigned phase = (unsigned)(n % 3u);
            const int dir = (((n / 3u) % 2u) == 0u) ? +1 : -1;
            if (phase == 1u)
                dtgt = (dir > 0) ? dup : ddn;
            else if (phase == 2u)
                dtgt = (dir > 0) ? ddn : dup;
        } else if (n >= ACT_N) {
            unsigned lo = (dcur > STEP) ? (dcur - STEP) : 1u;
            if (lo < dmin)
                lo = dmin;
            unsigned hi = std::min(MAX, dcur + STEP);
            const double pen_ref = std::max(drift_floor, lag_pred * LAG_M);
            const unsigned cap_hi = std::max(dmin, (unsigned)std::ceil(std::min(drift_ref, pen_ref) * CAP_R));
            if (hi > cap_hi)
                hi = cap_hi;
            if (hi < lo)
                lo = hi;
            const double scale = r.cost_scale_initialized ? r.cost_scale_ema : std::max(1e-9, cost);
            const double mp = MOVE_FR * scale, ap = ANC_FR * scale;
            const double sigma = std::sqrt(std::max(0.0, r.drift_rls.sigma2_ema));
            const double improve0 = std::max(IMP_FR * scale, 2.5 * sigma);
            const double rr = ((double)dcur - pen_ref) / pen_ref;
            const double bs = std::min(1.0, std::abs(rr));
            const double upb = (rr > 0.0) ? (1.0 + bs) : 1.0;
            const double dnb = (rr < 0.0) ? (1.0 + bs) : 1.0;
            double best = std::numeric_limits<double>::infinity();
            unsigned best_tot = dcur;
            const double inv = 1.0 / (double)std::max(1u, dcur);
            for (unsigned d = lo; d <= hi; ++d) {
                const unsigned w = apply_wait_policy_u(clamp_wait_i((int)d - lag_i, MAX), lag_pred, ur, lr, MAX);
                const unsigned tot = clamp_drift_u(w + lag_u, MAX);
                const double y = r.drift_rls.predict(tot, drift_ref);
                const double rel = (double)((int)tot - (int)dcur) * inv;
                const double score = y + mp * std::abs(rel) + ap * (rel * rel);
                if (score < best) {
                    best = score;
                    best_tot = tot;
                }
            }
            const double ycur = r.drift_rls.predict(dcur, drift_ref);
            double need = improve0;
            if (best_tot > dcur)
                need *= upb;
            else if (best_tot < dcur)
                need *= dnb;
            if (best <= ycur - need)
                dtgt = best_tot;
        }
    }
    if (aut) {
        const double cap_ref = std::max(drift_floor, lag_pred * LAG_M);
        const unsigned cap1 = std::max(dmin, (unsigned)std::ceil(drift_ref * CAP_R));
        const unsigned cap2 = std::max(dmin, (unsigned)std::ceil(cap_ref * CAP_R));
        const unsigned cap = std::min(cap1, cap2);
        if (dtgt > cap)
            dtgt = cap;
        const unsigned slo = (dcur > STEP) ? (dcur - STEP) : 1u;
        const unsigned shi = dcur + STEP;
        dtgt = std::clamp(dtgt, slo, shi);
    }
    dtgt = clamp_drift_u(std::max(dtgt, dmin), MAX);
    const unsigned wait = apply_wait_policy_u(clamp_wait_i((int)dtgt - lag_i, MAX), lag_pred, ur, lr, MAX);
    const unsigned total = clamp_drift_u(wait + lag_u, MAX);
    r.last_wait_cmd = wait;
    r.last_proposed = total;
    if (aut) {
        const unsigned cmd_out = clamp_drift_u((unsigned)std::ceil((double)total * safety), MAX);
        *perhapsIdealFutureDrift = cmd_out;
    }
    r.next_send_step = recv + (uint64_t)wait;
    r.next_send_wait = wait;
    r.pending_send = true;

    DEME_DEBUG_PRINTF(
        "[calibrateParams] recv=%llu send=%llu steps=%llu dt=%.6g cost=%.6g drift_total=%u lag=%u lag_ema=%.3g "
        "dcur=%u dtgt=%u wait=%u next_send=%llu drift_ref=%.3g\n",
        (unsigned long long)recv, (unsigned long long)send, (unsigned long long)steps, dt, cost, drift_total, lag_steps,
        r.lag_ema, dcur, dtgt, wait, (unsigned long long)r.next_send_step, drift_ref);
}

inline void DEMDynamicThread::ifProduceFreshThenUseItAndSendNewOrder() {
    auto& reg = futureDriftRegulator;
    const bool allow_blocking = (streamInfo.device != kT->streamInfo.device);
    if (!reg.pending_send) {
        // Consume fresh produce before force evaluation when possible (same-device path).
        tryConsumeKinematicProduce(allow_blocking, true, true);
    }
    // Refresh the regulator and schedule the next work order for kT (no-op unless receive_pending is set).
    reg.pending_total_time = getCycleElapsedSeconds();
    calibrateParams();

    // If a kT work order is scheduled (possibly due to kT being very fast), only send it when due.
    drainProgressEvents();
    const uint64_t now_stamp = static_cast<uint64_t>(pSchedSupport->currentStampOfDynamic.load());
    if (reg.pending_send && now_stamp >= reg.next_send_step &&
        !pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh.load(std::memory_order_acquire)) {
        timers.GetTimer("Send to kT buffer").start();
        determineSysVel();
        // Record the max drift value used for this work order, so the tuner can attribute the next observation.
        reg.last_sent_proposed = *perhapsIdealFutureDrift;
        reg.last_sent_true = reg.last_proposed;
        reg.last_sent_wait = reg.next_send_wait;
        sendToTheirBuffer();
        pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh.store(true, std::memory_order_release);
        pSchedSupport->schedulingStats.nKinematicUpdates++;
        timers.GetTimer("Send to kT buffer").stop();
        pSchedSupport->cv_KinematicCanProceed.notify_all();
        reg.pending_send = false;
    }
}

void DEMDynamicThread::workerThread() {
    // Set the gpu for this thread
    DEME_GPU_CALL(cudaSetDevice(streamInfo.device));
    DEME_NVTX_NAME_STREAM(streamInfo.stream, "dT_stream");

    // Allocate arrays whose length does not depend on user inputs
    initAllocation();

    while (!pSchedSupport->dynamicShouldJoin) {
        {
            std::unique_lock<std::mutex> lock(pSchedSupport->dynamicStartLock);
            while (!pSchedSupport->dynamicStarted) {
                pSchedSupport->cv_DynamicStartLock.wait(lock);
            }
            // Ensure that we wait for start signal on next iteration
            pSchedSupport->dynamicStarted = false;
            // The following is executed when kT and dT are being destroyed
            if (pSchedSupport->dynamicShouldJoin) {
                break;
            }
        }

        // There is only 2 situations where dT needs to wait for kT to provide one initial CD result...
        // Those are the `new-boot after previous sync' case, or the user significantly changed the simulation
        // environment; in any other situations, dT does not have `drift-into-future-too-much' problem here, b/c if it
        // has the problem then it would have been addressed at the end of last DoDynamics call, the final `ShouldWait'
        // check. Note: pendingCriticalUpdate is not fail-safe at all right now. The user still needs to sync before
        // making critical changes to the system to ensure safety.
        if (pSchedSupport->stampLastDynamicUpdateProdDate < 0 || pendingCriticalUpdate) {
            // This is possible: If it is after a user-manual sync
            ifProduceFreshThenUseIt(true);

            // If the user loaded contact manually, there is an extra thing we need to do: update kT prev_contact
            // arrays. Note the user can add anything only from a sync-ed stance anyway, so this check needs to be done
            // only here.
            if (new_contacts_loaded) {
                // If wildcard-less, then prev-contact arrays are not important
                if (!solverFlags.isHistoryless) {
                    // Note *solverScratchSpace.numContacts is now the num of contact after considering the newly
                    // added ones. Also note, when this method is called, there will be memory allocations, so it has to
                    // be done on kT's device.
                    DEME_GPU_CALL(cudaSetDevice(kT->streamInfo.device));
                    kT->updatePrevContactArrays(granData, *solverScratchSpace.numContacts);
                    DEME_GPU_CALL(cudaSetDevice(streamInfo.device));
                }
                new_contacts_loaded = false;
            }

            // In this `new-boot' case, we send kT a work order, b/c dT needs results from CD to proceed. After this one
            // instance, kT and dT may work in an async fashion.
            {
                // Seed drift attribution for the first received kT update.
                auto& reg = futureDriftRegulator;
                const unsigned int MAX_DRIFT = solverFlags.upperBoundFutureDrift;
                auto clamp_drift = [&](unsigned int v) { return std::min(std::max(1u, v), MAX_DRIFT); };
                const unsigned int cmd = std::max(1u, *perhapsIdealFutureDrift);
                reg.last_sent_proposed = cmd;
                const double de =
                    static_cast<double>(cmd) / static_cast<double>(solverFlags.futureDriftEffDriftSafetyFactor);
                const unsigned int true_target =
                    clamp_drift(static_cast<unsigned int>(std::max(1.0, std::floor(de + 0.5))));
                reg.last_sent_true = true_target;
                reg.last_sent_wait = 0;
                if (reg.last_proposed == 0) {
                    reg.last_proposed = true_target;
                }

                determineSysVel();
                sendToTheirBuffer();
            }
            pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh.store(true, std::memory_order_release);
            contactPairArr_isFresh = true;
            pSchedSupport->schedulingStats.nKinematicUpdates++;
            // Signal the kinematic that it has data for a new work order.
            pSchedSupport->cv_KinematicCanProceed.notify_all();
            // Then dT will wait for kT to finish one initial run
            {
                std::unique_lock<std::mutex> lock(pSchedSupport->dynamicCanProceed);
                while (!pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh.load(std::memory_order_acquire)) {
                    // loop to avoid spurious wakeups
                    pSchedSupport->cv_DynamicCanProceed.wait(lock);
                }
            }

            // We unpack it only when it is a `dry-run', meaning the user just wants to update this system, without
            // doing simulation; it also happens at system initialization. We do this so the kT-supplied contact info is
            // registered on dT.
            if (cycleDuration <= 0.0) {
                ifProduceFreshThenUseIt(true);
            }
        }
        startCycleStopwatch();
        for (double cycle = 0.0; cycle < cycleDuration; cycle += (double)(simParams->dyn.h)) {
            DEME_NVTX_RANGE("dT::cycle");
            throttleInFlightProgress();
            // If the produce is fresh, use it, and then send kT a new work order.
            // We used to send work order to kT whenever kT unpacks its buffer. This can lead to a situation where dT
            // sends a new work order and then immediately bails out (user asks it to do something else). A bit later
            // on, kT will update dT's buffer, and then kT will spot a new work order and work on the new order.
            // However! If kT finishes this new order before dT comes back, the persistent contact wildcard map will be
            // off (across 2 kT updates)! So, dT only send new work orders after kT finishes the old order and it
            // unpacks it.
            ifProduceFreshThenUseItAndSendNewOrder();

            // Check if we need to wait; i.e., if dynamic drifted too much into future, then we must wait a bit before
            // the next cycle begins

            // WAIT is removed for now...

            // If using variable ts size, only when a step is accepted can we move on
            bool step_accepted = false;
            do {
                // Pre-force catch-up: if kT finished during the previous step, unpack before using contacts.
                tryConsumeKinematicProduce(false, true, true);
                calculateForces();

                routineChecks();

                integrateOwnerMotions();

                timers.AccumulateGpuTimer("Clear force array");
                timers.AccumulateGpuTimer("Calculate contact forces");
                timers.AccumulateGpuTimer("Optional force reduction");
                timers.AccumulateGpuTimer("Integration");

                step_accepted = true;
            } while ((!solverFlags.isStepConst) || (!step_accepted));

            // CalculateForces is done, set contactPairArr_isFresh to false
            // This will be set to true next time it receives an update from kT
            contactPairArr_isFresh = false;

            // Late-cycle catch-up: if kT finished during this cycle, unpack now to avoid an extra-cycle delay.
            tryConsumeKinematicProduce(false, true, false);

            /*
            if (cycle == (cycleDuration - 1))
                pSchedSupport->dynamicDone = true;
            */

            // Dynamic wrapped up one cycle, record this fact into schedule support
            pSchedSupport->currentStampOfDynamic++;
            nTotalSteps++;

            //// TODO: make changes for variable time step size cases
            simParams->dyn.timeElapsed += (double)simParams->dyn.h;
            simParams.syncMemberToDeviceAsync<double>(
                offsetof(DEMSimParams, dyn) + offsetof(DEMSimParamsDynamic, timeElapsed), streamInfo.stream);
            DEME_DEBUG_PRINTF("Completed step %zu, time %.9g", nTotalSteps, simParams->dyn.timeElapsed);
        }

        // Unless the user did something critical, must we wait for a kT update before next step
        pendingCriticalUpdate = false;

        // When getting here, dT has finished one user call (although perhaps not at the end of the user script)
        {
            std::lock_guard<std::mutex> lock(pPagerToMain->mainCanProceed);
            pPagerToMain->userCallDone = true;
            pPagerToMain->cv_mainCanProceed.notify_all();
        }
    }
}

void DEMDynamicThread::getTiming(std::vector<std::string>& names, std::vector<double>& vals) {
    names = timer_names;
    for (const auto& name : timer_names) {
        vals.push_back(timers.GetTimer(name).GetTimeSeconds());
    }
}

void DEMDynamicThread::startThread() {
    std::lock_guard<std::mutex> lock(pSchedSupport->dynamicStartLock);
    pSchedSupport->dynamicStarted = true;
    pSchedSupport->cv_DynamicStartLock.notify_one();
}

void DEMDynamicThread::resetUserCallStat() {
    // Reset last kT-side data receiving cycle time stamp.
    pSchedSupport->stampLastDynamicUpdateProdDate = -1;
    pSchedSupport->currentStampOfDynamic = 0;
    pSchedSupport->completedStampOfDynamic = 0;
    progressEventHead = 0;
    progressEventCount = 0;
    // Reset dT stats variables, making ready for next user call
    pSchedSupport->dynamicDone = false;
    contactPairArr_isFresh = true;
    futureDriftRegulator.Clear();
    kT_numContacts_copy_pending = false;
    last_kT_produce_stamp = pSchedSupport->schedulingStats.nDynamicUpdates.load(std::memory_order_acquire);
    recv_stamp_override = -1;

    // Do not let user artificially set dynamicOwned_Prod2ConsBuffer_isFresh false. B/c only dT has the say on that. It
    // could be that kT has a new produce ready, but dT idled for long and do not want to use it and want a new produce.
    // Then dT needs to unpack this one first to get the contact mapping, then issue new work order, and that requires
    // no manually setting this to false.
    // pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = false;
}

size_t DEMDynamicThread::estimateDeviceMemUsage() const {
    return m_approxDeviceBytesUsed;
}

size_t DEMDynamicThread::estimateHostMemUsage() const {
    return m_approxHostBytesUsed;
}

void DEMDynamicThread::jitifyKernels(const std::unordered_map<std::string, std::string>& Subs,
                                     const std::vector<std::string>& JitifyOptions) {
    // Force calculation kernels
    {
        cal_force_kernels = std::make_shared<JitHelper::CachedProgram>(std::move(
            JitHelper::buildProgram("DEMCalcForceKernels_Primitive",
                                    JitHelper::KERNEL_DIR / "DEMCalcForceKernels_Primitive.cu", Subs, JitifyOptions)));
    }
    // Then patch-based force calculation kernels
    {
        cal_patch_force_kernels = std::make_shared<JitHelper::CachedProgram>(std::move(
            JitHelper::buildProgram("DEMCalcForceKernels_PatchBased",
                                    JitHelper::KERNEL_DIR / "DEMCalcForceKernels_PatchBased.cu", Subs, JitifyOptions)));
    }
    // Then force accumulation kernels
    {
        collect_force_kernels = std::make_shared<JitHelper::CachedProgram>(std::move(JitHelper::buildProgram(
            "DEMCollectForceKernels", JitHelper::KERNEL_DIR / "DEMCollectForceKernels.cu", Subs, JitifyOptions)));
    }
    // Then integration kernels
    {
        integrator_kernels = std::make_shared<JitHelper::CachedProgram>(JitHelper::buildProgram(
            "DEMIntegrationKernels", JitHelper::KERNEL_DIR / "DEMIntegrationKernels.cu", Subs, JitifyOptions));
    }
    // Then kernels that make on-the-fly changes to solver data
    if (solverFlags.canFamilyChangeOnDevice) {
        mod_kernels = std::make_shared<JitHelper::CachedProgram>(std::move(JitHelper::buildProgram(
            "DEMModeratorKernels", JitHelper::KERNEL_DIR / "DEMModeratorKernels.cu", Subs, JitifyOptions)));
    }

    // For now, the contact type to kernel map is known and hard-coded after jitification
    contactTypePrimitiveKernelMap[SPHERE_SPHERE_CONTACT] = {
        {cal_force_kernels, "calculatePrimitiveContactForces_SphSph"}};
    contactTypePrimitiveKernelMap[SPHERE_TRIANGLE_CONTACT] = {
        {cal_force_kernels, "calculatePrimitiveContactForces_SphTri"}};
    contactTypePrimitiveKernelMap[SPHERE_ANALYTICAL_CONTACT] = {
        {cal_force_kernels, "calculatePrimitiveContactForces_SphAnal"}};
    contactTypePrimitiveKernelMap[TRIANGLE_TRIANGLE_CONTACT] = {
        {cal_force_kernels, "calculatePrimitiveContactForces_TriTri"}};
    contactTypePrimitiveKernelMap[TRIANGLE_ANALYTICAL_CONTACT] = {
        {cal_force_kernels, "calculatePrimitiveContactForces_TriAnal"}};

    // Patch-based force kernel map for mesh-related contacts
    contactTypePatchKernelMap[SPHERE_TRIANGLE_CONTACT] = {
        {cal_patch_force_kernels, "calculatePatchContactForces_SphTri"}};
    contactTypePatchKernelMap[TRIANGLE_TRIANGLE_CONTACT] = {
        {cal_patch_force_kernels, "calculatePatchContactForces_TriTri"}};
    contactTypePatchKernelMap[TRIANGLE_ANALYTICAL_CONTACT] = {
        {cal_patch_force_kernels, "calculatePatchContactForces_TriAnal"}};
    prewarmKernels();
}

float* DEMDynamicThread::inspectCall(const std::shared_ptr<JitHelper::CachedProgram>& inspection_kernel,
                                     const std::string& kernel_name,
                                     INSPECT_ENTITY_TYPE thing_to_insp,
                                     CUB_REDUCE_FLAVOR reduce_flavor,
                                     bool all_domain,
                                     DualArray<scratch_t>& reduceResArr,
                                     DualArray<scratch_t>& reduceRes,
                                     bool return_device_ptr) {
    size_t n;
    ownerType_t owner_type = 0;
    switch (thing_to_insp) {
        case (INSPECT_ENTITY_TYPE::SPHERE):
            n = simParams->nSpheresGM;
            break;
        case (INSPECT_ENTITY_TYPE::CLUMP):
            n = simParams->nOwnerBodies;
            owner_type = OWNER_T_CLUMP;
            break;
        case (INSPECT_ENTITY_TYPE::EVERYTHING):
            n = simParams->nOwnerBodies;
            owner_type = OWNER_T_CLUMP | OWNER_T_MESH | OWNER_T_ANALYTICAL;
            break;
    }

    // This device set effectively bind the `master' thread, or say the API thread, to the dT device; but it is needed,
    // as the inspector will inspect dT data, most likely.
    DEME_GPU_CALL(cudaSetDevice(streamInfo.device));

    // We can use temp vectors as we please
    size_t quarryTempSize = n * sizeof(float);
    DEME_DUAL_ARRAY_RESIZE_NOVAL(reduceResArr, quarryTempSize);
    float* resArr = (float*)reduceResArr.device();
    size_t regionTempSize = n * sizeof(notStupidBool_t);
    // If this boolArrExclude is 1 at an element, that means this element is exluded in the reduction
    notStupidBool_t* boolArrExclude =
        (notStupidBool_t*)solverScratchSpace.allocateTempVector("boolArrExclude", regionTempSize);
    DEME_GPU_CALL(cudaMemset(boolArrExclude, 0, regionTempSize));

    // We may actually have 2 reduced returns: in regional reduction, key 0 and 1 give one return each.
    size_t returnSize = sizeof(float) * 2;
    DEME_DUAL_ARRAY_RESIZE_NOVAL(reduceRes, returnSize);
    float* res = (float*)reduceRes.device();
    size_t blocks_needed = (n + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    inspection_kernel->kernel(kernel_name)
        .instantiate()
        .configure(dim3(blocks_needed), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
        .launch(&granData, &simParams, resArr, boolArrExclude, n, owner_type);

    if (all_domain) {
        switch (reduce_flavor) {
            case (CUB_REDUCE_FLAVOR::MAX):
                cubMaxReduce<float>(resArr, res, n, streamInfo.stream, solverScratchSpace);
                break;
            case (CUB_REDUCE_FLAVOR::MIN):
                cubMinReduce<float>(resArr, res, n, streamInfo.stream, solverScratchSpace);
                break;
            case (CUB_REDUCE_FLAVOR::SUM):
                cubSumReduce<float, float>(resArr, res, n, streamInfo.stream, solverScratchSpace);
                break;
            case (CUB_REDUCE_FLAVOR::NONE):
                solverScratchSpace.finishUsingTempVector("boolArrExclude");
                if (return_device_ptr) {
                    return (float*)reduceResArr.device();
                } else {
                    reduceResArr.toHost();
                    return (float*)reduceResArr.host();
                }
        }
        // If this inspection is comfined in a region, then boolArrExclude and resArr need to be sorted and reduce by
        // key
    } else {
        // Extra arrays are needed for sort and reduce by key
        notStupidBool_t* boolArrExclude_sorted =
            (notStupidBool_t*)solverScratchSpace.allocateTempVector("boolArrExclude_sorted", regionTempSize);
        float* resArr_sorted = (float*)solverScratchSpace.allocateTempVector("resArr_sorted", quarryTempSize);
        size_t* num_unique_out = (size_t*)solverScratchSpace.allocateTempVector("num_unique_out", sizeof(size_t));
        switch (reduce_flavor) {
            case (CUB_REDUCE_FLAVOR::SUM):
                // Sort first
                cubSortByKey<notStupidBool_t, float>(boolArrExclude, boolArrExclude_sorted, resArr, resArr_sorted, n,
                                                     streamInfo.stream, solverScratchSpace);
                // Then reduce. We care about the sum for 0-marked entries only. Note boolArrExclude here is re-used for
                // storing d_unique_out.
                cubSumReduceByKey<notStupidBool_t, float>(boolArrExclude_sorted, boolArrExclude, resArr_sorted, res,
                                                          num_unique_out, n, streamInfo.stream, solverScratchSpace);
                break;
            case (CUB_REDUCE_FLAVOR::MAX):
                cubSortByKey<notStupidBool_t, float>(boolArrExclude, boolArrExclude_sorted, resArr, resArr_sorted, n,
                                                     streamInfo.stream, solverScratchSpace);
                cubMaxReduceByKey<notStupidBool_t, float>(boolArrExclude_sorted, boolArrExclude, resArr_sorted, res,
                                                          num_unique_out, n, streamInfo.stream, solverScratchSpace);
                break;
            case (CUB_REDUCE_FLAVOR::MIN):
                cubSortByKey<notStupidBool_t, float>(boolArrExclude, boolArrExclude_sorted, resArr, resArr_sorted, n,
                                                     streamInfo.stream, solverScratchSpace);
                cubMinReduceByKey<notStupidBool_t, float>(boolArrExclude_sorted, boolArrExclude, resArr_sorted, res,
                                                          num_unique_out, n, streamInfo.stream, solverScratchSpace);
                break;
            case (CUB_REDUCE_FLAVOR::NONE):
                solverScratchSpace.finishUsingTempVector("boolArrExclude");
                solverScratchSpace.finishUsingTempVector("boolArrExclude_sorted");
                solverScratchSpace.finishUsingTempVector("resArr_sorted");
                solverScratchSpace.finishUsingTempVector("num_unique_out");
                if (return_device_ptr) {
                    return (float*)reduceResArr.device();
                } else {
                    reduceResArr.toHost();
                    return (float*)reduceResArr.host();
                }
        }
    }

    solverScratchSpace.finishUsingTempVector("boolArrExclude");
    solverScratchSpace.finishUsingTempVector("boolArrExclude_sorted");
    solverScratchSpace.finishUsingTempVector("resArr_sorted");
    solverScratchSpace.finishUsingTempVector("num_unique_out");
    if (return_device_ptr) {
        return (float*)reduceRes.device();
    } else {
        reduceRes.toHost();
        return (float*)reduceRes.host();
    }
}

float* DEMDynamicThread::inspectCallDeviceNoReduce(const std::shared_ptr<JitHelper::CachedProgram>& inspection_kernel,
                                                   const std::string& kernel_name,
                                                   INSPECT_ENTITY_TYPE thing_to_insp,
                                                   CUB_REDUCE_FLAVOR reduce_flavor,
                                                   bool all_domain,
                                                   DualArray<scratch_t>& reduceResArr,
                                                   DualArray<scratch_t>& reduceRes) {
    return inspectCall(inspection_kernel, kernel_name, thing_to_insp, reduce_flavor, all_domain, reduceResArr,
                       reduceRes, true);
}

void DEMDynamicThread::initAllocation() {
    DEME_DUAL_ARRAY_RESIZE(familyExtraMarginSize, NUM_AVAL_FAMILIES, 0);
}

void DEMDynamicThread::deallocateEverything() {
    for (unsigned int i = 0; i < contactWildcards.size(); i++) {
        contactWildcards[i].reset();
    }
    for (unsigned int i = 0; i < ownerWildcards.size(); i++) {
        ownerWildcards[i].reset();
    }
    for (unsigned int i = 0; i < sphereWildcards.size(); i++) {
        sphereWildcards[i].reset();
    }
    for (unsigned int i = 0; i < analWildcards.size(); i++) {
        analWildcards[i].reset();
    }
    for (unsigned int i = 0; i < patchWildcards.size(); i++) {
        patchWildcards[i].reset();
    }
}

size_t DEMDynamicThread::getNumContacts() const {
    return *(solverScratchSpace.numContacts);
}

double DEMDynamicThread::getSimTime() const {
    return simParams->dyn.timeElapsed;
}

void DEMDynamicThread::setSimTime(double time) {
    simParams->dyn.timeElapsed = time;
    simParams.syncMemberToDeviceAsync<double>(offsetof(DEMSimParams, dyn) + offsetof(DEMSimParamsDynamic, timeElapsed),
                                              streamInfo.stream);
}

float DEMDynamicThread::getUpdateFreq() const {
    return (float)((pSchedSupport->dynamicMaxFutureDrift).load()) / 2.;
}

void DEMDynamicThread::setFamilyClumpMaterial(unsigned int N, unsigned int mat_id) {
    migrateFamilyToHost();
    for (size_t i = 0; i < simParams->nSpheresGM; i++) {
        bodyID_t owner_id = ownerClumpBody[i];  // No device-side change
        if (+(familyID[owner_id]) == N) {
            sphereMaterialOffset[i] = (materialsOffset_t)mat_id;
        }
    }
    sphereMaterialOffset.toDevice();
}
void DEMDynamicThread::setFamilyMeshMaterial(unsigned int N, unsigned int mat_id) {
    migrateFamilyToHost();
    for (size_t i = 0; i < simParams->nMeshPatches; i++) {
        bodyID_t owner_id = ownerPatchMesh[i];  // No device-side change
        if (+(familyID[owner_id]) == N) {
            patchMaterialOffset[i] = (materialsOffset_t)mat_id;
        }
    }
    patchMaterialOffset.toDevice();
}

size_t DEMDynamicThread::getOwnerContactForces(const std::vector<bodyID_t>& ownerIDs,
                                               std::vector<float3>& points,
                                               std::vector<float3>& forces) {
    // Set the gpu for this thread
    DEME_GPU_CALL(cudaSetDevice(streamInfo.device));
    // Allocate enough space
    size_t numCnt = *solverScratchSpace.numContacts;
    solverScratchSpace.allocateDualArray("points", numCnt * sizeof(float3));
    solverScratchSpace.allocateDualArray("forces", numCnt * sizeof(float3));
    solverScratchSpace.allocateDualArray("ownerIDs", ownerIDs.size() * sizeof(bodyID_t));
    solverScratchSpace.allocateDualStruct("numUsefulCnt");

    const std::vector<bodyID_t> ownerIDs_sorted = hostSort(ownerIDs);
    bodyID_t* h_ownerIDs = (bodyID_t*)solverScratchSpace.getDualArrayHost("ownerIDs");
    for (size_t i = 0; i < ownerIDs_sorted.size(); i++) {
        h_ownerIDs[i] = ownerIDs_sorted[i];
    }
    solverScratchSpace.syncDualArrayHostToDevice("ownerIDs");

    size_t* h_numUsefulCnt = solverScratchSpace.getDualStructHost("numUsefulCnt");
    *h_numUsefulCnt = 0;
    solverScratchSpace.syncDualStructHostToDevice("numUsefulCnt");
    size_t* d_numUsefulCnt = solverScratchSpace.getDualStructDevice("numUsefulCnt");
    bodyID_t* d_ownerIDs = (bodyID_t*)solverScratchSpace.getDualArrayDevice("ownerIDs");
    float3* d_points = (float3*)solverScratchSpace.getDualArrayDevice("points");
    float3* d_forces = (float3*)solverScratchSpace.getDualArrayDevice("forces");

    getContactForcesConcerningOwners(d_points, d_forces, nullptr, d_numUsefulCnt, d_ownerIDs, ownerIDs_sorted.size(),
                                     &simParams, &granData, numCnt, false, false, streamInfo.stream);

    // Bring back to host
    solverScratchSpace.syncDualStructDeviceToHost("numUsefulCnt");
    size_t numUsefulCnt = *h_numUsefulCnt;
    if (numUsefulCnt > 0) {
        solverScratchSpace.syncDualArrayDeviceToHost("points", 0, numUsefulCnt * sizeof(float3));
        solverScratchSpace.syncDualArrayDeviceToHost("forces", 0, numUsefulCnt * sizeof(float3));
    }
    float3* h_points = (float3*)solverScratchSpace.getDualArrayHost("points");
    float3* h_forces = (float3*)solverScratchSpace.getDualArrayHost("forces");
    points.resize(numUsefulCnt);
    forces.resize(numUsefulCnt);
    for (size_t i = 0; i < numUsefulCnt; i++) {
        points[i] = h_points[i];
        forces[i] = h_forces[i];
    }

    solverScratchSpace.finishUsingDualArray("points");
    solverScratchSpace.finishUsingDualArray("forces");
    solverScratchSpace.finishUsingDualArray("ownerIDs");
    solverScratchSpace.finishUsingDualStruct("numUsefulCnt");
    return numUsefulCnt;
}

size_t DEMDynamicThread::getOwnerContactForces(const std::vector<bodyID_t>& ownerIDs,
                                               std::vector<float3>& points,
                                               std::vector<float3>& forces,
                                               std::vector<float3>& torques,
                                               bool torque_in_local) {
    // Set the gpu for this thread
    DEME_GPU_CALL(cudaSetDevice(streamInfo.device));
    // Allocate enough space
    size_t numCnt = *solverScratchSpace.numContacts;
    solverScratchSpace.allocateDualArray("points", numCnt * sizeof(float3));
    solverScratchSpace.allocateDualArray("forces", numCnt * sizeof(float3));
    solverScratchSpace.allocateDualArray("torques", numCnt * sizeof(float3));
    solverScratchSpace.allocateDualArray("ownerIDs", ownerIDs.size() * sizeof(bodyID_t));
    solverScratchSpace.allocateDualStruct("numUsefulCnt");

    const std::vector<bodyID_t> ownerIDs_sorted = hostSort(ownerIDs);
    bodyID_t* h_ownerIDs = (bodyID_t*)solverScratchSpace.getDualArrayHost("ownerIDs");
    for (size_t i = 0; i < ownerIDs_sorted.size(); i++) {
        h_ownerIDs[i] = ownerIDs_sorted[i];
    }
    solverScratchSpace.syncDualArrayHostToDevice("ownerIDs");

    size_t* h_numUsefulCnt = solverScratchSpace.getDualStructHost("numUsefulCnt");
    *h_numUsefulCnt = 0;
    solverScratchSpace.syncDualStructHostToDevice("numUsefulCnt");
    size_t* d_numUsefulCnt = solverScratchSpace.getDualStructDevice("numUsefulCnt");
    bodyID_t* d_ownerIDs = (bodyID_t*)solverScratchSpace.getDualArrayDevice("ownerIDs");
    float3* d_points = (float3*)solverScratchSpace.getDualArrayDevice("points");
    float3* d_forces = (float3*)solverScratchSpace.getDualArrayDevice("forces");
    float3* d_torques = (float3*)solverScratchSpace.getDualArrayDevice("torques");

    getContactForcesConcerningOwners(d_points, d_forces, d_torques, d_numUsefulCnt, d_ownerIDs, ownerIDs_sorted.size(),
                                     &simParams, &granData, numCnt, true, torque_in_local, streamInfo.stream);

    // Bring back to host
    solverScratchSpace.syncDualStructDeviceToHost("numUsefulCnt");
    size_t numUsefulCnt = *h_numUsefulCnt;
    if (numUsefulCnt > 0) {
        solverScratchSpace.syncDualArrayDeviceToHost("points", 0, numUsefulCnt * sizeof(float3));
        solverScratchSpace.syncDualArrayDeviceToHost("forces", 0, numUsefulCnt * sizeof(float3));
        solverScratchSpace.syncDualArrayDeviceToHost("torques", 0, numUsefulCnt * sizeof(float3));
    }
    float3* h_points = (float3*)solverScratchSpace.getDualArrayHost("points");
    float3* h_forces = (float3*)solverScratchSpace.getDualArrayHost("forces");
    float3* h_torques = (float3*)solverScratchSpace.getDualArrayHost("torques");
    points.resize(numUsefulCnt);
    forces.resize(numUsefulCnt);
    torques.resize(numUsefulCnt);
    for (size_t i = 0; i < numUsefulCnt; i++) {
        points[i] = h_points[i];
        forces[i] = h_forces[i];
        torques[i] = h_torques[i];
    }

    solverScratchSpace.finishUsingDualArray("points");
    solverScratchSpace.finishUsingDualArray("forces");
    solverScratchSpace.finishUsingDualArray("torques");
    solverScratchSpace.finishUsingDualArray("ownerIDs");
    solverScratchSpace.finishUsingDualStruct("numUsefulCnt");
    return numUsefulCnt;
}

void DEMDynamicThread::setFamilyContactWildcardValue_impl(
    unsigned int N1,
    unsigned int N2,
    unsigned int wc_num,
    float val,
    const std::function<bool(unsigned int, unsigned int, unsigned int, unsigned int)>& condition) {
    // Get host updated then send all to device
    migrateFamilyToHost();
    contactWildcards[wc_num]->toHost();
    idPatchA.toHost();
    idPatchB.toHost();
    contactTypePatch.toHost();

    size_t numCnt = *solverScratchSpace.numContacts;
    for (size_t i = 0; i < numCnt; i++) {
        contact_t typeContact = contactTypePatch[i];
        bodyID_t geoA = idPatchA[i];
        bodyID_t ownerA = getPatchOwnerID(geoA, decodeTypeA(typeContact));
        bodyID_t geoB = idPatchB[i];
        bodyID_t ownerB = getPatchOwnerID(geoB, decodeTypeB(typeContact));

        unsigned int famA = +(familyID[ownerA]);
        unsigned int famB = +(familyID[ownerB]);

        if (condition(famA, famB, N1, N2)) {
            (*contactWildcards[wc_num])[i] = val;
        }
    }
    contactWildcards[wc_num]->toDevice();
}

void DEMDynamicThread::setFamilyContactWildcardValueEither(unsigned int N, unsigned int wc_num, float val) {
    setFamilyContactWildcardValue_impl(N, /*no use*/ 0, wc_num, val,
                                       [](unsigned int famA, unsigned int famB, unsigned int N1, unsigned int N2) {
                                           return N1 == famA || N1 == famB;
                                       });
}

void DEMDynamicThread::setFamilyContactWildcardValueBoth(unsigned int N, unsigned int wc_num, float val) {
    setFamilyContactWildcardValue_impl(N, /*no use*/ 0, wc_num, val,
                                       [](unsigned int famA, unsigned int famB, unsigned int N1, unsigned int N2) {
                                           return N1 == famA && N1 == famB;
                                       });
}

void DEMDynamicThread::setFamilyContactWildcardValue(unsigned int N1, unsigned int N2, unsigned int wc_num, float val) {
    setFamilyContactWildcardValue_impl(N1, N2, wc_num, val,
                                       [](unsigned int famA, unsigned int famB, unsigned int N1, unsigned int N2) {
                                           return (N1 == famA && N2 == famB) || (N2 == famA && N1 == famB);
                                       });
}

void DEMDynamicThread::setContactWildcardValue(unsigned int wc_num, float val) {
    // Get host updated then send all to device
    contactWildcards[wc_num]->toHost();
    size_t numCnt = *solverScratchSpace.numContacts;
    for (size_t i = 0; i < numCnt; i++) {
        (*contactWildcards[wc_num])[i] = val;
    }
    contactWildcards[wc_num]->toDevice();
}

void DEMDynamicThread::setOwnerWildcardValue(bodyID_t ownerID, unsigned int wc_num, const std::vector<float>& vals) {
    // `set' methods should in general use async-ed flavor, as it only matters when the next kernel is called, which is
    // serial to the memory transaction
    for (size_t i = 0; i < vals.size(); i++) {
        (*ownerWildcards[wc_num])[ownerID + i] = vals.at(i);
    }
    // Partial send to device
    ownerWildcards[wc_num]->toDevice(ownerID, vals.size());
}

void DEMDynamicThread::setPatchWildcardValue(bodyID_t geoID, unsigned int wc_num, const std::vector<float>& vals) {
    for (size_t i = 0; i < vals.size(); i++) {
        (*patchWildcards[wc_num])[geoID + i] = vals.at(i);
    }
    // Partial send to device
    patchWildcards[wc_num]->toDevice(geoID, vals.size());
}

void DEMDynamicThread::setSphWildcardValue(bodyID_t geoID, unsigned int wc_num, const std::vector<float>& vals) {
    for (size_t i = 0; i < vals.size(); i++) {
        (*sphereWildcards[wc_num])[geoID + i] = vals.at(i);
    }
    // Partial send to device
    sphereWildcards[wc_num]->toDevice(geoID, vals.size());
}

void DEMDynamicThread::setAnalWildcardValue(bodyID_t geoID, unsigned int wc_num, const std::vector<float>& vals) {
    for (size_t i = 0; i < vals.size(); i++) {
        (*analWildcards[wc_num])[geoID + i] = vals.at(i);
    }
    // Partial send to device
    analWildcards[wc_num]->toDevice(geoID, vals.size());
}

void DEMDynamicThread::setFamilyOwnerWildcardValue(unsigned int family_num,
                                                   unsigned int wc_num,
                                                   const std::vector<float>& vals) {
    // Get host updated then send all to device
    ownerWildcards[wc_num]->toHost();
    migrateFamilyToHost();
    size_t count = 0;
    for (size_t i = 0; i < simParams->nOwnerBodies; i++) {
        if (+(familyID[i]) == family_num) {
            (*ownerWildcards[wc_num])[i] = vals.at(count);
            if (count + 1 < vals.size()) {
                count++;
            }
        }
    }
    ownerWildcards[wc_num]->toDevice();
}

void DEMDynamicThread::getSphereWildcardValue(std::vector<float>& res, bodyID_t ID, unsigned int wc_num, size_t n) {
    res = std::move(sphereWildcards[wc_num]->getVal(ID, n));
}

void DEMDynamicThread::getPatchWildcardValue(std::vector<float>& res, bodyID_t ID, unsigned int wc_num, size_t n) {
    res = std::move(patchWildcards[wc_num]->getVal(ID, n));
}

void DEMDynamicThread::getAnalWildcardValue(std::vector<float>& res, bodyID_t ID, unsigned int wc_num, size_t n) {
    res = std::move(analWildcards[wc_num]->getVal(ID, n));
}

std::vector<float> DEMDynamicThread::getOwnerWildcardValue(bodyID_t ID, unsigned int wc_num, bodyID_t n) {
    return std::move(ownerWildcards[wc_num]->getVal(ID, n));
}

void DEMDynamicThread::getAllOwnerWildcardValue(std::vector<float>& res, unsigned int wc_num) {
    res = std::move(ownerWildcards[wc_num]->getVal(0, simParams->nOwnerBodies));
}

void DEMDynamicThread::getFamilyOwnerWildcardValue(std::vector<float>& res,
                                                   unsigned int family_num,
                                                   unsigned int wc_num) {
    // Get host updated then extract partial from it
    ownerWildcards[wc_num]->toHost();
    migrateFamilyToHost();
    res.resize(simParams->nOwnerBodies);
    size_t count = 0;
    for (size_t i = 0; i < simParams->nOwnerBodies; i++) {
        if (+(familyID[i]) == family_num) {
            res[count] = (*ownerWildcards[wc_num])[i];
            count++;
        }
    }
    res.resize(count);
}

std::vector<float3> DEMDynamicThread::getOwnerAngVel(bodyID_t ownerID, bodyID_t n) {
    std::vector<float3> angVel(n);
    auto X = omgBarX.getVal(ownerID, n);
    auto Y = omgBarY.getVal(ownerID, n);
    auto Z = omgBarZ.getVal(ownerID, n);
    for (bodyID_t i = 0; i < n; i++) {
        angVel[i] = make_float3(X[i], Y[i], Z[i]);
    }
    return angVel;
}

std::vector<float4> DEMDynamicThread::getOwnerOriQ(bodyID_t ownerID, bodyID_t n) {
    std::vector<float4> oriQ(n);
    auto W = oriQw.getVal(ownerID, n);
    auto X = oriQx.getVal(ownerID, n);
    auto Y = oriQy.getVal(ownerID, n);
    auto Z = oriQz.getVal(ownerID, n);
    for (bodyID_t i = 0; i < n; i++) {
        oriQ[i] = make_float4(X[i], Y[i], Z[i], W[i]);
    }
    return oriQ;
}

std::vector<float3> DEMDynamicThread::getOwnerAcc(bodyID_t ownerID, bodyID_t n) {
    std::vector<float3> acc(n);
    auto X = aX.getVal(ownerID, n);
    auto Y = aY.getVal(ownerID, n);
    auto Z = aZ.getVal(ownerID, n);
    for (bodyID_t i = 0; i < n; i++) {
        acc[i] = make_float3(X[i], Y[i], Z[i]);
    }
    return acc;
}

std::vector<float3> DEMDynamicThread::getOwnerAngAcc(bodyID_t ownerID, bodyID_t n) {
    std::vector<float3> aa(n);
    auto X = alphaX.getVal(ownerID, n);
    auto Y = alphaY.getVal(ownerID, n);
    auto Z = alphaZ.getVal(ownerID, n);
    for (bodyID_t i = 0; i < n; i++) {
        aa[i] = make_float3(X[i], Y[i], Z[i]);
    }
    return aa;
}

std::vector<float3> DEMDynamicThread::getOwnerVel(bodyID_t ownerID, bodyID_t n) {
    std::vector<float3> vel(n);
    auto X = vX.getVal(ownerID, n);
    auto Y = vY.getVal(ownerID, n);
    auto Z = vZ.getVal(ownerID, n);
    for (bodyID_t i = 0; i < n; i++) {
        vel[i] = make_float3(X[i], Y[i], Z[i]);
    }
    return vel;
}

std::vector<float3> DEMDynamicThread::getOwnerPos(bodyID_t ownerID, bodyID_t n) {
    std::vector<float3> pos(n);
    std::vector<voxelID_t> voxel = voxelID.getVal(ownerID, n);
    std::vector<subVoxelPos_t> subVoxX = locX.getVal(ownerID, n);
    std::vector<subVoxelPos_t> subVoxY = locY.getVal(ownerID, n);
    std::vector<subVoxelPos_t> subVoxZ = locZ.getVal(ownerID, n);
    for (bodyID_t i = 0; i < n; i++) {
        double X, Y, Z;
        voxelIDToPosition<double, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel[i], subVoxX[i], subVoxY[i], subVoxZ[i],
                                                            simParams->nvXp2, simParams->nvYp2, simParams->voxelSize,
                                                            simParams->l);
        pos[i] = make_float3(X + simParams->LBFX, Y + simParams->LBFY, Z + simParams->LBFZ);
    }
    return pos;
}

std::vector<unsigned int> DEMDynamicThread::getOwnerFamily(bodyID_t ownerID, bodyID_t n) {
    std::vector<unsigned int> fam(n);
    // Get from device by default, even not needed
    auto short_fam = familyID.getVal(ownerID, n);
    for (bodyID_t i = 0; i < n; i++) {
        fam[i] = (unsigned int)(+(short_fam[i]));
    }
    return fam;
}

void DEMDynamicThread::setOwnerAngVel(bodyID_t ownerID, const std::vector<float3>& angVel) {
    omgBarX.setVal(streamInfo.stream, RealTupleVectorToXComponentVector<float, float3>(angVel), ownerID);
    omgBarY.setVal(streamInfo.stream, RealTupleVectorToYComponentVector<float, float3>(angVel), ownerID);
    omgBarZ.setVal(streamInfo.stream, RealTupleVectorToZComponentVector<float, float3>(angVel), ownerID);
    syncMemoryTransfer();
}

void DEMDynamicThread::setOwnerPos(bodyID_t ownerID, const std::vector<float3>& pos) {
    std::vector<voxelID_t> vID(pos.size());
    std::vector<subVoxelPos_t> subIDx(pos.size()), subIDy(pos.size()), subIDz(pos.size());

    for (size_t i = 0; i < pos.size(); i++) {
        // Convert to relative pos wrt LBF point first
        double X = pos[i].x - simParams->LBFX;
        double Y = pos[i].y - simParams->LBFY;
        double Z = pos[i].z - simParams->LBFZ;
        positionToVoxelID<voxelID_t, subVoxelPos_t, double>(vID[i], subIDx[i], subIDy[i], subIDz[i], X, Y, Z,
                                                            simParams->nvXp2, simParams->nvYp2, simParams->voxelSize,
                                                            simParams->l);
    }

    voxelID.setVal(streamInfo.stream, vID, ownerID);
    locX.setVal(streamInfo.stream, subIDx, ownerID);
    locY.setVal(streamInfo.stream, subIDy, ownerID);
    locZ.setVal(streamInfo.stream, subIDz, ownerID);
    syncMemoryTransfer();
}

void DEMDynamicThread::setOwnerOriQ(bodyID_t ownerID, const std::vector<float4>& oriQ) {
    oriQw.setVal(streamInfo.stream, RealTupleVectorToWComponentVector<float, float4>(oriQ), ownerID);
    oriQx.setVal(streamInfo.stream, RealTupleVectorToXComponentVector<float, float4>(oriQ), ownerID);
    oriQy.setVal(streamInfo.stream, RealTupleVectorToYComponentVector<float, float4>(oriQ), ownerID);
    oriQz.setVal(streamInfo.stream, RealTupleVectorToZComponentVector<float, float4>(oriQ), ownerID);
    syncMemoryTransfer();
}

void DEMDynamicThread::setOwnerVel(bodyID_t ownerID, const std::vector<float3>& vel) {
    vX.setVal(streamInfo.stream, RealTupleVectorToXComponentVector<float, float3>(vel), ownerID);
    vY.setVal(streamInfo.stream, RealTupleVectorToYComponentVector<float, float3>(vel), ownerID);
    vZ.setVal(streamInfo.stream, RealTupleVectorToZComponentVector<float, float3>(vel), ownerID);
    syncMemoryTransfer();
}

void DEMDynamicThread::setOwnerFamily(bodyID_t ownerID, family_t fam, bodyID_t n) {
    familyID.setVal(std::vector<family_t>(n, fam), ownerID);
}

void DEMDynamicThread::setTriNodeRelPos(size_t start, const std::vector<DEMTriangle>& triangles) {
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
void DEMDynamicThread::updateTriNodeRelPos(size_t start, const std::vector<DEMTriangle>& updates) {
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

void DEMDynamicThread::addOwnerNextStepAcc(bodyID_t ownerID, const std::vector<float3>& acc) {
    accSpecified.setVal(streamInfo.stream, std::vector<notStupidBool_t>(acc.size(), 1), ownerID);
    aX.setVal(streamInfo.stream, RealTupleVectorToXComponentVector<float, float3>(acc), ownerID);
    aY.setVal(streamInfo.stream, RealTupleVectorToYComponentVector<float, float3>(acc), ownerID);
    aZ.setVal(streamInfo.stream, RealTupleVectorToZComponentVector<float, float3>(acc), ownerID);
    syncMemoryTransfer();
}

void DEMDynamicThread::addOwnerNextStepAngAcc(bodyID_t ownerID, const std::vector<float3>& angAcc) {
    angAccSpecified.setVal(streamInfo.stream, std::vector<notStupidBool_t>(angAcc.size(), 1), ownerID);
    alphaX.setVal(streamInfo.stream, RealTupleVectorToXComponentVector<float, float3>(angAcc), ownerID);
    alphaY.setVal(streamInfo.stream, RealTupleVectorToYComponentVector<float, float3>(angAcc), ownerID);
    alphaZ.setVal(streamInfo.stream, RealTupleVectorToZComponentVector<float, float3>(angAcc), ownerID);
    syncMemoryTransfer();
}

void DEMDynamicThread::prewarmKernels() {
    if (cal_force_kernels) {
        cal_force_kernels->kernel("calculatePrimitiveContactForces_SphSph").instantiate();
        cal_force_kernels->kernel("calculatePrimitiveContactForces_SphTri").instantiate();
        cal_force_kernels->kernel("calculatePrimitiveContactForces_SphAnal").instantiate();
        cal_force_kernels->kernel("calculatePrimitiveContactForces_TriTri").instantiate();
        cal_force_kernels->kernel("calculatePrimitiveContactForces_TriAnal").instantiate();
    }
    if (cal_patch_force_kernels) {
        cal_patch_force_kernels->kernel("calculatePatchContactForces_SphTri").instantiate();
        cal_patch_force_kernels->kernel("calculatePatchContactForces_TriTri").instantiate();
        cal_patch_force_kernels->kernel("calculatePatchContactForces_TriAnal").instantiate();
    }
    if (collect_force_kernels) {
        collect_force_kernels->kernel("forceToAcc").instantiate();
    }
    if (integrator_kernels) {
        integrator_kernels->kernel("integrateOwners").instantiate();
    }
    if (mod_kernels && solverFlags.canFamilyChangeOnDevice) {
        mod_kernels->kernel("applyFamilyChanges").instantiate();
    }
}

}  // namespace deme
