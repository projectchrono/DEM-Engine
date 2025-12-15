//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <cstring>
#include <iostream>
#include <thread>
#include <algorithm>

#ifdef DEME_USE_CHPF
    #include <chpf.hpp>
#endif
#include <core/ApiVersion.h>
#include <core/utils/JitHelper.h>
#include <DEM/dT.h>
#include <DEM/kT.h>
#include <DEM/HostSideHelpers.hpp>
#include <kernel/DEMHelperKernels.cuh>
#include <DEM/Defines.h>

#include <algorithms/DEMStaticDeviceSubroutines.h>
#include <kernel/DEMHelperKernels.cuh>

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
    idGeometryA.bindDevicePointer(&(granData->idGeometryA));
    idGeometryB.bindDevicePointer(&(granData->idGeometryB));
    contactType.bindDevicePointer(&(granData->contactType));
    familyMaskMatrix.bindDevicePointer(&(granData->familyMasks));
    familyExtraMarginSize.bindDevicePointer(&(granData->familyExtraMarginSize));

    contactForces.bindDevicePointer(&(granData->contactForces));
    contactTorque_convToForce.bindDevicePointer(&(granData->contactTorque_convToForce));
    contactPointGeometryA.bindDevicePointer(&(granData->contactPointGeometryA));
    contactPointGeometryB.bindDevicePointer(&(granData->contactPointGeometryB));
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
        triWildcards[i]->bindDevicePointer(&(granData->triWildcards[i]));
    }

    // The offset info that indexes into the template arrays
    ownerClumpBody.bindDevicePointer(&(granData->ownerClumpBody));
    clumpComponentOffset.bindDevicePointer(&(granData->clumpComponentOffset));
    clumpComponentOffsetExt.bindDevicePointer(&(granData->clumpComponentOffsetExt));
    sphereMaterialOffset.bindDevicePointer(&(granData->sphereMaterialOffset));
    volumeOwnerBody.bindDevicePointer(&(granData->volumeOwnerBody));

    // Mesh-related
    ownerMesh.bindDevicePointer(&(granData->ownerMesh));
    ownerAnalBody.bindDevicePointer(&(granData->ownerAnalBody));
    relPosNode1.bindDevicePointer(&(granData->relPosNode1));
    relPosNode2.bindDevicePointer(&(granData->relPosNode2));
    relPosNode3.bindDevicePointer(&(granData->relPosNode3));
    triMaterialOffset.bindDevicePointer(&(granData->triMaterialOffset));

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
    idGeometryA.toDeviceAsync(streamInfo.stream);
    idGeometryB.toDeviceAsync(streamInfo.stream);
    contactType.toDeviceAsync(streamInfo.stream);
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
        triWildcards[i]->toDeviceAsync(streamInfo.stream);
    }

    ownerClumpBody.toDeviceAsync(streamInfo.stream);
    clumpComponentOffset.toDeviceAsync(streamInfo.stream);
    clumpComponentOffsetExt.toDeviceAsync(streamInfo.stream);
    sphereMaterialOffset.toDeviceAsync(streamInfo.stream);
    volumeOwnerBody.toDeviceAsync(streamInfo.stream);

    ownerMesh.toDeviceAsync(streamInfo.stream);
    ownerAnalBody.toDeviceAsync(streamInfo.stream);
    relPosNode1.toDeviceAsync(streamInfo.stream);
    relPosNode2.toDeviceAsync(streamInfo.stream);
    relPosNode3.toDeviceAsync(streamInfo.stream);
    triMaterialOffset.toDeviceAsync(streamInfo.stream);

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
    migrateTriGeoWildcardToHost();
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
    idGeometryA.toHost();
    idGeometryB.toHost();
    contactType.toHost();
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
void DEMDynamicThread::migrateTriGeoWildcardToHost() {
    for (unsigned int i = 0; i < simParams->nGeoWildcards; i++) {
        triWildcards[i]->toHost();
    }
}
void DEMDynamicThread::migrateAnalGeoWildcardToHost() {
    for (unsigned int i = 0; i < simParams->nGeoWildcards; i++) {
        analWildcards[i]->toHost();
    }
}

bodyID_t DEMDynamicThread::getGeoOwnerID(const bodyID_t& geoB, const contact_t& type) const {
    // These arrays can't change on device
    switch (type) {
        case (NOT_A_CONTACT):
            return NULL_BODYID;
        case (SPHERE_SPHERE_CONTACT):
            return ownerClumpBody[geoB];
        case (SPHERE_MESH_CONTACT):
            return ownerMesh[geoB];
        default:  // Default is sphere--analytical
            return ownerAnalBody[geoB];
    }
}

// packTransferPointers
void DEMDynamicThread::packTransferPointers(DEMKinematicThread*& kT) {
    // These are the pointers for sending data to dT
    granData->pKTOwnedBuffer_absVel = kT->absVel_buffer.data();
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
        .instantiate("deme::DEMDataDT")
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

void DEMDynamicThread::allocateGPUArrays(size_t nOwnerBodies,
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
    // dT buffer arrays should be on dT and this is to ensure that
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
    DEME_DUAL_ARRAY_RESIZE(ownerMesh, nTriGM, 0);
    DEME_DUAL_ARRAY_RESIZE(relPosNode1, nTriGM, make_float3(0));
    DEME_DUAL_ARRAY_RESIZE(relPosNode2, nTriGM, make_float3(0));
    DEME_DUAL_ARRAY_RESIZE(relPosNode3, nTriGM, make_float3(0));
    DEME_DUAL_ARRAY_RESIZE(triMaterialOffset, nTriGM, 0);

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
    // The lengths of contact event-based arrays are just estimates. My estimate of total contact pairs is ~ 2n, and I
    // think the max is 6n (although I can't prove it). Note the estimate should be large enough to decrease the number
    // of reallocations in the simulation, but not too large that eats too much memory.
    {
        // In any case, in this initialization process we should not make contact arrays smaller than it used to be, or
        // we may lose data. Also, if this is a new-boot, we allocate this array for at least
        // nSpheresGM*DEME_INIT_CNT_MULTIPLIER elements.
        size_t cnt_arr_size =
            DEME_MAX(*solverScratchSpace.numContacts + nExtraContacts, nSpheresGM * DEME_INIT_CNT_MULTIPLIER);
        DEME_DUAL_ARRAY_RESIZE(idGeometryA, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(idGeometryB, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(contactType, cnt_arr_size, NOT_A_CONTACT);

        if (!solverFlags.useNoContactRecord) {
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
        triWildcards.resize(simParams->nGeoWildcards);
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
            triWildcards[i] =
                std::make_unique<DualArray<float>>(nTriGM, 0, &m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
        }
    }

    // You know what, let's not init dT buffers, since kT will change it when needed anyway. Besides, changing it here
    // will cause problems in the case of a re-init-ed simulation with more clumps added to system, since we may
    // accidentally clamp those arrays.
    /*
    buffer_size = DEME_MAX(buffer_size, nSpheresGM * DEME_INIT_CNT_MULTIPLIER);
    DEME_DEVICE_ARRAY_RESIZE(idGeometryA_buffer, buffer_size);
    DEME_DEVICE_ARRAY_RESIZE(idGeometryB_buffer, buffer_size);
    DEME_DEVICE_ARRAY_RESIZE(contactType_buffer, buffer_size);
    DEME_DEVICE_ARRAY_RESIZE(contactMapping_buffer, buffer_size);
    */
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
                                            const std::vector<std::shared_ptr<DEMMeshConnected>>& input_mesh_objs,
                                            const std::vector<float3>& input_mesh_obj_xyz,
                                            const std::vector<float4>& input_mesh_obj_rot,
                                            const std::vector<unsigned int>& input_mesh_obj_family,
                                            const std::vector<unsigned int>& mesh_facet_owner,
                                            const std::vector<materialsOffset_t>& mesh_facet_materials,
                                            const std::vector<DEMTriangle>& mesh_facets,
                                            const ClumpTemplateFlatten& clump_templates,
                                            const std::vector<float>& ext_obj_mass_types,
                                            const std::vector<float3>& ext_obj_moi_types,
                                            const std::vector<unsigned int>& ext_obj_comp_num,
                                            const std::vector<float>& mesh_obj_mass_types,
                                            const std::vector<float3>& mesh_obj_moi_types,
                                            size_t nExistOwners,
                                            size_t nExistSpheres,
                                            size_t nExistingFacets) {
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
                idGeometryA[cnt_arr_offset] = idPair.first + n_processed_sp_comp + nExistSpheres;
                idGeometryB[cnt_arr_offset] = idPair.second + n_processed_sp_comp + nExistSpheres;
                contactType[cnt_arr_offset] = SPHERE_SPHERE_CONTACT;  // Only sph--sph cnt for now
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

        //// TODO: and initial vel?

        family_t this_family_num = input_ext_obj_family.at(i);
        familyID[i + owner_offset_for_ext_obj] = this_family_num;
    }

    // Load in initial positions and mass properties for the owners of the meshed objects
    // They go after analytical object owners
    size_t owner_offset_for_mesh_obj = owner_offset_for_ext_obj + input_ext_obj_xyz.size();
    unsigned int offset_for_mesh_obj_mass_template = offset_for_ext_obj_mass_template + input_ext_obj_xyz.size();
    // k for indexing the triangle facets
    k = 0;
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
                    for (size_t jj = 0; jj < input_mesh_objs.at(i)->GetNumTriangles(); jj++) {
                        (*triWildcards[w_num])[nExistingFacets + k + jj] =
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

        inertiaPropOffsets[i + owner_offset_for_mesh_obj] = i + offset_for_mesh_obj_mass_template;
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

        //// TODO: and initial vel?

        // Per-facet info
        //// TODO: This flatten-then-init approach is historical and too ugly.
        size_t this_facet_owner = mesh_facet_owner.at(k);
        for (; k < mesh_facet_owner.size(); k++) {
            // mesh_facet_owner run length is the num of facets in this mesh entity
            if (mesh_facet_owner.at(k) != this_facet_owner)
                break;
            ownerMesh[nExistingFacets + k] = owner_offset_for_mesh_obj + this_facet_owner;
            triMaterialOffset[nExistingFacets + k] = mesh_facet_materials.at(k);
            DEMTriangle this_tri = mesh_facets.at(k);
            relPosNode1[nExistingFacets + k] = this_tri.p1;
            relPosNode2[nExistingFacets + k] = this_tri.p2;
            relPosNode3[nExistingFacets + k] = this_tri.p3;
        }

        family_t this_family_num = input_mesh_obj_family.at(i);
        familyID[i + owner_offset_for_mesh_obj] = this_family_num;

        // To save some mem
        m_meshes.back()->ClearWildcards();

        DEME_DEBUG_PRINTF("dT just loaded a mesh in family %u", +(this_family_num));
        DEME_DEBUG_PRINTF("This mesh is owner %zu", (i + owner_offset_for_mesh_obj));
        DEME_DEBUG_PRINTF("Number of triangle facets loaded thus far: %zu", k);
    }
}

void DEMDynamicThread::buildTrackedObjs(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                        const std::vector<unsigned int>& ext_obj_comp_num,
                                        const std::vector<std::shared_ptr<DEMMeshConnected>>& input_mesh_objs,
                                        std::vector<std::shared_ptr<DEMTrackedObj>>& tracked_objs,
                                        size_t nExistOwners,
                                        size_t nExistSpheres,
                                        size_t nExistingFacets,
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
    // Also take notes of num of facets of each mesh obj
    std::vector<size_t> prescans_mesh_size;
    prescans_mesh_size.push_back(0);
    for (const auto& a_mesh : input_mesh_objs) {
        prescans_mesh_size.push_back(prescans_mesh_size.back() + a_mesh->GetNumTriangles());
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
                tracked_obj->geoID = nExistingFacets + prescans_mesh_size.at(tracked_obj->load_order);
                tracked_obj->nGeos =
                    prescans_mesh_size.at(tracked_obj->load_order + 1) - prescans_mesh_size.at(tracked_obj->load_order);
                break;
            default:
                DEME_ERROR("A DEM tracked object has an unknown type.");
        }
    }
    nTrackersProcessed = tracked_objs.size();
    DEME_DEBUG_PRINTF("Total number of trackers on the record: %u", nTrackersProcessed);
}

void DEMDynamicThread::initGPUArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                     const std::vector<float3>& input_ext_obj_xyz,
                                     const std::vector<float4>& input_ext_obj_rot,
                                     const std::vector<unsigned int>& input_ext_obj_family,
                                     const std::vector<std::shared_ptr<DEMMeshConnected>>& input_mesh_objs,
                                     const std::vector<float3>& input_mesh_obj_xyz,
                                     const std::vector<float4>& input_mesh_obj_rot,
                                     const std::vector<unsigned int>& input_mesh_obj_family,
                                     const std::vector<unsigned int>& mesh_facet_owner,
                                     const std::vector<materialsOffset_t>& mesh_facet_materials,
                                     const std::vector<DEMTriangle>& mesh_facets,
                                     const std::unordered_map<unsigned int, std::string>& template_number_name_map,
                                     const ClumpTemplateFlatten& clump_templates,
                                     const std::vector<float>& ext_obj_mass_types,
                                     const std::vector<float3>& ext_obj_moi_types,
                                     const std::vector<unsigned int>& ext_obj_comp_num,
                                     const std::vector<float>& mesh_obj_mass_types,
                                     const std::vector<float3>& mesh_obj_moi_types,
                                     const std::vector<std::shared_ptr<DEMMaterial>>& loaded_materials,
                                     const std::vector<notStupidBool_t>& family_mask_matrix,
                                     const std::set<unsigned int>& no_output_families,
                                     std::vector<std::shared_ptr<DEMTrackedObj>>& tracked_objs) {
    // Get the info into the GPU memory from the host side. Can this process be more efficient? Maybe, but it's
    // initialization anyway.

    registerPolicies(template_number_name_map, clump_templates, ext_obj_mass_types, ext_obj_moi_types,
                     mesh_obj_mass_types, mesh_obj_moi_types, loaded_materials, family_mask_matrix, no_output_families);

    // For initialization, owner array offset is 0
    populateEntityArrays(input_clump_batches, input_ext_obj_xyz, input_ext_obj_rot, input_ext_obj_family,
                         input_mesh_objs, input_mesh_obj_xyz, input_mesh_obj_rot, input_mesh_obj_family,
                         mesh_facet_owner, mesh_facet_materials, mesh_facets, clump_templates, ext_obj_mass_types,
                         ext_obj_moi_types, ext_obj_comp_num, mesh_obj_mass_types, mesh_obj_moi_types, 0, 0, 0);

    buildTrackedObjs(input_clump_batches, ext_obj_comp_num, input_mesh_objs, tracked_objs, 0, 0, 0, 0);
}

void DEMDynamicThread::updateClumpMeshArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                             const std::vector<float3>& input_ext_obj_xyz,
                                             const std::vector<float4>& input_ext_obj_rot,
                                             const std::vector<unsigned int>& input_ext_obj_family,
                                             const std::vector<std::shared_ptr<DEMMeshConnected>>& input_mesh_objs,
                                             const std::vector<float3>& input_mesh_obj_xyz,
                                             const std::vector<float4>& input_mesh_obj_rot,
                                             const std::vector<unsigned int>& input_mesh_obj_family,
                                             const std::vector<unsigned int>& mesh_facet_owner,
                                             const std::vector<materialsOffset_t>& mesh_facet_materials,
                                             const std::vector<DEMTriangle>& mesh_facets,
                                             const ClumpTemplateFlatten& clump_templates,
                                             const std::vector<float>& ext_obj_mass_types,
                                             const std::vector<float3>& ext_obj_moi_types,
                                             const std::vector<unsigned int>& ext_obj_comp_num,
                                             const std::vector<float>& mesh_obj_mass_types,
                                             const std::vector<float3>& mesh_obj_moi_types,
                                             const std::vector<std::shared_ptr<DEMMaterial>>& loaded_materials,
                                             const std::vector<notStupidBool_t>& family_mask_matrix,
                                             const std::set<unsigned int>& no_output_families,
                                             std::vector<std::shared_ptr<DEMTrackedObj>>& tracked_objs,
                                             size_t nExistingOwners,
                                             size_t nExistingClumps,
                                             size_t nExistingSpheres,
                                             size_t nExistingTriMesh,
                                             size_t nExistingFacets,
                                             unsigned int nExistingObj,
                                             unsigned int nExistingAnalGM) {
    // No policy changes here

    // Analytical objects-related arrays should be empty
    populateEntityArrays(input_clump_batches, input_ext_obj_xyz, input_ext_obj_rot, input_ext_obj_family,
                         input_mesh_objs, input_mesh_obj_xyz, input_mesh_obj_rot, input_mesh_obj_family,
                         mesh_facet_owner, mesh_facet_materials, mesh_facets, clump_templates, ext_obj_mass_types,
                         ext_obj_moi_types, ext_obj_comp_num, mesh_obj_mass_types, mesh_obj_moi_types, nExistingOwners,
                         nExistingSpheres, nExistingFacets);

    // Make changes to tracked objects (potentially add more)
    buildTrackedObjs(input_clump_batches, ext_obj_comp_num, input_mesh_objs, tracked_objs, nExistingOwners,
                     nExistingSpheres, nExistingFacets, nExistingAnalGM);
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
        auto geoA = idGeometryA[i];
        auto geoB = idGeometryB[i];
        auto type = contactType[i];
        // We don't output fake contacts; but right now, no contact will be marked fake by kT, so no need to check that
        // if (type == NOT_A_CONTACT)
        //     continue;

        float3 forcexyz = contactForces[i];
        float3 torque = contactTorque_convToForce[i];
        // If this force+torque is too small, then it's not an active contact
        if (length(forcexyz + torque) < force_thres) {
            continue;
        }

        // geoA's owner must be a sphere
        auto ownerA = ownerClumpBody[geoA];
        bodyID_t ownerB;
        // geoB's owner depends...
        ownerB = getGeoOwnerID(geoB, type);

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

inline void DEMDynamicThread::contactEventArraysResize(size_t nContactPairs) {
    DEME_DUAL_ARRAY_RESIZE(idGeometryA, nContactPairs, 0);
    DEME_DUAL_ARRAY_RESIZE(idGeometryB, nContactPairs, 0);
    DEME_DUAL_ARRAY_RESIZE(contactType, nContactPairs, NOT_A_CONTACT);

    if (!solverFlags.useNoContactRecord) {
        DEME_DUAL_ARRAY_RESIZE(contactForces, nContactPairs, make_float3(0));
        DEME_DUAL_ARRAY_RESIZE(contactTorque_convToForce, nContactPairs, make_float3(0));
        DEME_DUAL_ARRAY_RESIZE(contactPointGeometryA, nContactPairs, make_float3(0));
        DEME_DUAL_ARRAY_RESIZE(contactPointGeometryB, nContactPairs, make_float3(0));
    }

    // Re-packing pointers now is automatic

    // Sync pointers to device can be delayed... we'll only need to do that before kernel calls
}

inline void DEMDynamicThread::unpackMyBuffer() {
    // Make a note on the contact number of the previous time step
    *solverScratchSpace.numPrevContacts = *solverScratchSpace.numContacts;
    // kT's batch of produce is made with this max drift in mind
    pSchedSupport->dynamicMaxFutureDrift = (pSchedSupport->kinematicMaxFutureDrift).load();
    // DEME_DEBUG_PRINTF("dynamicMaxFutureDrift is %u", (pSchedSupport->dynamicMaxFutureDrift).load());

    DEME_GPU_CALL(
        cudaMemcpy(&(solverScratchSpace.numContacts), &nContactPairs_buffer, sizeof(size_t), cudaMemcpyDeviceToDevice));
    solverScratchSpace.numContacts.toHost();
    // Need to resize those contact event-based arrays before usage
    if (*solverScratchSpace.numContacts > idGeometryA.size() || *solverScratchSpace.numContacts > buffer_size) {
        contactEventArraysResize(*solverScratchSpace.numContacts);
    }

    DEME_GPU_CALL(cudaMemcpy(granData->idGeometryA, idGeometryA_buffer.data(),
                             *solverScratchSpace.numContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->idGeometryB, idGeometryB_buffer.data(),
                             *solverScratchSpace.numContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->contactType, contactType_buffer.data(),
                             *solverScratchSpace.numContacts * sizeof(contact_t), cudaMemcpyDeviceToDevice));
    if (!solverFlags.isHistoryless) {
        // Note we don't have to use dedicated memory space for unpacking contactMapping_buffer contents, because we
        // only use it once per kT update, at the time of unpacking. So let us just use a temp vector to store it.
        size_t mapping_bytes = (*solverScratchSpace.numContacts) * sizeof(contactPairs_t);
        granData->contactMapping =
            (contactPairs_t*)solverScratchSpace.allocateTempVector("contactMapping", mapping_bytes);
        DEME_GPU_CALL(cudaMemcpy(granData->contactMapping, contactMapping_buffer.data(), mapping_bytes,
                                 cudaMemcpyDeviceToDevice));
    }
    // Prepare for kernel calls immediately after
    granData.toDevice();
}

inline void DEMDynamicThread::sendToTheirBuffer() {
    DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_voxelID, granData->voxelID,
                             simParams->nOwnerBodies * sizeof(voxelID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_locX, granData->locX,
                             simParams->nOwnerBodies * sizeof(subVoxelPos_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_locY, granData->locY,
                             simParams->nOwnerBodies * sizeof(subVoxelPos_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_locZ, granData->locZ,
                             simParams->nOwnerBodies * sizeof(subVoxelPos_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_oriQ0, granData->oriQw, simParams->nOwnerBodies * sizeof(oriQ_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_oriQ1, granData->oriQx, simParams->nOwnerBodies * sizeof(oriQ_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_oriQ2, granData->oriQy, simParams->nOwnerBodies * sizeof(oriQ_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_oriQ3, granData->oriQz, simParams->nOwnerBodies * sizeof(oriQ_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_absVel, pCycleVel, simParams->nOwnerBodies * sizeof(float),
                             cudaMemcpyDeviceToDevice));

    // Send simulation metrics for kT's reference.
    DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_ts, &(simParams->h), sizeof(double), cudaMemcpyHostToDevice));
    // Note that perhapsIdealFutureDrift is non-negative, and it will be used to determine the margin size; however, if
    // scheduleHelper is instructed to have negative future drift then perhapsIdealFutureDrift no longer affects them.
    DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_maxDrift, perhapsIdealFutureDrift.getHostPointer(),
                             sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Family number is a typical changable quantity on-the-fly. If this flag is on, dT is responsible for sending this
    // info to kT.
    if (solverFlags.canFamilyChangeOnDevice) {
        DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_familyID, granData->familyID,
                                 simParams->nOwnerBodies * sizeof(family_t), cudaMemcpyDeviceToDevice));
    }

    // May need to send updated mesh
    if (solverFlags.willMeshDeform) {
        DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_relPosNode1, granData->relPosNode1,
                                 simParams->nTriGM * sizeof(float3), cudaMemcpyDeviceToDevice));
        DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_relPosNode2, granData->relPosNode2,
                                 simParams->nTriGM * sizeof(float3), cudaMemcpyDeviceToDevice));
        DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_relPosNode3, granData->relPosNode3,
                                 simParams->nTriGM * sizeof(float3), cudaMemcpyDeviceToDevice));
        solverFlags.willMeshDeform = false;
        // kT can't be loading buffer when dT is sending, so it is safe
        kT->solverFlags.willMeshDeform = true;
    }

    // This subroutine also includes recording the time stamp of this batch ingredient dT sent to kT
    pSchedSupport->kinematicIngredProdDateStamp = (pSchedSupport->currentStampOfDynamic).load();
}

inline void DEMDynamicThread::migrateEnduringContacts() {
    // Use granData->contactMapping's information (stored in temp device vector) to map old and new contacts

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
    size_t blocks_needed_for_rearrange;
    if (verbosity >= VERBOSITY::STEP_METRIC) {
        if (*solverScratchSpace.numPrevContacts > 0) {
            // DEME_GPU_CALL(cudaMemset(contactSentry, 0, sentry_bytes));
            blocks_needed_for_rearrange =
                (*solverScratchSpace.numPrevContacts + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
            if (blocks_needed_for_rearrange > 0) {
                prep_force_kernels->kernel("markAliveContacts")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_rearrange), dim3(DEME_MAX_THREADS_PER_BLOCK), 0,
                               streamInfo.stream)
                    .launch(granData->contactWildcards[simParams->nContactWildcards - 1], contactSentry,
                            *solverScratchSpace.numPrevContacts);
                DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
            }
        }
    }

    // Rearrange contact histories based on kT instruction
    blocks_needed_for_rearrange =
        (*solverScratchSpace.numContacts + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed_for_rearrange > 0) {
        prep_force_kernels->kernel("rearrangeContactWildcards")
            .instantiate()
            .configure(dim3(blocks_needed_for_rearrange), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
            .launch(&granData, newWildcards[0], contactSentry, simParams->nContactWildcards,
                    *solverScratchSpace.numContacts);
        DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    }

    // Take a look, does the sentry indicate that there is an `alive' contact got lost?
    if (verbosity >= VERBOSITY::STEP_METRIC) {
        if (*solverScratchSpace.numPrevContacts > 0 && simParams->nContactWildcards > 0) {
            // Temp DualStruct defaults to size_t type
            solverScratchSpace.allocateDualStruct("lostContact");
            size_t* lostContact = solverScratchSpace.getDualStructDevice("lostContact");
            cubSumReduce<notStupidBool_t, size_t>(contactSentry, lostContact, *solverScratchSpace.numPrevContacts,
                                                  streamInfo.stream, solverScratchSpace);
            solverScratchSpace.syncDualStructDeviceToHost("lostContact");
            lostContact = solverScratchSpace.getDualStructHost("lostContact");
            if (*lostContact && solverFlags.isAsync) {
                DEME_STEP_METRIC(
                    "%zu contacts were active at time %.9g on dT, but they are not detected on kT, therefore being "
                    "removed unexpectedly!",
                    *lostContact, simParams->timeElapsed);
                DEME_STEP_DEBUG_PRINTF("New number of contacts: %zu", *solverScratchSpace.numContacts);
                DEME_STEP_DEBUG_PRINTF("Old number of contacts: %zu", *solverScratchSpace.numPrevContacts);
                DEME_STEP_DEBUG_PRINTF("New contact A:");
                DEME_STEP_DEBUG_EXEC(
                    displayDeviceArray<bodyID_t>(granData->idGeometryA, *solverScratchSpace.numContacts));
                DEME_STEP_DEBUG_PRINTF("New contact B:");
                DEME_STEP_DEBUG_EXEC(
                    displayDeviceArray<bodyID_t>(granData->idGeometryB, *solverScratchSpace.numContacts));
                DEME_STEP_DEBUG_PRINTF("Old version of the last contact wildcard:");
                DEME_STEP_DEBUG_EXEC(displayDeviceArray<float>(
                    granData->contactWildcards[simParams->nContactWildcards - 1], *solverScratchSpace.numPrevContacts));
                DEME_STEP_DEBUG_PRINTF("Old--new mapping:");
                DEME_STEP_DEBUG_EXEC(
                    displayDeviceArray<contactPairs_t>(granData->contactMapping, *solverScratchSpace.numContacts));
                DEME_STEP_DEBUG_PRINTF("Sentry:");
                DEME_STEP_DEBUG_EXEC(
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
        DEME_GPU_CALL(cudaMemcpy(granData->contactWildcards[i], newWildcards[i],
                                 (*solverScratchSpace.numContacts) * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    solverScratchSpace.finishUsingTempVector("newWildcards");
    solverScratchSpace.finishUsingTempVector("contactSentry");

    // granData may have changed in some of the earlier steps
    granData.toDevice();
}

inline void DEMDynamicThread::calculateForces() {
    // Reset force (acceleration) arrays for this time step
    size_t nContactPairs = *solverScratchSpace.numContacts;

    timers.GetTimer("Clear force array").start();
    {
        size_t blocks_needed_for_force_prep =
            (nContactPairs + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
        size_t blocks_needed_for_acc_prep =
            (simParams->nOwnerBodies + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;

        prep_force_kernels->kernel("prepareAccArrays")
            .instantiate()
            .configure(dim3(blocks_needed_for_acc_prep), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
            .launch(&simParams, &granData);
        DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // prepareForceArrays needs to clear contact force arrays, only if the user asks us to record contact forces.
        // So...
        if (!solverFlags.useNoContactRecord) {
            prep_force_kernels->kernel("prepareForceArrays")
                .instantiate()
                .configure(dim3(blocks_needed_for_force_prep), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
                .launch(&simParams, &granData, nContactPairs);
            DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        }
    }
    timers.GetTimer("Clear force array").stop();

    size_t blocks_needed_for_contacts =
        (nContactPairs + DT_FORCE_CALC_NTHREADS_PER_BLOCK - 1) / DT_FORCE_CALC_NTHREADS_PER_BLOCK;
    // If no contact then we don't have to calculate forces. Note there might still be forces, coming from prescription
    // or other sources.
    if (blocks_needed_for_contacts > 0) {
        timers.GetTimer("Calculate contact forces").start();
        // a custom kernel to compute forces
        cal_force_kernels->kernel("calculateContactForces")
            .instantiate()
            .configure(dim3(blocks_needed_for_contacts), dim3(DT_FORCE_CALC_NTHREADS_PER_BLOCK), 0, streamInfo.stream)
            .launch(&simParams, &granData, nContactPairs);
        DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        // displayDeviceFloat3(granData->contactForces, nContactPairs);
        // displayDeviceArray<contact_t>(granData->contactType, nContactPairs);
        // std::cout << "===========================" << std::endl;
        timers.GetTimer("Calculate contact forces").stop();

        if (!solverFlags.useForceCollectInPlace) {
            timers.GetTimer("Optional force reduction").start();
            // Reflect those body-wise forces on their owner clumps
            if (solverFlags.useCubForceCollect) {
                collectContactForcesThruCub(collect_force_kernels, granData, nContactPairs, simParams->nOwnerBodies,
                                            contactPairArr_isFresh, streamInfo.stream, solverScratchSpace, timers);
            } else {
                blocks_needed_for_contacts =
                    (nContactPairs + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
                // This does both acc and ang acc
                collect_force_kernels->kernel("forceToAcc")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_contacts), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
                    .launch(&granData, nContactPairs);
                DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
            }
            // displayDeviceArray<float>(granData->aZ, simParams->nOwnerBodies);
            // displayDeviceFloat3(granData->contactForces, nContactPairs);
            // std::cout << nContactPairs << std::endl;
            timers.GetTimer("Optional force reduction").stop();
        }
    }
}

inline void DEMDynamicThread::integrateOwnerMotions() {
    size_t blocks_needed_for_clumps =
        (simParams->nOwnerBodies + DEME_NUM_BODIES_PER_BLOCK - 1) / DEME_NUM_BODIES_PER_BLOCK;
    integrator_kernels->kernel("integrateOwners")
        .instantiate()
        .configure(dim3(blocks_needed_for_clumps), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
        .launch(&simParams, &granData);
    DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
}

inline void DEMDynamicThread::routineChecks() {
    if (solverFlags.canFamilyChangeOnDevice) {
        size_t blocks_needed_for_clumps =
            (simParams->nOwnerBodies + DEME_NUM_MODERATORS_PER_BLOCK - 1) / DEME_NUM_MODERATORS_PER_BLOCK;
        mod_kernels->kernel("applyFamilyChanges")
            .instantiate()
            .configure(dim3(blocks_needed_for_clumps), dim3(DEME_NUM_MODERATORS_PER_BLOCK), 0, streamInfo.stream)
            .launch(&simParams, &granData, simParams->nOwnerBodies);
        DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    }
}

inline float* DEMDynamicThread::determineSysVel() {
    return approxMaxVelFunc->dT_GetDeviceValue();
}

inline void DEMDynamicThread::unpack_impl() {
    {
        // Acquire lock and use the content of the dynamic-owned transfer buffer
        std::lock_guard<std::mutex> lock(pSchedSupport->dynamicOwnedBuffer_AccessCoordination);
        unpackMyBuffer();
        // Leave myself a mental note that I just obtained new produce from kT
        contactPairArr_isFresh = true;
        // pSchedSupport->schedulingStats.nDynamicReceives++;
    }
    // dT got the produce, now mark its buffer to be no longer fresh.
    pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = false;
    // Used for inspecting on average how stale kT's produce is.
    pSchedSupport->schedulingStats.accumKinematicLagSteps +=
        (pSchedSupport->currentStampOfDynamic).load() - (pSchedSupport->stampLastDynamicUpdateProdDate).load();
    // dT needs to know how fresh the contact pair info is, and that is determined by when kT received this batch of
    // ingredients.
    pSchedSupport->stampLastDynamicUpdateProdDate = (pSchedSupport->kinematicIngredProdDateStamp).load();

    // If this is a history-based run, then when contacts are received, we need to migrate the contact
    // history info, to match the structure of the new contact array
    if (!solverFlags.isHistoryless) {
        migrateEnduringContacts();
    }

    // With unpacking finished, contactMapping temp array is no longer needed
    solverScratchSpace.finishUsingTempVector("contactMapping");
}

inline void DEMDynamicThread::ifProduceFreshThenUseIt() {
    if (pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh) {
        unpack_impl();
    }
}

inline void DEMDynamicThread::calibrateParams() {
    // Unpacking is done; now we can use temp arrays again to derive max velocity and send to kT
    pCycleVel = determineSysVel();

    if (solverFlags.autoUpdateFreq) {
        unsigned int comfortable_drift;
        if (accumStepUpdater.Query(comfortable_drift)) {
            // If perhapsIdealFutureDrift needs to increase, then the following value much = perhapsIdealFutureDrift.
            comfortable_drift =
                (float)comfortable_drift * solverFlags.targetDriftMultipleOfAvg + solverFlags.targetDriftMoreThanAvg;
            if (*perhapsIdealFutureDrift > comfortable_drift) {
                *perhapsIdealFutureDrift -= FUTURE_DRIFT_TWEAK_STEP_SIZE;
            } else if (*perhapsIdealFutureDrift < comfortable_drift) {
                *perhapsIdealFutureDrift += FUTURE_DRIFT_TWEAK_STEP_SIZE;
            }
            *perhapsIdealFutureDrift = clampBetween<unsigned int, unsigned int>(*perhapsIdealFutureDrift, 0,
                                                                                solverFlags.upperBoundFutureDrift);

            DEME_DEBUG_PRINTF("Comfortable future drift is %u", comfortable_drift);
            DEME_DEBUG_PRINTF("Current future drift is %u", *perhapsIdealFutureDrift);
        }
    }
    // Actually, perhapsIdealFutureDrift seems to have no need to be on device... but I made it a DualStruct anyway
}

inline void DEMDynamicThread::ifProduceFreshThenUseItAndSendNewOrder() {
    if (pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh) {
        timers.GetTimer("Unpack updates from kT").start();
        unpack_impl();
        timers.GetTimer("Unpack updates from kT").stop();

        timers.GetTimer("Send to kT buffer").start();
        // Acquire lock and refresh the work order for the kinematic
        {
            calibrateParams();
            std::lock_guard<std::mutex> lock(pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
            sendToTheirBuffer();
        }
        pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = true;
        pSchedSupport->schedulingStats.nKinematicUpdates++;
        accumStepUpdater.AddUpdate();

        timers.GetTimer("Send to kT buffer").stop();
        // Signal the kinematic that it has data for a new work order
        pSchedSupport->cv_KinematicCanProceed.notify_all();
    }
}

void DEMDynamicThread::workerThread() {
    // Set the gpu for this thread
    DEME_GPU_CALL(cudaSetDevice(streamInfo.device));

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
            ifProduceFreshThenUseIt();

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
                pCycleVel = determineSysVel();
                std::lock_guard<std::mutex> lock(pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
                sendToTheirBuffer();
            }
            pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = true;
            contactPairArr_isFresh = true;
            pSchedSupport->schedulingStats.nKinematicUpdates++;
            accumStepUpdater.AddUpdate();
            // Signal the kinematic that it has data for a new work order.
            pSchedSupport->cv_KinematicCanProceed.notify_all();
            // Then dT will wait for kT to finish one initial run
            {
                std::unique_lock<std::mutex> lock(pSchedSupport->dynamicCanProceed);
                while (!pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh) {
                    // loop to avoid spurious wakeups
                    pSchedSupport->cv_DynamicCanProceed.wait(lock);
                }
            }

            // We unpack it only when it is a `dry-run', meaning the user just wants to update this system, without
            // doing simulation; it also happens at system initialization. We do this so the kT-supplied contact info is
            // registered on dT.
            if (cycleDuration <= 0.0) {
                ifProduceFreshThenUseIt();
            }
        }

        for (double cycle = 0.0; cycle < cycleDuration; cycle += (double)(simParams->h)) {
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
            if (pSchedSupport->dynamicShouldWait()) {
                timers.GetTimer("Wait for kT update").start();
                // Wait for a signal from kT to indicate that kT has caught up
                std::unique_lock<std::mutex> lock(pSchedSupport->dynamicCanProceed);
                while (!pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh) {
                    // Loop to avoid spurious wakeups
                    pSchedSupport->cv_DynamicCanProceed.wait(lock);
                }
                pSchedSupport->schedulingStats.nTimesDynamicHeldBack++;
                // If dT waits, it is penalized, since waiting means double-wait, very bad.
                if (solverFlags.autoUpdateFreq)
                    *perhapsIdealFutureDrift += FUTURE_DRIFT_TWEAK_STEP_SIZE;
                timers.GetTimer("Wait for kT update").stop();
            }
            // NOTE: This ShouldWait check should follow the ifProduceFreshThenUseItAndSendNewOrder call. Because we
            // need to avoid a scenario where dT is waiting here, and kT is also chilling waiting for an update. But
            // with this ShouldWait check being here, if dynamicOwned_Prod2ConsBuffer_isFresh is true so
            // ifProduceFreshThenUseItAndSendNewOrder is executed, then kT is is working for us, no worry; if
            // dynamicOwned_Prod2ConsBuffer_isFresh is false so ifProduceFreshThenUseItAndSendNewOrder didn't run, then
            // kT has to be in the process of doing a CD, we still will not be locked here.

            // If using variable ts size, only when a step is accepted can we move on
            bool step_accepted = false;
            do {
                calculateForces();

                routineChecks();

                timers.GetTimer("Integration").start();
                integrateOwnerMotions();
                timers.GetTimer("Integration").stop();

                step_accepted = true;
            } while ((!solverFlags.isStepConst) || (!step_accepted));

            // CalculateForces is done, set contactPairArr_isFresh to false
            // This will be set to true next time it receives an update from kT
            contactPairArr_isFresh = false;

            /*
            if (cycle == (cycleDuration - 1))
                pSchedSupport->dynamicDone = true;
            */

            // Dynamic wrapped up one cycle, record this fact into schedule support
            pSchedSupport->currentStampOfDynamic++;
            nTotalSteps++;
            accumStepUpdater.AddStep();

            //// TODO: make changes for variable time step size cases
            simParams->timeElapsed += (double)simParams->h;
            // timeElapsed needs to be updated to the device each time step
            // simParams.syncMemberToDevice<double>(offsetof(DEMSimParams, timeElapsed));
            simParams.toDevice();
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
    // Reset dT stats variables, making ready for next user call
    pSchedSupport->dynamicDone = false;
    contactPairArr_isFresh = true;
    accumStepUpdater.Clear();

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
    // First one is force array preparation kernels
    {
        prep_force_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMPrepForceKernels", JitHelper::KERNEL_DIR / "DEMPrepForceKernels.cu", Subs, JitifyOptions)));
    }
    // Then force calculation kernels
    {
        cal_force_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMCalcForceKernels", JitHelper::KERNEL_DIR / "DEMCalcForceKernels.cu", Subs, JitifyOptions)));
    }
    // Then force accumulation kernels
    if (solverFlags.useCubForceCollect) {
        collect_force_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMCollectForceKernels", JitHelper::KERNEL_DIR / "DEMCollectForceKernels.cu", Subs, JitifyOptions)));
    } else {
        collect_force_kernels = std::make_shared<jitify::Program>(std::move(
            JitHelper::buildProgram("DEMCollectForceKernels_Compact",
                                    JitHelper::KERNEL_DIR / "DEMCollectForceKernels_Compact.cu", Subs, JitifyOptions)));
    }
    // Then integration kernels
    {
        integrator_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMIntegrationKernels", JitHelper::KERNEL_DIR / "DEMIntegrationKernels.cu", Subs, JitifyOptions)));
    }
    // Then kernels that are... wildcards, which make on-the-fly changes to solver data
    if (solverFlags.canFamilyChangeOnDevice) {
        mod_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMModeratorKernels", JitHelper::KERNEL_DIR / "DEMModeratorKernels.cu", Subs, JitifyOptions)));
    }
    // Then misc kernels
    {
        misc_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMMiscKernels", JitHelper::KERNEL_DIR / "DEMMiscKernels.cu", Subs, JitifyOptions)));
    }
}

float* DEMDynamicThread::inspectCall(const std::shared_ptr<jitify::Program>& inspection_kernel,
                                     const std::string& kernel_name,
                                     INSPECT_ENTITY_TYPE thing_to_insp,
                                     CUB_REDUCE_FLAVOR reduce_flavor,
                                     bool all_domain,
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
    DEME_DUAL_ARRAY_RESIZE_NOVAL(m_reduceResArr, quarryTempSize);
    float* resArr = (float*)m_reduceResArr.device();
    size_t regionTempSize = n * sizeof(notStupidBool_t);
    // If this boolArrExclude is 1 at an element, that means this element is exluded in the reduction
    notStupidBool_t* boolArrExclude =
        (notStupidBool_t*)solverScratchSpace.allocateTempVector("boolArrExclude", regionTempSize);
    DEME_GPU_CALL(cudaMemset(boolArrExclude, 0, regionTempSize));

    // We may actually have 2 reduced returns: in regional reduction, key 0 and 1 give one return each.
    size_t returnSize = sizeof(float) * 2;
    DEME_DUAL_ARRAY_RESIZE_NOVAL(m_reduceRes, returnSize);
    float* res = (float*)m_reduceRes.device();
    size_t blocks_needed = (n + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    inspection_kernel->kernel(kernel_name)
        .instantiate()
        .configure(dim3(blocks_needed), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
        .launch(&granData, &simParams, resArr, boolArrExclude, n, owner_type);
    DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

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
                    return (float*)m_reduceResArr.device();
                } else {
                    m_reduceResArr.toHost();
                    return (float*)m_reduceResArr.host();
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
                    return (float*)m_reduceResArr.device();
                } else {
                    m_reduceResArr.toHost();
                    return (float*)m_reduceResArr.host();
                }
        }
    }

    solverScratchSpace.finishUsingTempVector("boolArrExclude");
    solverScratchSpace.finishUsingTempVector("boolArrExclude_sorted");
    solverScratchSpace.finishUsingTempVector("resArr_sorted");
    solverScratchSpace.finishUsingTempVector("num_unique_out");
    if (return_device_ptr) {
        return (float*)m_reduceRes.device();
    } else {
        m_reduceRes.toHost();
        return (float*)m_reduceRes.host();
    }
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
    for (unsigned int i = 0; i < triWildcards.size(); i++) {
        triWildcards[i].reset();
    }
}

size_t DEMDynamicThread::getNumContacts() const {
    return *(solverScratchSpace.numContacts);
}

double DEMDynamicThread::getSimTime() const {
    return simParams->timeElapsed;
}

void DEMDynamicThread::setSimTime(double time) {
    simParams->timeElapsed = time;
    // simParams.syncMemberToDevice<double>(offsetof(DEMSimParams, timeElapsed));
    simParams.toDevice();
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
    for (size_t i = 0; i < simParams->nTriGM; i++) {
        bodyID_t owner_id = ownerMesh[i];  // No device-side change
        if (+(familyID[owner_id]) == N) {
            triMaterialOffset[i] = (materialsOffset_t)mat_id;
        }
    }
    triMaterialOffset.toDevice();
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
    idGeometryA.toHost();
    idGeometryB.toHost();
    contactType.toHost();

    size_t numCnt = *solverScratchSpace.numContacts;
    for (size_t i = 0; i < numCnt; i++) {
        bodyID_t geoA = idGeometryA[i];
        bodyID_t ownerA = ownerClumpBody[geoA];
        bodyID_t geoB = idGeometryB[i];
        contact_t typeB = contactType[i];
        bodyID_t ownerB = getGeoOwnerID(geoB, typeB);

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

void DEMDynamicThread::setTriWildcardValue(bodyID_t geoID, unsigned int wc_num, const std::vector<float>& vals) {
    for (size_t i = 0; i < vals.size(); i++) {
        (*triWildcards[wc_num])[geoID + i] = vals.at(i);
    }
    // Partial send to device
    triWildcards[wc_num]->toDevice(geoID, vals.size());
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

void DEMDynamicThread::getTriWildcardValue(std::vector<float>& res, bodyID_t ID, unsigned int wc_num, size_t n) {
    res = std::move(triWildcards[wc_num]->getVal(ID, n));
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

}  // namespace deme
