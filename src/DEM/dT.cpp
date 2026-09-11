//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <cstring>
#include <cstdint>
#include <iostream>
#include <thread>
#include <algorithm>

#include <core/ApiVersion.h>
#include <core/utils/JitHelper.h>
#include <DEM/dT.h>
#include <DEM/kT.h>
#include <DEM/utils/HostSideHelpers.hpp>
#include <DEM/utils/DynamicThreadHelpers.hpp>
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
    idPrimitiveA.bindDevicePointer(&(granData->idPrimitiveA));
    idPrimitiveB.bindDevicePointer(&(granData->idPrimitiveB));
    contactTypePrimitive.bindDevicePointer(&(granData->contactTypePrimitive));
    geomToPatchMap.bindDevicePointer(&(granData->geomToPatchMap));

    // NEW: Bind separate patch ID and mapping array pointers
    idPatchA.bindDevicePointer(&(granData->idPatchA));
    idPatchB.bindDevicePointer(&(granData->idPatchB));
    contactTypePatch.bindDevicePointer(&(granData->contactTypePatch));
    contactPatchIsland.bindDevicePointer(&(granData->contactPatchIsland));

    familyMaskMatrix.bindDevicePointer(&(granData->familyMasks));
    familyExtraMarginSize.bindDevicePointer(&(granData->familyExtraMarginSize));
    ownerCombinedMaster.bindDevicePointer(&(granData->ownerCombinedMaster));
    ownerCombinedRelPos.bindDevicePointer(&(granData->ownerCombinedRelPos));
    ownerCombinedRelOriQ.bindDevicePointer(&(granData->ownerCombinedRelOriQ));
    ownerCombinedMasterMass.bindDevicePointer(&(granData->ownerCombinedMasterMass));
    ownerCombinedMasterMOI.bindDevicePointer(&(granData->ownerCombinedMasterMOI));

    contactForces.bindDevicePointer(&(granData->contactForces));
    contactTorque_convToForce.bindDevicePointer(&(granData->contactTorque_convToForce));
    contactNormals.bindDevicePointer(&(granData->contactNormals));
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

    // Mesh and analytical-related
    ownerTriMesh.bindDevicePointer(&(granData->ownerTriMesh));
    ownerPatchMesh.bindDevicePointer(&(granData->ownerPatchMesh));
    triPatchID.bindDevicePointer(&(granData->triPatchID));
    ownerAnalBody.bindDevicePointer(&(granData->ownerAnalBody));
    ownerMeshConvex.bindDevicePointer(&(granData->ownerMeshConvex));
    ownerMeshNeverWinner.bindDevicePointer(&(granData->ownerMeshNeverWinner));
    ownerMeshWatertight.bindDevicePointer(&(granData->ownerMeshWatertight));
    ownerMeshShellHalfThickness.bindDevicePointer(&(granData->ownerMeshShellHalfThickness));
    triNeighborIndex.bindDevicePointer(&(granData->triNeighborIndex));
    triNeighbor1.bindDevicePointer(&(granData->triNeighbor1));
    triNeighbor2.bindDevicePointer(&(granData->triNeighbor2));
    triNeighbor3.bindDevicePointer(&(granData->triNeighbor3));
    relPosNode1.bindDevicePointer(&(granData->relPosNode1));
    relPosNode2.bindDevicePointer(&(granData->relPosNode2));
    relPosNode3.bindDevicePointer(&(granData->relPosNode3));
    relPosPatch.bindDevicePointer(&(granData->relPosPatch));
    patchMaterialOffset.bindDevicePointer(&(granData->patchMaterialOffset));
    maxTriTriPenetration.bindDevicePointer(&(granData->maxTriTriPenetration));

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

    // Primitive contact info
    idPrimitiveA.toDeviceAsync(streamInfo.stream);
    idPrimitiveB.toDeviceAsync(streamInfo.stream);
    contactTypePrimitive.toDeviceAsync(streamInfo.stream);
    geomToPatchMap.toDeviceAsync(streamInfo.stream);

    // Separate patch contact info
    contactTypePatch.toDeviceAsync(streamInfo.stream);
    idPatchA.toDeviceAsync(streamInfo.stream);
    idPatchB.toDeviceAsync(streamInfo.stream);
    contactPatchIsland.toDeviceAsync(streamInfo.stream);

    familyMaskMatrix.toDeviceAsync(streamInfo.stream);
    familyExtraMarginSize.toDeviceAsync(streamInfo.stream);
    if (ownerCombinedMaster.size() > 0) {
        ownerCombinedMaster.toDeviceAsync(streamInfo.stream);
        ownerCombinedRelPos.toDeviceAsync(streamInfo.stream);
        ownerCombinedRelOriQ.toDeviceAsync(streamInfo.stream);
        ownerCombinedMasterMass.toDeviceAsync(streamInfo.stream);
        ownerCombinedMasterMOI.toDeviceAsync(streamInfo.stream);
    }

    contactForces.toDeviceAsync(streamInfo.stream);
    contactNormals.toDeviceAsync(streamInfo.stream);
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

    ownerTriMesh.toDeviceAsync(streamInfo.stream);
    ownerPatchMesh.toDeviceAsync(streamInfo.stream);
    triPatchID.toDeviceAsync(streamInfo.stream);
    ownerAnalBody.toDeviceAsync(streamInfo.stream);
    ownerMeshConvex.toDeviceAsync(streamInfo.stream);
    ownerMeshNeverWinner.toDeviceAsync(streamInfo.stream);
    ownerMeshWatertight.toDeviceAsync(streamInfo.stream);
    ownerMeshShellHalfThickness.toDeviceAsync(streamInfo.stream);
    triNeighborIndex.toDeviceAsync(streamInfo.stream);
    triNeighbor1.toDeviceAsync(streamInfo.stream);
    triNeighbor2.toDeviceAsync(streamInfo.stream);
    triNeighbor3.toDeviceAsync(streamInfo.stream);
    relPosNode1.toDeviceAsync(streamInfo.stream);
    relPosNode2.toDeviceAsync(streamInfo.stream);
    relPosNode3.toDeviceAsync(streamInfo.stream);
    relPosPatch.toDeviceAsync(streamInfo.stream);
    patchMaterialOffset.toDeviceAsync(streamInfo.stream);
    triPVGlobalTriToLocal.toDeviceAsync(streamInfo.stream);
    triPVStepP.toDeviceAsync(streamInfo.stream);
    triPVStepPV.toDeviceAsync(streamInfo.stream);
    triPVAccumP.toDeviceAsync(streamInfo.stream);
    triPVAccumPV.toDeviceAsync(streamInfo.stream);

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
    // Primitive contact info
    idPrimitiveA.toHost();
    idPrimitiveB.toHost();
    contactTypePrimitive.toHost();
    geomToPatchMap.toHost();

    // Separate patch contact info
    contactTypePatch.toHost();
    idPatchA.toHost();
    idPatchB.toHost();
    contactPatchIsland.toHost();

    // Contact results
    contactForces.toHost();
    contactNormals.toHost();
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
void DEMDynamicThread::packTransferPointers(DEMKinematicThread* kT) {
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
    granData->pKTOwnedBuffer_maxTriTriPenetration = kT->maxTriTriPenetration_buffer.data();
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
                                    float tritri_contact_rejection_ratio,
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
    simParams->dyn.binSize = binSize;
    simParams->dyn.inv_binSize = 1. / binSize;
    simParams->LBFX = LBFPoint.x;
    simParams->LBFY = LBFPoint.y;
    simParams->LBFZ = LBFPoint.z;
    simParams->Gx = G.x;
    simParams->Gy = G.y;
    simParams->Gz = G.z;
    simParams->dyn.h = ts_size;
    simParams->dyn.beta = expand_factor;  // If beta is auto-adapting, this assignment has no effect
    simParams->dyn.approxMaxVel = approx_max_vel;
    simParams->dyn.expSafetyMulti = expand_safety_param;
    simParams->dyn.expSafetyAdder = expand_safety_adder;
    simParams->capTriTriPenetration = max_tritri_penetration;
    simParams->triTriContactRejectionRatio = tritri_contact_rejection_ratio;
    simParams->useAngVelMargin = use_angvel_margin ? 1 : 0;
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

void DEMDynamicThread::allocateGPUArrays(size_t nOwnerBodies,
                                         size_t nOwnerClumps,
                                         unsigned int nExtObj,
                                         size_t nTriMeshes,
                                         size_t nSpheresGM,
                                         size_t nTriGM,
                                         size_t nTriNeighbors,
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
    DEME_DUAL_ARRAY_RESIZE(ownerMeshConvex, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(ownerMeshNeverWinner, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(ownerMeshWatertight, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(ownerMeshShellHalfThickness, nOwnerBodies, 0);

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
    DEME_DUAL_ARRAY_RESIZE(triNeighborIndex, nTriGM, NULL_BODYID);
    DEME_DUAL_ARRAY_RESIZE(triNeighbor1, nTriNeighbors, NULL_BODYID);
    DEME_DUAL_ARRAY_RESIZE(triNeighbor2, nTriNeighbors, NULL_BODYID);
    DEME_DUAL_ARRAY_RESIZE(triNeighbor3, nTriNeighbors, NULL_BODYID);
    DEME_DEVICE_ARRAY_RESIZE(maxTriTriPenetration, nTriGM);
    if (nTriGM > 0) {
        DEME_GPU_CALL(cudaMemset(maxTriTriPenetration.data(), 0, nTriGM * sizeof(float)));
    }
    triPVTrackingEnabled = false;
    triPVNumTrackedTriangles = 0;
    triPVWindowSteps = 0;
    triPVOwnerOrder.clear();
    triPVOwnerOffsets.clear();
    triPVOwnerCounts.clear();
    triPVOwnerToSlot.clear();
    DEME_DUAL_ARRAY_RESIZE(triPVGlobalTriToLocal, nTriGM, -1);
    DEME_DUAL_ARRAY_RESIZE(triPVStepP, 1, 0.f);
    DEME_DUAL_ARRAY_RESIZE(triPVStepPV, 1, 0.f);
    DEME_DUAL_ARRAY_RESIZE(triPVAccumP, 1, 0.f);
    DEME_DUAL_ARRAY_RESIZE(triPVAccumPV, 1, 0.f);

    // Resize to the number of mesh patches
    DEME_DUAL_ARRAY_RESIZE(ownerPatchMesh, nMeshPatches, 0);
    DEME_DUAL_ARRAY_RESIZE(patchMaterialOffset, nMeshPatches, 0);
    DEME_DUAL_ARRAY_RESIZE(relPosPatch, nMeshPatches, make_float3(0));

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

        DEME_DUAL_ARRAY_RESIZE(idPatchA, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(idPatchB, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(contactTypePatch, cnt_arr_size, NOT_A_CONTACT);
        DEME_DUAL_ARRAY_RESIZE(contactPatchIsland, cnt_arr_size, NULL_BODYID);

        // If there are meshes, then sph--mesh case always use force storage, no getting around; if no mesh, then if no
        // need to store forces, we can choose to not resize these arrays.
        if (!(solverFlags.useNoContactRecord && simParams->nTriGM == 0)) {
            DEME_DUAL_ARRAY_RESIZE(contactForces, cnt_arr_size, make_float3(0));
            DEME_DUAL_ARRAY_RESIZE(contactTorque_convToForce, cnt_arr_size, make_float3(0));
            DEME_DUAL_ARRAY_RESIZE(contactPointGeometryA, cnt_arr_size, make_float3(0));
            DEME_DUAL_ARRAY_RESIZE(contactPointGeometryB, cnt_arr_size, make_float3(0));
            if (simParams->storeNormal) {
                DEME_DUAL_ARRAY_RESIZE(contactNormals, cnt_arr_size, make_float3(0));
            }
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
                                            const std::vector<notStupidBool_t>& input_mesh_obj_convex,
                                            const std::vector<notStupidBool_t>& input_mesh_obj_never_winner,
                                            const std::vector<unsigned int>& mesh_facet_owner,
                                            const std::vector<bodyID_t>& mesh_facet_patch,
                                            const std::vector<bodyID_t>& mesh_facet_neighbor1,
                                            const std::vector<bodyID_t>& mesh_facet_neighbor2,
                                            const std::vector<bodyID_t>& mesh_facet_neighbor3,
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
                                            size_t nExistingMeshPatches,
                                            size_t nExistingTriNeighbors) {
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
    size_t neighbor_write = nExistingTriNeighbors;
    // p for indexing patches (flattened across all meshes)
    size_t p = 0;
    for (size_t i = 0; i < input_mesh_objs.size(); i++) {
        // If got here, it is a mesh
        const bodyID_t owner_id = i + owner_offset_for_mesh_obj;
        ownerTypes[owner_id] = OWNER_T_MESH;
        ownerMeshConvex[owner_id] = input_mesh_obj_convex.at(i);
        ownerMeshNeverWinner[owner_id] = input_mesh_obj_never_winner.at(i);
        ownerMeshWatertight[owner_id] = input_mesh_objs.at(i)->IsWatertight() ? 1 : 0;
        ownerMeshShellHalfThickness[owner_id] = std::max(input_mesh_objs.at(i)->GetShellHalfThickness(), 0.f);

        // Store inherent geo wildcards (per-triangle: one value per triangle facet).
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
        input_mesh_objs.at(i)->owner = owner_id;
        input_mesh_objs.at(i)->cache_offset = m_meshes.size();
        m_meshes.push_back(input_mesh_objs.at(i));

        inertiaPropOffsets[owner_id] = solverFlags.useMassJitify
                                           ? offset_for_mesh_obj_mass_template + mesh_obj_mass_offsets.at(i)
                                           : i + offset_for_mesh_obj_mass_template;
        if (!solverFlags.useMassJitify) {
            massOwnerBody[owner_id] = mesh_obj_mass_types.at(i);
            const float3 this_moi = mesh_obj_moi_types.at(i);
            mmiXX[owner_id] = this_moi.x;
            mmiYY[owner_id] = this_moi.y;
            mmiZZ[owner_id] = this_moi.z;
        }
        auto this_CoM_coord = input_mesh_obj_xyz.at(i) - LBF;
        positionToVoxelID<voxelID_t, subVoxelPos_t, double>(
            voxelID[owner_id], locX[owner_id], locY[owner_id], locZ[owner_id], (double)this_CoM_coord.x,
            (double)this_CoM_coord.y, (double)this_CoM_coord.z, simParams->nvXp2, simParams->nvYp2,
            simParams->voxelSize, simParams->l);

        // Set mesh owner's oriQ
        auto oriQ_of_this = input_mesh_obj_rot.at(i);
        oriQw[owner_id] = oriQ_of_this.w;
        oriQx[owner_id] = oriQ_of_this.x;
        oriQy[owner_id] = oriQ_of_this.y;
        oriQz[owner_id] = oriQ_of_this.z;

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
        const bool mesh_needs_neighbors =
            !(input_mesh_obj_convex.at(this_facet_owner) != 0 && input_mesh_obj_never_winner.at(this_facet_owner) != 0);
        for (; k < mesh_facet_owner.size(); k++) {
            // mesh_facet_owner run length is the num of facets in this mesh entity
            if (mesh_facet_owner.at(k) != this_facet_owner)
                break;
            const size_t global_tri = nExistingFacets + k;
            ownerTriMesh[global_tri] = owner_offset_for_mesh_obj + this_facet_owner;
            // Tri's patch belonging needs to take into account those patches that are previously added
            triPatchID[global_tri] = nExistingMeshPatches + mesh_facet_patch.at(k);
            if (mesh_needs_neighbors) {
                triNeighborIndex[global_tri] = neighbor_write;
                triNeighbor1[neighbor_write] = mesh_facet_neighbor1.at(k);
                triNeighbor2[neighbor_write] = mesh_facet_neighbor2.at(k);
                triNeighbor3[neighbor_write] = mesh_facet_neighbor3.at(k);
                neighbor_write++;
            } else {
                triNeighborIndex[global_tri] = NULL_BODYID;
            }
            DEMTriangle this_tri = mesh_facets.at(k);
            relPosNode1[global_tri] = this_tri.p1;
            relPosNode2[global_tri] = this_tri.p2;
            relPosNode3[global_tri] = this_tri.p3;
        }

        family_t this_family_num = input_mesh_obj_family.at(i);
        familyID[owner_id] = this_family_num;

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
    // Also take notes of num of triangles of each mesh obj (for per-triangle geo wildcard tracking)
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
                // For mesh, nGeos is the number of triangles (per-triangle geo wildcard).
                tracked_obj->nGeos =
                    prescans_mesh_size.at(tracked_obj->load_order + 1) - prescans_mesh_size.at(tracked_obj->load_order);
                break;
            default:
                DEME_ERROR(std::string("A DEM tracked object has an unknown type."));
        }
        if (tracked_obj->nSpanOwnersOverride > 0) {
            tracked_obj->nSpanOwners = tracked_obj->nSpanOwnersOverride;
        }
        if (tracked_obj->nGeosOverride > 0) {
            tracked_obj->nGeos = tracked_obj->nGeosOverride;
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
                                     const std::vector<notStupidBool_t>& input_mesh_obj_convex,
                                     const std::vector<notStupidBool_t>& input_mesh_obj_never_winner,
                                     const std::vector<unsigned int>& mesh_facet_owner,
                                     const std::vector<bodyID_t>& mesh_facet_patch,
                                     const std::vector<bodyID_t>& mesh_facet_neighbor1,
                                     const std::vector<bodyID_t>& mesh_facet_neighbor2,
                                     const std::vector<bodyID_t>& mesh_facet_neighbor3,
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
                         input_mesh_obj_convex, input_mesh_obj_never_winner, mesh_facet_owner, mesh_facet_patch,
                         mesh_facet_neighbor1, mesh_facet_neighbor2, mesh_facet_neighbor3, mesh_facets,
                         mesh_patch_owner, mesh_patch_materials, clump_templates, ext_obj_mass_types, ext_obj_moi_types,
                         ext_obj_comp_num, mesh_obj_mass_types, mesh_obj_moi_types, mesh_obj_mass_offsets, 0, 0, 0, 0,
                         0);

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
                                             const std::vector<notStupidBool_t>& input_mesh_obj_convex,
                                             const std::vector<notStupidBool_t>& input_mesh_obj_never_winner,
                                             const std::vector<unsigned int>& mesh_facet_owner,
                                             const std::vector<bodyID_t>& mesh_facet_patch,
                                             const std::vector<bodyID_t>& mesh_facet_neighbor1,
                                             const std::vector<bodyID_t>& mesh_facet_neighbor2,
                                             const std::vector<bodyID_t>& mesh_facet_neighbor3,
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
                                             size_t nExistingTriNeighbors,
                                             size_t nExistingPatches,
                                             unsigned int nExistingObj,
                                             unsigned int nExistingAnalGM) {
    // No policy changes here
    (void)mesh_obj_mass_jit_types;
    (void)mesh_obj_moi_jit_types;

    // Analytical objects-related arrays should be empty
    populateEntityArrays(input_clump_batches, input_ext_obj_xyz, input_ext_obj_rot, input_ext_obj_family,
                         input_mesh_objs, input_mesh_obj_xyz, input_mesh_obj_rot, input_mesh_obj_family,
                         input_mesh_obj_convex, input_mesh_obj_never_winner, mesh_facet_owner, mesh_facet_patch,
                         mesh_facet_neighbor1, mesh_facet_neighbor2, mesh_facet_neighbor3, mesh_facets,
                         mesh_patch_owner, mesh_patch_materials, clump_templates, ext_obj_mass_types, ext_obj_moi_types,
                         ext_obj_comp_num, mesh_obj_mass_types, mesh_obj_moi_types, mesh_obj_mass_offsets,
                         nExistingOwners, nExistingSpheres, nExistingFacets, nExistingPatches, nExistingTriNeighbors);

    // Make changes to tracked objects (potentially add more)
    buildTrackedObjs(input_clump_batches, ext_obj_comp_num, input_mesh_objs, tracked_objs, nExistingOwners,
                     nExistingSpheres, nExistingFacets, nExistingAnalGM);
}

void DEMDynamicThread::writeSpheresAsCsv(std::ofstream& ptFile) {
    migrateFamilyToHost();
    migrateClumpPosInfoToHost();
    migrateClumpHighOrderInfoToHost();
    migrateOwnerWildcardToHost();
    migrateSphGeoWildcardToHost();
    writeSpheresAsCsvFromHost(ptFile);
}

void DEMDynamicThread::writeSpheresAsCsvFromHost(std::ofstream& ptFile) {
    std::ostringstream outstrstream;

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

void DEMDynamicThread::writeSpheresAsVtk(std::ofstream& ptFile) {
    migrateFamilyToHost();
    migrateClumpPosInfoToHost();
    migrateClumpHighOrderInfoToHost();
    migrateOwnerWildcardToHost();
    migrateSphGeoWildcardToHost();
    writeSpheresAsVtkFromHost(ptFile);
}

void DEMDynamicThread::writeSpheresAsVtkFromHost(std::ofstream& ptFile) {
    // Keep one compact point record per component sphere. ParaView can turn these points into rendered spheres with a
    // Glyph filter using `r` as the scale array, without DEME writing a triangle tessellation for every sphere.
    struct SpherePointData {
        float3 pos;
        float radius;
        float3 vel;
        float3 ang_vel;
        float3 acc;
        float3 ang_acc;
        family_t family;
        std::vector<float> owner_wildcards;
        std::vector<float> geo_wildcards;
    };

    std::vector<SpherePointData> spheres;
    spheres.reserve(simParams->nSpheresGM);

    // Resolve owner-frame component locations and collect all enabled attributes before writing the VTK section
    // counts. Filtering is done here so every point-data array has exactly the same length as the POINTS section.
    for (size_t i = 0; i < simParams->nSpheresGM; i++) {
        const bodyID_t owner = ownerClumpBody[i];
        const family_t family = familyID[owner];
        if (familiesNoOutput.find(family) != familiesNoOutput.end()) {
            continue;
        }

        float X, Y, Z;
        voxelIDToPosition<float, voxelID_t, subVoxelPos_t>(X, Y, Z, voxelID[owner], locX[owner], locY[owner],
                                                           locZ[owner], simParams->nvXp2, simParams->nvYp2,
                                                           simParams->voxelSize, simParams->l);
        const float3 owner_pos = make_float3(X + simParams->LBFX, Y + simParams->LBFY, Z + simParams->LBFZ);
        const size_t component = solverFlags.useClumpJitify ? clumpComponentOffsetExt[i] : i;
        float3 rel_pos = make_float3(relPosSphereX[component], relPosSphereY[component], relPosSphereZ[component]);
        applyOriQToVector3<float, float>(rel_pos.x, rel_pos.y, rel_pos.z, oriQw[owner], oriQx[owner], oriQy[owner],
                                         oriQz[owner]);

        SpherePointData sphere;
        sphere.pos = owner_pos + rel_pos;
        sphere.radius = radiiSphere[component];
        sphere.vel = make_float3(vX[owner], vY[owner], vZ[owner]);
        sphere.ang_vel = make_float3(omgBarX[owner], omgBarY[owner], omgBarZ[owner]);
        sphere.acc = make_float3(aX[owner], aY[owner], aZ[owner]);
        sphere.ang_acc = make_float3(alphaX[owner], alphaY[owner], alphaZ[owner]);
        sphere.family = family;
        if (solverFlags.outputFlags & OUTPUT_CONTENT::OWNER_WILDCARD) {
            sphere.owner_wildcards.reserve(m_owner_wildcard_names.size());
            for (const auto& wildcard : ownerWildcards) {
                sphere.owner_wildcards.push_back((*wildcard)[owner]);
            }
        }
        if (solverFlags.outputFlags & OUTPUT_CONTENT::GEO_WILDCARD) {
            sphere.geo_wildcards.reserve(m_geo_wildcard_names.size());
            for (const auto& wildcard : sphereWildcards) {
                sphere.geo_wildcards.push_back((*wildcard)[i]);
            }
        }
        spheres.push_back(std::move(sphere));
    }

    std::ostringstream out;
    out << "# vtk DataFile Version 2.0\n";
    out << "DEME component spheres\n";
    out << "ASCII\n";
    out << "DATASET POLYDATA\n";
    out << "POINTS " << spheres.size() << " float\n";
    for (const auto& sphere : spheres) {
        out << sphere.pos.x << " " << sphere.pos.y << " " << sphere.pos.z << "\n";
    }
    out << "VERTICES " << spheres.size() << " " << 2 * spheres.size() << "\n";
    for (size_t i = 0; i < spheres.size(); i++) {
        out << "1 " << i << "\n";
    }
    out << "POINT_DATA " << spheres.size() << "\n";

    auto write_scalar = [&out, &spheres](const std::string& name, const auto& value) {
        out << "SCALARS " << name << " float 1\n";
        out << "LOOKUP_TABLE default\n";
        for (const auto& sphere : spheres) {
            out << value(sphere) << "\n";
        }
    };
    auto write_vector = [&out, &spheres](const std::string& name, const auto& value) {
        out << "VECTORS " << name << " float\n";
        for (const auto& sphere : spheres) {
            const float3 vector = value(sphere);
            out << vector.x << " " << vector.y << " " << vector.z << "\n";
        }
    };

    write_scalar("r", [](const SpherePointData& sphere) { return sphere.radius; });
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ABSV) {
        write_scalar("absv", [](const SpherePointData& sphere) { return length(sphere.vel); });
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::VEL) {
        write_vector("velocity", [](const SpherePointData& sphere) { return sphere.vel; });
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ANG_VEL) {
        write_vector("angular_velocity", [](const SpherePointData& sphere) { return sphere.ang_vel; });
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ABS_ACC) {
        write_scalar("abs_acc", [](const SpherePointData& sphere) { return length(sphere.acc); });
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ACC) {
        write_vector("acceleration", [](const SpherePointData& sphere) { return sphere.acc; });
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ANG_ACC) {
        write_vector("angular_acceleration", [](const SpherePointData& sphere) { return sphere.ang_acc; });
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
        write_scalar("family", [](const SpherePointData& sphere) { return +sphere.family; });
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::OWNER_WILDCARD) {
        size_t wildcard_index = 0;
        for (const auto& wildcard_name : m_owner_wildcard_names) {
            write_scalar(wildcard_name, [wildcard_index](const SpherePointData& sphere) {
                return sphere.owner_wildcards[wildcard_index];
            });
            wildcard_index++;
        }
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::GEO_WILDCARD) {
        size_t wildcard_index = 0;
        for (const auto& wildcard_name : m_geo_wildcard_names) {
            write_scalar(wildcard_name, [wildcard_index](const SpherePointData& sphere) {
                return sphere.geo_wildcards[wildcard_index];
            });
            wildcard_index++;
        }
    }

    ptFile << out.str();
}

void DEMDynamicThread::writeClumpsAsCsv(std::ofstream& ptFile, unsigned int accuracy) {
    migrateFamilyToHost();
    migrateClumpPosInfoToHost();
    migrateClumpHighOrderInfoToHost();
    migrateOwnerWildcardToHost();
    writeClumpsAsCsvFromHost(ptFile, accuracy);
}

void DEMDynamicThread::writeClumpsAsCsvFromHost(std::ofstream& ptFile, unsigned int accuracy) {
    std::ostringstream outstrstream;
    outstrstream.precision(accuracy);

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
    migrateFamilyToHost();
    migrateClumpPosInfoToHost();
    migrateContactInfoToHost();
    return generateContactInfoFromHost(force_thres);
}

std::shared_ptr<ContactInfoContainer> DEMDynamicThread::generateContactInfoFromHost(float force_thres) {
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
        if (type == NOT_A_CONTACT) {
            DEME_ERROR("NOT_A_CONTACT reached dT contact output; kT should compact null contacts before handoff.");
        }

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
            // If CNT_OUTPUT_CONTENT::NORMAL is on, then contactNormals is always stored
            contactInfo.Get<float3>("Normal")[useful_cnt] = contactNormals[i];
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
    migrateFamilyToHost();
    migrateClumpPosInfoToHost();
    migrateContactInfoToHost();
    writeContactsAsCsvFromHost(ptFile, force_thres);
}

void DEMDynamicThread::writeContactsAsCsvFromHost(std::ofstream& ptFile, float force_thres) {
    std::ostringstream outstrstream;

    std::shared_ptr<ContactInfoContainer> contactInfo = generateContactInfoFromHost(force_thres);

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
    migrateFamilyToHost();
    migrateClumpPosInfoToHost();
    writeMeshesAsVtkFromHost(ptFile);
}

void DEMDynamicThread::writeMeshesAsVtkFromHost(std::ofstream& ptFile) {
    std::ostringstream ostream;

    auto ownerPosFromHost = [this](bodyID_t owner) {
        double X, Y, Z;
        voxelID_t voxel = voxelID[owner];
        subVoxelPos_t subVoxX = locX[owner];
        subVoxelPos_t subVoxY = locY[owner];
        subVoxelPos_t subVoxZ = locZ[owner];
        voxelIDToPosition<double, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel, subVoxX, subVoxY, subVoxZ, simParams->nvXp2,
                                                            simParams->nvYp2, simParams->voxelSize, simParams->l);
        return make_float3(X + simParams->LBFX, Y + simParams->LBFY, Z + simParams->LBFZ);
    };
    auto ownerOriQFromHost = [this](bodyID_t owner) {
        return make_float4(oriQx[owner], oriQy[owner], oriQz[owner], oriQw[owner]);
    };

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
            float3 ownerPos = ownerPosFromHost(mowner);
            float4 ownerOriQ = ownerOriQFromHost(mowner);
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

    const unsigned int mesh_flags = solverFlags.meshOutFlags;
    if (mesh_flags != static_cast<unsigned int>(MESH_OUTPUT_CONTENT::XYZ)) {
        ostream << "\nCELL_DATA " << total_f << "\n";

        // Visit triangles in exactly the same order as the CELLS section. The global triangle ID advances through
        // skipped meshes as well, preserving IDs used by DEME contact and wildcard APIs.
        auto for_each_output_triangle = [&](const auto& callback) {
            size_t global_triangle = 0;
            for (size_t output_mesh = 0; output_mesh < m_meshes.size(); output_mesh++) {
                const auto& mesh = m_meshes[output_mesh];
                const size_t triangle_count = mesh->GetIndicesVertexes().size();
                if (!thisMeshSkip[output_mesh]) {
                    for (size_t local_triangle = 0; local_triangle < triangle_count; local_triangle++) {
                        callback(*mesh, output_mesh, local_triangle, global_triangle, mesh->owner);
                        global_triangle++;
                    }
                } else {
                    global_triangle += triangle_count;
                }
            }
        };
        auto write_scalar = [&](const std::string& name, const std::string& type, const auto& value) {
            ostream << "SCALARS " << name << " " << type << " 1\nLOOKUP_TABLE default\n";
            for_each_output_triangle([&](const DEMMesh& mesh, size_t mesh_id, size_t local_triangle,
                                         size_t global_triangle, bodyID_t owner) {
                ostream << value(mesh, mesh_id, local_triangle, global_triangle, owner) << "\n";
            });
        };
        auto write_vector = [&](const std::string& name, const auto& value) {
            ostream << "VECTORS " << name << " float\n";
            for_each_output_triangle([&](const DEMMesh& mesh, size_t mesh_id, size_t local_triangle,
                                         size_t global_triangle, bodyID_t owner) {
                const float3 vector = value(mesh, mesh_id, local_triangle, global_triangle, owner);
                ostream << vector.x << " " << vector.y << " " << vector.z << "\n";
            });
        };

        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::QUAT)) {
            ostream << "SCALARS quaternion float 4\nLOOKUP_TABLE default\n";
            for_each_output_triangle([&](const DEMMesh&, size_t, size_t, size_t, bodyID_t owner) {
                ostream << oriQw[owner] << " " << oriQx[owner] << " " << oriQy[owner] << " " << oriQz[owner] << "\n";
            });
        }
        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::ABSV)) {
            write_scalar("absv", "float", [&](const DEMMesh&, size_t, size_t, size_t, bodyID_t owner) {
                return length(make_float3(vX[owner], vY[owner], vZ[owner]));
            });
        }
        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::VEL)) {
            write_vector("velocity", [&](const DEMMesh&, size_t, size_t, size_t, bodyID_t owner) {
                return make_float3(vX[owner], vY[owner], vZ[owner]);
            });
        }
        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::ANG_VEL)) {
            write_vector("angular_velocity", [&](const DEMMesh&, size_t, size_t, size_t, bodyID_t owner) {
                return make_float3(omgBarX[owner], omgBarY[owner], omgBarZ[owner]);
            });
        }
        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::ABS_ACC)) {
            write_scalar("abs_acc", "float", [&](const DEMMesh&, size_t, size_t, size_t, bodyID_t owner) {
                return length(make_float3(aX[owner], aY[owner], aZ[owner]));
            });
        }
        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::ACC)) {
            write_vector("acceleration", [&](const DEMMesh&, size_t, size_t, size_t, bodyID_t owner) {
                return make_float3(aX[owner], aY[owner], aZ[owner]);
            });
        }
        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::ANG_ACC)) {
            write_vector("angular_acceleration", [&](const DEMMesh&, size_t, size_t, size_t, bodyID_t owner) {
                return make_float3(alphaX[owner], alphaY[owner], alphaZ[owner]);
            });
        }
        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::FAMILY)) {
            write_scalar("family", "int",
                         [&](const DEMMesh&, size_t, size_t, size_t, bodyID_t owner) { return +familyID[owner]; });
        }
        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::MAT)) {
            write_scalar("material", "int", [&](const DEMMesh&, size_t, size_t, size_t global_triangle, bodyID_t) {
                return +patchMaterialOffset[triPatchID[global_triangle]];
            });
        }
        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::OWNER)) {
            write_scalar("owner", "int", [&](const DEMMesh&, size_t, size_t, size_t, bodyID_t owner) { return owner; });
        }
        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::MESH_ID)) {
            write_scalar("mesh_id", "int",
                         [&](const DEMMesh&, size_t mesh_id, size_t, size_t, bodyID_t) { return mesh_id; });
        }
        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::TRI_ID)) {
            write_scalar("tri_id", "int", [&](const DEMMesh&, size_t, size_t, size_t global_triangle, bodyID_t) {
                return global_triangle;
            });
        }
        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::PATCH_ID)) {
            write_scalar("patch_id", "int", [&](const DEMMesh&, size_t, size_t, size_t global_triangle, bodyID_t) {
                return +triPatchID[global_triangle];
            });
        }
        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::OWNER_WILDCARD)) {
            size_t wildcard_index = 0;
            for (const auto& wildcard_name : m_owner_wildcard_names) {
                write_scalar(wildcard_name, "float",
                             [&, wildcard_index](const DEMMesh&, size_t, size_t, size_t, bodyID_t owner) {
                                 return (*ownerWildcards[wildcard_index])[owner];
                             });
                wildcard_index++;
            }
        }
        if (mesh_flags & static_cast<unsigned int>(MESH_OUTPUT_CONTENT::GEO_WILDCARD)) {
            size_t wildcard_index = 0;
            for (const auto& wildcard_name : m_geo_wildcard_names) {
                write_scalar(wildcard_name, "float",
                             [&, wildcard_index](const DEMMesh&, size_t, size_t, size_t global_triangle, bodyID_t) {
                                 return (*triWildcards[wildcard_index])[global_triangle];
                             });
                wildcard_index++;
            }
        }
    }

    ptFile << ostream.str();
}

void DEMDynamicThread::writeMeshesAsStl(std::ofstream& ptFile) {
    migrateFamilyToHost();
    migrateClumpPosInfoToHost();
    writeMeshesAsStlFromHost(ptFile);
}

void DEMDynamicThread::writeMeshesAsStlFromHost(std::ofstream& ptFile) {
    std::ostringstream ostream;

    auto ownerPosFromHost = [this](bodyID_t owner) {
        double X, Y, Z;
        voxelID_t voxel = voxelID[owner];
        subVoxelPos_t subVoxX = locX[owner];
        subVoxelPos_t subVoxY = locY[owner];
        subVoxelPos_t subVoxZ = locZ[owner];
        voxelIDToPosition<double, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel, subVoxX, subVoxY, subVoxZ, simParams->nvXp2,
                                                            simParams->nvYp2, simParams->voxelSize, simParams->l);
        return make_float3(X + simParams->LBFX, Y + simParams->LBFY, Z + simParams->LBFZ);
    };
    auto ownerOriQFromHost = [this](bodyID_t owner) {
        return make_float4(oriQx[owner], oriQy[owner], oriQz[owner], oriQw[owner]);
    };

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
            float3 ownerPos = ownerPosFromHost(mowner);
            float4 ownerOriQ = ownerOriQFromHost(mowner);
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

void DEMDynamicThread::writeMeshesAsPly(std::ofstream& ptFile, bool patch_colors) {
    migrateFamilyToHost();
    migrateClumpPosInfoToHost();
    writeMeshesAsPlyFromHost(ptFile, patch_colors);
}

void DEMDynamicThread::writeMeshesAsPlyFromHost(std::ofstream& ptFile, bool patch_colors) {
    std::ostringstream ostream;

    auto ownerPosFromHost = [this](bodyID_t owner) {
        double X, Y, Z;
        voxelID_t voxel = voxelID[owner];
        subVoxelPos_t subVoxX = locX[owner];
        subVoxelPos_t subVoxY = locY[owner];
        subVoxelPos_t subVoxZ = locZ[owner];
        voxelIDToPosition<double, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel, subVoxX, subVoxY, subVoxZ, simParams->nvXp2,
                                                            simParams->nvYp2, simParams->voxelSize, simParams->l);
        return make_float3(X + simParams->LBFX, Y + simParams->LBFY, Z + simParams->LBFZ);
    };
    auto ownerOriQFromHost = [this](bodyID_t owner) {
        return make_float4(oriQx[owner], oriQy[owner], oriQz[owner], oriQw[owner]);
    };

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
    if (patch_colors) {
        ostream << "property uchar red" << std::endl;
        ostream << "property uchar green" << std::endl;
        ostream << "property uchar blue" << std::endl;
    }
    ostream << "end_header" << std::endl;

    mesh_num = 0;
    for (const auto& mmesh : m_meshes) {
        if (!thisMeshSkip[mesh_num]) {
            bodyID_t mowner = mmesh->owner;
            float3 ownerPos = ownerPosFromHost(mowner);
            float4 ownerOriQ = ownerOriQFromHost(mowner);
            for (const auto& v : mmesh->GetCoordsVertices()) {
                float3 point = v;
                applyFrameTransformLocalToGlobal(point, ownerPos, ownerOriQ);
                ostream << point.x << " " << point.y << " " << point.z << std::endl;
            }
        }
        mesh_num++;
    }

    ostream << std::endl;
    auto hash32 = [](uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352d;
        x ^= x >> 15;
        x *= 0x846ca68b;
        x ^= x >> 16;
        return x;
    };

    mesh_num = 0;
    for (const auto& mmesh : m_meshes) {
        if (!thisMeshSkip[mesh_num]) {
            const auto& faces = mmesh->GetIndicesVertexes();
            const auto& patch_ids = mmesh->GetPatchIDs();
            bool has_patch_ids = (patch_ids.size() == faces.size());

            for (size_t fi = 0; fi < faces.size(); ++fi) {
                const auto& f = faces[fi];
                ostream << "3 " << (size_t)f.x + vertexOffset[mesh_num] << " " << (size_t)f.y + vertexOffset[mesh_num]
                        << " " << (size_t)f.z + vertexOffset[mesh_num];
                if (patch_colors) {
                    uint32_t patch_id = has_patch_ids ? static_cast<uint32_t>(patch_ids[fi]) : 0u;
                    uint32_t key = patch_id + 0x9e3779b9u * (mesh_num + 1u);
                    uint32_t h = hash32(key);
                    unsigned int r = (h >> 16) & 0xFFu;
                    unsigned int g = (h >> 8) & 0xFFu;
                    unsigned int b = h & 0xFFu;
                    ostream << " " << r << " " << g << " " << b;
                }
                ostream << std::endl;
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
        if (simParams->storeNormal) {
            DEME_DUAL_ARRAY_RESIZE(contactNormals, nContactPairs, make_float3(0));
        }
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
    DEME_DUAL_ARRAY_RESIZE(contactPatchIsland, nPatchPairs, NULL_BODYID);

    // Re-packing pointers to device now is automatic
    // Sync pointers to device can be delayed... we'll only need to do that before kernel calls
}

inline void DEMDynamicThread::unpackMyBuffer() {
    // Make a note on the contact number of the previous time step
    *solverScratchSpace.numPrevContacts = *solverScratchSpace.numContacts;
    *solverScratchSpace.numPrevPrimitiveContacts = *solverScratchSpace.numPrimitiveContacts;
    // kT's batch of produce is made with this max drift in mind
    pSchedSupport->dynamicMaxFutureDrift = (pSchedSupport->kinematicMaxFutureDrift).load();
    // DEME_DEBUG_PRINTF("dynamicMaxFutureDrift is %u", (pSchedSupport->dynamicMaxFutureDrift).load());

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

    DEME_GPU_CALL(cudaMemcpy(granData->idPrimitiveA, idPrimitiveA_buffer.data(),
                             *solverScratchSpace.numPrimitiveContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->idPrimitiveB, idPrimitiveB_buffer.data(),
                             *solverScratchSpace.numPrimitiveContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->contactTypePrimitive, contactTypePrimitive_buffer.data(),
                             *solverScratchSpace.numPrimitiveContacts * sizeof(contact_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->geomToPatchMap, geomToPatchMap_buffer.data(),
                             *solverScratchSpace.numPrimitiveContacts * sizeof(contactPairs_t),
                             cudaMemcpyDeviceToDevice));

    // Unpack separate patch ID arrays
    DEME_GPU_CALL(cudaMemcpy(granData->idPatchA, idPatchA_buffer.data(),
                             *solverScratchSpace.numContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->idPatchB, idPatchB_buffer.data(),
                             *solverScratchSpace.numContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->contactTypePatch, contactTypePatch_buffer.data(),
                             *solverScratchSpace.numContacts * sizeof(contact_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(granData->contactPatchIsland, contactPatchIsland_buffer.data(),
                             *solverScratchSpace.numContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));

    if (!solverFlags.isHistoryless) {
        // Note we don't have to use dedicated memory space for unpacking contactMapping_buffer contents, because we
        // only use it once per kT update, at the time of unpacking. So let us just use a temp vector to store it.
        size_t mapping_bytes = (*solverScratchSpace.numContacts) * sizeof(contactPairs_t);
        granData->contactMapping =
            (contactPairs_t*)solverScratchSpace.allocateTempVector("contactMapping", mapping_bytes);
        DEME_GPU_CALL(cudaMemcpy(granData->contactMapping, contactMapping_buffer.data(), mapping_bytes,
                                 cudaMemcpyDeviceToDevice));
        // std::cout << "Unpacked contactMapping: " << std::endl;
        // displayDeviceArray<contactPairs_t>(granData->contactMapping, *solverScratchSpace.numContacts);
    }
    // Prepare for kernel calls immediately after; queue this pointer-bundle refresh after the unpack copies.
    granData.toDeviceAsync(streamInfo.stream);
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
    DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_absAngVel, pCycleAngVel, simParams->nOwnerBodies * sizeof(float),
                             cudaMemcpyDeviceToDevice));

    // Send simulation metrics for kT's reference.
    DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_ts, &(simParams->dyn.h), sizeof(float), cudaMemcpyHostToDevice));
    // Note that perhapsIdealFutureDrift is non-negative, and it will be used to determine the margin size; however, if
    // scheduleHelper is instructed to have negative future drift then perhapsIdealFutureDrift no longer affects them.
    DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_maxDrift, perhapsIdealFutureDrift.getHostPointer(),
                             sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Send per-triangle max tri-tri penetration values for kT's margin computation.
    if (!simParams->meshParticlesLowPoly && simParams->nTriGM > 0) {
        DEME_GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_maxTriTriPenetration, maxTriTriPenetration.data(),
                                 (size_t)simParams->nTriGM * sizeof(float), cudaMemcpyDeviceToDevice));
    }

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

inline void DEMDynamicThread::reportLostContactDebugDetails(const LostContactDebugSnapshot& oldContactSnapshot,
                                                            const notStupidBool_t* contactSentry,
                                                            float* const* newWildcards,
                                                            size_t lostContactCount) {
    if (DEME_GET_VERBOSITY() < VERBOSITY_DEBUG || lostContactCount == 0) {
        return;
    }

    // This method is intentionally DEBUG-only. It performs blocking host copies and prints detailed identity data for
    // old live patch contacts that failed to find a migration partner in the newly received kT contact array.
    const size_t nPrevContacts = *solverScratchSpace.numPrevContacts;
    const size_t nNewContacts = *solverScratchSpace.numContacts;
    const size_t nPrevPrimitiveContacts = *solverScratchSpace.numPrevPrimitiveContacts;
    const size_t nNewPrimitiveContacts = *solverScratchSpace.numPrimitiveContacts;

    std::vector<notStupidBool_t> sentryHost(nPrevContacts);
    DEME_GPU_CALL(
        cudaMemcpy(sentryHost.data(), contactSentry, nPrevContacts * sizeof(notStupidBool_t), cudaMemcpyDeviceToHost));

    std::vector<std::string> wildcardNames(m_contact_wildcard_names.begin(), m_contact_wildcard_names.end());
    auto contactTypeName = [](contact_t type) {
        switch (type) {
            case SPHERE_SPHERE_CONTACT:
                return "SPHERE_SPHERE_CONTACT";
            case SPHERE_TRIANGLE_CONTACT:
                return "SPHERE_TRIANGLE_CONTACT";
            case SPHERE_ANALYTICAL_CONTACT:
                return "SPHERE_ANALYTICAL_CONTACT";
            case TRIANGLE_TRIANGLE_CONTACT:
                return "TRIANGLE_TRIANGLE_CONTACT";
            case TRIANGLE_ANALYTICAL_CONTACT:
                return "TRIANGLE_ANALYTICAL_CONTACT";
            case NOT_A_CONTACT:
                return "NOT_A_CONTACT";
            default:
                return "UNKNOWN_CONTACT";
        }
    };
    auto geoTypeName = [](geoType_t type) {
        switch (type) {
            case GEO_T_SPHERE:
                return "sphere";
            case GEO_T_TRIANGLE:
                return "triangle";
            case GEO_T_ANALYTICAL:
                return "analytical";
            default:
                return "unknown";
        }
    };
    auto safePatchOwner = [&](bodyID_t patchID, geoType_t type) {
        if (patchID == NULL_BODYID) {
            return NULL_BODYID;
        }
        switch (type) {
            case GEO_T_SPHERE:
                return patchID < ownerClumpBody.size() ? ownerClumpBody[patchID] : NULL_BODYID;
            case GEO_T_TRIANGLE:
                return patchID < ownerPatchMesh.size() ? ownerPatchMesh[patchID] : NULL_BODYID;
            case GEO_T_ANALYTICAL:
                return patchID < ownerAnalBody.size() ? ownerAnalBody[patchID] : NULL_BODYID;
            default:
                return NULL_BODYID;
        }
    };
    auto safeGeoOwner = [&](bodyID_t geoID, geoType_t type) {
        if (geoID == NULL_BODYID) {
            return NULL_BODYID;
        }
        switch (type) {
            case GEO_T_SPHERE:
                return geoID < ownerClumpBody.size() ? ownerClumpBody[geoID] : NULL_BODYID;
            case GEO_T_TRIANGLE:
                return geoID < ownerTriMesh.size() ? ownerTriMesh[geoID] : NULL_BODYID;
            case GEO_T_ANALYTICAL:
                return geoID < ownerAnalBody.size() ? ownerAnalBody[geoID] : NULL_BODYID;
            default:
                return NULL_BODYID;
        }
    };
    auto safeCombinedMaster = [&](bodyID_t ownerID) {
        if (ownerID == NULL_BODYID || ownerID >= ownerCombinedMaster.size()) {
            return NULL_BODYID;
        }
        return ownerCombinedMaster[ownerID];
    };
    auto printBodyID = [](std::ostream& os, bodyID_t id) -> std::ostream& {
        if (id == NULL_BODYID) {
            os << "NULL/invalid";
        } else {
            os << id;
        }
        return os;
    };
    auto printWildcardAverages = [&](const char* label, const std::vector<double>& averages, size_t count) {
        std::cout << "[DEME LOST CONTACT DEBUG] " << label << "ContactWildcardAverages(count=" << count << ")={";
        for (unsigned int wc = 0; wc < simParams->nContactWildcards; wc++) {
            if (wc > 0) {
                std::cout << ", ";
            }
            if (wc < wildcardNames.size()) {
                std::cout << wildcardNames[wc] << "#" << wc << "=" << averages[wc];
            } else {
                std::cout << "#" << wc << "=" << averages[wc];
            }
        }
        std::cout << "}\n";
    };
    auto unpackStoredDouble = [](const float3& storage) {
        static_assert(sizeof(double) == 2 * sizeof(float),
                      "Double must be exactly twice the size of float for contact-geometry debug "
                      "unpacking.");
        union {
            double d;
            float f[2];
        } converter;
        converter.f[0] = storage.x;
        converter.f[1] = storage.y;
        return converter.d;
    };

    std::vector<std::vector<float>> oldWildcardValues(simParams->nContactWildcards);
    std::vector<double> oldWildcardAverages(simParams->nContactWildcards, 0.0);
    std::vector<double> migratedNewWildcardAverages(simParams->nContactWildcards, 0.0);
    for (unsigned int wc = 0; wc < simParams->nContactWildcards; wc++) {
        oldWildcardValues[wc].resize(nPrevContacts);
        DEME_GPU_CALL(cudaMemcpy(oldWildcardValues[wc].data(), granData->contactWildcards[wc],
                                 nPrevContacts * sizeof(float), cudaMemcpyDeviceToHost));
        double oldSum = 0.0;
        for (float val : oldWildcardValues[wc]) {
            oldSum += val;
        }
        if (nPrevContacts > 0) {
            oldWildcardAverages[wc] = oldSum / static_cast<double>(nPrevContacts);
        }

        if (nNewContacts > 0) {
            std::vector<float> migratedValues(nNewContacts);
            DEME_GPU_CALL(cudaMemcpy(migratedValues.data(), newWildcards[wc], nNewContacts * sizeof(float),
                                     cudaMemcpyDeviceToHost));
            double migratedSum = 0.0;
            for (float val : migratedValues) {
                migratedSum += val;
            }
            migratedNewWildcardAverages[wc] = migratedSum / static_cast<double>(nNewContacts);
        }
    }

    std::cout << "\n[DEME LOST CONTACT DEBUG] " << lostContactCount
              << " old live patch contact(s) had no migration partner at sim time " << simParams->dyn.timeElapsed
              << ". This detailed dump is DEBUG-verbosity only.\n";
    std::cout << "[DEME LOST CONTACT DEBUG] oldPatchContacts=" << nPrevContacts << ", newPatchContacts=" << nNewContacts
              << ", oldPrimitiveContacts=" << nPrevPrimitiveContacts
              << ", newPrimitiveContacts=" << nNewPrimitiveContacts
              << ", dTStamp=" << pSchedSupport->currentStampOfDynamic.load()
              << ", lastKinematicIngredStamp=" << pSchedSupport->stampLastDynamicUpdateProdDate.load()
              << ", async=" << solverFlags.isAsync << "\n";
    printWildcardAverages("oldAll", oldWildcardAverages, nPrevContacts);
    printWildcardAverages("migratedNewAll", migratedNewWildcardAverages, nNewContacts);

    if (oldContactSnapshot.nPatchContacts != nPrevContacts ||
        oldContactSnapshot.nPrimitiveContacts != nPrevPrimitiveContacts) {
        std::cout << "[DEME LOST CONTACT DEBUG] WARNING: old-contact snapshot size mismatch: "
                  << "snapshotPatch=" << oldContactSnapshot.nPatchContacts << ", sentryPatch=" << nPrevContacts
                  << ", snapshotPrimitive=" << oldContactSnapshot.nPrimitiveContacts
                  << ", expectedPrimitive=" << nPrevPrimitiveContacts << "\n";
    }

    for (size_t oldCnt = 0; oldCnt < sentryHost.size(); oldCnt++) {
        if (!sentryHost[oldCnt]) {
            continue;
        }
        bool havePatchIdentity = false;
        contact_t patchType = NOT_A_CONTACT;
        geoType_t patchTypeA = GEO_T_SPHERE;
        geoType_t patchTypeB = GEO_T_SPHERE;
        bodyID_t patchA = NULL_BODYID;
        bodyID_t patchB = NULL_BODYID;
        bodyID_t patchOwnerA = NULL_BODYID;
        bodyID_t patchOwnerB = NULL_BODYID;

        std::cout << "[DEME LOST CONTACT DEBUG] lostOldPatchIndex=" << oldCnt;
        if (oldCnt < oldContactSnapshot.contactTypePatch.size() && oldCnt < oldContactSnapshot.idPatchA.size() &&
            oldCnt < oldContactSnapshot.idPatchB.size() && oldCnt < oldContactSnapshot.contactPatchIsland.size()) {
            havePatchIdentity = true;
            patchType = oldContactSnapshot.contactTypePatch[oldCnt];
            patchTypeA = decodeTypeA<contact_t, geoType_t>(patchType);
            patchTypeB = decodeTypeB<contact_t, geoType_t>(patchType);
            patchA = oldContactSnapshot.idPatchA[oldCnt];
            patchB = oldContactSnapshot.idPatchB[oldCnt];
            patchOwnerA = safePatchOwner(patchA, patchTypeA);
            patchOwnerB = safePatchOwner(patchB, patchTypeB);

            std::cout << ", type=" << contactTypeName(patchType) << "(" << static_cast<unsigned int>(patchType) << ")"
                      << ", patchA=" << patchA << "[" << geoTypeName(patchTypeA) << "]"
                      << ", patchB=" << patchB << "[" << geoTypeName(patchTypeB) << "]"
                      << ", ownerA=";
            printBodyID(std::cout, patchOwnerA);
            std::cout << ", ownerB=";
            printBodyID(std::cout, patchOwnerB);
            std::cout << ", combinedMasterA=";
            printBodyID(std::cout, safeCombinedMaster(patchOwnerA));
            std::cout << ", combinedMasterB=";
            printBodyID(std::cout, safeCombinedMaster(patchOwnerB));
            std::cout << ", patchIsland=";
            printBodyID(std::cout, oldContactSnapshot.contactPatchIsland[oldCnt]);
        } else {
            std::cout << ", old patch identity unavailable";
        }

        std::cout << ", wildcards={";
        for (unsigned int wc = 0; wc < simParams->nContactWildcards; wc++) {
            if (wc > 0) {
                std::cout << ", ";
            }
            float wildcardVal = oldCnt < oldWildcardValues[wc].size() ? oldWildcardValues[wc][oldCnt] : 0.f;
            if (wc < wildcardNames.size()) {
                std::cout << wildcardNames[wc] << "#" << wc << "=" << wildcardVal;
            } else {
                std::cout << "#" << wc << "=" << wildcardVal;
            }
        }
        std::cout << "}\n";

        size_t primitiveContributors = 0;
        size_t penetrationSamples = 0;
        size_t positivePenetrationSamples = 0;
        double minPenetration = 0.0;
        double maxPenetration = 0.0;
        double sumPenetration = 0.0;
        double sumArea = 0.0;
        double maxArea = 0.0;
        for (size_t prim = 0; prim < oldContactSnapshot.geomToPatchMap.size(); prim++) {
            if (oldContactSnapshot.geomToPatchMap[prim] != oldCnt) {
                continue;
            }
            primitiveContributors++;
            const bool hasPenetration = prim < oldContactSnapshot.primitivePenetrationStorage.size();
            const bool hasArea = prim < oldContactSnapshot.primitiveAreaStorage.size();
            double penetration = 0.0;
            double area = 0.0;
            if (hasPenetration) {
                penetration = unpackStoredDouble(oldContactSnapshot.primitivePenetrationStorage[prim]);
                if (penetrationSamples == 0) {
                    minPenetration = penetration;
                    maxPenetration = penetration;
                } else {
                    minPenetration = std::min(minPenetration, penetration);
                    maxPenetration = std::max(maxPenetration, penetration);
                }
                sumPenetration += penetration;
                penetrationSamples++;
                if (penetration > 0.0) {
                    positivePenetrationSamples++;
                }
            }
            if (hasArea) {
                area = unpackStoredDouble(oldContactSnapshot.primitiveAreaStorage[prim]);
                sumArea += area;
                maxArea = std::max(maxArea, area);
            }
            if (prim < oldContactSnapshot.contactTypePrimitive.size() &&
                prim < oldContactSnapshot.idPrimitiveA.size() && prim < oldContactSnapshot.idPrimitiveB.size()) {
                contact_t primType = oldContactSnapshot.contactTypePrimitive[prim];
                geoType_t primTypeA = decodeTypeA<contact_t, geoType_t>(primType);
                geoType_t primTypeB = decodeTypeB<contact_t, geoType_t>(primType);
                bodyID_t primA = oldContactSnapshot.idPrimitiveA[prim];
                bodyID_t primB = oldContactSnapshot.idPrimitiveB[prim];
                bodyID_t primOwnerA = safeGeoOwner(primA, primTypeA);
                bodyID_t primOwnerB = safeGeoOwner(primB, primTypeB);
                std::cout << "[DEME LOST CONTACT DEBUG]   primitiveIndex=" << prim
                          << ", primitiveType=" << contactTypeName(primType) << "("
                          << static_cast<unsigned int>(primType) << ")"
                          << ", geomA=" << primA << "[" << geoTypeName(primTypeA) << "]"
                          << ", geomB=" << primB << "[" << geoTypeName(primTypeB) << "]"
                          << ", ownerA=";
                printBodyID(std::cout, primOwnerA);
                std::cout << ", ownerB=";
                printBodyID(std::cout, primOwnerB);
                std::cout << ", patchMap=" << oldContactSnapshot.geomToPatchMap[prim];
            } else {
                std::cout << "[DEME LOST CONTACT DEBUG]   primitiveIndex=" << prim
                          << ", primitive identity unavailable, patchMap=" << oldContactSnapshot.geomToPatchMap[prim];
            }
            if (hasPenetration) {
                std::cout << ", penetration=" << penetration;
            } else {
                std::cout << ", penetration=unavailable";
            }
            if (hasArea) {
                std::cout << ", area=" << area;
            } else {
                std::cout << ", area=unavailable";
            }
            std::cout << "\n";
        }
        if (primitiveContributors == 0) {
            std::cout << "[DEME LOST CONTACT DEBUG]   no old primitive contributors pointed to lost "
                         "patch index "
                      << oldCnt << "\n";
        } else if (penetrationSamples > 0) {
            const double avgPenetration = sumPenetration / static_cast<double>(penetrationSamples);
            std::cout << "[DEME LOST CONTACT DEBUG]   primitivePenetrationSummary(count=" << penetrationSamples
                      << ", positive=" << positivePenetrationSamples << ", min=" << minPenetration
                      << ", max=" << maxPenetration << ", avg=" << avgPenetration << ", areaSum=" << sumArea
                      << ", areaMax=" << maxArea << ")\n";
        }
    }
}

inline void DEMDynamicThread::migrateEnduringContacts(const LostContactDebugSnapshot& oldContactSnapshot) {
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
            if (*lostContact) {
                if (solverFlags.isAsync) {
                    // This prints when verbosity is METRIC or higher. Detailed records are DEBUG-only.
                    DEME_STATUS(
                        "ALIVE_CONTACT_NOT_DETECTED",
                        "%zu contacts were active at time %.9g on dT, but they are not detected on kT, therefore being "
                        "removed unexpectedly!",
                        *lostContact, simParams->dyn.timeElapsed);
                }
                reportLostContactDebugDetails(oldContactSnapshot, contactSentry, newWildcards, *lostContact);
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

    // granData may have changed in some of the earlier steps. Queue the pointer-bundle refresh before the following
    // kernels instead of forcing a host barrier here.
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
    DEME_GPU_DEBUG_SYNC(streamInfo.stream);
}

inline void DEMDynamicThread::dispatchPatchBasedForceCorrections(
    const ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>>& typeStartCountPrimitiveMap,
    const ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>>& typeStartCountPatchMap,
    const ContactTypeMap<std::vector<std::pair<std::shared_ptr<JitHelper::CachedProgram>, std::string>>>&
        typeKernelMap) {
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

            // Vote for the contact direction; voting power depends on the contact area.
            // Multiple contact quantities (areas, penetrations, contact points) are computed by a single fused kernel,
            // then reduced independently per contact type.
            if (countPrimitive > 0) {
                // geomToPatchMap maps each primitive contact to its patch pair. Contacts are type-sorted, so the
                // segment beginning at startOffsetPrimitive can be used directly as the reduce key stream.
                contactPairs_t* keys = granData->geomToPatchMap + startOffsetPrimitive;

                // Allocate arrays for reduce-by-key results (uniqueKeys uses contactPairs_t, not patchIDPair_t)
                contactPairs_t* uniqueKeys = (contactPairs_t*)solverScratchSpace.allocateTempVector(
                    "uniqueKeys", countPrimitive * sizeof(contactPairs_t));
                solverScratchSpace.allocateDualStruct("numUniqueKeys");
                size_t* numUniqueKeys = solverScratchSpace.getDualStructDevice("numUniqueKeys");

                // Step 1: Prepare weighted normals for voting (normal * area).
                float3* weightedNormals =
                    (float3*)solverScratchSpace.allocateTempVector("weightedNormals", countPrimitive * sizeof(float3));
                prepareWeightedNormalsForVoting(&granData, weightedNormals, startOffsetPrimitive, countPrimitive,
                                                streamInfo.stream);

                // Step 2: Reduce-by-key for weighted normals (sum)
                // The keys are geomToPatchMap values (contactPairs_t), which group primitives by patch pair
                float3* votedWeightedNormals =
                    (float3*)solverScratchSpace.allocateTempVector("votedWeightedNormals", countPatch * sizeof(float3));
                cubSumReduceByKey<contactPairs_t, float3>(keys, uniqueKeys, weightedNormals, votedWeightedNormals,
                                                          numUniqueKeys, countPrimitive, streamInfo.stream,
                                                          solverScratchSpace);
                solverScratchSpace.finishUsingTempVector("weightedNormals");
                DEME_DEBUG_EXEC({
                    solverScratchSpace.syncDualStructDeviceToHost("numUniqueKeys");
                    size_t numUniqueKeysHost = *(solverScratchSpace.getDualStructHost("numUniqueKeys"));
                    if (numUniqueKeysHost != countPatch) {
                        DEME_ERROR(
                            "Patch-based contact voting produced %zu unique patch pairs, but expected %zu pairs for "
                            "contact type %d!",
                            numUniqueKeysHost, countPatch, contact_type);
                    }
                });

                // Step 3: Normalize the voted normals by total area and scatter back to a temp array.
                float3* votedNormals =
                    (float3*)solverScratchSpace.allocateTempVector("votedNormals", countPatch * sizeof(float3));
                normalizeAndScatterVotedNormals(votedWeightedNormals, votedNormals, countPatch, streamInfo.stream);
                solverScratchSpace.finishUsingTempVector("votedWeightedNormals");
                // displayDeviceFloat3(votedNormals, countPatch);

                // Step 4: Fused kernel: compute projected areas, projected penetrations, contact-point weights, and
                // weighted contact points in one pass.
                double* primitiveProjectedAreas = (double*)solverScratchSpace.allocateTempVector(
                    "primitiveProjectedAreas", countPrimitive * sizeof(double));
                double* primitiveProjectedPens = (double*)solverScratchSpace.allocateTempVector(
                    "primitiveProjectedPens", countPrimitive * sizeof(double));
                double* primitiveWeights =
                    (double*)solverScratchSpace.allocateTempVector("primitiveWeights", countPrimitive * sizeof(double));
                double3* primitiveWeightedCPs = (double3*)solverScratchSpace.allocateTempVector(
                    "primitiveWeightedCPs", countPrimitive * sizeof(double3));
                computePerPrimitiveWeightedQuantities(
                    &granData, votedNormals, keys, primitiveProjectedAreas, primitiveProjectedPens, primitiveWeights,
                    primitiveWeightedCPs, startOffsetPrimitive, startOffsetPatch, countPrimitive, streamInfo.stream);

                // Step 5a: Sum-reduce-by-key for total projected areas per patch.
                double* totalProjAreas =
                    (double*)solverScratchSpace.allocateTempVector("totalProjAreas", countPatch * sizeof(double));
                cubSumReduceByKey<contactPairs_t, double>(keys, uniqueKeys, primitiveProjectedAreas, totalProjAreas,
                                                          numUniqueKeys, countPrimitive, streamInfo.stream,
                                                          solverScratchSpace);
                solverScratchSpace.finishUsingTempVector("primitiveProjectedAreas");

                // Step 5b: Max-reduce-by-key for per-patch max projected penetration.
                // finalPen is the max (deepest) projected penetration among all primitives in the patch.
                double* maxProjPens =
                    (double*)solverScratchSpace.allocateTempVector("maxProjPens", countPatch * sizeof(double));
                cubMaxReduceByKey<contactPairs_t, double>(keys, uniqueKeys, primitiveProjectedPens, maxProjPens,
                                                          numUniqueKeys, countPrimitive, streamInfo.stream,
                                                          solverScratchSpace);
                solverScratchSpace.finishUsingTempVector("primitiveProjectedPens");

                // Step 5c: Sum-reduce-by-key for total weights per patch (used for contact point averaging).
                double* totalWeights =
                    (double*)solverScratchSpace.allocateTempVector("totalWeights", countPatch * sizeof(double));
                cubSumReduceByKey<contactPairs_t, double>(keys, uniqueKeys, primitiveWeights, totalWeights,
                                                          numUniqueKeys, countPrimitive, streamInfo.stream,
                                                          solverScratchSpace);

                // Step 5d: Sum-reduce-by-key for total weighted contact points per patch.
                double3* totalWeightedCPs =
                    (double3*)solverScratchSpace.allocateTempVector("totalWeightedCPs", countPatch * sizeof(double3));
                cubSumReduceByKey<contactPairs_t, double3>(keys, uniqueKeys, primitiveWeightedCPs, totalWeightedCPs,
                                                           numUniqueKeys, countPrimitive, streamInfo.stream,
                                                           solverScratchSpace);
                solverScratchSpace.finishUsingTempVector("primitiveWeightedCPs");

                // Step 6a: Extract primitive penetrations for the zero-area fallback (max-negative reduce).
                double* primitivePenetrations = (double*)solverScratchSpace.allocateTempVector(
                    "primitivePenetrations", countPrimitive * sizeof(double));
                extractPrimitivePenetrations(&granData, primitivePenetrations, startOffsetPrimitive, countPrimitive,
                                             streamInfo.stream);

                // Step 6b: Max-negative-reduce-by-key to get max negative penetration per patch.
                // This finds the largest negative value (smallest absolute value among negatives).
                // Positive values are treated as very negative to indicate invalid/non-physical state.
                double* maxPenetrations =
                    (double*)solverScratchSpace.allocateTempVector("maxPenetrations", countPatch * sizeof(double));
                cubMaxNegativeReduceByKey<contactPairs_t, double>(keys, uniqueKeys, primitivePenetrations,
                                                                  maxPenetrations, numUniqueKeys, countPrimitive,
                                                                  streamInfo.stream, solverScratchSpace);
                solverScratchSpace.finishUsingTempVector("primitivePenetrations");

                // Step 6c: Find max-penetration primitives for zero-area patches and extract their normals,
                // penetrations, and contact points.
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

                // Clean up: uniqueKeys and numUniqueKeys are no longer needed after all reductions.
                // keys is a direct pointer into granData->geomToPatchMap and does not need cleanup.
                solverScratchSpace.finishUsingTempVector("uniqueKeys");
                solverScratchSpace.finishUsingDualStruct("numUniqueKeys");

                // Step 7: Finalize patch results.
                // finalPen = max projected penetration (maxProjPens), not an average.
                // finalCP  = weight-averaged contact point (weight = projArea * projPen).
                // Zero-area patches use the max-penetration primitive's fallback values.
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
                finalizePatchResults(totalProjAreas, maxProjPens, totalWeights, votedNormals, totalWeightedCPs,
                                     zeroAreaNormals, zeroAreaPenetrations, zeroAreaContactPoints, finalAreas,
                                     finalNormals, finalPenetrations.data(), finalContactPoints, countPatch,
                                     streamInfo.stream);
                solverScratchSpace.finishUsingTempVector("zeroAreaNormals");
                solverScratchSpace.finishUsingTempVector("zeroAreaPenetrations");
                solverScratchSpace.finishUsingTempVector("zeroAreaContactPoints");
                solverScratchSpace.finishUsingTempVector("totalProjAreas");
                solverScratchSpace.finishUsingTempVector("maxProjPens");
                solverScratchSpace.finishUsingTempVector("totalWeightedCPs");

                // Now we have:
                // - finalAreas: final contact area per patch pair (countPatch elements)
                // - finalNormals: final normal direction per patch pair (countPatch elements)
                // - finalPenetrations: final penetration depth per patch pair (countPatch elements)
                // - finalContactPoints: final contact point per patch pair (countPatch elements)
                // These can be used for subsequent force calculations
                // std::cout << "Patch-based contact penetration, area, normal, contact point for contact type "
                //           << (int)contact_type << ":" << std::endl;
                // displayDeviceArray<double>(finalPenetrations.data(), countPatch);
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
                } else {
                    DEME_ERROR("Patch contact type %d has no associated force kernel to execute!", contact_type);
                }
                DEME_GPU_DEBUG_SYNC(streamInfo.stream);

                // Optional per-triangle wear/tracking diagnostics. This runs after patch force correction so
                // granData->contactForces contains patch-level forces, while primitiveWeights/totalWeights still
                // describe how each primitive contributed to its patch contact.
                if (triPVTrackingEnabled && triPVNumTrackedTriangles > 0) {
                    float* patchNormalForce =
                        (float*)solverScratchSpace.allocateTempVector("patchNormalForce", countPatch * sizeof(float));
                    float* patchSlipSpeed =
                        (float*)solverScratchSpace.allocateTempVector("patchSlipSpeed", countPatch * sizeof(float));
                    computePatchPVScalars(&simParams, &granData, finalNormals, finalContactPoints, startOffsetPatch,
                                          countPatch, patchNormalForce, patchSlipSpeed, streamInfo.stream);
                    accumulateTrianglePVFromPatchContacts(
                        &simParams, &granData, keys, primitiveWeights, totalWeights, patchNormalForce, patchSlipSpeed,
                        startOffsetPrimitive, startOffsetPatch, countPrimitive, triPVGlobalTriToLocal.device(),
                        triPVAccumP.device(), triPVAccumPV.device(), streamInfo.stream);
                    solverScratchSpace.finishUsingTempVector("patchNormalForce");
                    solverScratchSpace.finishUsingTempVector("patchSlipSpeed");
                }

                // Final clean up
                solverScratchSpace.finishUsingTempVector("totalWeights");
                solverScratchSpace.finishUsingTempVector("primitiveWeights");
                solverScratchSpace.finishUsingTempVector("votedNormals");
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
    // Reset force (acceleration) arrays for this time step
    size_t nContactPairs = *solverScratchSpace.numContacts;
    size_t nPrimitiveContactPairs = *solverScratchSpace.numPrimitiveContacts;
    if (!simParams->meshParticlesLowPoly && simParams->nTriGM > 0) {
        DEME_GPU_CALL(cudaMemset(maxTriTriPenetration.data(), 0, (size_t)simParams->nTriGM * sizeof(float)));
    }

    timers.StartGpuTimer("Clear force array", streamInfo.stream);
    {
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
            timers.StartGpuTimer("Optional force reduction", streamInfo.stream);
            // Reflect those body-wise forces on their owner clumps
            size_t blocks_needed_for_contacts =
                (nContactPairs + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
            // This does both acc and ang acc
            collect_force_kernels->kernel("forceToAcc")
                .instantiate()
                .configure(dim3(blocks_needed_for_contacts), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
                .launch(&simParams, &granData, nContactPairs);
            DEME_GPU_DEBUG_SYNC(streamInfo.stream);
            // displayDeviceArray<float>(granData->aZ, simParams->nOwnerBodies);
            // displayDeviceFloat3(granData->contactForces, nContactPairs);
            // std::cout << nContactPairs << std::endl;
            timers.StopGpuTimer("Optional force reduction", streamInfo.stream);
        }
    }

    if (simParams->nCombinedOwners > 0) {
        // Fold non-master member accelerations into their master before integration. This also covers prescribed
        // accelerations in no-contact steps because it runs outside the contact-count branch.
        timers.StartGpuTimer("Optional force reduction", streamInfo.stream);
        constexpr unsigned int COMBINED_OWNER_AGGREGATION_BLOCK = 512;
        const size_t blocks_needed_for_owners =
            (simParams->nOwnerBodies + COMBINED_OWNER_AGGREGATION_BLOCK - 1) / COMBINED_OWNER_AGGREGATION_BLOCK;
        collect_force_kernels->kernel("aggregateCombinedOwnersAcc")
            .instantiate()
            .configure(dim3(blocks_needed_for_owners), dim3(COMBINED_OWNER_AGGREGATION_BLOCK), 0, streamInfo.stream)
            .launch(&simParams, &granData, simParams->nOwnerBodies);
        DEME_GPU_DEBUG_SYNC(streamInfo.stream);
        timers.StopGpuTimer("Optional force reduction", streamInfo.stream);
    }

    finalizeTrianglePVWindowStep();
}

void DEMDynamicThread::finalizeTrianglePVWindowStep() {
    if (!triPVTrackingEnabled || triPVNumTrackedTriangles == 0) {
        return;
    }
    triPVWindowSteps++;
}

inline void DEMDynamicThread::integrateOwnerMotions() {
    timers.StartGpuTimer("Integration", streamInfo.stream);
    size_t blocks_needed_for_clumps =
        (simParams->nOwnerBodies + DEME_NUM_BODIES_PER_BLOCK - 1) / DEME_NUM_BODIES_PER_BLOCK;
    integrator_kernels->kernel("integrateOwners")
        .instantiate()
        .configure(dim3(blocks_needed_for_clumps), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
        .launch(&simParams, &granData, (double)simParams->dyn.timeElapsed);
    DEME_GPU_DEBUG_SYNC(streamInfo.stream);

    if (simParams->nCombinedOwners > 0) {
        // Re-impose each non-master member from the integrated master state and its fixed relative transform.
        constexpr unsigned int COMBINED_OWNER_REIMPOSITION_BLOCK = 512;
        const size_t blocks_needed_for_owners =
            (simParams->nOwnerBodies + COMBINED_OWNER_REIMPOSITION_BLOCK - 1) / COMBINED_OWNER_REIMPOSITION_BLOCK;
        collect_force_kernels->kernel("reimposeCombinedOwners")
            .instantiate()
            .configure(dim3(blocks_needed_for_owners), dim3(COMBINED_OWNER_REIMPOSITION_BLOCK), 0, streamInfo.stream)
            .launch(&simParams, &granData, simParams->nOwnerBodies);
        DEME_GPU_DEBUG_SYNC(streamInfo.stream);
    }
    timers.StopGpuTimer("Integration", streamInfo.stream);
}

inline void DEMDynamicThread::routineChecks() {
    if (solverFlags.canFamilyChangeOnDevice) {
        size_t blocks_needed_for_clumps =
            (simParams->nOwnerBodies + DEME_NUM_MODERATORS_PER_BLOCK - 1) / DEME_NUM_MODERATORS_PER_BLOCK;
        mod_kernels->kernel("applyFamilyChanges")
            .instantiate()
            .configure(dim3(blocks_needed_for_clumps), dim3(DEME_NUM_MODERATORS_PER_BLOCK), 0, streamInfo.stream)
            .launch(&simParams, &granData, simParams->nOwnerBodies);
        DEME_GPU_DEBUG_SYNC(streamInfo.stream);
    }
}

inline void DEMDynamicThread::determineSysVel() {
    // Get linear velocity
    pCycleVel = approxVelFunc->dT_GetDeviceValues();
    // Get angular velocity magnitude
    pCycleAngVel = approxAngVelFunc->dT_GetDeviceValues();
}

inline void DEMDynamicThread::unpack_impl() {
    LostContactDebugSnapshot oldContactSnapshot;
    if (!solverFlags.isHistoryless && DEME_GET_VERBOSITY() >= VERBOSITY_DEBUG) {
        oldContactSnapshot.nPatchContacts = *solverScratchSpace.numContacts;
        oldContactSnapshot.nPrimitiveContacts = *solverScratchSpace.numPrimitiveContacts;

        // kT unpack overwrites dT's contact identity arrays before history migration checks the old wildcard sentry.
        // DEBUG lost-contact diagnostics therefore snapshot the old identities before receiving the new contact arrays.
        // The sync is DEBUG-only: it makes the host-side snapshot a reliable diagnostic without changing normal runs.
        DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        auto copySnapshot = [](auto& dst, auto* src, size_t count) {
            dst.resize(count);
            if (count > 0) {
                DEME_GPU_CALL(cudaMemcpy(dst.data(), src, count * sizeof(*dst.data()), cudaMemcpyDeviceToHost));
            }
        };
        copySnapshot(oldContactSnapshot.idPatchA, idPatchA.data(), oldContactSnapshot.nPatchContacts);
        copySnapshot(oldContactSnapshot.idPatchB, idPatchB.data(), oldContactSnapshot.nPatchContacts);
        copySnapshot(oldContactSnapshot.contactTypePatch, contactTypePatch.data(), oldContactSnapshot.nPatchContacts);
        copySnapshot(oldContactSnapshot.contactPatchIsland, contactPatchIsland.data(),
                     oldContactSnapshot.nPatchContacts);
        copySnapshot(oldContactSnapshot.idPrimitiveA, idPrimitiveA.data(), oldContactSnapshot.nPrimitiveContacts);
        copySnapshot(oldContactSnapshot.idPrimitiveB, idPrimitiveB.data(), oldContactSnapshot.nPrimitiveContacts);
        copySnapshot(oldContactSnapshot.contactTypePrimitive, contactTypePrimitive.data(),
                     oldContactSnapshot.nPrimitiveContacts);
        copySnapshot(oldContactSnapshot.geomToPatchMap, geomToPatchMap.data(), oldContactSnapshot.nPrimitiveContacts);
        copySnapshot(oldContactSnapshot.primitivePenetrationStorage, contactPointGeometryA.data(),
                     oldContactSnapshot.nPrimitiveContacts);
        copySnapshot(oldContactSnapshot.primitiveAreaStorage, contactPointGeometryB.data(),
                     oldContactSnapshot.nPrimitiveContacts);
    }

    {
        // Acquire lock and use the content of the dynamic-owned transfer buffer
        std::lock_guard<std::mutex> lock(pSchedSupport->dynamicOwnedBuffer_AccessCoordination);
        unpackMyBuffer();
        // Leave myself a mental note that I just obtained new produce from kT
        contactPairArr_isFresh = true;
        // pSchedSupport->schedulingStats.nDynamicReceives++;
    }
    // dT got the produce, now mark its buffer to be no longer fresh.
    pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh.store(false, std::memory_order_release);
    // Used for inspecting on average how stale kT's produce is.
    const int64_t current_dynamic_stamp = (pSchedSupport->currentStampOfDynamic).load();
    const int64_t previous_contact_prod_stamp = (pSchedSupport->stampLastDynamicUpdateProdDate).load();
    pSchedSupport->schedulingStats.accumKinematicLagSteps += current_dynamic_stamp - previous_contact_prod_stamp;
    if (previous_contact_prod_stamp >= 0 && current_dynamic_stamp >= previous_contact_prod_stamp) {
        // The contact set just replaced had to cover this many dT steps. This is the safest signal for future-drift
        // tuning because it measures the actual two-thread handoff, not just kT's isolated runtime.
        const int64_t covered_drift = current_dynamic_stamp - previous_contact_prod_stamp;
        futureDriftRegulator.RecordContactSetCoverage(static_cast<unsigned int>(
            std::min<int64_t>(covered_drift, static_cast<int64_t>(solverFlags.upperBoundFutureDrift))));
    }
    // dT needs to know how fresh the contact pair info is, and that is determined by when kT received this batch of
    // ingredients.
    pSchedSupport->stampLastDynamicUpdateProdDate = (pSchedSupport->kinematicIngredProdDateStamp).load();

    // If this is a history-based run, then when contacts are received, we need to migrate the contact
    // history info, to match the structure of the new contact array
    if (!solverFlags.isHistoryless) {
        migrateEnduringContacts(oldContactSnapshot);
    }

    // With unpacking finished, contactMapping temp array is no longer needed
    solverScratchSpace.finishUsingTempVector("contactMapping");

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

inline void DEMDynamicThread::ifProduceFreshThenUseIt() {
    if (pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh.load(std::memory_order_acquire)) {
        unpack_impl();
    }
}

inline void DEMDynamicThread::calibrateParams() {
    // Unpacking is done; now we can use temp arrays again to derive max velocity and send to kT
    determineSysVel();  // This will set pCycleVel and pCycleAngVel

    if (solverFlags.autoUpdateFreq) {
        if (futureDriftRegulator.HasContactSetCoverageSample()) {
            *perhapsIdealFutureDrift = futureDriftRegulator.Recommend(
                *perhapsIdealFutureDrift, solverFlags.targetDriftMultipleOfAvg, solverFlags.targetDriftMoreThanAvg,
                solverFlags.upperBoundFutureDrift, FUTURE_DRIFT_TWEAK_STEP_SIZE);

            DEME_DEBUG_PRINTF("Observed covered future drift is %u",
                              futureDriftRegulator.LastObservedContactSetCoverage());
            DEME_DEBUG_PRINTF("Current future drift is %u", *perhapsIdealFutureDrift);
        }
    }
    // Actually, perhapsIdealFutureDrift seems to have no need to be on device... but I made it a DualStruct anyway
}

inline void DEMDynamicThread::ifProduceFreshThenUseItAndSendNewOrder() {
    if (pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh.load(std::memory_order_acquire)) {
        timers.GetTimer("Unpack updates from kT").start();
        unpack_impl();
        timers.GetTimer("Unpack updates from kT").stop();

        timers.GetTimer("Send to kT buffer").start();
        // Acquire lock and refresh the work order for the kinematic
        {
            // Scheduling safety invariant: consuming one kT product and posting the next work order stay coupled. The
            // future-drift regulator may change the margin command, but it must not defer this send; deferring it lets
            // dT and kT span multiple unmatched contact arrays and breaks contact-history migration.
            calibrateParams();
            std::lock_guard<std::mutex> lock(pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
            sendToTheirBuffer();
        }
        pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh.store(true, std::memory_order_release);
        pSchedSupport->schedulingStats.nKinematicUpdates++;

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
                determineSysVel();  // This will set pCycleVel and pCycleAngVel
                std::lock_guard<std::mutex> lock(pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
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
                ifProduceFreshThenUseIt();
            }
        }

        for (double cycle = 0.0; cycle < cycleDuration; cycle += (double)(simParams->dyn.h)) {
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
                while (!pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh.load(std::memory_order_acquire)) {
                    // Loop to avoid spurious wakeups
                    pSchedSupport->cv_DynamicCanProceed.wait(lock);
                }
                pSchedSupport->schedulingStats.nTimesDynamicHeldBack++;
                // If dT waits, it is penalized, since waiting means double-wait, very bad.
                if (solverFlags.autoUpdateFreq) {
                    *perhapsIdealFutureDrift = futureDriftRegulator.BumpAfterWait(
                        *perhapsIdealFutureDrift, solverFlags.upperBoundFutureDrift, FUTURE_DRIFT_TWEAK_STEP_SIZE);
                }
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

                integrateOwnerMotions();

                timers.AccumulateGpuTimer("Clear force array");
                timers.AccumulateGpuTimer("Calculate contact forces");
                timers.AccumulateGpuTimer("Optional force reduction");
                timers.AccumulateGpuTimer("Integration");

                step_accepted = true;
            } while ((!solverFlags.isStepConst) || (!step_accepted));

            //// TODO: make changes for variable time step size cases
            // Keep the host copy authoritative for API queries. integrateOwners also writes the same next time into
            // device simParams so downstream stream work can observe the advanced time before this host-to-device
            // refresh executes.
            simParams->dyn.timeElapsed += (double)simParams->dyn.h;
            simParams.syncMemberToDeviceAsync<double>(
                offsetof(DEMSimParams, dyn) + offsetof(DEMSimParamsDynamic, timeElapsed), streamInfo.stream);

            // Required step-completion barrier.
            //
            // dT/kT scheduling treats currentStampOfDynamic, nTotalSteps, and the kT handoff buffers as facts about
            // completed dynamics steps. CUDA launches above are asynchronous, so without this barrier the CPU thread
            // can advance those counters and hand stale positions/velocities to kT while the GPU is still finishing
            // this step. That makes kT derive contact margins from an inflated future drift, which is especially
            // dangerous for mesh contacts because enlarged triangle margins can explode the primitive contact count.
            //
            // This synchronization is therefore part of the solver's correctness contract, not debug error checking.
            DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

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
    // Reset dT stats variables, making ready for next user call
    pSchedSupport->dynamicDone = false;
    contactPairArr_isFresh = true;
    futureDriftRegulator.Clear();

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
        integrator_kernels = std::make_shared<JitHelper::CachedProgram>(std::move(JitHelper::buildProgram(
            "DEMIntegrationKernels", JitHelper::KERNEL_DIR / "DEMIntegrationKernels.cu", Subs, JitifyOptions)));
    }
    // Then kernels that make on-the-fly changes to solver data
    {
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

void DEMDynamicThread::prewarmKernels() {
    // Prewarm force, integration, and moderation kernels so cached JIT artifacts are loaded before the first dynamics
    // step. This is especially noticeable for mesh contacts, whose primitive and patch kernels may otherwise compile at
    // first contact.
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
        collect_force_kernels->kernel("aggregateCombinedOwnersAcc").instantiate();
        collect_force_kernels->kernel("reimposeCombinedOwners").instantiate();
    }
    if (integrator_kernels) {
        integrator_kernels->kernel("integrateOwners").instantiate();
    }
    if (mod_kernels && solverFlags.canFamilyChangeOnDevice) {
        mod_kernels->kernel("applyFamilyChanges").instantiate();
    }
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
    DEME_GPU_DEBUG_SYNC(streamInfo.stream);

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
    for (unsigned int i = 0; i < triWildcards.size(); i++) {
        triWildcards[i].reset();
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

size_t DEMDynamicThread::getOwnerContactForcesToDevice(const std::vector<bodyID_t>& ownerIDs,
                                                       float3* points,
                                                       float3* forces,
                                                       float3* torques,
                                                       size_t capacity,
                                                       int destination_device,
                                                       bool need_torque,
                                                       bool torque_in_local) {
    ScopedCudaDevice device_scope(streamInfo.device);
    if (ownerIDs.empty()) {
        DEME_ERROR("Contact-force device retrieval requires at least one owner ID.");
    }
    for (bodyID_t owner : ownerIDs) {
        if (owner >= simParams->nOwnerBodies) {
            DEME_ERROR("Contact-force device retrieval owner ID %zu exceeds the %zu owners in the simulation.",
                       (size_t)owner, (size_t)simParams->nOwnerBodies);
        }
    }
    const size_t num_contacts = *solverScratchSpace.numContacts;
    if (capacity < num_contacts) {
        DEME_ERROR(
            "Contact-force device retrieval needs capacity for at least %zu entries (the current contact count), but "
            "the destination capacity is %zu.",
            num_contacts, capacity);
    }

    const size_t full_bytes = num_contacts * sizeof(float3);
    DEME_GPU_CALL(device_data::ValidateOutputPointer(points, full_bytes, destination_device));
    DEME_GPU_CALL(device_data::ValidateOutputPointer(forces, full_bytes, destination_device));
    if (need_torque) {
        DEME_GPU_CALL(device_data::ValidateOutputPointer(torques, full_bytes, destination_device));
    }
    if (num_contacts == 0) {
        return 0;
    }

    solverScratchSpace.allocateDualArray("device_contact_owner_ids", ownerIDs.size() * sizeof(bodyID_t));
    solverScratchSpace.allocateDualStruct("device_contact_count");
    const std::vector<bodyID_t> sorted_owner_ids = hostSort(ownerIDs);
    bodyID_t* host_owner_ids =
        reinterpret_cast<bodyID_t*>(solverScratchSpace.getDualArrayHost("device_contact_owner_ids"));
    std::copy(sorted_owner_ids.begin(), sorted_owner_ids.end(), host_owner_ids);
    solverScratchSpace.syncDualArrayHostToDevice("device_contact_owner_ids");

    size_t* host_count = solverScratchSpace.getDualStructHost("device_contact_count");
    *host_count = 0;
    solverScratchSpace.syncDualStructHostToDevice("device_contact_count");

    const bool same_device = destination_device == streamInfo.device;
    float3* packed_points = points;
    float3* packed_forces = forces;
    float3* packed_torques = torques;
    if (!same_device) {
        packed_points =
            reinterpret_cast<float3*>(solverScratchSpace.allocateTempVector("device_contact_points", full_bytes));
        packed_forces =
            reinterpret_cast<float3*>(solverScratchSpace.allocateTempVector("device_contact_forces", full_bytes));
        if (need_torque) {
            packed_torques =
                reinterpret_cast<float3*>(solverScratchSpace.allocateTempVector("device_contact_torques", full_bytes));
        }
    }

    getContactForcesConcerningOwners(
        packed_points, packed_forces, packed_torques, solverScratchSpace.getDualStructDevice("device_contact_count"),
        reinterpret_cast<bodyID_t*>(solverScratchSpace.getDualArrayDevice("device_contact_owner_ids")),
        sorted_owner_ids.size(), &simParams, &granData, num_contacts, need_torque, torque_in_local, streamInfo.stream);
    DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    solverScratchSpace.syncDualStructDeviceToHost("device_contact_count");
    const size_t useful_count = *host_count;

    if (!same_device && useful_count > 0) {
        const size_t useful_bytes = useful_count * sizeof(float3);
        DEME_GPU_CALL(
            ownerDataTransferBuffer.Copy(points, destination_device, packed_points, streamInfo.device, useful_bytes));
        DEME_GPU_CALL(
            ownerDataTransferBuffer.Copy(forces, destination_device, packed_forces, streamInfo.device, useful_bytes));
        if (need_torque) {
            DEME_GPU_CALL(ownerDataTransferBuffer.Copy(torques, destination_device, packed_torques, streamInfo.device,
                                                       useful_bytes));
        }
    }

    if (!same_device) {
        solverScratchSpace.finishUsingTempVector("device_contact_points");
        solverScratchSpace.finishUsingTempVector("device_contact_forces");
        if (need_torque) {
            solverScratchSpace.finishUsingTempVector("device_contact_torques");
        }
    }
    solverScratchSpace.finishUsingDualArray("device_contact_owner_ids");
    solverScratchSpace.finishUsingDualStruct("device_contact_count");
    return useful_count;
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

void DEMDynamicThread::getOwnerDataToDevice(void* destination,
                                            size_t capacity,
                                            int destination_device,
                                            bodyID_t ownerID,
                                            bodyID_t n,
                                            OwnerDataField field,
                                            unsigned int wildcard_index,
                                            bool validate) {
    if (validate && (ownerID > simParams->nOwnerBodies || n > simParams->nOwnerBodies - ownerID)) {
        DEME_ERROR("Owner device retrieval range [%zu, %zu) exceeds the %zu owners in the simulation.", (size_t)ownerID,
                   (size_t)(ownerID + n), (size_t)simParams->nOwnerBodies);
    }
    if (validate && capacity < n) {
        DEME_ERROR("Owner device retrieval needs capacity for %zu elements, but the destination has %zu.", (size_t)n,
                   capacity);
    }

    const size_t bytes = static_cast<size_t>(n) * OwnerDataElementSize(field);
    if (validate) {
        DEME_GPU_CALL(device_data::ValidateOutputPointer(destination, bytes, destination_device));
    }
    if (n == 0) {
        return;
    }

    ScopedCudaDevice device_scope(streamInfo.device);
    if (destination_device == streamInfo.device) {
        PackOwnerData(destination, field, ownerID, n, &simParams, &granData, solverFlags.useMassJitify, wildcard_index,
                      streamInfo.stream);
        DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        return;
    }

    // Cross-device output is packed once on dT's GPU, then CUDA selects the available inter-device transfer route.
    const std::string scratch_name = "owner_device_retrieval";
    void* packed = solverScratchSpace.allocateTempVector(scratch_name, bytes);
    PackOwnerData(packed, field, ownerID, n, &simParams, &granData, solverFlags.useMassJitify, wildcard_index,
                  streamInfo.stream);
    DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    DEME_GPU_CALL(ownerDataTransferBuffer.Copy(destination, destination_device, packed, streamInfo.device, bytes));
    solverScratchSpace.finishUsingTempVector(scratch_name);
}

void DEMDynamicThread::setOwnerDataFromDevice(bodyID_t ownerID,
                                              const void* source,
                                              size_t count,
                                              int source_device,
                                              OwnerStateField field,
                                              bool validate) {
    if (validate && (ownerID > simParams->nOwnerBodies || count > simParams->nOwnerBodies - ownerID)) {
        DEME_ERROR("Owner device update range [%zu, %zu) exceeds the %zu owners in the simulation.", (size_t)ownerID,
                   (size_t)(ownerID + count), (size_t)simParams->nOwnerBodies);
    }
    if (count == 0) {
        return;
    }
    const size_t element_size = field == OwnerStateField::ORIENTATION ? sizeof(float4) : sizeof(float3);
    const size_t bytes = count * element_size;
    if (validate) {
        DEME_GPU_CALL(device_data::ValidateOutputPointer(source, bytes, source_device));
    }
    ScopedCudaDevice device_scope(streamInfo.device);
    const bool same_device = source_device == streamInfo.device;
    constexpr const char* scratch_name = "owner_device_input";
    const void* unpack_source = source;
    if (!same_device) {
        void* local_source = solverScratchSpace.allocateTempVector(scratch_name, bytes);
        DEME_GPU_CALL(ownerDataTransferBuffer.Copy(local_source, streamInfo.device, source, source_device, bytes));
        unpack_source = local_source;
    }
    if (validate && field == OwnerStateField::ORIENTATION) {
        const std::string validation_name = "owner_orientation_validation";
        solverScratchSpace.allocateDualStruct(validation_name);
        size_t* host_invalid = solverScratchSpace.getDualStructHost(validation_name);
        *host_invalid = 0;
        solverScratchSpace.syncDualStructHostToDevice(validation_name);
        ValidateOwnerOrientations(
            static_cast<const float4*>(unpack_source), count,
            reinterpret_cast<unsigned int*>(solverScratchSpace.getDualStructDevice(validation_name)),
            streamInfo.stream);
        solverScratchSpace.syncDualStructDeviceToHost(validation_name);
        const bool invalid = *host_invalid != 0;
        solverScratchSpace.finishUsingDualStruct(validation_name);
        if (invalid) {
            if (!same_device) {
                solverScratchSpace.finishUsingTempVector(scratch_name);
            }
            DEME_ERROR("Owner orientation updates require finite, nonzero-length quaternions.");
        }
    }
    UnpackOwnerState(unpack_source, field, ownerID, count, &simParams, &granData, streamInfo.stream);
    DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    if (!same_device) {
        solverScratchSpace.finishUsingTempVector(scratch_name);
    }
}

void DEMDynamicThread::getOwnerContactWrenchToDevice(float3* forces,
                                                     float3* torques,
                                                     size_t capacity,
                                                     int destination_device,
                                                     bodyID_t ownerID,
                                                     bodyID_t count) {
    if (ownerID > simParams->nOwnerBodies || count > simParams->nOwnerBodies - ownerID) {
        DEME_ERROR("Owner contact-wrench range [%zu, %zu) exceeds the %zu owners in the simulation.", (size_t)ownerID,
                   (size_t)(ownerID + count), (size_t)simParams->nOwnerBodies);
    }
    if (capacity < count) {
        DEME_ERROR("Owner contact-wrench retrieval needs capacity for %zu owners, but the destination has %zu.",
                   (size_t)count, capacity);
    }

    const size_t bytes = static_cast<size_t>(count) * sizeof(float3);
    DEME_GPU_CALL(device_data::ValidateOutputPointer(forces, bytes, destination_device));
    DEME_GPU_CALL(device_data::ValidateOutputPointer(torques, bytes, destination_device));
    if (count == 0) {
        return;
    }
    if (solverFlags.useNoContactRecord) {
        DEME_ERROR("Owner contact-wrench retrieval requires contact recording; do not enable SetNoForceRecord().");
    }

    ScopedCudaDevice device_scope(streamInfo.device);
    const bool same_device = destination_device == streamInfo.device;
    float3* reduced_forces = forces;
    float3* reduced_torques = torques;
    if (!same_device) {
        reduced_forces = reinterpret_cast<float3*>(solverScratchSpace.allocateTempVector("owner_wrench_forces", bytes));
        reduced_torques =
            reinterpret_cast<float3*>(solverScratchSpace.allocateTempVector("owner_wrench_torques", bytes));
    }

    ReduceOwnerContactWrenches(reduced_forces, reduced_torques, ownerID, count, &granData,
                               *solverScratchSpace.numContacts, streamInfo.stream);
    DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    if (!same_device) {
        DEME_GPU_CALL(
            ownerDataTransferBuffer.Copy(forces, destination_device, reduced_forces, streamInfo.device, bytes));
        DEME_GPU_CALL(
            ownerDataTransferBuffer.Copy(torques, destination_device, reduced_torques, streamInfo.device, bytes));
        solverScratchSpace.finishUsingTempVector("owner_wrench_forces");
        solverScratchSpace.finishUsingTempVector("owner_wrench_torques");
    }
}

void DEMDynamicThread::getOwnerContactWrench(std::vector<float3>& forces,
                                             std::vector<float3>& torques,
                                             bodyID_t ownerID,
                                             bodyID_t count) {
    forces.resize(count);
    torques.resize(count);
    if (count == 0) {
        return;
    }

    const std::string force_name = "owner_wrench_host_forces";
    const std::string torque_name = "owner_wrench_host_torques";
    solverScratchSpace.allocateDualArray(force_name, static_cast<size_t>(count) * sizeof(float3));
    solverScratchSpace.allocateDualArray(torque_name, static_cast<size_t>(count) * sizeof(float3));
    auto* device_forces = reinterpret_cast<float3*>(solverScratchSpace.getDualArrayDevice(force_name));
    auto* device_torques = reinterpret_cast<float3*>(solverScratchSpace.getDualArrayDevice(torque_name));

    // Reuse the device reduction so host and device APIs have identical filtering, frame, and accumulation semantics.
    getOwnerContactWrenchToDevice(device_forces, device_torques, count, streamInfo.device, ownerID, count);
    solverScratchSpace.syncDualArrayDeviceToHost(force_name);
    solverScratchSpace.syncDualArrayDeviceToHost(torque_name);
    const auto* host_forces = reinterpret_cast<const float3*>(solverScratchSpace.getDualArrayHost(force_name));
    const auto* host_torques = reinterpret_cast<const float3*>(solverScratchSpace.getDualArrayHost(torque_name));
    std::copy(host_forces, host_forces + count, forces.begin());
    std::copy(host_torques, host_torques + count, torques.begin());
    solverScratchSpace.finishUsingDualArray(force_name);
    solverScratchSpace.finishUsingDualArray(torque_name);
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
    std::vector<float4> normalized(oriQ.size());
    for (size_t i = 0; i < oriQ.size(); i++) {
        const double norm_squared =
            static_cast<double>(oriQ[i].x) * oriQ[i].x + static_cast<double>(oriQ[i].y) * oriQ[i].y +
            static_cast<double>(oriQ[i].z) * oriQ[i].z + static_cast<double>(oriQ[i].w) * oriQ[i].w;
        if (!std::isfinite(norm_squared) || norm_squared == 0.0) {
            DEME_ERROR("Owner orientation updates require finite, nonzero-length quaternions.");
        }
        const float inverse_norm = static_cast<float>(1.0 / std::sqrt(norm_squared));
        normalized[i] = oriQ[i] * inverse_norm;
    }
    oriQw.setVal(streamInfo.stream, RealTupleVectorToWComponentVector<float, float4>(normalized), ownerID);
    oriQx.setVal(streamInfo.stream, RealTupleVectorToXComponentVector<float, float4>(normalized), ownerID);
    oriQy.setVal(streamInfo.stream, RealTupleVectorToYComponentVector<float, float4>(normalized), ownerID);
    oriQz.setVal(streamInfo.stream, RealTupleVectorToZComponentVector<float, float4>(normalized), ownerID);
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
    ++visualizationRevision;
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
    ++visualizationRevision;
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

void DEMDynamicThread::configureTrianglePVTracking(const std::vector<bodyID_t>& mesh_owner_ids) {
    if (mesh_owner_ids.empty()) {
        disableTrianglePVTracking();
        return;
    }

    ownerTriMesh.toHost();
    std::vector<bodyID_t> owner_order;
    std::vector<size_t> offsets;
    std::vector<size_t> counts;
    std::unordered_map<bodyID_t, size_t> owner_to_slot;
    owner_order.reserve(mesh_owner_ids.size());
    offsets.reserve(mesh_owner_ids.size());
    counts.reserve(mesh_owner_ids.size());

    for (bodyID_t owner : mesh_owner_ids) {
        if (owner_to_slot.find(owner) != owner_to_slot.end()) {
            continue;
        }
        size_t count = 0;
        for (size_t tri = 0; tri < ownerTriMesh.size(); tri++) {
            if (ownerTriMesh[tri] == owner) {
                count++;
            }
        }
        if (count == 0) {
            DEME_ERROR("Triangle PV tracking owner %zu is not a mesh owner or has no triangles.", (size_t)owner);
        }
        const size_t slot = owner_order.size();
        owner_to_slot.emplace(owner, slot);
        owner_order.push_back(owner);
        offsets.push_back(0);
        counts.push_back(count);
    }

    size_t n_tracked_triangles = 0;
    for (size_t i = 0; i < owner_order.size(); i++) {
        offsets[i] = n_tracked_triangles;
        n_tracked_triangles += counts[i];
    }
    if (n_tracked_triangles == 0) {
        disableTrianglePVTracking();
        return;
    }

    DEME_DUAL_ARRAY_RESIZE(triPVGlobalTriToLocal, simParams->nTriGM, -1);
    for (size_t tri = 0; tri < ownerTriMesh.size(); tri++) {
        const bodyID_t owner = ownerTriMesh[tri];
        auto it = owner_to_slot.find(owner);
        if (it == owner_to_slot.end()) {
            triPVGlobalTriToLocal[tri] = -1;
            continue;
        }
        const size_t slot = it->second;
        size_t local_idx = offsets[slot];
        for (size_t prev = 0; prev < tri; prev++) {
            if (ownerTriMesh[prev] == owner) {
                local_idx++;
            }
        }
        triPVGlobalTriToLocal[tri] = static_cast<int>(local_idx);
    }

    const size_t alloc_size = DEME_MAX((size_t)1, n_tracked_triangles);
    DEME_DUAL_ARRAY_RESIZE(triPVStepP, alloc_size, 0.f);
    DEME_DUAL_ARRAY_RESIZE(triPVStepPV, alloc_size, 0.f);
    DEME_DUAL_ARRAY_RESIZE(triPVAccumP, alloc_size, 0.f);
    DEME_DUAL_ARRAY_RESIZE(triPVAccumPV, alloc_size, 0.f);

    triPVGlobalTriToLocal.toDeviceAsync(streamInfo.stream);
    triPVStepP.toDeviceAsync(streamInfo.stream);
    triPVStepPV.toDeviceAsync(streamInfo.stream);
    triPVAccumP.toDeviceAsync(streamInfo.stream);
    triPVAccumPV.toDeviceAsync(streamInfo.stream);
    syncMemoryTransfer();

    triPVOwnerOrder = std::move(owner_order);
    triPVOwnerOffsets = std::move(offsets);
    triPVOwnerCounts = std::move(counts);
    triPVOwnerToSlot = std::move(owner_to_slot);
    triPVTrackingEnabled = true;
    triPVNumTrackedTriangles = n_tracked_triangles;
    triPVWindowSteps = 0;
}

void DEMDynamicThread::disableTrianglePVTracking() {
    triPVTrackingEnabled = false;
    triPVNumTrackedTriangles = 0;
    triPVWindowSteps = 0;
    triPVOwnerOrder.clear();
    triPVOwnerOffsets.clear();
    triPVOwnerCounts.clear();
    triPVOwnerToSlot.clear();

    if (triPVGlobalTriToLocal.size() > 0) {
        for (size_t i = 0; i < triPVGlobalTriToLocal.size(); i++) {
            triPVGlobalTriToLocal[i] = -1;
        }
        triPVGlobalTriToLocal.toDeviceAsync(streamInfo.stream);
    }
    if (triPVStepP.size() > 0) {
        DEME_GPU_CALL(cudaMemsetAsync(triPVStepP.device(), 0, triPVStepP.size() * sizeof(float), streamInfo.stream));
    }
    if (triPVStepPV.size() > 0) {
        DEME_GPU_CALL(cudaMemsetAsync(triPVStepPV.device(), 0, triPVStepPV.size() * sizeof(float), streamInfo.stream));
    }
    if (triPVAccumP.size() > 0) {
        DEME_GPU_CALL(cudaMemsetAsync(triPVAccumP.device(), 0, triPVAccumP.size() * sizeof(float), streamInfo.stream));
    }
    if (triPVAccumPV.size() > 0) {
        DEME_GPU_CALL(
            cudaMemsetAsync(triPVAccumPV.device(), 0, triPVAccumPV.size() * sizeof(float), streamInfo.stream));
    }
    syncMemoryTransfer();
}

bool DEMDynamicThread::getTrackedOwnerTrianglePV(bodyID_t ownerID,
                                                 std::vector<float>& avgP,
                                                 std::vector<float>& avgV,
                                                 std::vector<float>& avgPV,
                                                 bool reset_window) {
    if (!triPVTrackingEnabled) {
        return false;
    }
    auto it = triPVOwnerToSlot.find(ownerID);
    if (it == triPVOwnerToSlot.end()) {
        return false;
    }
    const size_t slot = it->second;
    const size_t offset = triPVOwnerOffsets[slot];
    const size_t count = triPVOwnerCounts[slot];
    avgP.assign(count, 0.f);
    avgV.assign(count, 0.f);
    avgPV.assign(count, 0.f);

    if (count > 0 && triPVWindowSteps > 0) {
        triPVAccumP.toHost();
        triPVAccumPV.toHost();
        const float inv_steps = 1.f / static_cast<float>(triPVWindowSteps);
        for (size_t i = 0; i < count; i++) {
            avgP[i] = triPVAccumP[offset + i] * inv_steps;
            avgPV[i] = triPVAccumPV[offset + i] * inv_steps;
            if (!std::isfinite(avgP[i]) || avgP[i] < 0.f) {
                avgP[i] = 0.f;
            }
            if (!std::isfinite(avgPV[i]) || avgPV[i] < 0.f) {
                avgPV[i] = 0.f;
            }
            avgV[i] = (avgP[i] > DEME_TINY_FLOAT) ? (avgPV[i] / avgP[i]) : 0.f;
            if (!std::isfinite(avgV[i]) || avgV[i] < 0.f) {
                avgV[i] = 0.f;
            }
        }
    }

    if (reset_window) {
        resetTrackedTrianglePVWindow();
    }
    return true;
}

void DEMDynamicThread::resetTrackedTrianglePVWindow() {
    triPVWindowSteps = 0;
    if (triPVAccumP.size() > 0) {
        DEME_GPU_CALL(cudaMemsetAsync(triPVAccumP.device(), 0, triPVAccumP.size() * sizeof(float), streamInfo.stream));
    }
    if (triPVAccumPV.size() > 0) {
        DEME_GPU_CALL(
            cudaMemsetAsync(triPVAccumPV.device(), 0, triPVAccumPV.size() * sizeof(float), streamInfo.stream));
    }
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
