//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cstring>
#include <iostream>
#include <thread>
#include <algorithm>

#include <core/ApiVersion.h>
#include <core/utils/chpf/particle_writer.hpp>
#include <DEM/dT.h>
#include <DEM/kT.h>
#include <DEM/HostSideHelpers.cpp>
#include <nvmath/helper_math.cuh>
#include <DEM/DEMDefines.h>

#include <algorithms/DEMCubBasedSubroutines.h>

namespace sgps {

// Put sim data array pointers in place
void DEMDynamicThread::packDataPointers() {
    granData->inertiaPropOffsets = inertiaPropOffsets.data();
    granData->familyID = familyID.data();
    granData->voxelID = voxelID.data();
    granData->locX = locX.data();
    granData->locY = locY.data();
    granData->locZ = locZ.data();
    granData->aX = aX.data();
    granData->aY = aY.data();
    granData->aZ = aZ.data();
    granData->vX = vX.data();
    granData->vY = vY.data();
    granData->vZ = vZ.data();
    granData->oriQ0 = oriQ0.data();
    granData->oriQ1 = oriQ1.data();
    granData->oriQ2 = oriQ2.data();
    granData->oriQ3 = oriQ3.data();
    granData->omgBarX = omgBarX.data();
    granData->omgBarY = omgBarY.data();
    granData->omgBarZ = omgBarZ.data();
    granData->alphaX = alphaX.data();
    granData->alphaY = alphaY.data();
    granData->alphaZ = alphaZ.data();
    granData->idGeometryA = idGeometryA.data();
    granData->idGeometryB = idGeometryB.data();
    granData->contactType = contactType.data();

    granData->idGeometryA_buffer = idGeometryA_buffer.data();
    granData->idGeometryB_buffer = idGeometryB_buffer.data();
    granData->contactType_buffer = contactType_buffer.data();
    granData->contactMapping_buffer = contactMapping_buffer.data();

    granData->contactForces = contactForces.data();
    granData->contactPointGeometryA = contactPointGeometryA.data();
    granData->contactPointGeometryB = contactPointGeometryB.data();
    granData->contactHistory = contactHistory.data();

    // The offset info that indexes into the template arrays
    granData->ownerClumpBody = ownerClumpBody.data();
    granData->clumpComponentOffset = clumpComponentOffset.data();
    granData->clumpComponentOffsetExt = clumpComponentOffsetExt.data();
    granData->materialTupleOffset = materialTupleOffset.data();

    // Template array pointers
    granData->radiiSphere = radiiSphere.data();
    granData->relPosSphereX = relPosSphereX.data();
    granData->relPosSphereY = relPosSphereY.data();
    granData->relPosSphereZ = relPosSphereZ.data();
    granData->massOwnerBody = massOwnerBody.data();
    granData->mmiXX = mmiXX.data();
    granData->mmiYY = mmiYY.data();
    granData->mmiZZ = mmiZZ.data();
    granData->EProxy = EProxy.data();
    granData->nuProxy = nuProxy.data();
    granData->CoRProxy = CoRProxy.data();
    granData->muProxy = muProxy.data();
    granData->CrrProxy = CrrProxy.data();
}

void DEMDynamicThread::packTransferPointers(DEMKinematicThread* kT) {
    // These are the pointers for sending data to dT
    granData->pKTOwnedBuffer_voxelID = kT->granData->voxelID_buffer;
    granData->pKTOwnedBuffer_locX = kT->granData->locX_buffer;
    granData->pKTOwnedBuffer_locY = kT->granData->locY_buffer;
    granData->pKTOwnedBuffer_locZ = kT->granData->locZ_buffer;
    granData->pKTOwnedBuffer_oriQ0 = kT->granData->oriQ0_buffer;
    granData->pKTOwnedBuffer_oriQ1 = kT->granData->oriQ1_buffer;
    granData->pKTOwnedBuffer_oriQ2 = kT->granData->oriQ2_buffer;
    granData->pKTOwnedBuffer_oriQ3 = kT->granData->oriQ3_buffer;
    granData->pKTOwnedBuffer_familyID = kT->granData->familyID_buffer;
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
    simParams->nbX = nbX;
    simParams->nbY = nbY;
    simParams->nbZ = nbZ;
}

float DEMDynamicThread::getKineticEnergy() {
    // We can use temp vectors as we please. Allocate num_of_clumps floats.
    size_t quarryTempSize = (size_t)simParams->nOwnerBodies * sizeof(float);
    float* KEArr = (float*)stateOfSolver_resources.allocateTempVector1(quarryTempSize);
    size_t returnSize = sizeof(float);
    float* KE = (float*)stateOfSolver_resources.allocateTempVector2(returnSize);
    size_t blocks_needed_for_KE =
        (simParams->nOwnerBodies + SGPS_DEM_NUM_BODIES_PER_BLOCK - 1) / SGPS_DEM_NUM_BODIES_PER_BLOCK;
    quarry_stats_kernels->kernel("computeKE")
        .instantiate()
        .configure(dim3(blocks_needed_for_KE), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
        .launch(granData, KEArr);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    // displayArray<float>(KEArr, simParams->nOwnerBodies);
    sumReduce(KEArr, KE, simParams->nOwnerBodies, streamInfo.stream, stateOfSolver_resources);
    return *KE;
}

void DEMDynamicThread::allocateManagedArrays(size_t nOwnerBodies,
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
    SGPS_DEM_TRACKED_RESIZE(inertiaPropOffsets, nOwnerBodies, "inertiaPropOffsets", 0);
    SGPS_DEM_TRACKED_RESIZE(familyID, nOwnerBodies, "familyID", 0);
    SGPS_DEM_TRACKED_RESIZE(voxelID, nOwnerBodies, "voxelID", 0);
    SGPS_DEM_TRACKED_RESIZE(locX, nOwnerBodies, "locX", 0);
    SGPS_DEM_TRACKED_RESIZE(locY, nOwnerBodies, "locY", 0);
    SGPS_DEM_TRACKED_RESIZE(locZ, nOwnerBodies, "locZ", 0);
    SGPS_DEM_TRACKED_RESIZE(oriQ0, nOwnerBodies, "oriQ0", 1);
    SGPS_DEM_TRACKED_RESIZE(oriQ1, nOwnerBodies, "oriQ1", 0);
    SGPS_DEM_TRACKED_RESIZE(oriQ2, nOwnerBodies, "oriQ2", 0);
    SGPS_DEM_TRACKED_RESIZE(oriQ3, nOwnerBodies, "oriQ3", 0);
    SGPS_DEM_TRACKED_RESIZE(vX, nOwnerBodies, "vX", 0);
    SGPS_DEM_TRACKED_RESIZE(vY, nOwnerBodies, "vY", 0);
    SGPS_DEM_TRACKED_RESIZE(vZ, nOwnerBodies, "vZ", 0);
    SGPS_DEM_TRACKED_RESIZE(omgBarX, nOwnerBodies, "omgBarX", 0);
    SGPS_DEM_TRACKED_RESIZE(omgBarY, nOwnerBodies, "omgBarY", 0);
    SGPS_DEM_TRACKED_RESIZE(omgBarZ, nOwnerBodies, "omgBarZ", 0);
    SGPS_DEM_TRACKED_RESIZE(aX, nOwnerBodies, "aX", 0);
    SGPS_DEM_TRACKED_RESIZE(aY, nOwnerBodies, "aY", 0);
    SGPS_DEM_TRACKED_RESIZE(aZ, nOwnerBodies, "aZ", 0);
    SGPS_DEM_TRACKED_RESIZE(alphaX, nOwnerBodies, "alphaX", 0);
    SGPS_DEM_TRACKED_RESIZE(alphaY, nOwnerBodies, "alphaY", 0);
    SGPS_DEM_TRACKED_RESIZE(alphaZ, nOwnerBodies, "alphaZ", 0);

    // Resize to the number of spheres
    SGPS_DEM_TRACKED_RESIZE(ownerClumpBody, nSpheresGM, "ownerClumpBody", 0);
    SGPS_DEM_TRACKED_RESIZE(clumpComponentOffset, nSpheresGM, "clumpComponentOffset", 0);
    // This extended component offset array can hold offset numbers even for big clumps (whereas clumpComponentOffset is
    // typically uint_8, so it may not). If a sphere's component offset index falls in this range then it is not
    // jitified, and the kernel needs to look for it in the global memory.
    SGPS_DEM_TRACKED_RESIZE(clumpComponentOffsetExt, nSpheresGM, "clumpComponentOffsetExt", 0);
    SGPS_DEM_TRACKED_RESIZE(materialTupleOffset, nSpheresGM, "materialTupleOffset", 0);

    // Resize to the length of the clump templates
    SGPS_DEM_TRACKED_RESIZE(massOwnerBody, nMassProperties, "massOwnerBody", 0);
    SGPS_DEM_TRACKED_RESIZE(mmiXX, nMassProperties, "mmiXX", 0);
    SGPS_DEM_TRACKED_RESIZE(mmiYY, nMassProperties, "mmiYY", 0);
    SGPS_DEM_TRACKED_RESIZE(mmiZZ, nMassProperties, "mmiZZ", 0);
    SGPS_DEM_TRACKED_RESIZE(radiiSphere, nClumpComponents, "radiiSphere", 0);
    SGPS_DEM_TRACKED_RESIZE(relPosSphereX, nClumpComponents, "relPosSphereX", 0);
    SGPS_DEM_TRACKED_RESIZE(relPosSphereY, nClumpComponents, "relPosSphereY", 0);
    SGPS_DEM_TRACKED_RESIZE(relPosSphereZ, nClumpComponents, "relPosSphereZ", 0);
    SGPS_DEM_TRACKED_RESIZE(EProxy, nMatTuples, "EProxy", 0);
    SGPS_DEM_TRACKED_RESIZE(nuProxy, nMatTuples, "nuProxy", 0);
    SGPS_DEM_TRACKED_RESIZE(CoRProxy, nMatTuples, "CoRProxy", 0);
    SGPS_DEM_TRACKED_RESIZE(muProxy, nMatTuples, "muProxy", 0);
    SGPS_DEM_TRACKED_RESIZE(CrrProxy, nMatTuples, "CrrProxy", 0);

    // Arrays for contact info
    // The lengths of contact event-based arrays are just estimates. My estimate of total contact pairs is 2n, and I
    // think the max is 6n (although I can't prove it). Note the estimate should be large enough to decrease the number
    // of reallocations in the simulation, but not too large that eats too much memory.
    SGPS_DEM_TRACKED_RESIZE(idGeometryA, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "idGeometryA", 0);
    SGPS_DEM_TRACKED_RESIZE(idGeometryB, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "idGeometryB", 0);
    SGPS_DEM_TRACKED_RESIZE(contactForces, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "contactForces",
                            make_float3(0));
    SGPS_DEM_TRACKED_RESIZE(contactHistory, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "contactHistory",
                            make_float3(0));
    SGPS_DEM_TRACKED_RESIZE(contactType, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "contactType", DEM_NOT_A_CONTACT);
    SGPS_DEM_TRACKED_RESIZE(contactPointGeometryA, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "contactPointGeometryA",
                            make_float3(0));
    SGPS_DEM_TRACKED_RESIZE(contactPointGeometryB, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "contactPointGeometryB",
                            make_float3(0));

    // Transfer buffer arrays
    // The following several arrays will have variable sizes, so here we only used an estimate.
    SGPS_DEM_TRACKED_RESIZE(idGeometryA_buffer, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "idGeometryA_buffer", 0);
    SGPS_DEM_TRACKED_RESIZE(idGeometryB_buffer, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "idGeometryB_buffer", 0);
    SGPS_DEM_TRACKED_RESIZE(contactType_buffer, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "contactType_buffer",
                            DEM_NOT_A_CONTACT);
    if (!solverFlags.isHistoryless) {
        SGPS_DEM_TRACKED_RESIZE(contactMapping_buffer, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER,
                                "contactMapping_buffer", DEM_NULL_MAPPING_PARTNER);
    }
}

void DEMDynamicThread::initManagedArrays(const std::vector<inertiaOffset_t>& input_clump_types,
                                         const std::vector<float3>& input_clump_xyz,
                                         const std::vector<float3>& input_clump_vel,
                                         const std::vector<unsigned int>& input_clump_family,
                                         const std::vector<float3>& input_ext_obj_xyz,
                                         const std::vector<unsigned int>& input_ext_obj_family,
                                         const std::unordered_map<unsigned int, family_t>& family_user_impl_map,
                                         const std::unordered_map<family_t, unsigned int>& family_impl_user_map,
                                         const std::vector<std::vector<unsigned int>>& input_clumps_sp_mat_ids,
                                         const std::vector<float>& clumps_mass_types,
                                         const std::vector<float3>& clumps_moi_types,
                                         const std::vector<std::vector<float>>& clumps_sp_radii_types,
                                         const std::vector<std::vector<float3>>& clumps_sp_location_types,
                                         const std::vector<float>& mat_E,
                                         const std::vector<float>& mat_nu,
                                         const std::vector<float>& mat_CoR,
                                         const std::vector<float>& mat_mu,
                                         const std::vector<float>& mat_Crr,
                                         const std::set<unsigned int>& no_output_families,
                                         std::vector<std::shared_ptr<DEMTrackedObj>>& tracked_objs) {
    // Get the info into the managed memory from the host side. Can this process be more efficient? Maybe, but it's
    // initialization anyway.

    // First, load in material properties
    for (unsigned int i = 0; i < mat_E.size(); i++) {
        EProxy.at(i) = mat_E.at(i);
        nuProxy.at(i) = mat_nu.at(i);
        CoRProxy.at(i) = mat_CoR.at(i);
        muProxy.at(i) = mat_mu.at(i);
        CrrProxy.at(i) = mat_Crr.at(i);
    }

    // Then load in clump mass and MOI
    for (unsigned int i = 0; i < simParams->nDistinctMassProperties; i++) {
        massOwnerBody.at(i) = clumps_mass_types.at(i);
        float3 this_moi = clumps_moi_types.at(i);
        mmiXX.at(i) = this_moi.x;
        mmiYY.at(i) = this_moi.y;
        mmiZZ.at(i) = this_moi.z;
    }

    // dT will need this family number map, store it
    familyUserImplMap = family_user_impl_map;
    familyImplUserMap = family_impl_user_map;

    // Take notes of the families that should not be outputted
    {
        std::set<unsigned int>::iterator it;
        unsigned int i = 0;
        familiesNoOutput.resize(no_output_families.size());
        for (it = no_output_families.begin(); it != no_output_families.end(); it++, i++) {
            familiesNoOutput.at(i) = familyUserImplMap.at(*it);
        }
        std::sort(familiesNoOutput.begin(), familiesNoOutput.end());
        SGPS_DEM_DEBUG_PRINTF("Impl-level families that will not be outputted:");
        SGPS_DEM_DEBUG_EXEC(displayArray<family_t>(familiesNoOutput.data(), familiesNoOutput.size()));
    }

    // Then, load in clump type info
    // All flattened to single arrays, so a prescans is needed so we know each input clump's component offsets
    size_t k = 0;
    std::vector<unsigned int> prescans;

    prescans.push_back(0);
    for (auto elem : clumps_sp_radii_types) {
        for (auto radius : elem) {
            radiiSphere.at(k) = radius;
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
    }
    k = 0;

    // Then, load in input clumps
    float3 LBF;
    LBF.x = simParams->LBFX;
    LBF.y = simParams->LBFY;
    LBF.z = simParams->LBFZ;
    for (size_t i = 0; i < simParams->nOwnerClumps; i++) {
        auto type_of_this_clump = input_clump_types.at(i);
        inertiaPropOffsets.at(i) = type_of_this_clump;
        auto this_CoM_coord = input_clump_xyz.at(i) - LBF;
        // std::cout << "CoM position: " << this_CoM_coord.x << ", " << this_CoM_coord.y << ", " << this_CoM_coord.z <<
        // std::endl;
        auto this_clump_no_sp_radii = clumps_sp_radii_types.at(type_of_this_clump);
        auto this_clump_no_sp_relPos = clumps_sp_location_types.at(type_of_this_clump);
        auto this_clump_no_sp_mat_ids = input_clumps_sp_mat_ids.at(type_of_this_clump);

        for (size_t j = 0; j < this_clump_no_sp_radii.size(); j++) {
            materialTupleOffset.at(k) = this_clump_no_sp_mat_ids.at(j);
            ownerClumpBody.at(k) = i;

            // This component offset, is it too large that can't live in the jitified array?
            unsigned int this_comp_offset = prescans.at(type_of_this_clump) + j;
            clumpComponentOffsetExt.at(k) = this_comp_offset;
            if (this_comp_offset < simParams->nJitifiableClumpComponents) {
                clumpComponentOffset.at(k) = this_comp_offset;
            } else {
                // If not, an indicator will be put there
                clumpComponentOffset.at(k) = DEM_RESERVED_CLUMP_COMPONENT_OFFSET;
            }

            k++;
            // std::cout << "Sphere Rel Pos offset: " << this_clump_no_sp_loc_offsets.at(j) << std::endl;
        }

        voxelID_t voxelNumX = (double)this_CoM_coord.x / simParams->voxelSize;
        voxelID_t voxelNumY = (double)this_CoM_coord.y / simParams->voxelSize;
        voxelID_t voxelNumZ = (double)this_CoM_coord.z / simParams->voxelSize;
        locX.at(i) = ((double)this_CoM_coord.x - (double)voxelNumX * simParams->voxelSize) / simParams->l;
        locY.at(i) = ((double)this_CoM_coord.y - (double)voxelNumY * simParams->voxelSize) / simParams->l;
        locZ.at(i) = ((double)this_CoM_coord.z - (double)voxelNumZ * simParams->voxelSize) / simParams->l;
        // std::cout << "Clump voxel num: " << voxelNumX << ", " << voxelNumY << ", " << voxelNumZ << std::endl;

        voxelID.at(i) += voxelNumX;
        voxelID.at(i) += voxelNumY << simParams->nvXp2;
        voxelID.at(i) += voxelNumZ << (simParams->nvXp2 + simParams->nvYp2);
        // std::cout << "Computed voxel num: " << voxelID.at(i) << std::endl;

        // Set initial velocity
        auto vel_of_this_clump = input_clump_vel.at(i);
        vX.at(i) = vel_of_this_clump.x;
        vY.at(i) = vel_of_this_clump.y;
        vZ.at(i) = vel_of_this_clump.z;

        // Set family code
        family_t this_family_num = family_user_impl_map.at(input_clump_family.at(i));
        familyID.at(i) = this_family_num;
    }

    // Load in initial positions and mass properties for the owners of those external objects
    // They go after clump owners
    size_t offset_for_ext_obj = simParams->nOwnerClumps;
    unsigned int offset_for_ext_obj_mass_template = simParams->nDistinctClumpBodyTopologies;
    for (size_t i = 0; i < simParams->nExtObj; i++) {
        inertiaPropOffsets.at(i + offset_for_ext_obj) = i + offset_for_ext_obj_mass_template;
        auto this_CoM_coord = input_ext_obj_xyz.at(i) - LBF;
        voxelID_t voxelNumX = (double)this_CoM_coord.x / simParams->voxelSize;
        voxelID_t voxelNumY = (double)this_CoM_coord.y / simParams->voxelSize;
        voxelID_t voxelNumZ = (double)this_CoM_coord.z / simParams->voxelSize;
        locX.at(i + offset_for_ext_obj) =
            ((double)this_CoM_coord.x - (double)voxelNumX * simParams->voxelSize) / simParams->l;
        locY.at(i + offset_for_ext_obj) =
            ((double)this_CoM_coord.y - (double)voxelNumY * simParams->voxelSize) / simParams->l;
        locZ.at(i + offset_for_ext_obj) =
            ((double)this_CoM_coord.z - (double)voxelNumZ * simParams->voxelSize) / simParams->l;

        voxelID.at(i + offset_for_ext_obj) += voxelNumX;
        voxelID.at(i + offset_for_ext_obj) += voxelNumY << simParams->nvXp2;
        voxelID.at(i + offset_for_ext_obj) += voxelNumZ << (simParams->nvXp2 + simParams->nvYp2);
        // TODO: and initial rot?
        // TODO: and initial vel?

        family_t this_family_num = family_user_impl_map.at(input_ext_obj_family.at(i));
        familyID.at(i + offset_for_ext_obj) = this_family_num;
    }

    SGPS_DEM_DEBUG_PRINTF("Compare clumpComponentOffset and clumpComponentOffsetExt:");
    SGPS_DEM_DEBUG_EXEC(displayArray<clumpComponentOffset_t>(clumpComponentOffset.data(), simParams->nSpheresGM));
    SGPS_DEM_DEBUG_EXEC(displayArray<clumpComponentOffsetExt_t>(clumpComponentOffsetExt.data(), simParams->nSpheresGM));

    // Provide feedback to the tracked objects, tell them the owner numbers they are looking for
    // Little computation is needed, as long as we know the structure of our owner array: nOwnerClumps go first, then
    // nExtObj
    for (auto& tracked_obj : tracked_objs) {
        switch (tracked_obj->type) {
            case (DEM_OWNER_TYPE::CLUMP):
                tracked_obj->ownerID = tracked_obj->load_order;
                break;
            case (DEM_OWNER_TYPE::ANALYTICAL):
                tracked_obj->ownerID = tracked_obj->load_order + simParams->nOwnerClumps;
                break;
            default:
                SGPS_DEM_ERROR("A DEM tracked object has an unknown type.");
        }
    }
}

void DEMDynamicThread::writeChpfAsSpheres(std::ofstream& ptFile) const {
    ParticleFormatWriter pw;
    // pw.write(ptFile, ParticleFormatWriter::CompressionType::NONE, mass);
    std::vector<float> posX(simParams->nSpheresGM);
    std::vector<float> posY(simParams->nSpheresGM);
    std::vector<float> posZ(simParams->nSpheresGM);
    std::vector<float> spRadii(simParams->nSpheresGM);
    std::vector<unsigned int> families;
    if (solverFlags.outputFlags & DEM_OUTPUT_CONTENT::FAMILY) {
        families.resize(simParams->nSpheresGM);
    }
    size_t num_output_spheres = 0;

    for (size_t i = 0; i < simParams->nSpheresGM; i++) {
        auto this_owner = ownerClumpBody.at(i);
        family_t this_family = familyID.at(this_owner);
        // If this (impl-level) family is in the no-output list, skip it
        if (std::binary_search(familiesNoOutput.begin(), familiesNoOutput.end(), this_family)) {
            continue;
        }

        float3 CoM;
        float X, Y, Z;
        voxelID_t voxel = voxelID.at(this_owner);
        subVoxelPos_t subVoxX = locX.at(this_owner);
        subVoxelPos_t subVoxY = locY.at(this_owner);
        subVoxelPos_t subVoxZ = locZ.at(this_owner);
        hostVoxelID2Position<float, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel, subVoxX, subVoxY, subVoxZ,
                                                              simParams->nvXp2, simParams->nvYp2, simParams->voxelSize,
                                                              simParams->l);
        CoM.x = X + simParams->LBFX;
        CoM.y = Y + simParams->LBFY;
        CoM.z = Z + simParams->LBFZ;

        // Must use clumpComponentOffsetExt, b/c clumpComponentOffset may not be a faithful representation of clump
        // component offset numbers
        float this_sp_deviation_x = relPosSphereX.at(clumpComponentOffsetExt.at(i));
        float this_sp_deviation_y = relPosSphereY.at(clumpComponentOffsetExt.at(i));
        float this_sp_deviation_z = relPosSphereZ.at(clumpComponentOffsetExt.at(i));
        float this_sp_rot_0 = oriQ0.at(this_owner);
        float this_sp_rot_1 = oriQ1.at(this_owner);
        float this_sp_rot_2 = oriQ2.at(this_owner);
        float this_sp_rot_3 = oriQ3.at(this_owner);
        hostApplyOriQ2Vector3<float, float>(this_sp_deviation_x, this_sp_deviation_y, this_sp_deviation_z,
                                            this_sp_rot_0, this_sp_rot_1, this_sp_rot_2, this_sp_rot_3);
        posX.at(num_output_spheres) = CoM.x + this_sp_deviation_x;
        posY.at(num_output_spheres) = CoM.y + this_sp_deviation_y;
        posZ.at(num_output_spheres) = CoM.z + this_sp_deviation_z;
        // std::cout << "Sphere Pos: " << posX.at(i) << ", " << posY.at(i) << ", " << posZ.at(i) << std::endl;

        spRadii.at(num_output_spheres) = radiiSphere.at(clumpComponentOffsetExt.at(i));

        // Family number needs to be user number
        if (solverFlags.outputFlags & DEM_OUTPUT_CONTENT::FAMILY) {
            families.at(num_output_spheres) = familyImplUserMap.at(this_family);
        }

        num_output_spheres++;
    }
    // Write basics
    posX.resize(num_output_spheres);
    posY.resize(num_output_spheres);
    posZ.resize(num_output_spheres);
    spRadii.resize(num_output_spheres);
    pw.write(ptFile, ParticleFormatWriter::CompressionType::NONE, posX, posY, posZ, spRadii);
    // Write family numbers
    if (solverFlags.outputFlags & DEM_OUTPUT_CONTENT::FAMILY) {
        families.resize(num_output_spheres);
        pw.write(ptFile, ParticleFormatWriter::CompressionType::NONE, families);
    }
}

void DEMDynamicThread::writeCsvAsSpheres(std::ofstream& ptFile) const {
    std::ostringstream outstrstream;

    outstrstream << "x,y,z,r";
    if (solverFlags.outputFlags & DEM_OUTPUT_CONTENT::ABSV) {
        outstrstream << ",|v|";
    }
    if (solverFlags.outputFlags & DEM_OUTPUT_CONTENT::VEL) {
        outstrstream << ",v_x,v_y,v_z";
    }
    if (solverFlags.outputFlags & DEM_OUTPUT_CONTENT::ANG_VEL) {
        outstrstream << ",w_x,w_y,w_z";
    }
    if (solverFlags.outputFlags & DEM_OUTPUT_CONTENT::FORCE) {
        outstrstream << ",F_x,F_y,F_z";
    }
    if (solverFlags.outputFlags & DEM_OUTPUT_CONTENT::FAMILY) {
        outstrstream << ",family";
    }
    outstrstream << "\n";

    for (size_t i = 0; i < simParams->nSpheresGM; i++) {
        auto this_owner = ownerClumpBody.at(i);
        family_t this_family = familyID.at(this_owner);
        // If this (impl-level) family is in the no-output list, skip it
        if (std::binary_search(familiesNoOutput.begin(), familiesNoOutput.end(), this_family)) {
            continue;
        }

        float3 CoM;
        float3 pos;
        float radius;
        float X, Y, Z;
        voxelID_t voxel = voxelID.at(this_owner);
        subVoxelPos_t subVoxX = locX.at(this_owner);
        subVoxelPos_t subVoxY = locY.at(this_owner);
        subVoxelPos_t subVoxZ = locZ.at(this_owner);
        hostVoxelID2Position<float, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel, subVoxX, subVoxY, subVoxZ,
                                                              simParams->nvXp2, simParams->nvYp2, simParams->voxelSize,
                                                              simParams->l);
        CoM.x = X + simParams->LBFX;
        CoM.y = Y + simParams->LBFY;
        CoM.z = Z + simParams->LBFZ;

        // Must use clumpComponentOffsetExt, b/c clumpComponentOffset may not be a faithful representation of clump
        // component offset numbers
        float3 this_sp_deviation;
        this_sp_deviation.x = relPosSphereX.at(clumpComponentOffsetExt.at(i));
        this_sp_deviation.y = relPosSphereY.at(clumpComponentOffsetExt.at(i));
        this_sp_deviation.z = relPosSphereZ.at(clumpComponentOffsetExt.at(i));
        float this_sp_rot_0 = oriQ0.at(this_owner);
        float this_sp_rot_1 = oriQ1.at(this_owner);
        float this_sp_rot_2 = oriQ2.at(this_owner);
        float this_sp_rot_3 = oriQ3.at(this_owner);
        hostApplyOriQ2Vector3<float, float>(this_sp_deviation.x, this_sp_deviation.y, this_sp_deviation.z,
                                            this_sp_rot_0, this_sp_rot_1, this_sp_rot_2, this_sp_rot_3);
        pos = CoM + this_sp_deviation;
        outstrstream << pos.x << "," << pos.y << "," << pos.z;

        radius = radiiSphere.at(clumpComponentOffsetExt.at(i));
        outstrstream << "," << radius;

        // Family number needs to be user number
        if (solverFlags.outputFlags & DEM_OUTPUT_CONTENT::FAMILY) {
            outstrstream << "," << familyImplUserMap.at(this_family);
        }

        outstrstream << "\n";
    }

    ptFile << outstrstream.str();
}

inline void DEMDynamicThread::contactEventArraysResize(size_t nContactPairs) {
    SGPS_DEM_TRACKED_RESIZE_NOPRINT(idGeometryA, nContactPairs);
    SGPS_DEM_TRACKED_RESIZE_NOPRINT(idGeometryB, nContactPairs);
    SGPS_DEM_TRACKED_RESIZE_NOPRINT(contactType, nContactPairs);
    SGPS_DEM_TRACKED_RESIZE_NOPRINT(contactForces, nContactPairs);
    SGPS_DEM_TRACKED_RESIZE_NOPRINT(contactPointGeometryA, nContactPairs);
    SGPS_DEM_TRACKED_RESIZE_NOPRINT(contactPointGeometryB, nContactPairs);

    // Re-pack pointers in case the arrays got reallocated
    granData->idGeometryA = idGeometryA.data();
    granData->idGeometryB = idGeometryB.data();
    granData->contactType = contactType.data();
    granData->contactForces = contactForces.data();
    granData->contactPointGeometryA = contactPointGeometryA.data();
    granData->contactPointGeometryB = contactPointGeometryB.data();

    // Note that my buffer array sizes may have been changed by kT, so I need to repack those pointers too
    granData->idGeometryA_buffer = idGeometryA_buffer.data();
    granData->idGeometryB_buffer = idGeometryB_buffer.data();
    granData->contactType_buffer = contactType_buffer.data();
    // If not historyless, then contact history map needs to be updated in case kT made modifications
    if (!solverFlags.isHistoryless) {
        granData->contactMapping_buffer = contactMapping_buffer.data();
    }
}

inline void DEMDynamicThread::unpackMyBuffer() {
    // Make a note on the contact number of the previous time step
    *stateOfSolver_resources.pNumPrevContacts = *stateOfSolver_resources.pNumContacts;

    GPU_CALL(cudaMemcpy(stateOfSolver_resources.pNumContacts, &(granData->nContactPairs_buffer), sizeof(size_t),
                        cudaMemcpyDeviceToDevice));

    // Need to resize those contact event-based arrays before usage
    if (*stateOfSolver_resources.pNumContacts > idGeometryA.size() ||
        *stateOfSolver_resources.pNumContacts > idGeometryA_buffer.size()) {
        contactEventArraysResize(*stateOfSolver_resources.pNumContacts);
    }

    GPU_CALL(cudaMemcpy(granData->idGeometryA, granData->idGeometryA_buffer,
                        *stateOfSolver_resources.pNumContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->idGeometryB, granData->idGeometryB_buffer,
                        *stateOfSolver_resources.pNumContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->contactType, granData->contactType_buffer,
                        *stateOfSolver_resources.pNumContacts * sizeof(contact_t), cudaMemcpyDeviceToDevice));
    if (!solverFlags.isHistoryless) {
        // Note we don't have to use dedicated memory space for unpacking contactMapping_buffer contents, because we
        // only use it once per kT update, at the time of unpacking. So let us just use a temp vector to store it.
        size_t mapping_bytes = (*stateOfSolver_resources.pNumContacts) * sizeof(contactPairs_t);
        granData->contactMapping = (contactPairs_t*)stateOfSolver_resources.allocateTempVector1(mapping_bytes);
        GPU_CALL(cudaMemcpy(granData->contactMapping, granData->contactMapping_buffer, mapping_bytes,
                            cudaMemcpyDeviceToDevice));
    }
}

inline void DEMDynamicThread::sendToTheirBuffer() {
    GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_voxelID, granData->voxelID,
                        simParams->nOwnerBodies * sizeof(voxelID_t), cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_locX, granData->locX, simParams->nOwnerBodies * sizeof(subVoxelPos_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_locY, granData->locY, simParams->nOwnerBodies * sizeof(subVoxelPos_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_locZ, granData->locZ, simParams->nOwnerBodies * sizeof(subVoxelPos_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_oriQ0, granData->oriQ0, simParams->nOwnerBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_oriQ1, granData->oriQ1, simParams->nOwnerBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_oriQ2, granData->oriQ2, simParams->nOwnerBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_oriQ3, granData->oriQ3, simParams->nOwnerBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));

    // Family number is a typical changable quantity on-the-fly. If this flag is on, dT is responsible for sending this
    // info to kT.
    if (solverFlags.canFamilyChange) {
        GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_familyID, granData->familyID,
                            simParams->nOwnerBodies * sizeof(family_t), cudaMemcpyDeviceToDevice));
    }
}

inline void DEMDynamicThread::migrateContactHistory() {
    // Use this newHistory to store temporarily the rearranged contact history
    size_t history_arr_bytes = (*stateOfSolver_resources.pNumContacts) * sizeof(float3);
    float3* newHistory = (float3*)stateOfSolver_resources.allocateTempVector2(history_arr_bytes);

    // A sentry array is here to see if there exist a contact that dT thinks it's alive but kT doesn't map it to the new
    // history array
    size_t sentry_bytes = (*stateOfSolver_resources.pNumPrevContacts) * sizeof(notStupidBool_t);
    notStupidBool_t* contactSentry = (notStupidBool_t*)stateOfSolver_resources.allocateTempVector3(sentry_bytes);
    // GPU_CALL(cudaMemset(contactSentry, 0, sentry_bytes));
    size_t blocks_needed_for_rearrange =
        (*stateOfSolver_resources.pNumPrevContacts + SGPS_DEM_MAX_THREADS_PER_BLOCK - 1) /
        SGPS_DEM_MAX_THREADS_PER_BLOCK;
    if (blocks_needed_for_rearrange > 0) {
        prep_force_kernels->kernel("markAliveContacts")
            .instantiate()
            .configure(dim3(blocks_needed_for_rearrange), dim3(SGPS_DEM_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
            .launch(granData->contactHistory, contactSentry, *stateOfSolver_resources.pNumPrevContacts);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    }

    blocks_needed_for_rearrange =
        (*stateOfSolver_resources.pNumContacts + SGPS_DEM_MAX_THREADS_PER_BLOCK - 1) / SGPS_DEM_MAX_THREADS_PER_BLOCK;
    if (blocks_needed_for_rearrange > 0) {
        prep_force_kernels->kernel("rearrangeContactHistory")
            .instantiate()
            .configure(dim3(blocks_needed_for_rearrange), dim3(SGPS_DEM_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
            .launch(granData->contactMapping, granData->contactHistory, newHistory, contactSentry,
                    *stateOfSolver_resources.pNumContacts);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    }

    // Take a look, does the sentry indicate that there is an `alive' contact got lost?
    if (*stateOfSolver_resources.pNumPrevContacts > 0) {
        notStupidBool_t* lostContact =
            (notStupidBool_t*)stateOfSolver_resources.allocateTempVector4(sizeof(notStupidBool_t));
        flagMaxReduce(contactSentry, lostContact, *stateOfSolver_resources.pNumPrevContacts, streamInfo.stream,
                      stateOfSolver_resources);
        if (*lostContact && solverFlags.isAsync) {
            SGPS_DEM_INFO_STEP_WARN(
                "At least one contact is active at time %.9g on dT, but it is not detected on kT, therefore being "
                "removed unexpectedly!",
                timeElapsed);
            SGPS_DEM_DEBUG_PRINTF("New contact A:");
            SGPS_DEM_DEBUG_EXEC(displayArray<bodyID_t>(granData->idGeometryA, *stateOfSolver_resources.pNumContacts));
            SGPS_DEM_DEBUG_PRINTF("New contact B:");
            SGPS_DEM_DEBUG_EXEC(displayArray<bodyID_t>(granData->idGeometryB, *stateOfSolver_resources.pNumContacts));
            SGPS_DEM_DEBUG_PRINTF("Old contact history:");
            SGPS_DEM_DEBUG_EXEC(displayFloat3(granData->contactHistory, *stateOfSolver_resources.pNumPrevContacts));
            SGPS_DEM_DEBUG_PRINTF("History mapping:");
            SGPS_DEM_DEBUG_EXEC(
                displayArray<contactPairs_t>(granData->contactMapping, *stateOfSolver_resources.pNumContacts));
            SGPS_DEM_DEBUG_PRINTF("Sentry:");
            SGPS_DEM_DEBUG_EXEC(
                displayArray<notStupidBool_t>(contactSentry, *stateOfSolver_resources.pNumPrevContacts));
        }
    }

    // Copy new history back to history array (after resizing the `main' history array)
    if (*stateOfSolver_resources.pNumContacts > contactHistory.size()) {
        SGPS_DEM_TRACKED_RESIZE_NOPRINT(contactHistory, *stateOfSolver_resources.pNumContacts);
        granData->contactHistory = contactHistory.data();
    }
    GPU_CALL(cudaMemcpy(granData->contactHistory, newHistory, (*stateOfSolver_resources.pNumContacts) * sizeof(float3),
                        cudaMemcpyDeviceToDevice));
}

inline void DEMDynamicThread::calculateForces() {
    // reset force (acceleration) arrays for this time step and apply gravity
    size_t threads_needed_for_prep = simParams->nOwnerBodies > *stateOfSolver_resources.pNumContacts
                                         ? simParams->nOwnerBodies
                                         : *stateOfSolver_resources.pNumContacts;
    size_t blocks_needed_for_prep =
        (threads_needed_for_prep + SGPS_DEM_MAX_THREADS_PER_BLOCK - 1) / SGPS_DEM_MAX_THREADS_PER_BLOCK;

    prep_force_kernels->kernel("prepareForceArrays")
        .instantiate()
        .configure(dim3(blocks_needed_for_prep), dim3(SGPS_DEM_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
        .launch(simParams, granData, *stateOfSolver_resources.pNumContacts);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

    // TODO: is there a better way??? Like memset?
    // GPU_CALL(cudaMemset(granData->contactForces, zeros, *stateOfSolver_resources.pNumContacts * sizeof(float3)));
    // GPU_CALL(cudaMemset(granData->alphaX, 0, simParams->nOwnerBodies * sizeof(float)));
    // GPU_CALL(cudaMemset(granData->alphaY, 0, simParams->nOwnerBodies * sizeof(float)));
    // GPU_CALL(cudaMemset(granData->alphaZ, 0, simParams->nOwnerBodies * sizeof(float)));
    // GPU_CALL(cudaMemset(granData->aX,
    //                     (double)simParams->h * (double)simParams->h * (double)simParams->Gx / (double)simParams->l,
    //                     simParams->nOwnerBodies * sizeof(float)));
    // GPU_CALL(cudaMemset(granData->aY,
    //                     (double)simParams->h * (double)simParams->h * (double)simParams->Gy / (double)simParams->l,
    //                     simParams->nOwnerBodies * sizeof(float)));
    // GPU_CALL(cudaMemset(granData->aZ,
    //                     (double)simParams->h * (double)simParams->h * (double)simParams->Gz / (double)simParams->l,
    //                     simParams->nOwnerBodies * sizeof(float)));

    size_t blocks_needed_for_contacts =
        (*stateOfSolver_resources.pNumContacts + SGPS_DEM_NUM_BODIES_PER_BLOCK - 1) / SGPS_DEM_NUM_BODIES_PER_BLOCK;
    // If no contact then we don't have to calculate forces. Note there might still be forces, coming from prescription
    // or other sources.
    if (blocks_needed_for_contacts > 0) {
        // a custom kernel to compute forces
        cal_force_kernels->kernel("calculateContactForces")
            .instantiate()
            .configure(dim3(blocks_needed_for_contacts), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
            .launch(simParams, granData, *stateOfSolver_resources.pNumContacts);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        // displayFloat3(granData->contactForces, *stateOfSolver_resources.pNumContacts);
        // std::cout << "===========================" << std::endl;

        // Reflect those body-wise forces on their owner clumps
        // hostCollectForces(granData->inertiaPropOffsets, granData->idGeometryA, granData->idGeometryB,
        //                   granData->contactForces, granData->aX, granData->aY, granData->aZ,
        //                   granData->ownerClumpBody, granData->massOwnerBody, simParams->h,
        //                   *stateOfSolver_resources.pNumContacts,simParams->l);
        collectContactForces(collect_force_kernels, granData->inertiaPropOffsets, granData->idGeometryA,
                             granData->idGeometryB, granData->contactType, granData->contactForces,
                             granData->contactPointGeometryA, granData->contactPointGeometryB, granData->aX,
                             granData->aY, granData->aZ, granData->alphaX, granData->alphaY, granData->alphaZ,
                             granData->ownerClumpBody, *stateOfSolver_resources.pNumContacts, simParams->nOwnerBodies,
                             contactPairArr_isFresh, streamInfo.stream, stateOfSolver_resources);
        // displayArray<float>(granData->aX, simParams->nOwnerBodies);
        // displayFloat3(granData->contactForces, *stateOfSolver_resources.pNumContacts);
        // std::cout << *stateOfSolver_resources.pNumContacts << std::endl;

        // Calculate the torque on owner clumps from those body-wise forces
        // hostCollectTorques(granData->inertiaPropOffsets, granData->idGeometryA, granData->idGeometryB,
        //                    granData->contactForces, granData->contactPointGeometryA, granData->contactPointGeometryB,
        //                    granData->alphaX, granData->alphaY, granData->alphaZ, granData->ownerClumpBody,
        //                    granData->mmiXX, granData->mmiYY, granData->mmiZZ, simParams->h,
        //                    *stateOfSolver_resources.pNumContacts, simParams->l);
        // displayArray<float>(granData->oriQ0, simParams->nOwnerBodies);
        // displayArray<float>(granData->oriQ1, simParams->nOwnerBodies);
        // displayArray<float>(granData->oriQ2, simParams->nOwnerBodies);
        // displayArray<float>(granData->oriQ3, simParams->nOwnerBodies);
        // displayArray<float>(granData->alphaX, simParams->nOwnerBodies);
        // displayArray<float>(granData->alphaY, simParams->nOwnerBodies);
        // displayArray<float>(granData->alphaZ, simParams->nOwnerBodies);
    }
}

inline void DEMDynamicThread::integrateClumpMotions() {
    size_t blocks_needed_for_clumps =
        (simParams->nOwnerBodies + SGPS_DEM_NUM_BODIES_PER_BLOCK - 1) / SGPS_DEM_NUM_BODIES_PER_BLOCK;
    integrator_kernels->kernel("integrateClumps")
        .instantiate()
        .configure(dim3(blocks_needed_for_clumps), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
        .launch(granData, simParams->h, timeElapsed);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
}

inline void DEMDynamicThread::routineChecks() {
    if (solverFlags.canFamilyChange) {
        size_t blocks_needed_for_clumps =
            (simParams->nOwnerBodies + SGPS_DEM_NUM_BODIES_PER_BLOCK - 1) / SGPS_DEM_NUM_BODIES_PER_BLOCK;
        mod_kernels->kernel("applyFamilyChanges")
            .instantiate()
            .configure(dim3(blocks_needed_for_clumps), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
            .launch(granData, simParams->h, timeElapsed);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    }
}

void DEMDynamicThread::workerThread() {
    // Set the gpu for this thread
    cudaSetDevice(streamInfo.device);
    cudaStreamCreate(&streamInfo.stream);
    {
        int totGPU;
        cudaGetDeviceCount(&totGPU);
        SGPS_DEM_INFO("Number of total active devices: %d", totGPU);
    }

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

        // There is only one situation where dT needs to wait for kT to provide one initial CD result...
        // This is the `new-boot' case, where stampLastUpdateOfDynamic == -1; in any other situations, dT does not have
        // `drift-into-future-too-much' problem here, b/c if it has the problem then it would have been addressed at the
        // end of last DoStepDynamics call, the final `ShouldWait' check.
        if (pSchedSupport->stampLastUpdateOfDynamic < 0) {
            // In this `new-boot' case, we send kT a work order, b/c dT needs results from CD to proceed. After this one
            // instance, kT and dT may work in an async fashion.
            {
                std::lock_guard<std::mutex> lock(pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
                sendToTheirBuffer();
            }
            pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = true;
            contactPairArr_isFresh = true;
            pSchedSupport->schedulingStats.nKinematicUpdates++;
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
        }

        for (int cycle = 0; cycle < nDynamicCycles; cycle++) {
            // If the produce is fresh, use it
            if (pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh) {
                {
                    // Acquire lock and use the content of the dynamic-owned transfer buffer
                    std::lock_guard<std::mutex> lock(pSchedSupport->dynamicOwnedBuffer_AccessCoordination);
                    unpackMyBuffer();
                    // Leave myself a mental note that I just obtained new produce from kT
                    contactPairArr_isFresh = true;
                }
                // dT got the produce, now mark its buffer to be no longer fresh
                pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = false;
                pSchedSupport->stampLastUpdateOfDynamic = (pSchedSupport->currentStampOfDynamic).load();

                // If this is a history-based run, then when contacts are received, we need to migrate the contact
                // history info, to match the structure of the new contact array
                if (!solverFlags.isHistoryless) {
                    migrateContactHistory();
                }
            }

            calculateForces();
            routineChecks();
            integrateClumpMotions();

            // TODO: make changes for variable time step size cases
            timeElapsed += simParams->h;

            // CalculateForces is done, set it to false
            // This will be set to true next time it receives an update from kT
            contactPairArr_isFresh = false;

            /*
            if (cycle == (nDynamicCycles - 1))
                pSchedSupport->dynamicDone = true;
            */

            // Dynamic wrapped up one cycle
            pSchedSupport->currentStampOfDynamic++;

            // If the kinematic is idle, give it the opportunity to get busy again
            if (!pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh) {
                // Acquire lock and refresh the work order for the kinematic
                {
                    std::lock_guard<std::mutex> lock(pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
                    sendToTheirBuffer();
                }
                pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = true;
                pSchedSupport->schedulingStats.nKinematicUpdates++;
                // Signal the kinematic that it has data for a new work order
                pSchedSupport->cv_KinematicCanProceed.notify_all();
            }

            // Check if we need to wait; i.e., if dynamic drifted too much into future, then we must wait a bit before
            // the next cycle begins
            if (pSchedSupport->dynamicShouldWait()) {
                // Wait for a signal from kT to indicate that kT has caught up
                pSchedSupport->schedulingStats.nTimesDynamicHeldBack++;
                std::unique_lock<std::mutex> lock(pSchedSupport->dynamicCanProceed);
                while (!pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh) {
                    // Loop to avoid spurious wakeups
                    pSchedSupport->cv_DynamicCanProceed.wait(lock);
                }
            }
            // NOTE: This ShouldWait check is at the end of a dT cycle, not at the beginning, because dT could start a
            // cycle and immediately be too much into future. If this is at the beginning, dT will stale while kT could
            // also be chilling waiting for an update; but since it's at the end of a cycle, kT should got busy already
            // due to that kinematicOwned_Cons2ProdBuffer_isFresh got set to true just before.
        }

        // When getting here, dT has finished one user call (although perhaps not at the end of the user script)
        pPagerToMain->userCallDone = true;
        pPagerToMain->cv_mainCanProceed.notify_all();
    }
}

void DEMDynamicThread::startThread() {
    std::lock_guard<std::mutex> lock(pSchedSupport->dynamicStartLock);
    pSchedSupport->dynamicStarted = true;
    pSchedSupport->cv_DynamicStartLock.notify_one();
}

void DEMDynamicThread::resetUserCallStat() {
    // Reset last kT-side data receiving cycle time stamp.
    pSchedSupport->stampLastUpdateOfDynamic = -1;
    pSchedSupport->currentStampOfDynamic = 0;
    // Reset dT stats variables, making ready for next user call
    pSchedSupport->dynamicDone = false;
    pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = false;
    contactPairArr_isFresh = true;
}

size_t DEMDynamicThread::estimateMemUsage() const {
    return m_approx_bytes_used;
}

void DEMDynamicThread::jitifyKernels(const std::unordered_map<std::string, std::string>& templateSubs,
                                     const std::unordered_map<std::string, std::string>& templateAcqSubs,
                                     const std::unordered_map<std::string, std::string>& simParamSubs,
                                     const std::unordered_map<std::string, std::string>& massMatSubs,
                                     const std::unordered_map<std::string, std::string>& familyMaskSubs,
                                     const std::unordered_map<std::string, std::string>& familyPrescribeSubs,
                                     const std::unordered_map<std::string, std::string>& familyChangesSubs,
                                     const std::unordered_map<std::string, std::string>& analGeoSubs,
                                     const std::unordered_map<std::string, std::string>& forceModelSubs) {
    // First one is force array preparation kernels
    {
        std::unordered_map<std::string, std::string> pfSubs = templateSubs;
        pfSubs.insert(simParamSubs.begin(), simParamSubs.end());
        prep_force_kernels = std::make_shared<jitify::Program>(
            std::move(JitHelper::buildProgram("DEMPrepForceKernels", JitHelper::KERNEL_DIR / "DEMPrepForceKernels.cu",
                                              pfSubs, {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
    // Then force calculation kernels
    // Depending on historyless-ness, we may want jitify different versions of kernels
    {
        std::unordered_map<std::string, std::string> cfSubs = templateSubs;
        cfSubs.insert(templateAcqSubs.begin(), templateAcqSubs.end());
        cfSubs.insert(simParamSubs.begin(), simParamSubs.end());
        cfSubs.insert(massMatSubs.begin(), massMatSubs.end());
        cfSubs.insert(analGeoSubs.begin(), analGeoSubs.end());
        cfSubs.insert(forceModelSubs.begin(), forceModelSubs.end());
        if (solverFlags.isHistoryless) {
            cal_force_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
                "DEMHistorylessForceKernels", JitHelper::KERNEL_DIR / "DEMHistorylessForceKernels.cu", cfSubs,
                {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
        } else {
            cal_force_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
                "DEMHistoryBasedForceKernels", JitHelper::KERNEL_DIR / "DEMHistoryBasedForceKernels.cu", cfSubs,
                {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
        }
    }
    // Then force accumulation kernels
    {
        std::unordered_map<std::string, std::string> cfSubs = massMatSubs;
        cfSubs.insert(simParamSubs.begin(), simParamSubs.end());
        cfSubs.insert(analGeoSubs.begin(), analGeoSubs.end());
        collect_force_kernels = std::make_shared<jitify::Program>(std::move(
            JitHelper::buildProgram("DEMCollectForceKernels", JitHelper::KERNEL_DIR / "DEMCollectForceKernels.cu",
                                    cfSubs, {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
    // Then integration kernels
    {
        std::unordered_map<std::string, std::string> intSubs = simParamSubs;
        intSubs.insert(familyPrescribeSubs.begin(), familyPrescribeSubs.end());
        integrator_kernels = std::make_shared<jitify::Program>(std::move(
            JitHelper::buildProgram("DEMIntegrationKernels", JitHelper::KERNEL_DIR / "DEMIntegrationKernels.cu",
                                    intSubs, {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
    // Then kernels that are... wildcards, which make on-the-fly changes to solver data
    if (solverFlags.canFamilyChange) {
        std::unordered_map<std::string, std::string> modSubs = simParamSubs;
        modSubs.insert(massMatSubs.begin(), massMatSubs.end());
        modSubs.insert(familyChangesSubs.begin(), familyChangesSubs.end());
        mod_kernels = std::make_shared<jitify::Program>(
            std::move(JitHelper::buildProgram("DEMModeratorKernels", JitHelper::KERNEL_DIR / "DEMModeratorKernels.cu",
                                              modSubs, {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
    // Then quarrying kernels
    {
        std::unordered_map<std::string, std::string> qSubs = massMatSubs;
        qSubs.insert(simParamSubs.begin(), simParamSubs.end());
        quarry_stats_kernels = std::make_shared<jitify::Program>(
            std::move(JitHelper::buildProgram("DEMQuarryKernels", JitHelper::KERNEL_DIR / "DEMQuarryKernels.cu", qSubs,
                                              {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
}

float3 DEMDynamicThread::getOwnerAngVel(bodyID_t ownerID) const {
    float3 angVel;
    angVel.x = omgBarX.at(ownerID);
    angVel.y = omgBarY.at(ownerID);
    angVel.z = omgBarZ.at(ownerID);
    return angVel;
}

float3 DEMDynamicThread::getOwnerPosition(bodyID_t ownerID) const {
    float3 pos;
    double X, Y, Z;
    voxelID_t voxel = voxelID.at(ownerID);
    subVoxelPos_t subVoxX = locX.at(ownerID);
    subVoxelPos_t subVoxY = locY.at(ownerID);
    subVoxelPos_t subVoxZ = locZ.at(ownerID);
    hostVoxelID2Position<double, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel, subVoxX, subVoxY, subVoxZ, simParams->nvXp2,
                                                           simParams->nvYp2, simParams->voxelSize, simParams->l);
    pos.x = X + simParams->LBFX;
    pos.y = Y + simParams->LBFY;
    pos.z = Z + simParams->LBFZ;
    return pos;
}

}  // namespace sgps
