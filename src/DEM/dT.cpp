//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <cstring>
#include <iostream>
#include <thread>
#include <algorithm>

#include <chpf.hpp>
#include <core/ApiVersion.h>
#include <core/utils/JitHelper.h>
#include <DEM/dT.h>
#include <DEM/kT.h>
#include <DEM/HostSideHelpers.hpp>
#include <nvmath/helper_math.cuh>
#include <DEM/Defines.h>

#include <algorithms/DEMCubBasedSubroutines.h>

namespace deme {

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
    granData->oriQw = oriQw.data();
    granData->oriQx = oriQx.data();
    granData->oriQy = oriQy.data();
    granData->oriQz = oriQz.data();
    granData->omgBarX = omgBarX.data();
    granData->omgBarY = omgBarY.data();
    granData->omgBarZ = omgBarZ.data();
    granData->alphaX = alphaX.data();
    granData->alphaY = alphaY.data();
    granData->alphaZ = alphaZ.data();
    granData->idGeometryA = idGeometryA.data();
    granData->idGeometryB = idGeometryB.data();
    granData->contactType = contactType.data();
    granData->familyMasks = familyMaskMatrix.data();

    // granData->idGeometryA_buffer = idGeometryA_buffer.data();
    // granData->idGeometryB_buffer = idGeometryB_buffer.data();
    // granData->contactType_buffer = contactType_buffer.data();
    // granData->contactMapping_buffer = contactMapping_buffer.data();

    granData->contactForces = contactForces.data();
    granData->contactTorque_convToForce = contactTorque_convToForce.data();
    granData->contactPointGeometryA = contactPointGeometryA.data();
    granData->contactPointGeometryB = contactPointGeometryB.data();
    // granData->contactHistory = contactHistory.data();
    // granData->contactDuration = contactDuration.data();
    for (unsigned int i = 0; i < simParams->nContactWildcards; i++) {
        granData->contactWildcards[i] = contactWildcards[i].data();
    }
    for (unsigned int i = 0; i < simParams->nOwnerWildcards; i++) {
        granData->ownerWildcards[i] = ownerWildcards[i].data();
    }

    // The offset info that indexes into the template arrays
    granData->ownerClumpBody = ownerClumpBody.data();
    granData->clumpComponentOffset = clumpComponentOffset.data();
    granData->clumpComponentOffsetExt = clumpComponentOffsetExt.data();
    granData->sphereMaterialOffset = sphereMaterialOffset.data();
    granData->volumeOwnerBody = volumeOwnerBody.data();

    // Mesh-related
    granData->ownerMesh = ownerMesh.data();
    granData->relPosNode1 = relPosNode1.data();
    granData->relPosNode2 = relPosNode2.data();
    granData->relPosNode3 = relPosNode3.data();
    granData->triMaterialOffset = triMaterialOffset.data();

    // Template array pointers
    granData->radiiSphere = radiiSphere.data();
    granData->relPosSphereX = relPosSphereX.data();
    granData->relPosSphereY = relPosSphereY.data();
    granData->relPosSphereZ = relPosSphereZ.data();
    granData->massOwnerBody = massOwnerBody.data();
    granData->mmiXX = mmiXX.data();
    granData->mmiYY = mmiYY.data();
    granData->mmiZZ = mmiZZ.data();
}

void DEMDynamicThread::packTransferPointers(DEMKinematicThread*& kT) {
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

void DEMDynamicThread::changeFamily(unsigned int ID_from, unsigned int ID_to) {
    family_t ID_from_impl = ID_from;
    family_t ID_to_impl = ID_to;
    std::replace_if(
        familyID.begin(), familyID.end(), [ID_from_impl](family_t& i) { return i == ID_from_impl; }, ID_to_impl);
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
                                    float expand_factor,
                                    float approx_max_vel,
                                    float expand_safety_param,
                                    const std::set<std::string>& contact_wildcards,
                                    const std::set<std::string>& owner_wildcards) {
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

    simParams->nContactWildcards = contact_wildcards.size();
    simParams->nOwnerWildcards = owner_wildcards.size();

    m_contact_wildcard_names = contact_wildcards;
    m_owner_wildcard_names = owner_wildcards;
}

float DEMDynamicThread::getKineticEnergy() {
    // // We can use temp vectors as we please
    // size_t quarryTempSize = (size_t)simParams->nOwnerBodies * sizeof(double);
    // double* KEArr = (double*)stateOfSolver_resources.allocateTempVector(1, quarryTempSize);
    // size_t returnSize = sizeof(double);
    // double* KE = (double*)stateOfSolver_resources.allocateTempVector(2, returnSize);
    // size_t blocks_needed_for_KE =
    //     (simParams->nOwnerBodies + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    // quarry_stats_kernels->kernel("computeKE")
    //     .instantiate()
    //     .configure(dim3(blocks_needed_for_KE), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
    //     .launch(granData, simParams->nOwnerBodies, KEArr);
    // GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    // // displayArray<double>(KEArr, simParams->nOwnerBodies);
    // doubleSumReduce(KEArr, KE, simParams->nOwnerBodies, streamInfo.stream, stateOfSolver_resources);
    // return *KE;
    return 0;
}

void DEMDynamicThread::changeOwnerSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors) {
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
    size_t blocks_needed_for_marking = (IDs.size() + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;

    // Mark on the bool array those owners that need a change
    misc_kernels->kernel("markOwnerToChange")
        .instantiate()
        .configure(dim3(blocks_needed_for_marking), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
        .launch(idBool, ownerFactors, dIDs, dFactors, IDs.size());
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

    // Change the size of the sphere components in question
    size_t blocks_needed_for_changing =
        (simParams->nSpheresGM + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    misc_kernels->kernel("dTModifyComponents")
        .instantiate()
        .configure(dim3(blocks_needed_for_changing), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
        .launch(granData, idBool, ownerFactors, simParams->nSpheresGM);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

    // cudaStreamDestroy(new_stream);
}

void DEMDynamicThread::allocateManagedArrays(size_t nOwnerBodies,
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
    GPU_CALL(cudaSetDevice(streamInfo.device));

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
    DEME_TRACKED_RESIZE(familyID, nOwnerBodies, "familyID", 0);
    DEME_TRACKED_RESIZE(voxelID, nOwnerBodies, "voxelID", 0);
    DEME_TRACKED_RESIZE(locX, nOwnerBodies, "locX", 0);
    DEME_TRACKED_RESIZE(locY, nOwnerBodies, "locY", 0);
    DEME_TRACKED_RESIZE(locZ, nOwnerBodies, "locZ", 0);
    DEME_TRACKED_RESIZE(oriQw, nOwnerBodies, "oriQw", 1);
    DEME_TRACKED_RESIZE(oriQx, nOwnerBodies, "oriQx", 0);
    DEME_TRACKED_RESIZE(oriQy, nOwnerBodies, "oriQy", 0);
    DEME_TRACKED_RESIZE(oriQz, nOwnerBodies, "oriQz", 0);
    DEME_TRACKED_RESIZE(vX, nOwnerBodies, "vX", 0);
    DEME_TRACKED_RESIZE(vY, nOwnerBodies, "vY", 0);
    DEME_TRACKED_RESIZE(vZ, nOwnerBodies, "vZ", 0);
    DEME_TRACKED_RESIZE(omgBarX, nOwnerBodies, "omgBarX", 0);
    DEME_TRACKED_RESIZE(omgBarY, nOwnerBodies, "omgBarY", 0);
    DEME_TRACKED_RESIZE(omgBarZ, nOwnerBodies, "omgBarZ", 0);
    DEME_TRACKED_RESIZE(aX, nOwnerBodies, "aX", 0);
    DEME_TRACKED_RESIZE(aY, nOwnerBodies, "aY", 0);
    DEME_TRACKED_RESIZE(aZ, nOwnerBodies, "aZ", 0);
    DEME_TRACKED_RESIZE(alphaX, nOwnerBodies, "alphaX", 0);
    DEME_TRACKED_RESIZE(alphaY, nOwnerBodies, "alphaY", 0);
    DEME_TRACKED_RESIZE(alphaZ, nOwnerBodies, "alphaZ", 0);

    // Resize the family mask `matrix' (in fact it is flattened)
    DEME_TRACKED_RESIZE(familyMaskMatrix, (NUM_AVAL_FAMILIES - 1) * NUM_AVAL_FAMILIES / 2, "familyMaskMatrix",
                        DONT_PREVENT_CONTACT);

    // Resize to the number of geometries
    DEME_TRACKED_RESIZE(ownerClumpBody, nSpheresGM, "ownerClumpBody", 0);
    DEME_TRACKED_RESIZE(sphereMaterialOffset, nSpheresGM, "sphereMaterialOffset", 0);
    // For clump component offset, it's only needed if clump components are jitified
    if (solverFlags.useClumpJitify) {
        DEME_TRACKED_RESIZE(clumpComponentOffset, nSpheresGM, "clumpComponentOffset", 0);
        // This extended component offset array can hold offset numbers even for big clumps (whereas
        // clumpComponentOffset is typically uint_8, so it may not). If a sphere's component offset index falls in this
        // range then it is not jitified, and the kernel needs to look for it in the global memory.
        DEME_TRACKED_RESIZE(clumpComponentOffsetExt, nSpheresGM, "clumpComponentOffsetExt", 0);
        DEME_TRACKED_RESIZE(radiiSphere, nClumpComponents, "radiiSphere", 0);
        DEME_TRACKED_RESIZE(relPosSphereX, nClumpComponents, "relPosSphereX", 0);
        DEME_TRACKED_RESIZE(relPosSphereY, nClumpComponents, "relPosSphereY", 0);
        DEME_TRACKED_RESIZE(relPosSphereZ, nClumpComponents, "relPosSphereZ", 0);
    } else {
        DEME_TRACKED_RESIZE(radiiSphere, nSpheresGM, "radiiSphere", 0);
        DEME_TRACKED_RESIZE(relPosSphereX, nSpheresGM, "relPosSphereX", 0);
        DEME_TRACKED_RESIZE(relPosSphereY, nSpheresGM, "relPosSphereY", 0);
        DEME_TRACKED_RESIZE(relPosSphereZ, nSpheresGM, "relPosSphereZ", 0);
    }

    // Resize to the number of triangle facets
    DEME_TRACKED_RESIZE(ownerMesh, nTriGM, "ownerMesh", 0);
    DEME_TRACKED_RESIZE(relPosNode1, nTriGM, "relPosNode1", make_float3(0));
    DEME_TRACKED_RESIZE(relPosNode2, nTriGM, "relPosNode2", make_float3(0));
    DEME_TRACKED_RESIZE(relPosNode3, nTriGM, "relPosNode3", make_float3(0));
    DEME_TRACKED_RESIZE(triMaterialOffset, nTriGM, "triMaterialOffset", 0);

    // Resize to the number of analytical geometries
    DEME_TRACKED_RESIZE(ownerAnalBody, nAnalGM, "ownerAnalBody", 0);

    // Resize to number of owners
    DEME_TRACKED_RESIZE(ownerTypes, nOwnerBodies, "ownerTypes", 0);
    DEME_TRACKED_RESIZE(inertiaPropOffsets, nOwnerBodies, "inertiaPropOffsets", 0);
    // If we jitify mass properties, then
    if (solverFlags.useMassJitify) {
        DEME_TRACKED_RESIZE(massOwnerBody, nMassProperties, "massOwnerBody", 0);
        DEME_TRACKED_RESIZE(mmiXX, nMassProperties, "mmiXX", 0);
        DEME_TRACKED_RESIZE(mmiYY, nMassProperties, "mmiYY", 0);
        DEME_TRACKED_RESIZE(mmiZZ, nMassProperties, "mmiZZ", 0);
    } else {
        DEME_TRACKED_RESIZE(massOwnerBody, nOwnerBodies, "massOwnerBody", 0);
        DEME_TRACKED_RESIZE(mmiXX, nOwnerBodies, "mmiXX", 0);
        DEME_TRACKED_RESIZE(mmiYY, nOwnerBodies, "mmiYY", 0);
        DEME_TRACKED_RESIZE(mmiZZ, nOwnerBodies, "mmiZZ", 0);
    }
    // Volume info is jitified
    DEME_TRACKED_RESIZE(volumeOwnerBody, nMassProperties, "volumeOwnerBody", 0);

    // Arrays for contact info
    // The lengths of contact event-based arrays are just estimates. My estimate of total contact pairs is ~ 2n, and I
    // think the max is 6n (although I can't prove it). Note the estimate should be large enough to decrease the number
    // of reallocations in the simulation, but not too large that eats too much memory.
    {
        // In any case, in this initialization process we should not make contact arrays smaller than it used to be, or
        // we may lose data. Also, if this is a new-boot, we allocate this array for at least
        // nSpheresGM*DEME_INIT_CNT_MULTIPLIER elements.
        size_t cnt_arr_size =
            DEME_MAX(*stateOfSolver_resources.pNumContacts + nExtraContacts, nSpheresGM * DEME_INIT_CNT_MULTIPLIER);
        DEME_TRACKED_RESIZE(idGeometryA, cnt_arr_size, "idGeometryA", 0);
        DEME_TRACKED_RESIZE(idGeometryB, cnt_arr_size, "idGeometryB", 0);
        DEME_TRACKED_RESIZE(contactType, cnt_arr_size, "contactType", NOT_A_CONTACT);

        if (!solverFlags.useNoContactRecord) {
            DEME_TRACKED_RESIZE(contactForces, cnt_arr_size, "contactForces", make_float3(0));
            DEME_TRACKED_RESIZE(contactTorque_convToForce, cnt_arr_size, "contactTorque_convToForce", make_float3(0));
            DEME_TRACKED_RESIZE(contactPointGeometryA, cnt_arr_size, "contactPointGeometryA", make_float3(0));
            DEME_TRACKED_RESIZE(contactPointGeometryB, cnt_arr_size, "contactPointGeometryB", make_float3(0));
        }
        // Allocate memory for each wildcard array
        contactWildcards.resize(simParams->nContactWildcards);
        ownerWildcards.resize(simParams->nOwnerWildcards);
        for (unsigned int i = 0; i < simParams->nContactWildcards; i++) {
            DEME_TRACKED_RESIZE_FLOAT(contactWildcards[i], cnt_arr_size, 0);
        }
        for (unsigned int i = 0; i < simParams->nOwnerWildcards; i++) {
            DEME_TRACKED_RESIZE_FLOAT(ownerWildcards[i], nOwnerBodies, 0);
        }
    }

    // Transfer buffer arrays
    // The following several arrays will have variable sizes, so here we only used an estimate.
    // It is cudaMalloc-ed memory, not managed, because we want explicit locality control of buffers
    buffer_size = nSpheresGM * DEME_INIT_CNT_MULTIPLIER;
    DEME_DEVICE_PTR_ALLOC(granData->idGeometryA_buffer, buffer_size);
    DEME_DEVICE_PTR_ALLOC(granData->idGeometryB_buffer, buffer_size);
    DEME_DEVICE_PTR_ALLOC(granData->contactType_buffer, buffer_size);
    // DEME_TRACKED_RESIZE(idGeometryA_buffer, nSpheresGM * DEME_INIT_CNT_MULTIPLIER, "idGeometryA_buffer",
    // 0); DEME_TRACKED_RESIZE(idGeometryB_buffer, nSpheresGM * DEME_INIT_CNT_MULTIPLIER,
    // "idGeometryB_buffer", 0); DEME_TRACKED_RESIZE(contactType_buffer, nSpheresGM *
    // DEME_INIT_CNT_MULTIPLIER, "contactType_buffer", NOT_A_CONTACT);
    // DEME_ADVISE_DEVICE(idGeometryA_buffer, streamInfo.device);
    // DEME_ADVISE_DEVICE(idGeometryB_buffer, streamInfo.device);
    // DEME_ADVISE_DEVICE(contactType_buffer, streamInfo.device);
    if (!solverFlags.isHistoryless) {
        // DEME_TRACKED_RESIZE(contactMapping_buffer, nSpheresGM * DEME_INIT_CNT_MULTIPLIER,
        //                         "contactMapping_buffer", NULL_MAPPING_PARTNER);
        // DEME_ADVISE_DEVICE(contactMapping_buffer, streamInfo.device);
        DEME_DEVICE_PTR_ALLOC(granData->contactMapping_buffer, buffer_size);
    }
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
            massOwnerBody.at(k) = clump_templates.mass.at(i);
            float3 this_moi = clump_templates.MOI.at(i);
            mmiXX.at(k) = this_moi.x;
            mmiYY.at(k) = this_moi.y;
            mmiZZ.at(k) = this_moi.z;
        }
        // Volume info is always registered, and even if the user does not use mass/MOI jitify, volume info may be
        // needed in void ratio computation
        volumeOwnerBody.at(k) = clump_templates.volume.at(i);
        k++;
    }
    for (unsigned int i = 0; i < ext_obj_mass_types.size(); i++) {
        if (solverFlags.useMassJitify) {
            massOwnerBody.at(k) = ext_obj_mass_types.at(i);
            float3 this_moi = ext_obj_moi_types.at(i);
            mmiXX.at(k) = this_moi.x;
            mmiYY.at(k) = this_moi.y;
            mmiZZ.at(k) = this_moi.z;
        }
        // Currently analytical object volume is not used
        k++;
    }
    for (unsigned int i = 0; i < mesh_obj_mass_types.size(); i++) {
        if (solverFlags.useMassJitify) {
            massOwnerBody.at(k) = mesh_obj_mass_types.at(i);
            float3 this_moi = mesh_obj_moi_types.at(i);
            mmiXX.at(k) = this_moi.x;
            mmiYY.at(k) = this_moi.y;
            mmiZZ.at(k) = this_moi.z;
        }
        // Currently mesh volume is not used
        k++;
    }

    // Store family mask
    for (size_t i = 0; i < family_mask_matrix.size(); i++)
        familyMaskMatrix.at(i) = family_mask_matrix.at(i);

    // Store clump naming map
    templateNumNameMap = template_number_name_map;

    // Take notes of the families that should not be outputted
    {
        std::set<unsigned int>::iterator it;
        unsigned int i = 0;
        familiesNoOutput.resize(no_output_families.size());
        for (it = no_output_families.begin(); it != no_output_families.end(); it++, i++) {
            familiesNoOutput.at(i) = *it;
        }
        std::sort(familiesNoOutput.begin(), familiesNoOutput.end());
        DEME_DEBUG_PRINTF("Impl-level families that will not be outputted:");
        DEME_DEBUG_EXEC(displayArray<family_t>(familiesNoOutput.data(), familiesNoOutput.size()));
    }
}

void DEMDynamicThread::populateEntityArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                            const std::vector<float3>& input_ext_obj_xyz,
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
                radiiSphere.at(k) = radius;
                k++;
            }
            prescans_comp.push_back(k);
        }
        prescans_comp.pop_back();
        k = 0;

        for (const auto& elem : clump_templates.spRelPos) {
            for (const auto& loc : elem) {
                relPosSphereX.at(k) = loc.x;
                relPosSphereY.at(k) = loc.y;
                relPosSphereZ.at(k) = loc.z;
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
        // Pop family number-related warning only once
        bool pop_family_msg = false;
        // Keep tab of the number of sphere components processed in this initialization call, especially if there are
        // multiple batches loaded for this initialization call
        size_t n_processed_sp_comp = 0;
        // This number serves as an offset for loading existing contact pairs/history. Contact array should have been
        // enlarged for loading these user-manually added contact pairs. Those pairs go after existing contact pairs.
        size_t cnt_arr_offset = *stateOfSolver_resources.pNumContacts;
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
            if ((!a_batch->family_isSpecified) && (!pop_family_msg)) {
                DEME_WARNING("Some clumps do not have their family numbers specified, so defaulted to %u",
                             DEFAULT_CLUMP_FAMILY_NUM);
                pop_family_msg = true;
            }
            const std::vector<unsigned int>& input_clump_family = a_batch->families;

            for (size_t j = 0; j < a_batch->GetNumClumps(); j++) {
                // If got here, this is a clump
                ownerTypes.at(nExistOwners + i) = OWNER_T_CLUMP;

                auto type_of_this_clump = type_marks.at(j);
                inertiaPropOffsets.at(nExistOwners + i) = type_of_this_clump;
                if (!solverFlags.useMassJitify) {
                    massOwnerBody.at(nExistOwners + i) = clump_templates.mass.at(type_of_this_clump);
                    const float3 this_moi = clump_templates.MOI.at(type_of_this_clump);
                    mmiXX.at(nExistOwners + i) = this_moi.x;
                    mmiYY.at(nExistOwners + i) = this_moi.y;
                    mmiZZ.at(nExistOwners + i) = this_moi.z;
                }

                auto this_CoM_coord = input_clump_xyz.at(j) - LBF;

                auto this_clump_no_sp_radii = clump_templates.spRadii.at(type_of_this_clump);
                auto this_clump_no_sp_relPos = clump_templates.spRelPos.at(type_of_this_clump);
                auto this_clump_no_sp_mat_ids = clump_templates.matIDs.at(type_of_this_clump);

                for (size_t jj = 0; jj < this_clump_no_sp_radii.size(); jj++) {
                    sphereMaterialOffset.at(nExistSpheres + k) = this_clump_no_sp_mat_ids.at(jj);
                    ownerClumpBody.at(nExistSpheres + k) = nExistOwners + i;

                    // Depending on whether we jitify or flatten
                    if (solverFlags.useClumpJitify) {
                        // This component offset, is it too large that can't live in the jitified array?
                        unsigned int this_comp_offset = prescans_comp.at(type_of_this_clump) + jj;
                        clumpComponentOffsetExt.at(nExistSpheres + k) = this_comp_offset;
                        if (this_comp_offset < simParams->nJitifiableClumpComponents) {
                            clumpComponentOffset.at(nExistSpheres + k) = this_comp_offset;
                        } else {
                            // If not, an indicator will be put there
                            clumpComponentOffset.at(nExistSpheres + k) = RESERVED_CLUMP_COMPONENT_OFFSET;
                        }
                    } else {
                        radiiSphere.at(nExistSpheres + k) = this_clump_no_sp_radii.at(jj);
                        const float3 relPos = this_clump_no_sp_relPos.at(jj);
                        relPosSphereX.at(nExistSpheres + k) = relPos.x;
                        relPosSphereY.at(nExistSpheres + k) = relPos.y;
                        relPosSphereZ.at(nExistSpheres + k) = relPos.z;
                    }

                    k++;
                    // std::cout << "Sphere Rel Pos offset: " << this_clump_no_sp_loc_offsets.at(j) << std::endl;
                }

                hostPositionToVoxelID<voxelID_t, subVoxelPos_t, double>(
                    voxelID.at(nExistOwners + i), locX.at(nExistOwners + i), locY.at(nExistOwners + i),
                    locZ.at(nExistOwners + i), (double)this_CoM_coord.x, (double)this_CoM_coord.y,
                    (double)this_CoM_coord.z, simParams->nvXp2, simParams->nvYp2, simParams->voxelSize, simParams->l);

                // Set initial oriQ
                auto oriQ_of_this_clump = input_clump_oriQ.at(j);
                oriQw.at(nExistOwners + i) = oriQ_of_this_clump.w;
                oriQx.at(nExistOwners + i) = oriQ_of_this_clump.x;
                oriQy.at(nExistOwners + i) = oriQ_of_this_clump.y;
                oriQz.at(nExistOwners + i) = oriQ_of_this_clump.z;

                // Set initial velocity
                auto vel_of_this_clump = input_clump_vel.at(j);
                vX.at(nExistOwners + i) = vel_of_this_clump.x;
                vY.at(nExistOwners + i) = vel_of_this_clump.y;
                vZ.at(nExistOwners + i) = vel_of_this_clump.z;

                // Set initial angular velocity
                auto angVel_of_this_clump = input_clump_angVel.at(j);
                omgBarX.at(nExistOwners + i) = angVel_of_this_clump.x;
                omgBarY.at(nExistOwners + i) = angVel_of_this_clump.y;
                omgBarZ.at(nExistOwners + i) = angVel_of_this_clump.z;

                // Set family code
                family_t this_family_num = input_clump_family.at(j);
                familyID.at(nExistOwners + i) = this_family_num;

                i++;
            }
            // If this batch as owner wildcards, we load it in
            {
                unsigned int w_num = 0;
                for (const auto& w_name : m_owner_wildcard_names) {
                    if (a_batch->owner_wildcards.find(w_name) == a_batch->owner_wildcards.end()) {
                        // No such wildcard loaded
                        DEME_WARNING(
                            "Owner wildcard %s is needed by force model, yet not specified for a batch of "
                            "clumps.\nTheir initial values are defauled to 0.",
                            w_name.c_str());
                    } else {
                        for (size_t jj = 0; jj < a_batch->GetNumClumps(); jj++) {
                            ownerWildcards[w_num].at(nExistOwners + nTotalClumpsThisCall + jj) =
                                a_batch->owner_wildcards[w_name].at(jj);
                        }
                    }
                    w_num++;
                }
            }

            DEME_DEBUG_PRINTF("Loaded a batch of %zu clumps.", a_batch->GetNumClumps());

            // Write the extra contact pairs to memory
            for (size_t jj = 0; jj < a_batch->GetNumContacts(); jj++) {
                const auto& idPair = a_batch->contact_pairs.at(jj);
                // idPair.first + n_processed_sp_comp can take into account the sphere components that have been loaded
                // in previous batches, makes this loading process scalable.
                idGeometryA.at(cnt_arr_offset) = idPair.first + n_processed_sp_comp + nExistSpheres;
                idGeometryB.at(cnt_arr_offset) = idPair.second + n_processed_sp_comp + nExistSpheres;
                contactType.at(cnt_arr_offset) = SPHERE_SPHERE_CONTACT;  // Only sph--sph cnt for now
                unsigned int w_num = 0;
                for (const auto& w_name : m_contact_wildcard_names) {
                    contactWildcards[w_num].at(cnt_arr_offset) = a_batch->contact_wildcards.at(w_name).at(jj);
                    w_num++;
                }
                cnt_arr_offset++;
            }

            // Make ready for the next batch, update contact history offset
            n_processed_sp_comp = k;
        }

        DEME_DEBUG_PRINTF("Total number of transferred clumps this time: %zu", i);
        DEME_DEBUG_PRINTF("Total number of existing owners in simulation: %zu", nExistOwners);
        DEME_DEBUG_PRINTF("Total number of owners in simulation after this init call: %zu", simParams->nOwnerBodies);
        nTotalClumpsThisCall = i;

        // If user loaded contact pairs, we need to inform kT on the first time step...
        if (cnt_arr_offset > *stateOfSolver_resources.pNumContacts) {
            *stateOfSolver_resources.pNumContacts = cnt_arr_offset;
            new_contacts_loaded = true;
            DEME_DEBUG_PRINTF("Total number of contact pairs this sim starts with: %zu",
                              *stateOfSolver_resources.pNumContacts);
        }
    }

    // Load in initial positions and mass properties for the owners of those external objects
    // They go after clump owners
    k = 0;
    size_t owner_offset_for_ext_obj = nExistOwners + nTotalClumpsThisCall;
    unsigned int offset_for_ext_obj_mass_template = simParams->nDistinctClumpBodyTopologies;
    for (size_t i = 0; i < input_ext_obj_xyz.size(); i++) {
        // If got here, it is an analytical obj
        ownerTypes.at(i + owner_offset_for_ext_obj) = OWNER_T_ANALYTICAL;
        // For each analytical geometry component of this obj, it needs to know its owner number
        for (size_t j = 0; j < ext_obj_comp_num.at(i); j++) {
            ownerAnalBody.at(k) = i + owner_offset_for_ext_obj;
            k++;
        }

        // Analytical object mass properties are useful in force collection, but not useful in force calculation:
        // analytical component masses are jitified into kernels directly.
        inertiaPropOffsets.at(i + owner_offset_for_ext_obj) = i + offset_for_ext_obj_mass_template;
        if (!solverFlags.useMassJitify) {
            massOwnerBody.at(i + owner_offset_for_ext_obj) = ext_obj_mass_types.at(i);
            const float3 this_moi = ext_obj_moi_types.at(i);
            mmiXX.at(i + owner_offset_for_ext_obj) = this_moi.x;
            mmiYY.at(i + owner_offset_for_ext_obj) = this_moi.y;
            mmiZZ.at(i + owner_offset_for_ext_obj) = this_moi.z;
        }
        auto this_CoM_coord = input_ext_obj_xyz.at(i) - LBF;
        hostPositionToVoxelID<voxelID_t, subVoxelPos_t, double>(
            voxelID.at(i + owner_offset_for_ext_obj), locX.at(i + owner_offset_for_ext_obj),
            locY.at(i + owner_offset_for_ext_obj), locZ.at(i + owner_offset_for_ext_obj), (double)this_CoM_coord.x,
            (double)this_CoM_coord.y, (double)this_CoM_coord.z, simParams->nvXp2, simParams->nvYp2,
            simParams->voxelSize, simParams->l);
        //// TODO: and initial rot?
        //// TODO: and initial vel?

        family_t this_family_num = input_ext_obj_family.at(i);
        familyID.at(i + owner_offset_for_ext_obj) = this_family_num;
    }

    // Load in initial positions and mass properties for the owners of the meshed objects
    // They go after analytical object owners
    size_t owner_offset_for_mesh_obj = owner_offset_for_ext_obj + input_ext_obj_xyz.size();
    unsigned int offset_for_mesh_obj_mass_template = offset_for_ext_obj_mass_template + input_ext_obj_xyz.size();
    // k for indexing the triangle facets
    k = 0;
    for (size_t i = 0; i < input_mesh_objs.size(); i++) {
        // If got here, it is a mesh
        ownerTypes.at(i + owner_offset_for_mesh_obj) = OWNER_T_MESH;

        // Store this mesh in dT's cache
        input_mesh_objs.at(i)->owner = i + owner_offset_for_mesh_obj;
        m_meshes.push_back(input_mesh_objs.at(i));

        inertiaPropOffsets.at(i + owner_offset_for_mesh_obj) = i + offset_for_mesh_obj_mass_template;
        if (!solverFlags.useMassJitify) {
            massOwnerBody.at(i + owner_offset_for_mesh_obj) = mesh_obj_mass_types.at(i);
            const float3 this_moi = mesh_obj_moi_types.at(i);
            mmiXX.at(i + owner_offset_for_mesh_obj) = this_moi.x;
            mmiYY.at(i + owner_offset_for_mesh_obj) = this_moi.y;
            mmiZZ.at(i + owner_offset_for_mesh_obj) = this_moi.z;
        }
        auto this_CoM_coord = input_mesh_obj_xyz.at(i) - LBF;
        hostPositionToVoxelID<voxelID_t, subVoxelPos_t, double>(
            voxelID.at(i + owner_offset_for_mesh_obj), locX.at(i + owner_offset_for_mesh_obj),
            locY.at(i + owner_offset_for_mesh_obj), locZ.at(i + owner_offset_for_mesh_obj), (double)this_CoM_coord.x,
            (double)this_CoM_coord.y, (double)this_CoM_coord.z, simParams->nvXp2, simParams->nvYp2,
            simParams->voxelSize, simParams->l);

        // Set mesh owner's oriQ
        auto oriQ_of_this = input_mesh_obj_rot.at(i);
        oriQw.at(i + owner_offset_for_mesh_obj) = oriQ_of_this.w;
        oriQx.at(i + owner_offset_for_mesh_obj) = oriQ_of_this.x;
        oriQy.at(i + owner_offset_for_mesh_obj) = oriQ_of_this.y;
        oriQz.at(i + owner_offset_for_mesh_obj) = oriQ_of_this.z;

        //// TODO: and initial vel?

        // Per-facet info
        size_t this_facet_owner = mesh_facet_owner.at(k);
        for (; k < mesh_facet_owner.size(); k++) {
            // mesh_facet_owner run length is the num of facets in this mesh entity
            if (mesh_facet_owner.at(k) != this_facet_owner)
                break;
            ownerMesh.at(nExistingFacets + k) = owner_offset_for_mesh_obj + this_facet_owner;
            triMaterialOffset.at(nExistingFacets + k) = mesh_facet_materials.at(k);
            DEMTriangle this_tri = mesh_facets.at(k);
            relPosNode1.at(nExistingFacets + k) = this_tri.p1;
            relPosNode2.at(nExistingFacets + k) = this_tri.p2;
            relPosNode3.at(nExistingFacets + k) = this_tri.p3;
        }

        family_t this_family_num = input_mesh_obj_family.at(i);
        familyID.at(i + owner_offset_for_mesh_obj) = this_family_num;

        DEME_DEBUG_PRINTF("dT just loaded a mesh in family %u", +(this_family_num));
        DEME_DEBUG_PRINTF("This mesh is owner %u", (i + owner_offset_for_mesh_obj));
        DEME_DEBUG_PRINTF("Number of triangle facets loaded thus far: %zu", k);
    }
}

void DEMDynamicThread::buildTrackedObjs(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                        const std::vector<float3>& input_ext_obj_xyz,
                                        const std::vector<std::shared_ptr<DEMMeshConnected>>& input_mesh_objs,
                                        std::vector<std::shared_ptr<DEMTrackedObj>>& tracked_objs,
                                        size_t nExistOwners,
                                        size_t nExistingFacets) {
    // We take notes on how many clumps each batch has, it will be useful when we assemble the tracker information
    std::vector<size_t> prescans_batch_size;
    prescans_batch_size.push_back(0);
    for (const auto& a_batch : input_clump_batches) {
        prescans_batch_size.push_back(prescans_batch_size.back() + a_batch->GetNumClumps());
    }
    // Also take notes on num of facets of each mesh obj
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
        switch (tracked_obj->type) {
            case (OWNER_TYPE::CLUMP):
                tracked_obj->ownerID = nExistOwners + prescans_batch_size.at(tracked_obj->load_order);
                tracked_obj->nSpanOwners = prescans_batch_size.at(tracked_obj->load_order + 1) -
                                           prescans_batch_size.at(tracked_obj->load_order);
                break;
            case (OWNER_TYPE::ANALYTICAL):
                // prescans_batch_size.back() is the total num of loaded clumps this time
                tracked_obj->ownerID = nExistOwners + tracked_obj->load_order + prescans_batch_size.back();
                tracked_obj->nSpanOwners = 1;
                break;
            case (OWNER_TYPE::MESH):
                tracked_obj->ownerID =
                    nExistOwners + input_ext_obj_xyz.size() + prescans_batch_size.back() + tracked_obj->load_order;
                tracked_obj->nSpanOwners = 1;
                tracked_obj->facetID = nExistingFacets + prescans_mesh_size.at(tracked_obj->load_order);
                tracked_obj->nFacets =
                    prescans_mesh_size.at(tracked_obj->load_order + 1) - prescans_mesh_size.at(tracked_obj->load_order);
                break;
            default:
                DEME_ERROR("A DEM tracked object has an unknown type.");
        }
    }
    nTrackersProcessed = tracked_objs.size();
    DEME_DEBUG_PRINTF("Total number of trackers on the record: %u", nTrackersProcessed);
}

void DEMDynamicThread::initManagedArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                         const std::vector<float3>& input_ext_obj_xyz,
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
    // Get the info into the managed memory from the host side. Can this process be more efficient? Maybe, but it's
    // initialization anyway.

    registerPolicies(template_number_name_map, clump_templates, ext_obj_mass_types, ext_obj_moi_types,
                     mesh_obj_mass_types, mesh_obj_moi_types, loaded_materials, family_mask_matrix, no_output_families);

    // For initialization, owner array offset is 0
    populateEntityArrays(input_clump_batches, input_ext_obj_xyz, input_ext_obj_family, input_mesh_objs,
                         input_mesh_obj_xyz, input_mesh_obj_rot, input_mesh_obj_family, mesh_facet_owner,
                         mesh_facet_materials, mesh_facets, clump_templates, ext_obj_mass_types, ext_obj_moi_types,
                         ext_obj_comp_num, mesh_obj_mass_types, mesh_obj_moi_types, 0, 0, 0);

    buildTrackedObjs(input_clump_batches, input_ext_obj_xyz, input_mesh_objs, tracked_objs, 0, 0);
}

void DEMDynamicThread::updateClumpMeshArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                             const std::vector<float3>& input_ext_obj_xyz,
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
                                             size_t nExistingFacets) {
    // No policy changes here

    // Analytical objects-related arrays should be empty
    populateEntityArrays(input_clump_batches, input_ext_obj_xyz, input_ext_obj_family, input_mesh_objs,
                         input_mesh_obj_xyz, input_mesh_obj_rot, input_mesh_obj_family, mesh_facet_owner,
                         mesh_facet_materials, mesh_facets, clump_templates, ext_obj_mass_types, ext_obj_moi_types,
                         ext_obj_comp_num, mesh_obj_mass_types, mesh_obj_moi_types, nExistingOwners, nExistingSpheres,
                         nExistingFacets);

    // Make changes to tracked objects (potentially add more)
    buildTrackedObjs(input_clump_batches, input_ext_obj_xyz, input_mesh_objs, tracked_objs, nExistingOwners,
                     nExistingFacets);
}

void DEMDynamicThread::writeSpheresAsChpf(std::ofstream& ptFile) const {
    chpf::Writer pw;
    // pw.write(ptFile, chpf::Compressor::Type::USE_DEFAULT, mass);
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
        hostVoxelIDToPosition<float, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel, subVoxX, subVoxY, subVoxZ,
                                                               simParams->nvXp2, simParams->nvYp2, simParams->voxelSize,
                                                               simParams->l);
        CoM.x = X + simParams->LBFX;
        CoM.y = Y + simParams->LBFY;
        CoM.z = Z + simParams->LBFZ;

        size_t compOffset = (solverFlags.useClumpJitify) ? clumpComponentOffsetExt.at(i) : i;
        float this_sp_deviation_x = relPosSphereX.at(compOffset);
        float this_sp_deviation_y = relPosSphereY.at(compOffset);
        float this_sp_deviation_z = relPosSphereZ.at(compOffset);
        float this_sp_rot_0 = oriQw.at(this_owner);
        float this_sp_rot_1 = oriQx.at(this_owner);
        float this_sp_rot_2 = oriQy.at(this_owner);
        float this_sp_rot_3 = oriQz.at(this_owner);
        hostApplyOriQToVector3<float, float>(this_sp_deviation_x, this_sp_deviation_y, this_sp_deviation_z,
                                             this_sp_rot_0, this_sp_rot_1, this_sp_rot_2, this_sp_rot_3);
        posX.at(num_output_spheres) = CoM.x + this_sp_deviation_x;
        posY.at(num_output_spheres) = CoM.y + this_sp_deviation_y;
        posZ.at(num_output_spheres) = CoM.z + this_sp_deviation_z;
        // std::cout << "Sphere Pos: " << posX.at(i) << ", " << posY.at(i) << ", " << posZ.at(i) << std::endl;

        spRadii.at(num_output_spheres) = radiiSphere.at(compOffset);

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
        // TODO: How to do that?
        // pw.write(ptFile, chpf::Compressor::Type::USE_DEFAULT, {}, families);
    }
}

void DEMDynamicThread::writeSpheresAsCsv(std::ofstream& ptFile) const {
    std::ostringstream outstrstream;

    outstrstream << OUTPUT_FILE_X_COL_NAME + "," + OUTPUT_FILE_Y_COL_NAME + "," + OUTPUT_FILE_Z_COL_NAME + "," +
                        OUTPUT_FILE_R_COL_NAME;

    if (solverFlags.outputFlags & OUTPUT_CONTENT::ABSV) {
        outstrstream << ",absv";
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::VEL) {
        outstrstream << ",v_x,v_y,v_z";
    }
    // if (solverFlags.outputFlags & OUTPUT_CONTENT::ANG_VEL) {
    //     outstrstream << ",w_x,w_y,w_z";
    // }
    // if (solverFlags.outputFlags & OUTPUT_CONTENT::ACC) {
    //     outstrstream << ",a_x,a_y,a_z";
    // }
    // if (solverFlags.outputFlags & OUTPUT_CONTENT::ANG_ACC) {
    //     outstrstream << ",alpha_x,alpha_y,alpha_z";
    // }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
        outstrstream << ",family";
    }
    // if (solverFlags.outputFlags & OUTPUT_CONTENT::MAT) {
    //     outstrstream << ",material";
    // }
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
        hostVoxelIDToPosition<float, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel, subVoxX, subVoxY, subVoxZ,
                                                               simParams->nvXp2, simParams->nvYp2, simParams->voxelSize,
                                                               simParams->l);
        CoM.x = X + simParams->LBFX;
        CoM.y = Y + simParams->LBFY;
        CoM.z = Z + simParams->LBFZ;

        size_t compOffset = (solverFlags.useClumpJitify) ? clumpComponentOffsetExt.at(i) : i;
        float3 this_sp_deviation;
        this_sp_deviation.x = relPosSphereX.at(compOffset);
        this_sp_deviation.y = relPosSphereY.at(compOffset);
        this_sp_deviation.z = relPosSphereZ.at(compOffset);
        float this_sp_rot_0 = oriQw.at(this_owner);
        float this_sp_rot_1 = oriQx.at(this_owner);
        float this_sp_rot_2 = oriQy.at(this_owner);
        float this_sp_rot_3 = oriQz.at(this_owner);
        hostApplyOriQToVector3<float, float>(this_sp_deviation.x, this_sp_deviation.y, this_sp_deviation.z,
                                             this_sp_rot_0, this_sp_rot_1, this_sp_rot_2, this_sp_rot_3);
        pos = CoM + this_sp_deviation;
        outstrstream << pos.x << "," << pos.y << "," << pos.z;

        radius = radiiSphere.at(compOffset);
        outstrstream << "," << radius;

        // Only linear velocity
        float3 vxyz;
        vxyz.x = vX.at(this_owner);
        vxyz.y = vY.at(this_owner);
        vxyz.z = vZ.at(this_owner);
        if (solverFlags.outputFlags & OUTPUT_CONTENT::ABSV) {
            outstrstream << "," << length(vxyz);
        }
        if (solverFlags.outputFlags & OUTPUT_CONTENT::VEL) {
            outstrstream << "," << vxyz.x << "," << vxyz.y << "," << vxyz.z;
        }

        // Family number needs to be user number
        if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
            outstrstream << "," << +(this_family);
        }

        outstrstream << "\n";
    }

    ptFile << outstrstream.str();
}

void DEMDynamicThread::writeClumpsAsChpf(std::ofstream& ptFile, unsigned int accuracy) const {}

void DEMDynamicThread::writeClumpsAsCsv(std::ofstream& ptFile, unsigned int accuracy) const {
    std::ostringstream outstrstream;
    outstrstream.precision(accuracy);

    // xyz and quaternion are always there
    outstrstream << OUTPUT_FILE_X_COL_NAME + "," + OUTPUT_FILE_Y_COL_NAME + "," + OUTPUT_FILE_Z_COL_NAME +
                        ",Qw,Qx,Qy,Qz," + OUTPUT_FILE_CLUMP_TYPE_NAME;
    if (solverFlags.outputFlags & OUTPUT_CONTENT::ABSV) {
        outstrstream << ",absv";
    }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::VEL) {
        outstrstream << ",v_x,v_y,v_z";
    }
    // if (solverFlags.outputFlags & OUTPUT_CONTENT::ANG_VEL) {
    //     outstrstream << ",w_x,w_y,w_z";
    // }
    // if (solverFlags.outputFlags & OUTPUT_CONTENT::ACC) {
    //     outstrstream << ",a_x,a_y,a_z";
    // }
    // if (solverFlags.outputFlags & OUTPUT_CONTENT::ANG_ACC) {
    //     outstrstream << ",alpha_x,alpha_y,alpha_z";
    // }
    if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
        outstrstream << ",family";
    }
    outstrstream << "\n";

    for (size_t i = 0; i < simParams->nOwnerBodies; i++) {
        // i is this owner's number. And if it is not a clump, we can move on.
        if (ownerTypes.at(i) != OWNER_T_CLUMP)
            continue;

        family_t this_family = familyID.at(i);
        // If this (impl-level) family is in the no-output list, skip it
        if (std::binary_search(familiesNoOutput.begin(), familiesNoOutput.end(), this_family)) {
            continue;
        }

        float3 CoM;
        float X, Y, Z;
        voxelID_t voxel = voxelID.at(i);
        subVoxelPos_t subVoxX = locX.at(i);
        subVoxelPos_t subVoxY = locY.at(i);
        subVoxelPos_t subVoxZ = locZ.at(i);
        hostVoxelIDToPosition<float, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel, subVoxX, subVoxY, subVoxZ,
                                                               simParams->nvXp2, simParams->nvYp2, simParams->voxelSize,
                                                               simParams->l);
        CoM.x = X + simParams->LBFX;
        CoM.y = Y + simParams->LBFY;
        CoM.z = Z + simParams->LBFZ;
        // Output position
        outstrstream << CoM.x << "," << CoM.y << "," << CoM.z;

        // Then quaternions
        outstrstream << "," << oriQw.at(i) << "," << oriQx.at(i) << "," << oriQy.at(i) << "," << oriQz.at(i);

        // Then type of clump
        unsigned int clump_mark = inertiaPropOffsets.at(i);
        outstrstream << "," << templateNumNameMap.at(clump_mark);

        // Only linear velocity
        float3 vxyz;
        vxyz.x = vX.at(i);
        vxyz.y = vY.at(i);
        vxyz.z = vZ.at(i);
        if (solverFlags.outputFlags & OUTPUT_CONTENT::ABSV) {
            outstrstream << "," << length(vxyz);
        }
        if (solverFlags.outputFlags & OUTPUT_CONTENT::VEL) {
            outstrstream << "," << vxyz.x << "," << vxyz.y << "," << vxyz.z;
        }

        // Family number needs to be user number
        if (solverFlags.outputFlags & OUTPUT_CONTENT::FAMILY) {
            outstrstream << "," << +(this_family);
        }

        outstrstream << "\n";
    }

    ptFile << outstrstream.str();
}

void DEMDynamicThread::writeContactsAsCsv(std::ofstream& ptFile, float force_thres) const {
    std::ostringstream outstrstream;

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
    if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::POINT) {
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
    if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::TORQUE_ONLY_FORCE) {
        outstrstream << "," + OUTPUT_FILE_TOF_X_NAME + "," + OUTPUT_FILE_TOF_Y_NAME + "," + OUTPUT_FILE_TOF_Z_NAME;
    }
    if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::WILDCARD) {
        // Write all wildcard names as header
        for (const auto& w_name : m_contact_wildcard_names) {
            outstrstream << "," + w_name;
        }
    }
    outstrstream << "\n";

    for (size_t i = 0; i < *(stateOfSolver_resources.pNumContacts); i++) {
        // Geos that are involved in this contact
        auto geoA = idGeometryA.at(i);
        auto geoB = idGeometryB.at(i);
        auto type = contactType.at(i);
        // We don't output fake contacts; but right now, no contact will be marked fake by kT, so no need to check that
        // if (type == NOT_A_CONTACT)
        //     continue;

        float3 forcexyz = contactForces.at(i);
        float3 torque = contactTorque_convToForce.at(i);
        // If this force+torque is too small, then it's not an active contact
        if (length(forcexyz + torque) < force_thres) {
            continue;
        }

        // geoA's owner must be a sphere
        auto ownerA = ownerClumpBody.at(geoA);
        bodyID_t ownerB;
        // geoB's owner depends...
        switch (type) {
            case (SPHERE_SPHERE_CONTACT):
                ownerB = ownerClumpBody.at(geoB);
                break;
            case (SPHERE_MESH_CONTACT):
                ownerB = ownerMesh.at(geoB);
                break;
            default:  // Default is sphere--analytical
                ownerB = ownerAnalBody.at(geoB);
        }

        // Type is mapped to SS, SM and such....
        outstrstream << contact_type_out_name_map.at(type);

        // (Internal) ownerID and/or geometry ID
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::OWNER) {
            outstrstream << "," << ownerA << "," << ownerB;
        }
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::GEO_ID) {
            outstrstream << "," << geoA << "," << geoB;
        }

        // Force is already in global...
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::FORCE) {
            outstrstream << "," << forcexyz.x << "," << forcexyz.y << "," << forcexyz.z;
        }

        // Contact point is in local frame. To make it global, first map that vector to axis-aligned global frame, then
        // add the location of body A CoM
        float4 oriQA;
        float3 CoM, cntPntA;  // A's CoM
        {
            oriQA.w = oriQw.at(ownerA);
            oriQA.x = oriQx.at(ownerA);
            oriQA.y = oriQy.at(ownerA);
            oriQA.z = oriQz.at(ownerA);
            voxelID_t voxel = voxelID.at(ownerA);
            subVoxelPos_t subVoxX = locX.at(ownerA);
            subVoxelPos_t subVoxY = locY.at(ownerA);
            subVoxelPos_t subVoxZ = locZ.at(ownerA);
            hostVoxelIDToPosition<float, voxelID_t, subVoxelPos_t>(CoM.x, CoM.y, CoM.z, voxel, subVoxX, subVoxY,
                                                                   subVoxZ, simParams->nvXp2, simParams->nvYp2,
                                                                   simParams->voxelSize, simParams->l);
            CoM.x += simParams->LBFX;
            CoM.y += simParams->LBFY;
            CoM.z += simParams->LBFZ;
            cntPntA = contactPointGeometryA.at(i);
            hostApplyOriQToVector3(cntPntA.x, cntPntA.y, cntPntA.z, oriQA.w, oriQA.x, oriQA.y, oriQA.z);
            cntPntA += CoM;
        }
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::POINT) {
            // oriQ is updated already... whereas the contact point is effectively last step's... That's unfortunate.
            // Should we do somthing ahout it?
            outstrstream << "," << cntPntA.x << "," << cntPntA.y << "," << cntPntA.z;
        }

        // To get contact normal: it's just contact point - sphereA center, that gives you the outward normal for body A
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::NORMAL) {
            size_t compOffset = (solverFlags.useClumpJitify) ? clumpComponentOffsetExt.at(geoA) : geoA;
            float3 this_sp_deviation;
            this_sp_deviation.x = relPosSphereX.at(compOffset);
            this_sp_deviation.y = relPosSphereY.at(compOffset);
            this_sp_deviation.z = relPosSphereZ.at(compOffset);
            hostApplyOriQToVector3<float, float>(this_sp_deviation.x, this_sp_deviation.y, this_sp_deviation.z, oriQA.w,
                                                 oriQA.x, oriQA.y, oriQA.z);
            float3 pos = CoM + this_sp_deviation;
            float3 normal = normalize(cntPntA - pos);
            outstrstream << "," << normal.x << "," << normal.y << "," << normal.z;
        }

        // Torque is in global already...
        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::TORQUE_ONLY_FORCE) {
            outstrstream << "," << torque.x << "," << torque.y << "," << torque.z;
        }

        if (solverFlags.cntOutFlags & CNT_OUTPUT_CONTENT::WILDCARD) {
            // The order shouldn't be an issue... the same set is being processed here and in equip_contact_wildcards
            for (unsigned int j = 0; j < m_contact_wildcard_names.size(); j++) {
                outstrstream << "," << contactWildcards[j][i];
            }
        }

        outstrstream << "\n";
    }

    ptFile << outstrstream.str();
}

void DEMDynamicThread::writeMeshesAsVtk(std::ofstream& ptFile) {
    std::ostringstream ostream;

    std::vector<size_t> vertexOffset(m_meshes.size() + 1, 0);
    size_t total_f = 0;
    size_t total_v = 0;

    ostream << "# vtk DataFile Version 2.0\n";
    ostream << "VTK from DEM simulation\n";
    ostream << "ASCII\n";
    ostream << "\n\n";

    ostream << "DATASET UNSTRUCTURED_GRID\n";

    // Prescan the V and F: to write all meshes to one file, we need vertex number offset info
    unsigned int mesh_num = 0;
    for (const auto& mmesh : m_meshes) {
        vertexOffset[mesh_num + 1] = mmesh->getCoordsVertices().size();
        total_v += mmesh->getCoordsVertices().size();
        total_f += mmesh->getIndicesVertexes().size();
        mesh_num++;
    }
    for (unsigned int i = 1; i < m_meshes.size(); i++)
        vertexOffset[i] = vertexOffset[i] + vertexOffset[i - 1];

    // Writing m_vertices
    ostream << "POINTS " << total_v << " float" << std::endl;
    mesh_num = 0;
    for (const auto& mmesh : m_meshes) {
        bodyID_t mowner = mmesh->getOwner();
        float3 ownerPos = this->getOwnerPos(mowner);
        float4 ownerOriQ = this->getOwnerOriQ(mowner);
        for (const auto& v : mmesh->getCoordsVertices()) {
            float3 point = v;
            hostApplyFrameTransform(point, ownerPos, ownerOriQ);
            ostream << point.x << " " << point.y << " " << point.z << std::endl;
        }
        mesh_num++;
    }

    // Writing faces
    ostream << "\n\n";
    ostream << "CELLS " << total_f << " " << 4 * total_f << std::endl;
    mesh_num = 0;
    for (const auto& mmesh : m_meshes) {
        for (const auto& f : mmesh->getIndicesVertexes()) {
            ostream << "3 " << (size_t)f.x + vertexOffset[mesh_num] << " " << (size_t)f.y + vertexOffset[mesh_num]
                    << " " << (size_t)f.z + vertexOffset[mesh_num] << std::endl;
        }
        mesh_num++;
    }

    // Writing face types. Type 5 is generally triangles
    ostream << "\n\n";
    ostream << "CELL_TYPES " << total_f << std::endl;
    for (const auto& mmesh : m_meshes) {
        auto nfaces = mmesh->getIndicesVertexes().size();
        for (size_t j = 0; j < nfaces; j++)
            ostream << "5 " << std::endl;
    }

    ptFile << ostream.str();
}

inline void DEMDynamicThread::contactEventArraysResize(size_t nContactPairs) {
    DEME_TRACKED_RESIZE_NOPRINT(idGeometryA, nContactPairs, 0);
    DEME_TRACKED_RESIZE_NOPRINT(idGeometryB, nContactPairs, 0);
    DEME_TRACKED_RESIZE_NOPRINT(contactType, nContactPairs, NOT_A_CONTACT);

    if (!solverFlags.useNoContactRecord) {
        DEME_TRACKED_RESIZE_NOPRINT(contactForces, nContactPairs, make_float3(0));
        DEME_TRACKED_RESIZE_NOPRINT(contactTorque_convToForce, nContactPairs, make_float3(0));
        DEME_TRACKED_RESIZE_NOPRINT(contactPointGeometryA, nContactPairs, make_float3(0));
        DEME_TRACKED_RESIZE_NOPRINT(contactPointGeometryB, nContactPairs, make_float3(0));
    }

    // Re-pack pointers in case the arrays got reallocated
    granData->idGeometryA = idGeometryA.data();
    granData->idGeometryB = idGeometryB.data();
    granData->contactType = contactType.data();
    granData->contactForces = contactForces.data();
    granData->contactTorque_convToForce = contactTorque_convToForce.data();
    granData->contactPointGeometryA = contactPointGeometryA.data();
    granData->contactPointGeometryB = contactPointGeometryB.data();
}

inline void DEMDynamicThread::unpackMyBuffer() {
    // Make a note on the contact number of the previous time step
    *stateOfSolver_resources.pNumPrevContacts = *stateOfSolver_resources.pNumContacts;

    GPU_CALL(cudaMemcpy(stateOfSolver_resources.pNumContacts, &(granData->nContactPairs_buffer), sizeof(size_t),
                        cudaMemcpyDeviceToDevice));

    // Need to resize those contact event-based arrays before usage
    if (*stateOfSolver_resources.pNumContacts > idGeometryA.size() ||
        *stateOfSolver_resources.pNumContacts > buffer_size) {
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
        // only use it once per kT update, at the time of unpacking. So let us just use a temp vector to store it. Note
        // we cannot use vector 0 since it may hold critical flattened owner ID info.
        size_t mapping_bytes = (*stateOfSolver_resources.pNumContacts) * sizeof(contactPairs_t);
        granData->contactMapping = (contactPairs_t*)stateOfSolver_resources.allocateTempVector(1, mapping_bytes);
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
    GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_oriQ0, granData->oriQw, simParams->nOwnerBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_oriQ1, granData->oriQx, simParams->nOwnerBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_oriQ2, granData->oriQy, simParams->nOwnerBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_oriQ3, granData->oriQz, simParams->nOwnerBodies * sizeof(oriQ_t),
                        cudaMemcpyDeviceToDevice));

    // Family number is a typical changable quantity on-the-fly. If this flag is on, dT is responsible for sending this
    // info to kT.
    if (solverFlags.canFamilyChange) {
        GPU_CALL(cudaMemcpy(granData->pKTOwnedBuffer_familyID, granData->familyID,
                            simParams->nOwnerBodies * sizeof(family_t), cudaMemcpyDeviceToDevice));
    }
}

inline void DEMDynamicThread::migratePersistentContacts() {
    // Use this newHistory and newDuration to store temporarily the rearranged contact history.  Note we cannot use
    // vector 0 or 1 since they may be in use (1 is used by granData->contactMapping).

    // All contact wildcards are the same type, so we can just allocate one temp array for all of them
    float* newWildcards[DEME_MAX_WILDCARD_NUM];
    size_t wildcard_arr_bytes = (*stateOfSolver_resources.pNumContacts) * sizeof(float) * simParams->nContactWildcards;
    newWildcards[0] = (float*)stateOfSolver_resources.allocateTempVector(2, wildcard_arr_bytes);
    for (unsigned int i = 1; i < simParams->nContactWildcards; i++) {
        newWildcards[i] = newWildcards[i - 1] + (*stateOfSolver_resources.pNumContacts);
    }

    // This is used for checking if there are contact history got lost in the transition by surprise. But no need to
    // check if the user did not ask for it.
    size_t sentry_bytes = (*stateOfSolver_resources.pNumPrevContacts) * sizeof(notStupidBool_t);
    notStupidBool_t* contactSentry = (notStupidBool_t*)stateOfSolver_resources.allocateTempVector(3, sentry_bytes);

    // A sentry array is here to see if there exist a contact that dT thinks it's alive but kT doesn't map it to the new
    // history array. This is just a quick and rough check: we only look at the last contact wildcard to see if it is
    // non-0, whatever it represents.
    size_t blocks_needed_for_rearrange;
    if (verbosity >= VERBOSITY::STEP_METRIC) {
        if (*stateOfSolver_resources.pNumPrevContacts > 0) {
            // GPU_CALL(cudaMemset(contactSentry, 0, sentry_bytes));
            blocks_needed_for_rearrange = (*stateOfSolver_resources.pNumPrevContacts + DEME_MAX_THREADS_PER_BLOCK - 1) /
                                          DEME_MAX_THREADS_PER_BLOCK;
            if (blocks_needed_for_rearrange > 0) {
                prep_force_kernels->kernel("markAliveContacts")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_rearrange), dim3(DEME_MAX_THREADS_PER_BLOCK), 0,
                               streamInfo.stream)
                    .launch(granData->contactWildcards[simParams->nContactWildcards - 1], contactSentry,
                            *stateOfSolver_resources.pNumPrevContacts);
                GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
            }
        }
    }

    // Rearrange contact histories based on kT instruction
    blocks_needed_for_rearrange =
        (*stateOfSolver_resources.pNumContacts + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed_for_rearrange > 0) {
        prep_force_kernels->kernel("rearrangeContactWildcards")
            .instantiate()
            .configure(dim3(blocks_needed_for_rearrange), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
            .launch(granData, newWildcards[0], contactSentry, simParams->nContactWildcards,
                    *stateOfSolver_resources.pNumContacts);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    }

    // Take a look, does the sentry indicate that there is an `alive' contact got lost?
    if (verbosity >= VERBOSITY::STEP_METRIC) {
        if (*stateOfSolver_resources.pNumPrevContacts > 0 && simParams->nContactWildcards > 0) {
            size_t* lostContact = (size_t*)stateOfSolver_resources.allocateTempVector(4, sizeof(size_t));
            boolSumReduce(contactSentry, lostContact, *stateOfSolver_resources.pNumPrevContacts, streamInfo.stream,
                          stateOfSolver_resources);
            if (*lostContact && solverFlags.isAsync) {
                DEME_STEP_METRIC(
                    "%zu contacts were active at time %.9g on dT, but they are not detected on kT, therefore being "
                    "removed unexpectedly!",
                    *lostContact, simParams->timeElapsed);
                DEME_DEBUG_PRINTF("New number of contacts: %zu", *stateOfSolver_resources.pNumContacts);
                DEME_DEBUG_PRINTF("Old number of contacts: %zu", *stateOfSolver_resources.pNumPrevContacts);
                DEME_DEBUG_PRINTF("New contact A:");
                DEME_DEBUG_EXEC(displayArray<bodyID_t>(granData->idGeometryA, *stateOfSolver_resources.pNumContacts));
                DEME_DEBUG_PRINTF("New contact B:");
                DEME_DEBUG_EXEC(displayArray<bodyID_t>(granData->idGeometryB, *stateOfSolver_resources.pNumContacts));
                DEME_DEBUG_PRINTF("Old version of the last contact wildcard:");
                DEME_DEBUG_EXEC(displayArray<float>(granData->contactWildcards[simParams->nContactWildcards - 1],
                                                    *stateOfSolver_resources.pNumPrevContacts));
                DEME_DEBUG_PRINTF("Old--new mapping:");
                DEME_DEBUG_EXEC(
                    displayArray<contactPairs_t>(granData->contactMapping, *stateOfSolver_resources.pNumContacts));
                DEME_DEBUG_PRINTF("Sentry:");
                DEME_DEBUG_EXEC(
                    displayArray<notStupidBool_t>(contactSentry, *stateOfSolver_resources.pNumPrevContacts));
            }
        }
    }

    // Copy new history back to history array (after resizing the `main' history array)
    if (*stateOfSolver_resources.pNumContacts > contactWildcards[0].size()) {
        for (unsigned int i = 0; i < simParams->nContactWildcards; i++) {
            DEME_TRACKED_RESIZE_FLOAT(contactWildcards[i], *stateOfSolver_resources.pNumContacts, 0);
            granData->contactWildcards[i] = contactWildcards[i].data();
        }
    }
    for (unsigned int i = 0; i < simParams->nContactWildcards; i++) {
        GPU_CALL(cudaMemcpy(granData->contactWildcards[i], newWildcards[i],
                            (*stateOfSolver_resources.pNumContacts) * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

inline void DEMDynamicThread::calculateForces() {
    // reset force (acceleration) arrays for this time step and apply gravity
    size_t nContactPairs = *stateOfSolver_resources.pNumContacts;
    size_t threads_needed_for_prep =
        (simParams->nOwnerBodies > nContactPairs) ? simParams->nOwnerBodies : nContactPairs;
    size_t blocks_needed_for_prep =
        (threads_needed_for_prep + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;

    // prepareForceArrays needs to clear contact force arrays, only if the user asks us to record contact forces. So...
    {
        size_t nContactThatMatters = (solverFlags.useNoContactRecord) ? 0 : nContactPairs;
        prep_force_kernels->kernel("prepareForceArrays")
            .instantiate()
            .configure(dim3(blocks_needed_for_prep), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
            .launch(simParams, granData, nContactThatMatters);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    }

    // TODO: is there a better way??? Like memset?
    // GPU_CALL(cudaMemset(granData->contactForces, zeros, nContactPairs * sizeof(float3)));
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
        (nContactPairs + DT_FORCE_CALC_NTHREADS_PER_BLOCK - 1) / DT_FORCE_CALC_NTHREADS_PER_BLOCK;
    // If no contact then we don't have to calculate forces. Note there might still be forces, coming from prescription
    // or other sources.
    if (blocks_needed_for_contacts > 0) {
        timers.GetTimer("Calculate contact forces").start();
        // a custom kernel to compute forces
        cal_force_kernels->kernel("calculateContactForces")
            .instantiate()
            .configure(dim3(blocks_needed_for_contacts), dim3(DT_FORCE_CALC_NTHREADS_PER_BLOCK), 0, streamInfo.stream)
            .launch(simParams, granData, nContactPairs);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        // displayFloat3(granData->contactForces, nContactPairs);
        // displayArray<contact_t>(granData->contactType, nContactPairs);
        // std::cout << "===========================" << std::endl;
        timers.GetTimer("Calculate contact forces").stop();

        if (!solverFlags.useForceCollectInPlace) {
            timers.GetTimer("Collect contact forces").start();
            // Reflect those body-wise forces on their owner clumps
            if (solverFlags.useCubForceCollect) {
                collectContactForcesThruCub(collect_force_kernels, granData, nContactPairs, simParams->nOwnerBodies,
                                            contactPairArr_isFresh, streamInfo.stream, stateOfSolver_resources, timers);
            } else {
                blocks_needed_for_contacts =
                    (nContactPairs + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
                // This does both acc and ang acc
                collect_force_kernels->kernel("forceToAcc")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_contacts), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
                    .launch(granData, nContactPairs);
                GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
            }
            // displayArray<float>(granData->aZ, simParams->nOwnerBodies);
            // displayFloat3(granData->contactForces, nContactPairs);
            // std::cout << nContactPairs << std::endl;
            timers.GetTimer("Collect contact forces").stop();
        }
    }
}

inline void DEMDynamicThread::integrateOwnerMotions() {
    size_t blocks_needed_for_clumps =
        (simParams->nOwnerBodies + DEME_NUM_BODIES_PER_BLOCK - 1) / DEME_NUM_BODIES_PER_BLOCK;
    integrator_kernels->kernel("integrateOwners")
        .instantiate()
        .configure(dim3(blocks_needed_for_clumps), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
        .launch(simParams, granData);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
}

inline void DEMDynamicThread::routineChecks() {
    if (solverFlags.canFamilyChange) {
        size_t blocks_needed_for_clumps =
            (simParams->nOwnerBodies + DEME_NUM_BODIES_PER_BLOCK - 1) / DEME_NUM_BODIES_PER_BLOCK;
        mod_kernels->kernel("applyFamilyChanges")
            .instantiate()
            .configure(dim3(blocks_needed_for_clumps), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
            .launch(simParams, granData, simParams->nOwnerBodies);
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    }
}

inline void DEMDynamicThread::ifProduceFreshThenUseItAndSendNewOrder() {
    if (pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh) {
        timers.GetTimer("Unpack updates from kT").start();
        {
            // Acquire lock and use the content of the dynamic-owned transfer buffer
            std::lock_guard<std::mutex> lock(pSchedSupport->dynamicOwnedBuffer_AccessCoordination);
            unpackMyBuffer();
            // Leave myself a mental note that I just obtained new produce from kT
            contactPairArr_isFresh = true;
            // pSchedSupport->schedulingStats.nDynamicReceives++;
        }
        // dT got the produce, now mark its buffer to be no longer fresh
        pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = false;
        pSchedSupport->stampLastUpdateOfDynamic = (pSchedSupport->currentStampOfDynamic).load();

        // If this is a history-based run, then when contacts are received, we need to migrate the contact
        // history info, to match the structure of the new contact array
        if (!solverFlags.isHistoryless) {
            migratePersistentContacts();
        }
        timers.GetTimer("Unpack updates from kT").stop();

        timers.GetTimer("Send to kT buffer").start();
        // Acquire lock and refresh the work order for the kinematic
        {
            std::lock_guard<std::mutex> lock(pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
            sendToTheirBuffer();
        }
        pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = true;
        pSchedSupport->schedulingStats.nKinematicUpdates++;
        timers.GetTimer("Send to kT buffer").stop();
        // Signal the kinematic that it has data for a new work order
        pSchedSupport->cv_KinematicCanProceed.notify_all();
    }
}

void DEMDynamicThread::workerThread() {
    // Set the gpu for this thread
    GPU_CALL(cudaSetDevice(streamInfo.device));
    GPU_CALL(cudaStreamCreate(&streamInfo.stream));

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
        if (pSchedSupport->stampLastUpdateOfDynamic < 0 || pendingCriticalUpdate) {
            // If the user loaded contact manually, there is an extra thing we need to do: update kT prev_contact
            // arrays. Note the user can add anything only from a sync-ed stance anyway, so this check needs to be done
            // only here.
            if (new_contacts_loaded) {
                // If wildcard-less, then prev-contact arrays are not important
                if (!solverFlags.isHistoryless) {
                    // Note *stateOfSolver_resources.pNumContacts is now the num of contact after considering the newly
                    // added ones
                    kT->updatePrevContactArrays(granData, *stateOfSolver_resources.pNumContacts);
                }
                new_contacts_loaded = false;
            }

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

        for (double cycle = 0.0; cycle < cycleDuration; cycle += simParams->h) {
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
                pSchedSupport->schedulingStats.nTimesDynamicHeldBack++;
                std::unique_lock<std::mutex> lock(pSchedSupport->dynamicCanProceed);
                while (!pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh) {
                    // Loop to avoid spurious wakeups
                    pSchedSupport->cv_DynamicCanProceed.wait(lock);
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

            //// TODO: make changes for variable time step size cases
            simParams->timeElapsed += (double)simParams->h;
        }

        // Unless the user did something critical, must we wait for a kT update before next step
        pendingCriticalUpdate = false;

        // When getting here, dT has finished one user call (although perhaps not at the end of the user script)
        pPagerToMain->userCallDone = true;
        pPagerToMain->cv_mainCanProceed.notify_all();
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

void DEMDynamicThread::jitifyKernels(const std::unordered_map<std::string, std::string>& Subs) {
    // First one is force array preparation kernels
    {
        prep_force_kernels = std::make_shared<jitify::Program>(
            std::move(JitHelper::buildProgram("DEMPrepForceKernels", JitHelper::KERNEL_DIR / "DEMPrepForceKernels.cu",
                                              Subs, {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
    // Then force calculation kernels
    {
        cal_force_kernels = std::make_shared<jitify::Program>(
            std::move(JitHelper::buildProgram("DEMCalcForceKernels", JitHelper::KERNEL_DIR / "DEMCalcForceKernels.cu",
                                              Subs, {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
    // Then force accumulation kernels
    if (solverFlags.useCubForceCollect) {
        collect_force_kernels = std::make_shared<jitify::Program>(std::move(
            JitHelper::buildProgram("DEMCollectForceKernels", JitHelper::KERNEL_DIR / "DEMCollectForceKernels.cu", Subs,
                                    {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    } else {
        collect_force_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMCollectForceKernels_Compact", JitHelper::KERNEL_DIR / "DEMCollectForceKernels_Compact.cu", Subs,
            {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
    // Then integration kernels
    {
        integrator_kernels = std::make_shared<jitify::Program>(std::move(
            JitHelper::buildProgram("DEMIntegrationKernels", JitHelper::KERNEL_DIR / "DEMIntegrationKernels.cu", Subs,
                                    {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
    // Then kernels that are... wildcards, which make on-the-fly changes to solver data
    if (solverFlags.canFamilyChange) {
        mod_kernels = std::make_shared<jitify::Program>(
            std::move(JitHelper::buildProgram("DEMModeratorKernels", JitHelper::KERNEL_DIR / "DEMModeratorKernels.cu",
                                              Subs, {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
    // Then misc kernels
    {
        misc_kernels = std::make_shared<jitify::Program>(
            std::move(JitHelper::buildProgram("DEMMiscKernels", JitHelper::KERNEL_DIR / "DEMMiscKernels.cu", Subs,
                                              {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
}

float* DEMDynamicThread::inspectCall(const std::shared_ptr<jitify::Program>& inspection_kernel,
                                     const std::string& kernel_name,
                                     size_t n,
                                     CUB_REDUCE_FLAVOR reduce_flavor,
                                     bool all_domain) {
    // We can use temp vectors as we please
    size_t quarryTempSize = n * sizeof(float);
    float* resArr = (float*)stateOfSolver_resources.allocateTempVector(1, quarryTempSize);
    size_t regionTempSize = n * sizeof(notStupidBool_t);
    // If this boolArrExclude is 1 at an element, that means this element is exluded in the reduction
    notStupidBool_t* boolArrExclude = (notStupidBool_t*)stateOfSolver_resources.allocateTempVector(2, regionTempSize);
    GPU_CALL(cudaMemset(boolArrExclude, 0, regionTempSize));

    // We may actually have 2 reduced returns: in regional reduction, key 0 and 1 give one return each.
    size_t returnSize = sizeof(float) * 2;
    float* res = (float*)stateOfSolver_resources.allocateTempVector(3, returnSize);
    size_t blocks_needed = (n + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    inspection_kernel->kernel(kernel_name)
        .instantiate()
        .configure(dim3(blocks_needed), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
        .launch(granData, simParams, resArr, boolArrExclude, n);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

    if (all_domain) {
        switch (reduce_flavor) {
            case (CUB_REDUCE_FLAVOR::MAX):
                floatMaxReduce(resArr, res, n, streamInfo.stream, stateOfSolver_resources);
                break;
            case (CUB_REDUCE_FLAVOR::SUM):
                floatSumReduce(resArr, res, n, streamInfo.stream, stateOfSolver_resources);
                break;
            case (CUB_REDUCE_FLAVOR::NONE):
                //// TODO: Query a full array w/o reducing doesn't seem like something useful...
                return resArr;
        }
        // If this inspection is comfined in a region, then boolArrExclude and resArr need to be sorted and reduce by
        // key
    } else {
        // Extra arrays are needed for sort and reduce by key
        notStupidBool_t* boolArrExclude_sorted =
            (notStupidBool_t*)stateOfSolver_resources.allocateTempVector(4, regionTempSize);
        float* resArr_sorted = (float*)stateOfSolver_resources.allocateTempVector(5, quarryTempSize);
        size_t* num_unique_out = (size_t*)stateOfSolver_resources.allocateTempVector(6, sizeof(size_t));
        switch (reduce_flavor) {
            case (CUB_REDUCE_FLAVOR::MAX):
                //// TODO: Implement it
                break;
            case (CUB_REDUCE_FLAVOR::SUM):
                // Sort first
                floatSortByKey(boolArrExclude, boolArrExclude_sorted, resArr, resArr_sorted, n, streamInfo.stream,
                               stateOfSolver_resources);
                // Then reduce. We care about the sum for 0-marked entries only. Note boolArrExclude here is re-used for
                // storing d_unique_out.
                floatSumReduceByKey(boolArrExclude_sorted, boolArrExclude, resArr_sorted, res, num_unique_out, n,
                                    streamInfo.stream, stateOfSolver_resources);
                break;
        }
    }

    return res;
}

void DEMDynamicThread::setOwnerWildcardValue(unsigned int wc_num, float val) {
    for (size_t i = 0; i < simParams->nOwnerBodies; i++) {
        ownerWildcards[wc_num].at(i) = val;
    }
}

void DEMDynamicThread::setFamilyOwnerWildcardValue(unsigned int family_num, unsigned int wc_num, float val) {
    for (size_t i = 0; i < simParams->nOwnerBodies; i++) {
        if (familyID[i] == family_num) {
            ownerWildcards[wc_num].at(i) = val;
        }
    }
}

float3 DEMDynamicThread::getOwnerAngVel(bodyID_t ownerID) const {
    float3 angVel;
    angVel.x = omgBarX.at(ownerID);
    angVel.y = omgBarY.at(ownerID);
    angVel.z = omgBarZ.at(ownerID);
    return angVel;
}

float4 DEMDynamicThread::getOwnerOriQ(bodyID_t ownerID) const {
    float4 oriQ;
    oriQ.w = oriQw.at(ownerID);
    oriQ.x = oriQx.at(ownerID);
    oriQ.y = oriQy.at(ownerID);
    oriQ.z = oriQz.at(ownerID);
    return oriQ;
}

float3 DEMDynamicThread::getOwnerAcc(bodyID_t ownerID) const {
    float3 acc;
    acc.x = aX.at(ownerID);
    acc.y = aY.at(ownerID);
    acc.z = aZ.at(ownerID);
    return acc;
}

float3 DEMDynamicThread::getOwnerAngAcc(bodyID_t ownerID) const {
    float3 aa;
    aa.x = alphaX.at(ownerID);
    aa.y = alphaY.at(ownerID);
    aa.z = alphaZ.at(ownerID);
    return aa;
}

float3 DEMDynamicThread::getOwnerVel(bodyID_t ownerID) const {
    float3 vel;
    vel.x = vX.at(ownerID);
    vel.y = vY.at(ownerID);
    vel.z = vZ.at(ownerID);
    return vel;
}

float3 DEMDynamicThread::getOwnerPos(bodyID_t ownerID) const {
    float3 pos;
    double X, Y, Z;
    voxelID_t voxel = voxelID.at(ownerID);
    subVoxelPos_t subVoxX = locX.at(ownerID);
    subVoxelPos_t subVoxY = locY.at(ownerID);
    subVoxelPos_t subVoxZ = locZ.at(ownerID);
    hostVoxelIDToPosition<double, voxelID_t, subVoxelPos_t>(X, Y, Z, voxel, subVoxX, subVoxY, subVoxZ, simParams->nvXp2,
                                                            simParams->nvYp2, simParams->voxelSize, simParams->l);
    pos.x = X + simParams->LBFX;
    pos.y = Y + simParams->LBFY;
    pos.z = Z + simParams->LBFZ;
    return pos;
}

void DEMDynamicThread::setOwnerAngVel(bodyID_t ownerID, float3 angVel) {
    omgBarX.at(ownerID) = angVel.x;
    omgBarY.at(ownerID) = angVel.y;
    omgBarZ.at(ownerID) = angVel.z;
}

void DEMDynamicThread::setOwnerPos(bodyID_t ownerID, float3 pos) {
    // Convert to relative pos wrt LBF point first
    double X, Y, Z;
    X = pos.x - simParams->LBFX;
    Y = pos.y - simParams->LBFY;
    Z = pos.z - simParams->LBFZ;
    hostPositionToVoxelID<voxelID_t, subVoxelPos_t, double>(voxelID.at(ownerID), locX.at(ownerID), locY.at(ownerID),
                                                            locZ.at(ownerID), X, Y, Z, simParams->nvXp2,
                                                            simParams->nvYp2, simParams->voxelSize, simParams->l);
}

void DEMDynamicThread::setOwnerOriQ(bodyID_t ownerID, float4 oriQ) {
    oriQw.at(ownerID) = oriQ.w;
    oriQx.at(ownerID) = oriQ.x;
    oriQy.at(ownerID) = oriQ.y;
    oriQz.at(ownerID) = oriQ.z;
}

void DEMDynamicThread::setOwnerVel(bodyID_t ownerID, float3 vel) {
    vX.at(ownerID) = vel.x;
    vY.at(ownerID) = vel.y;
    vZ.at(ownerID) = vel.z;
}

void DEMDynamicThread::setTriNodeRelPos(size_t start, const std::vector<DEMTriangle>& triangles, bool overwrite) {
    if (overwrite) {
        for (size_t i = 0; i < triangles.size(); i++) {
            relPosNode1[start + i] = triangles[i].p1;
            relPosNode2[start + i] = triangles[i].p2;
            relPosNode3[start + i] = triangles[i].p3;
        }
    } else {
        for (size_t i = 0; i < triangles.size(); i++) {
            relPosNode1[start + i] += triangles[i].p1;
            relPosNode2[start + i] += triangles[i].p2;
            relPosNode3[start + i] += triangles[i].p3;
        }
    }
}

}  // namespace deme
