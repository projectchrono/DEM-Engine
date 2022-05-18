//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cstring>
#include <iostream>
#include <thread>

#include <core/ApiVersion.h>
#include <core/utils/Macros.h>
#include <core/utils/chpf/particle_writer.hpp>
#include <DEM/kT.h>
#include <DEM/dT.h>
#include <DEM/HostSideHelpers.cpp>
#include <DEM/DEMDefines.h>

#include <algorithms/DEMCubBasedSubroutines.h>

namespace sgps {

inline void DEMKinematicThread::transferArraysResize(size_t nContactPairs) {
    // This memory usage is not tracked... How can I track the size changes on my friend's end??
    dT->idGeometryA_buffer.resize(nContactPairs);
    dT->idGeometryB_buffer.resize(nContactPairs);
    dT->contactType_buffer.resize(nContactPairs);
    granData->pDTOwnedBuffer_idGeometryA = dT->idGeometryA_buffer.data();
    granData->pDTOwnedBuffer_idGeometryB = dT->idGeometryB_buffer.data();
    granData->pDTOwnedBuffer_contactType = dT->contactType_buffer.data();
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

            contactDetection(bin_occupation_kernels, contact_detection_kernels, history_kernels, granData, simParams,
                             solverFlags, verbosity, idGeometryA, idGeometryB, contactType, streamInfo.stream,
                             stateOfSolver_resources);

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
    granData->pDTOwnedBuffer_contactType = dT->granData->contactType_buffer;
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

void DEMKinematicThread::allocateManagedArrays(size_t nOwnerBodies,
                                               size_t nOwnerClumps,
                                               unsigned int nExtObj,
                                               size_t nTriEntities,
                                               size_t nSpheresGM,
                                               size_t nTriGM,
                                               unsigned int nAnalGM,
                                               unsigned int nClumpTopo,
                                               unsigned int nClumpComponents,
                                               unsigned int nMatTuples) {
    // Sizes of these arrays
    simParams->nSpheresGM = nSpheresGM;
    simParams->nTriGM = nTriGM;
    simParams->nAnalGM = nAnalGM;
    simParams->nOwnerBodies = nOwnerBodies;
    simParams->nOwnerClumps = nOwnerClumps;
    simParams->nExtObj = nExtObj;
    simParams->nTriEntities = nTriEntities;
    simParams->nDistinctClumpBodyTopologies = nClumpTopo;
    simParams->nDistinctClumpComponents = nClumpComponents;
    simParams->nMatTuples = nMatTuples;

    // Resize to the number of clumps
    TRACKED_VECTOR_RESIZE(familyID, nOwnerBodies, "familyID", 0);
    TRACKED_VECTOR_RESIZE(voxelID, nOwnerBodies, "voxelID", 0);
    TRACKED_VECTOR_RESIZE(locX, nOwnerBodies, "locX", 0);
    TRACKED_VECTOR_RESIZE(locY, nOwnerBodies, "locY", 0);
    TRACKED_VECTOR_RESIZE(locZ, nOwnerBodies, "locZ", 0);
    TRACKED_VECTOR_RESIZE(oriQ0, nOwnerBodies, "oriQ0", 1);
    TRACKED_VECTOR_RESIZE(oriQ1, nOwnerBodies, "oriQ1", 0);
    TRACKED_VECTOR_RESIZE(oriQ2, nOwnerBodies, "oriQ2", 0);
    TRACKED_VECTOR_RESIZE(oriQ3, nOwnerBodies, "oriQ3", 0);

    // Transfer buffer arrays
    TRACKED_VECTOR_RESIZE(voxelID_buffer, nOwnerBodies, "voxelID_buffer", 0);
    TRACKED_VECTOR_RESIZE(locX_buffer, nOwnerBodies, "locX_buffer", 0);
    TRACKED_VECTOR_RESIZE(locY_buffer, nOwnerBodies, "locY_buffer", 0);
    TRACKED_VECTOR_RESIZE(locZ_buffer, nOwnerBodies, "locZ_buffer", 0);
    TRACKED_VECTOR_RESIZE(oriQ0_buffer, nOwnerBodies, "oriQ0_buffer", 0);
    TRACKED_VECTOR_RESIZE(oriQ1_buffer, nOwnerBodies, "oriQ1_buffer", 0);
    TRACKED_VECTOR_RESIZE(oriQ2_buffer, nOwnerBodies, "oriQ2_buffer", 0);
    TRACKED_VECTOR_RESIZE(oriQ3_buffer, nOwnerBodies, "oriQ3_buffer", 0);

    // Resize to the number of spheres
    TRACKED_VECTOR_RESIZE(ownerClumpBody, nSpheresGM, "ownerClumpBody", 0);
    TRACKED_VECTOR_RESIZE(clumpComponentOffset, nSpheresGM, "clumpComponentOffset", 0);

    // Resize to the length of the clump templates
    TRACKED_VECTOR_RESIZE(radiiSphere, nClumpComponents, "radiiSphere", 0);
    TRACKED_VECTOR_RESIZE(relPosSphereX, nClumpComponents, "relPosSphereX", 0);
    TRACKED_VECTOR_RESIZE(relPosSphereY, nClumpComponents, "relPosSphereY", 0);
    TRACKED_VECTOR_RESIZE(relPosSphereZ, nClumpComponents, "relPosSphereZ", 0);
    // TRACKED_VECTOR_RESIZE(inflatedRadiiVoxelRatio, nClumpComponents, "inflatedRadiiVoxelRatio", 0);

    // Arrays for kT produced contact info
    // The following several arrays will have variable sizes, so here we only used an estimate. My estimate of total
    // contact pairs is 2n, and I think the max is 6n (although I can't prove it). Note the estimate should be large
    // enough to decrease the number of reallocations in the simulation, but not too large that eats too much memory.
    TRACKED_VECTOR_RESIZE(idGeometryA, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "idGeometryA", 0);
    TRACKED_VECTOR_RESIZE(idGeometryB, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "idGeometryB", 0);
    TRACKED_VECTOR_RESIZE(contactType, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "contactType", DEM_NOT_A_CONTACT);
    if (!solverFlags.isFrictionless) {
        TRACKED_VECTOR_RESIZE(previous_idGeometryA, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "previous_idGeometryA",
                              0);
        // In the first iteration, if the this array is all-zero then the run-length is a huge number. I know cub don't
        // usually care about overflow so it should be no problem even if no treatment is applied, but let's just
        // randomize the init state of this array so no surprises ever happen.
        for (size_t i = 0; i < previous_idGeometryA.size(); i++) {
            previous_idGeometryA.at(i) = (i / SGPS_DEM_INIT_CNT_MULTIPLIER) + (i % SGPS_DEM_INIT_CNT_MULTIPLIER);
        }
        TRACKED_VECTOR_RESIZE(previous_idGeometryB, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "previous_idGeometryB",
                              0);
        TRACKED_VECTOR_RESIZE(previous_contactType, nOwnerBodies * SGPS_DEM_INIT_CNT_MULTIPLIER, "previous_contactType",
                              DEM_NOT_A_CONTACT);
    }
}

void DEMKinematicThread::populateManagedArrays(const std::vector<unsigned int>& input_clump_types,
                                               const std::vector<unsigned int>& input_clump_family,
                                               const std::vector<unsigned int>& input_ext_obj_family,
                                               const std::unordered_map<unsigned int, family_t>& family_user_impl_map,
                                               const std::vector<float>& clumps_mass_types,
                                               const std::vector<std::vector<float>>& clumps_sp_radii_types,
                                               const std::vector<std::vector<float3>>& clumps_sp_location_types) {
    // Use some temporary hacks to get the info in the managed mem
    // All the input vectors should have the same length, nClumpTopo
    size_t k = 0;
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

    float3 LBF;
    LBF.x = simParams->LBFX;
    LBF.y = simParams->LBFY;
    LBF.z = simParams->LBFZ;
    // float3 LBF = make_float3(simParams->LBFX, simParams->LBFY, simParams->LBFZ);
    for (size_t i = 0; i < simParams->nOwnerClumps; i++) {
        auto type_of_this_clump = input_clump_types.at(i);

        // auto this_CoM_coord = input_clump_xyz.at(i) - LBF; // kT don't have to init owner xyz
        auto this_clump_no_sp_radii = clumps_sp_radii_types.at(type_of_this_clump);
        auto this_clump_no_sp_relPos = clumps_sp_location_types.at(type_of_this_clump);

        for (size_t j = 0; j < this_clump_no_sp_radii.size(); j++) {
            clumpComponentOffset.at(k) = prescans.at(type_of_this_clump) + j;
            ownerClumpBody.at(k) = i;
            k++;
        }

        family_t this_family_num = family_user_impl_map.at(input_clump_family.at(i));
        familyID.at(i) = this_family_num;
    }

    size_t offset_for_ext_obj = simParams->nOwnerClumps;
    for (size_t i = 0; i < simParams->nExtObj; i++) {
        family_t this_family_num = family_user_impl_map.at(input_ext_obj_family.at(i));
        familyID.at(i + offset_for_ext_obj) = this_family_num;
    }
}

void DEMKinematicThread::jitifyKernels(const std::unordered_map<std::string, std::string>& templateSubs,
                                       const std::unordered_map<std::string, std::string>& simParamSubs,
                                       const std::unordered_map<std::string, std::string>& massMatSubs,
                                       const std::unordered_map<std::string, std::string>& familyMaskSubs,
                                       const std::unordered_map<std::string, std::string>& familyPrescribeSubs,
                                       const std::unordered_map<std::string, std::string>& analGeoSubs) {
    // First one is bin_occupation_kernels kernels, which figure out the bin--sphere touch pairs
    {
        std::unordered_map<std::string, std::string> boSubs = templateSubs;
        boSubs.insert(simParamSubs.begin(), simParamSubs.end());
        boSubs.insert(analGeoSubs.begin(), analGeoSubs.end());
        boSubs.insert(familyMaskSubs.begin(), familyMaskSubs.end());
        // bin_occupation_kernels = JitHelper::buildProgram(
        //     "DEMBinSphereKernels", JitHelper::KERNEL_DIR / "DEMBinSphereKernels.cu",
        //     std::unordered_map<std::string, std::string>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});
        bin_occupation_kernels = std::make_shared<jitify::Program>(
            std::move(JitHelper::buildProgram("DEMBinSphereKernels", JitHelper::KERNEL_DIR / "DEMBinSphereKernels.cu",
                                              boSubs, {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
    // Then CD kernels
    {
        std::unordered_map<std::string, std::string> cdSubs = templateSubs;
        cdSubs.insert(simParamSubs.begin(), simParamSubs.end());
        cdSubs.insert(familyMaskSubs.begin(), familyMaskSubs.end());
        contact_detection_kernels = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMContactKernels", JitHelper::KERNEL_DIR / "DEMContactKernels.cu", cdSubs,
            {"-I" + (JitHelper::KERNEL_DIR / "..").string(), "-I/opt/apps/cuda/x86_64/11.6.0/default/include"})));
    }
    // Then contact history mapping kernels
    {
        std::unordered_map<std::string, std::string> hSubs;
        history_kernels = std::make_shared<jitify::Program>(std::move(
            JitHelper::buildProgram("DEMHistoryMappingKernels", JitHelper::KERNEL_DIR / "DEMHistoryMappingKernels.cu",
                                    hSubs, {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
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
