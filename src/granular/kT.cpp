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
#include <granular/kT.h>
#include <granular/dT.h>
#include <granular/HostSideHelpers.cpp>

#include <algorithms/DEMCubHelperFunctions.h>

namespace sgps {

inline void DEMKinematicThread::contactEventArraysResize(size_t nContactPairs) {
    TRACKED_QUICK_VECTOR_RESIZE(idGeometryA, nContactPairs);
    TRACKED_QUICK_VECTOR_RESIZE(idGeometryB, nContactPairs);
    TRACKED_QUICK_VECTOR_RESIZE(contactType, nContactPairs);

    // Re-pack pointers in case the arrays got reallocated
    granData->idGeometryA = idGeometryA.data();
    granData->idGeometryB = idGeometryB.data();
    granData->contactType = contactType.data();
}

inline void DEMKinematicThread::transferArraysResize(size_t nContactPairs) {
    // This memory usage is not tracked... How can I track the size changes on my friend's end??
    dT->idGeometryA_buffer.resize(nContactPairs);
    dT->idGeometryB_buffer.resize(nContactPairs);
    dT->contactType_buffer.resize(nContactPairs);
    granData->pDTOwnedBuffer_idGeometryA = dT->idGeometryA_buffer.data();
    granData->pDTOwnedBuffer_idGeometryB = dT->idGeometryB_buffer.data();
    granData->pDTOwnedBuffer_contactType = dT->contactType_buffer.data();
}

void DEMKinematicThread::contactDetection() {
    // total bytes needed for temp arrays in contact detection
    size_t CD_temp_arr_bytes = 0;

    // 1st step: register the number of sphere--bin touching pairs for each sphere for further processing
    CD_temp_arr_bytes = simParams->nSpheresGM * sizeof(binsSphereTouches_t);
    binsSphereTouches_t* numBinsSphereTouches =
        (binsSphereTouches_t*)stateOfSolver_resources.allocateTempVector1(CD_temp_arr_bytes);
    // This kernel is also tasked to find how many analytical objects each sphere touches
    // We'll use a new vector 3 to store this
    CD_temp_arr_bytes = simParams->nSpheresGM * sizeof(objID_t);
    objID_t* numAnalGeoSphereTouches = (objID_t*)stateOfSolver_resources.allocateTempVector3(CD_temp_arr_bytes);
    size_t blocks_needed_for_bodies =
        (simParams->nSpheresGM + SGPS_DEM_NUM_BODIES_PER_BLOCK - 1) / SGPS_DEM_NUM_BODIES_PER_BLOCK;

    bin_occupation->kernel("getNumberOfBinsEachSphereTouches")
        .instantiate()
        .configure(dim3(blocks_needed_for_bodies), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
        .launch(granData, numBinsSphereTouches, numAnalGeoSphereTouches);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

    // 2nd step: prefix scan sphere--bin touching pairs
    CD_temp_arr_bytes = simParams->nSpheresGM * sizeof(binSphereTouchPairs_t);
    binSphereTouchPairs_t* numBinsSphereTouchesScan =
        (binSphereTouchPairs_t*)stateOfSolver_resources.allocateTempVector2(CD_temp_arr_bytes);
    cubPrefixScan_binSphere(numBinsSphereTouches, numBinsSphereTouchesScan, simParams->nSpheresGM, streamInfo.stream,
                            stateOfSolver_resources);
    stateOfSolver_resources.setNumBinSphereTouchPairs((size_t)numBinsSphereTouchesScan[simParams->nSpheresGM - 1] +
                                                      (size_t)numBinsSphereTouches[simParams->nSpheresGM - 1]);
    // The same process is done for sphere--analytical geometry pairs as well. Use vector 4 for this.
    CD_temp_arr_bytes = simParams->nSpheresGM * sizeof(binSphereTouchPairs_t);
    binSphereTouchPairs_t* numAnalGeoSphereTouchesScan =
        (binSphereTouchPairs_t*)stateOfSolver_resources.allocateTempVector4(CD_temp_arr_bytes);
    cubPrefixScan_sphereGeo(numAnalGeoSphereTouches, numAnalGeoSphereTouchesScan, simParams->nSpheresGM,
                            streamInfo.stream, stateOfSolver_resources);
    stateOfSolver_resources.setNumContacts((size_t)numAnalGeoSphereTouches[simParams->nSpheresGM - 1] +
                                           (size_t)numAnalGeoSphereTouchesScan[simParams->nSpheresGM - 1]);
    if (stateOfSolver_resources.getNumContacts() > idGeometryA.size()) {
        contactEventArraysResize(stateOfSolver_resources.getNumContacts());
    }
    // std::cout << stateOfSolver_resources.getNumBinSphereTouchPairs() << std::endl;
    // displayArray<binsSphereTouches_t>(numBinsSphereTouches, simParams->nSpheresGM);
    // displayArray<binSphereTouchPairs_t>(numBinsSphereTouchesScan, simParams->nSpheresGM);

    // 3rd step: use a custom kernel to figure out all sphere--bin touching pairs. Note numBinsSphereTouches can retire
    // now so we allocate on temp vector 1 and re-use vector 3.
    CD_temp_arr_bytes = stateOfSolver_resources.getNumBinSphereTouchPairs() * sizeof(binID_t);
    binID_t* binIDsEachSphereTouches = (binID_t*)stateOfSolver_resources.allocateTempVector1(CD_temp_arr_bytes);
    CD_temp_arr_bytes = stateOfSolver_resources.getNumBinSphereTouchPairs() * sizeof(bodyID_t);
    bodyID_t* sphereIDsEachBinTouches = (bodyID_t*)stateOfSolver_resources.allocateTempVector3(CD_temp_arr_bytes);
    // This kernel is also responsible of figuring out sphere--analytical geometry pairs
    bin_occupation->kernel("populateBinSphereTouchingPairs")
        .instantiate()
        .configure(dim3(blocks_needed_for_bodies), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, streamInfo.stream)
        .launch(granData, numBinsSphereTouchesScan, numAnalGeoSphereTouchesScan, binIDsEachSphereTouches,
                sphereIDsEachBinTouches, granData->idGeometryA, granData->idGeometryB, granData->contactType);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    // std::cout << "idGeometryB: ";
    // displayArray<bodyID_t>(granData->idGeometryB, stateOfSolver_resources.getNumContacts());
    // std::cout << "contactType: ";
    // displayArray<contact_t>(granData->contactType, stateOfSolver_resources.getNumContacts());
    // std::cout << "Unsorted bin IDs: ";
    // displayArray<binID_t>(binIDsEachSphereTouches, stateOfSolver_resources.getNumBinSphereTouchPairs());
    // std::cout << "Corresponding sphere IDs: ";
    // displayArray<bodyID_t>(sphereIDsEachBinTouches, stateOfSolver_resources.getNumBinSphereTouchPairs());

    // 4th step: allocate and populate SORTED binIDsEachSphereTouches and sphereIDsEachBinTouches. Note
    // numBinsSphereTouchesScan can retire now so we allocate on vector 2 and re-use vector 4.
    CD_temp_arr_bytes = stateOfSolver_resources.getNumBinSphereTouchPairs() * sizeof(bodyID_t);
    bodyID_t* sphereIDsEachBinTouches_sorted =
        (bodyID_t*)stateOfSolver_resources.allocateTempVector2(CD_temp_arr_bytes);
    CD_temp_arr_bytes = stateOfSolver_resources.getNumBinSphereTouchPairs() * sizeof(binID_t);
    binID_t* binIDsEachSphereTouches_sorted = (binID_t*)stateOfSolver_resources.allocateTempVector4(CD_temp_arr_bytes);
    // hostSortByKey<binID_t, bodyID_t>(granData->binIDsEachSphereTouches, granData->sphereIDsEachBinTouches,
    //                                  stateOfSolver_resources.getNumBinSphereTouchPairs());
    cubSortByKeys(binIDsEachSphereTouches, binIDsEachSphereTouches_sorted, sphereIDsEachBinTouches,
                  sphereIDsEachBinTouches_sorted, stateOfSolver_resources.getNumBinSphereTouchPairs(),
                  streamInfo.stream, stateOfSolver_resources);
    // std::cout << "Sorted bin IDs: ";
    // displayArray<binID_t>(binIDsEachSphereTouches_sorted, stateOfSolver_resources.getNumBinSphereTouchPairs());
    // std::cout << "Corresponding sphere IDs: ";
    // displayArray<bodyID_t>(sphereIDsEachBinTouches_sorted, stateOfSolver_resources.getNumBinSphereTouchPairs());

    // 5th step: use DeviceRunLengthEncode to identify those active (that have bodies in them) bins.
    // Also, binIDsEachSphereTouches is large enough for a unique scan because total sphere--bin pairs are more than
    // active bins.
    binID_t* binIDsUnique = (binID_t*)binIDsEachSphereTouches;
    cubUnique(binIDsEachSphereTouches_sorted, binIDsUnique, stateOfSolver_resources.getNumActiveBinsPointer(),
              stateOfSolver_resources.getNumBinSphereTouchPairs(), streamInfo.stream, stateOfSolver_resources);
    // Allocate space for encoding output, and run it. Note the (unsorted) binIDsEachSphereTouches and
    // sphereIDsEachBinTouches can retire now, so we allocate on temp vectors 1 and 3.
    CD_temp_arr_bytes = stateOfSolver_resources.getNumActiveBins() * sizeof(binID_t);
    binID_t* activeBinIDs = (binID_t*)stateOfSolver_resources.allocateTempVector1(CD_temp_arr_bytes);
    CD_temp_arr_bytes = stateOfSolver_resources.getNumActiveBins() * sizeof(spheresBinTouches_t);
    spheresBinTouches_t* numSpheresBinTouches =
        (spheresBinTouches_t*)stateOfSolver_resources.allocateTempVector3(CD_temp_arr_bytes);
    cubRunLengthEncode(binIDsEachSphereTouches_sorted, activeBinIDs, numSpheresBinTouches,
                       stateOfSolver_resources.getNumActiveBinsPointer(),
                       stateOfSolver_resources.getNumBinSphereTouchPairs(), streamInfo.stream, stateOfSolver_resources);
    // std::cout << "numActiveBins: " << stateOfSolver_resources.getNumActiveBins() << std::endl;
    // std::cout << "activeBinIDs: ";
    // displayArray<binID_t>(activeBinIDs, stateOfSolver_resources.getNumActiveBins());
    // std::cout << "numSpheresBinTouches: ";
    // displayArray<spheresBinTouches_t>(numSpheresBinTouches, stateOfSolver_resources.getNumActiveBins());
    // std::cout << "binIDsEachSphereTouches_sorted: ";
    // displayArray<binID_t>(binIDsEachSphereTouches_sorted, stateOfSolver_resources.getNumBinSphereTouchPairs());

    // Then, scan to find the offsets that are used to index into sphereIDsEachBinTouches_sorted to obtain bin-wise
    // spheres. Note binIDsEachSphereTouches_sorted can retire so we allocate on temp vector 4.
    CD_temp_arr_bytes = stateOfSolver_resources.getNumActiveBins() * sizeof(binSphereTouchPairs_t);
    binSphereTouchPairs_t* sphereIDsLookUpTable =
        (binSphereTouchPairs_t*)stateOfSolver_resources.allocateTempVector4(CD_temp_arr_bytes);
    cubPrefixScan_binSphere(numSpheresBinTouches, sphereIDsLookUpTable, stateOfSolver_resources.getNumActiveBins(),
                            streamInfo.stream, stateOfSolver_resources);
    // std::cout << "sphereIDsLookUpTable: ";
    // displayArray<binSphereTouchPairs_t>(sphereIDsLookUpTable, stateOfSolver_resources.getNumActiveBins());

    // 6th step: find the contact pairs. One-two punch: first find num of contacts in each bin, then prescan, then find
    // the actual pair names. A new temp array is needed for this numContactsInEachBin. Note we assume the number of
    // contact in each bin is the same level as the number of spheres in each bin (capped by the same data type).
    CD_temp_arr_bytes = stateOfSolver_resources.getNumActiveBins() * sizeof(spheresBinTouches_t);
    spheresBinTouches_t* numContactsInEachBin =
        (spheresBinTouches_t*)stateOfSolver_resources.allocateTempVector5(CD_temp_arr_bytes);
    size_t blocks_needed_for_bins =
        (stateOfSolver_resources.getNumActiveBins() + SGPS_DEM_NUM_BINS_PER_BLOCK - 1) / SGPS_DEM_NUM_BINS_PER_BLOCK;
    if (blocks_needed_for_bins > 0) {
        contact_detection->kernel("getNumberOfContactsEachBin")
            .instantiate()
            .configure(dim3(blocks_needed_for_bins), dim3(SGPS_DEM_NUM_BINS_PER_BLOCK), 0, streamInfo.stream)
            .launch(granData, sphereIDsEachBinTouches_sorted, activeBinIDs, numSpheresBinTouches, sphereIDsLookUpTable,
                    numContactsInEachBin, stateOfSolver_resources.getNumActiveBins());
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));

        // TODO: sphere should have jitified and non-jitified part. Use a component ID > max_comp_id to signal bringing
        // data from global memory.
        // TODO: Add tri--sphere CD kernel (if mesh support is to be added). This kernel integrates tri--boundary CD.
        // Note triangle facets can have jitified (many bodies of the same type) and non-jitified (a big meshed body)
        // part. Use a component ID > max_comp_id to signal bringing data from global memory.
        // TODO: Add tri--tri CD kernel (in the far future, should mesh-rerpesented geometry to be supported). This
        // kernel integrates tri--boundary CD.
        // TODO: remember that boundary types are either all jitified or non-jitified. In principal, they should be all
        // jitified.

        // Prescan numContactsInEachBin to get the final contactReportOffsets. A new vector is needed.
        CD_temp_arr_bytes = stateOfSolver_resources.getNumActiveBins() * sizeof(contactPairs_t);
        contactPairs_t* contactReportOffsets =
            (contactPairs_t*)stateOfSolver_resources.allocateTempVector6(CD_temp_arr_bytes);
        cubPrefixScan_contacts(numContactsInEachBin, contactReportOffsets, stateOfSolver_resources.getNumActiveBins(),
                               streamInfo.stream, stateOfSolver_resources);
        // displayArray<contactPairs_t>(contactReportOffsets, stateOfSolver_resources.getNumActiveBins());

        // Add sphere--sphere contacts together with sphere--analytical geometry contacts
        size_t nSphereGeoContact = stateOfSolver_resources.getNumContacts();
        size_t nSphereSphereContact = (size_t)numContactsInEachBin[stateOfSolver_resources.getNumActiveBins() - 1] +
                                      (size_t)contactReportOffsets[stateOfSolver_resources.getNumActiveBins() - 1];
        stateOfSolver_resources.setNumContacts(nSphereSphereContact + nSphereGeoContact);
        if (stateOfSolver_resources.getNumContacts() > idGeometryA.size()) {
            contactEventArraysResize(stateOfSolver_resources.getNumContacts());
        }
        // std::cout << "NumContacts: " << stateOfSolver_resources.getNumContacts() << std::endl;

        // Sphere--sphere contact pairs go after sphere--anal-geo contacts
        bodyID_t* idSphA = (granData->idGeometryA + nSphereGeoContact);
        bodyID_t* idSphB = (granData->idGeometryB + nSphereGeoContact);
        // In next kernel call, all contacts registered there will be sphere--sphere contacts
        GPU_CALL(cudaMemset((void*)(granData->contactType + nSphereGeoContact), DEM_SPHERE_SPHERE_CONTACT,
                            nSphereSphereContact * sizeof(contact_t)));
        // Then fill in those contacts
        contact_detection->kernel("populateContactPairsEachBin")
            .instantiate()
            .configure(dim3(blocks_needed_for_bins), dim3(SGPS_DEM_NUM_BINS_PER_BLOCK), 0, streamInfo.stream)
            .launch(granData, sphereIDsEachBinTouches_sorted, activeBinIDs, numSpheresBinTouches, sphereIDsLookUpTable,
                    contactReportOffsets, idSphA, idSphB, stateOfSolver_resources.getNumActiveBins());
        GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
        // displayArray<bodyID_t>(granData->idGeometryA, stateOfSolver_resources.getNumContacts());
        // displayArray<bodyID_t>(granData->idGeometryB, stateOfSolver_resources.getNumContacts());

        // TODO: Now, sort idGeometryAB by their owners. This is to increase dT shmem use rate.
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
}

inline void DEMKinematicThread::sendToTheirBuffer() {
    GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_nContactPairs, stateOfSolver_resources.getNumContactsPointer(),
                        sizeof(size_t), cudaMemcpyDeviceToDevice));
    // Resize dT owned buffers before usage
    if (stateOfSolver_resources.getNumContacts() > dT->idGeometryA_buffer.size()) {
        transferArraysResize(stateOfSolver_resources.getNumContacts());
    }
    GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_idGeometryA, granData->idGeometryA,
                        stateOfSolver_resources.getNumContacts() * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_idGeometryB, granData->idGeometryB,
                        stateOfSolver_resources.getNumContacts() * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_contactType, granData->contactType,
                        stateOfSolver_resources.getNumContacts() * sizeof(contact_t), cudaMemcpyDeviceToDevice));
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

            contactDetection();

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

void DEMKinematicThread::useFrictionlessModel(bool useFrictionless) {
    isFrictionless = useFrictionless;
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
    // First one is bin_occupation kernels, which figure out the bin--sphere touch pairs
    {
        std::unordered_map<std::string, std::string> boSubs = templateSubs;
        boSubs.insert(simParamSubs.begin(), simParamSubs.end());
        boSubs.insert(analGeoSubs.begin(), analGeoSubs.end());
        boSubs.insert(familyMaskSubs.begin(), familyMaskSubs.end());
        // bin_occupation = JitHelper::buildProgram(
        //     "DEMBinSphereKernels", JitHelper::KERNEL_DIR / "DEMBinSphereKernels.cu",
        //     std::unordered_map<std::string, std::string>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});
        bin_occupation = std::make_shared<jitify::Program>(
            std::move(JitHelper::buildProgram("DEMBinSphereKernels", JitHelper::KERNEL_DIR / "DEMBinSphereKernels.cu",
                                              boSubs, {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
    // Then CD kernels
    {
        std::unordered_map<std::string, std::string> cdSubs = templateSubs;
        cdSubs.insert(simParamSubs.begin(), simParamSubs.end());
        cdSubs.insert(familyMaskSubs.begin(), familyMaskSubs.end());
        contact_detection = std::make_shared<jitify::Program>(
            std::move(JitHelper::buildProgram("DEMContactKernels", JitHelper::KERNEL_DIR / "DEMContactKernels.cu",
                                              cdSubs, {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
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
