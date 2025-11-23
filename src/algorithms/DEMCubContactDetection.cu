//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <cub/cub.cuh>
#include <kernel/DEMHelperKernels.cuh>

#include <algorithms/DEMStaticDeviceSubroutines.h>
#include <algorithms/DEMStaticDeviceUtilities.cuh>
#include <algorithms/DEMKinematicMisc.cu>
#include <algorithms/DEMCubWrappers.cu>

#include <DEM/utils/HostSideHelpers.hpp>
#include <core/utils/GpuError.h>

namespace deme {

inline void contactEventArraysResize(size_t nContactPairs,
                                     DualArray<bodyID_t>& idGeometryA,
                                     DualArray<bodyID_t>& idGeometryB,
                                     DualArray<contact_t>& contactType,
                                     DualArray<notStupidBool_t>& contactPersistency,
                                     DualStruct<DEMDataKT>& granData) {
    // Note these resizing are automatically on kT's device
    DEME_DUAL_ARRAY_RESIZE_NOVAL(idGeometryA, nContactPairs);
    DEME_DUAL_ARRAY_RESIZE_NOVAL(idGeometryB, nContactPairs);
    DEME_DUAL_ARRAY_RESIZE_NOVAL(contactType, nContactPairs);

    // In the case of user-loaded contacts, if the persistency array is not long enough then we have to manually
    // extend it.
    DEME_DUAL_ARRAY_RESIZE(contactPersistency, nContactPairs, CONTACT_NOT_PERSISTENT);

    // Re-packing pointers now is automatic

    // It's safe to toDevice even though kT is working now and dT may write to its buffer
    // This is because all buffer arrays are not used in kernels so their pointers are only meaningfully stored on host,
    // so writing from host to device won't change the destination where dT writes
    granData.toDevice();
}

inline void meshPatchPairsResize(size_t nMeshInvolvedContactPairs,
                                 DualArray<patchIDPair_t>& contactPatchPairs,
                                 DualStruct<DEMDataKT>& granData) {
    // Note these resizing are automatically on kT's device
    DEME_DUAL_ARRAY_RESIZE_NOVAL(contactPatchPairs, nMeshInvolvedContactPairs);

    // Re-packing pointers now is automatic

    // It's safe to toDevice even though kT is working now and dT may write to its buffer
    // This is because all buffer arrays are not used in kernels so their pointers are only meaningfully stored on host,
    // so writing from host to device won't change the destination where dT writes
    granData.toDevice();
}

inline void removeDuplicateContacts(DualStruct<DEMDataKT>& granData,
                                    bodyID_t* idA_sorted,
                                    bodyID_t* idB_sorted,
                                    contact_t* contactType_sorted,
                                    notStupidBool_t* persistency_sorted,
                                    DualArray<bodyID_t>& idGeometryA,
                                    DualArray<bodyID_t>& idGeometryB,
                                    DualArray<contact_t>& contactType,
                                    DualArray<notStupidBool_t>& contactPersistency,
                                    bool process_persistency,
                                    bodyID_t safe_entity_count,
                                    size_t numTotalCnts,
                                    cudaStream_t& this_stream,
                                    DEMSolverScratchData& scratchPad) {
    // First we run-length it based on idA
    size_t run_length_bytes = safe_entity_count * sizeof(geoSphereTouches_t);
    geoSphereTouches_t* idA_runlength =
        (geoSphereTouches_t*)scratchPad.allocateTempVector("idA_runlength", run_length_bytes);
    size_t unique_id_bytes = safe_entity_count * sizeof(bodyID_t);
    bodyID_t* unique_idA = (bodyID_t*)scratchPad.allocateTempVector("unique_idA", unique_id_bytes);
    scratchPad.allocateDualStruct("numUniqueA");
    cubDEMRunLengthEncode<bodyID_t, geoSphereTouches_t>(idA_sorted, unique_idA, idA_runlength,
                                                        scratchPad.getDualStructDevice("numUniqueA"), numTotalCnts,
                                                        this_stream, scratchPad);
    scratchPad.syncDualStructDeviceToHost("numUniqueA");
    size_t* pNumUniqueA = scratchPad.getDualStructHost("numUniqueA");
    size_t scanned_runlength_bytes = (*pNumUniqueA) * sizeof(contactPairs_t);
    contactPairs_t* idA_scanned_runlength =
        (contactPairs_t*)scratchPad.allocateTempVector("idA_scanned_runlength", scanned_runlength_bytes);
    cubDEMPrefixScan<geoSphereTouches_t, contactPairs_t>(idA_runlength, idA_scanned_runlength, *pNumUniqueA,
                                                         this_stream, scratchPad);

    // Then each thread will take care of an id in A to mark redundency...
    size_t retain_flags_size = numTotalCnts * sizeof(notStupidBool_t);
    notStupidBool_t* retain_flags = (notStupidBool_t*)scratchPad.allocateTempVector("retain_flags", retain_flags_size);
    size_t blocks_needed_for_setting_1 = (numTotalCnts + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed_for_setting_1 > 0) {
        setArr<<<dim3(blocks_needed_for_setting_1), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, this_stream>>>(
            retain_flags, numTotalCnts, (notStupidBool_t)1);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }

    size_t blocks_needed_for_flagging = (*pNumUniqueA + DEME_NUM_BODIES_PER_BLOCK - 1) / DEME_NUM_BODIES_PER_BLOCK;
    if (blocks_needed_for_flagging > 0) {
        markDuplicateContacts<<<dim3(blocks_needed_for_flagging), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, this_stream>>>(
            idA_runlength, idA_scanned_runlength, idB_sorted, contactType_sorted, persistency_sorted, retain_flags,
            *pNumUniqueA, process_persistency);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    }
    // std::cout << "Marked retainers: " << std::endl;
    // displayDeviceArray<notStupidBool_t>(retain_flags, numTotalCnts);
    scratchPad.finishUsingDualStruct("numUniqueA");

    // Then remove redundency based on the flag array...
    // Note the contactPersistency array is managed by the current contact arr. It will also be copied over.
    scratchPad.allocateDualStruct("numRetainedCnts");
    cubDEMSum<notStupidBool_t, size_t>(retain_flags, scratchPad.getDualStructDevice("numRetainedCnts"), numTotalCnts,
                                       this_stream, scratchPad);
    scratchPad.syncDualStructDeviceToHost("numRetainedCnts");
    size_t* pNumRetainedCnts = scratchPad.getDualStructHost("numRetainedCnts");
    // DEME_DEBUG_PRINTF("Found %zu contacts, including user-specified persistent contacts.",
    //                        *pNumRetainedCnts);
    // Potentially need to resize the contact arrays
    if (*pNumRetainedCnts > idGeometryA.size()) {
        contactEventArraysResize(*pNumRetainedCnts, idGeometryA, idGeometryB, contactType, contactPersistency,
                                 granData);
    }
    // Then select those needed contacts
    cubDEMSelectFlagged<bodyID_t, notStupidBool_t>(idA_sorted, granData->idGeometryA, retain_flags,
                                                   scratchPad.getDualStructDevice("numRetainedCnts"), numTotalCnts,
                                                   this_stream, scratchPad);
    cubDEMSelectFlagged<bodyID_t, notStupidBool_t>(idB_sorted, granData->idGeometryB, retain_flags,
                                                   scratchPad.getDualStructDevice("numRetainedCnts"), numTotalCnts,
                                                   this_stream, scratchPad);
    cubDEMSelectFlagged<contact_t, notStupidBool_t>(contactType_sorted, granData->contactType, retain_flags,
                                                    scratchPad.getDualStructDevice("numRetainedCnts"), numTotalCnts,
                                                    this_stream, scratchPad);
    if (process_persistency) {
        cubDEMSelectFlagged<notStupidBool_t, notStupidBool_t>(
            persistency_sorted, granData->contactPersistency, retain_flags,
            scratchPad.getDualStructDevice("numRetainedCnts"), numTotalCnts, this_stream, scratchPad);
    }
    scratchPad.syncDualStructDeviceToHost(
        "numRetainedCnts");  // In theory no need, but when CONTACT_IS_PERSISTENT is not 1...
    // DEME_DEBUG_PRINTF("CUB confirms there are %zu contacts, including user-specified persistent
    // contacts.", *pNumRetainedCnts);
    // std::cout << "Contacts after duplication check: " << std::endl;
    // displayDeviceArray<bodyID_t>(granData->idGeometryA, *pNumRetainedCnts);
    // displayDeviceArray<bodyID_t>(granData->idGeometryB, *pNumRetainedCnts);
    // displayDeviceArray<contact_t>(granData->contactType, *pNumRetainedCnts);
    // displayDeviceArray<notStupidBool_t>(granData->contactPersistency, *pNumRetainedCnts);

    // And update the number of contacts.
    *scratchPad.numContacts = *pNumRetainedCnts;
    scratchPad.finishUsingDualStruct("numRetainedCnts");

    // Unclaim all temp vectors
    scratchPad.finishUsingTempVector("idA_runlength");
    scratchPad.finishUsingTempVector("unique_idA");
    scratchPad.finishUsingTempVector("idA_scanned_runlength");
    scratchPad.finishUsingTempVector("retain_flags");
}

void contactDetection(std::shared_ptr<jitify::Program>& bin_sphere_kernels,
                      std::shared_ptr<jitify::Program>& bin_triangle_kernels,
                      std::shared_ptr<jitify::Program>& sphere_contact_kernels,
                      std::shared_ptr<jitify::Program>& sphTri_contact_kernels,
                      DualStruct<DEMDataKT>& granData,
                      DualStruct<DEMSimParams>& simParams,
                      SolverFlags& solverFlags,
                      verbosity_t& verbosity,
                      DualArray<bodyID_t>& idGeometryA,
                      DualArray<bodyID_t>& idGeometryB,
                      DualArray<contact_t>& contactType,
                      DualArray<bodyID_t>& previous_idGeometryA,
                      DualArray<bodyID_t>& previous_idGeometryB,
                      DualArray<contact_t>& previous_contactType,
                      DualArray<notStupidBool_t>& contactPersistency,
                      DualArray<patchIDPair_t>& contactPatchPairs,
                      DualArray<contactPairs_t>& contactMapping,
                      cudaStream_t& this_stream,
                      DEMSolverScratchData& scratchPad,
                      SolverTimers& timers,
                      kTStateParams& stateParams) {
    // A dumb check
    if (simParams->nSpheresGM == 0 && simParams->nTriGM == 0) {
        *scratchPad.numContacts = 0;
        *scratchPad.numPrevContacts = 0;
        *scratchPad.numPrevSpheres = 0;
        *scratchPad.numPrevTriangles = 0;

        scratchPad.numContacts.toDevice();
        scratchPad.numPrevContacts.toDevice();
        scratchPad.numPrevSpheres.toDevice();
        scratchPad.numPrevTriangles.toDevice();
        return;
    }
    // These are needed for the solver to keep tab... But you know, we may have no triangles or no contacts, so
    // initializing them is needed.
    stateParams.maxSphFoundInBin = 0;
    stateParams.maxTriFoundInBin = 0;
    stateParams.avgCntsPerSphere = 0;

    // A special flag that marks if the contact arrays are sorted based on idGeometryA; if so, we can choose to not do
    // another sort at some points
    bool contactArraysAreSortedByA = false;

    // total bytes needed for temp arrays in contact detection
    size_t CD_temp_arr_bytes = 0;

    // Contact counts
    size_t nSphereSphereContact = 0, nTriSphereContact = 0, nTriTriContact = 0, nSphereGeoContact = 0,
           nTriGeoContact = 0;

    {
        timers.GetTimer("Discretize domain").start();
        // -----------------------------------------------------------------------------------------------------------
        // Sphere-related discretization & sphere--analytical contact detection
        // -----------------------------------------------------------------------------------------------------------

        // If there are spheres, the following information needs to be extracted and kept...
        scratchPad.allocateDualStruct("numActiveBins");
        *scratchPad.getDualStructHost("numActiveBins") = 0;  // Need a default value on host for no-sphere case
        size_t* pNumActiveBins = scratchPad.getDualStructHost("numActiveBins");  // Can later be host or device pointers
        bodyID_t* sphereIDsEachBinTouches_sorted;
        binID_t* activeBinIDs;
        spheresBinTouches_t* numSpheresBinTouches;
        binSphereTouchPairs_t* sphereIDsLookUpTable;
        if (simParams->nSpheresGM > 0) {
            // 1st step: register the number of sphere--bin touching pairs for each sphere for further processing
            CD_temp_arr_bytes = simParams->nSpheresGM * sizeof(binsSphereTouches_t);
            binsSphereTouches_t* numBinsSphereTouches =
                (binsSphereTouches_t*)scratchPad.allocateTempVector("numBinsSphereTouches", CD_temp_arr_bytes);
            // This kernel is also tasked to find how many analytical objects each sphere touches
            // We'll use a new vector 2 to store this
            CD_temp_arr_bytes = simParams->nSpheresGM * sizeof(objID_t);
            objID_t* numAnalGeoSphereTouches =
                (objID_t*)scratchPad.allocateTempVector("numAnalGeoSphereTouches", CD_temp_arr_bytes);
            size_t blocks_needed_for_bodies =
                (simParams->nSpheresGM + DEME_NUM_BODIES_PER_BLOCK - 1) / DEME_NUM_BODIES_PER_BLOCK;

            bin_sphere_kernels->kernel("getNumberOfBinsEachSphereTouches")
                .instantiate()
                .configure(dim3(blocks_needed_for_bodies), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, this_stream)
                .launch(&simParams, &granData, numBinsSphereTouches, numAnalGeoSphereTouches);
            DEME_GPU_CALL(cudaStreamSynchronize(this_stream));

            // 2nd step: prefix scan sphere--bin touching pairs
            // The last element of this scanned array is useful: it can be used to check if the 2 sweeps reach the same
            // conclusion on bin--sph touch pairs
            CD_temp_arr_bytes = (simParams->nSpheresGM + 1) * sizeof(binSphereTouchPairs_t);
            binSphereTouchPairs_t* numBinsSphereTouchesScan =
                (binSphereTouchPairs_t*)scratchPad.allocateTempVector("numBinsSphereTouchesScan", CD_temp_arr_bytes);
            cubDEMPrefixScan<binsSphereTouches_t, binSphereTouchPairs_t>(
                numBinsSphereTouches, numBinsSphereTouchesScan, simParams->nSpheresGM, this_stream, scratchPad);
            // If there are temp variables that need both device and host copies, we just create DualStruct on-spot
            scratchPad.allocateDualStruct("numBinSphereTouchPairs");
            size_t* pNumBinSphereTouchPairs = scratchPad.getDualStructDevice("numBinSphereTouchPairs");
            deviceAdd<size_t, binSphereTouchPairs_t, binsSphereTouches_t>(
                pNumBinSphereTouchPairs, &(numBinsSphereTouchesScan[simParams->nSpheresGM - 1]),
                &(numBinsSphereTouches[simParams->nSpheresGM - 1]), this_stream);
            deviceAssign<binSphereTouchPairs_t, size_t>(&(numBinsSphereTouchesScan[simParams->nSpheresGM]),
                                                        pNumBinSphereTouchPairs, this_stream);
            // numBinSphereTouchPairs stores data on device, but now we need it on host, so a migrantion is needed. This
            // cannot be forgotten.
            scratchPad.syncDualStructDeviceToHost("numBinSphereTouchPairs");
            // Now pNumBinSphereTouchPairs is host pointer and exclusively used on host
            pNumBinSphereTouchPairs = scratchPad.getDualStructHost("numBinSphereTouchPairs");
            // The same process is done for sphere--analytical geometry pairs as well.
            // One extra elem is used for storing the final elem in scan result.
            CD_temp_arr_bytes = (simParams->nSpheresGM + 1) * sizeof(binSphereTouchPairs_t);
            binSphereTouchPairs_t* numAnalGeoSphereTouchesScan =
                (binSphereTouchPairs_t*)scratchPad.allocateTempVector("numAnalGeoSphereTouchesScan", CD_temp_arr_bytes);
            cubDEMPrefixScan<objID_t, binSphereTouchPairs_t>(numAnalGeoSphereTouches, numAnalGeoSphereTouchesScan,
                                                             simParams->nSpheresGM, this_stream, scratchPad);
            deviceAdd<binSphereTouchPairs_t, objID_t, binSphereTouchPairs_t>(
                &(numAnalGeoSphereTouchesScan[simParams->nSpheresGM]),
                &(numAnalGeoSphereTouches[simParams->nSpheresGM - 1]),
                &(numAnalGeoSphereTouchesScan[simParams->nSpheresGM - 1]), this_stream);
            deviceAssign<size_t, binSphereTouchPairs_t>(
                &(scratchPad.numContacts), &(numAnalGeoSphereTouchesScan[simParams->nSpheresGM]), this_stream);
            // numContact is updated (with geo--sphere pair number), get it to host
            scratchPad.numContacts.toHost();
            nSphereGeoContact = *scratchPad.numContacts;
            if (*scratchPad.numContacts > idGeometryA.size()) {
                contactEventArraysResize(*(scratchPad.numContacts), idGeometryA, idGeometryB, contactType,
                                         contactPersistency, granData);
            }
            // std::cout << *pNumBinSphereTouchPairs << std::endl;
            // displayDeviceArray<binsSphereTouches_t>(numBinsSphereTouches, simParams->nSpheresGM);
            // displayDeviceArray<binSphereTouchPairs_t>(numBinsSphereTouchesScan, simParams->nSpheresGM);

            // 3rd step: use a custom kernel to figure out all sphere--bin touching pairs. Note numBinsSphereTouches can
            // retire now.
            scratchPad.finishUsingTempVector("numBinsSphereTouches");
            scratchPad.finishUsingTempVector("numAnalGeoSphereTouches");

            CD_temp_arr_bytes = (*pNumBinSphereTouchPairs) * sizeof(binID_t);
            binID_t* binIDsEachSphereTouches =
                (binID_t*)scratchPad.allocateTempVector("binIDsEachSphereTouches", CD_temp_arr_bytes);
            CD_temp_arr_bytes = (*pNumBinSphereTouchPairs) * sizeof(bodyID_t);
            bodyID_t* sphereIDsEachBinTouches =
                (bodyID_t*)scratchPad.allocateTempVector("sphereIDsEachBinTouches", CD_temp_arr_bytes);
            // This kernel is also responsible of figuring out sphere--analytical geometry pairs
            bin_sphere_kernels->kernel("populateBinSphereTouchingPairs")
                .instantiate()
                .configure(dim3(blocks_needed_for_bodies), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, this_stream)
                .launch(&simParams, &granData, numBinsSphereTouchesScan, numAnalGeoSphereTouchesScan,
                        binIDsEachSphereTouches, sphereIDsEachBinTouches, granData->idGeometryA, granData->idGeometryB,
                        granData->contactType);
            DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            // std::cout << "Unsorted bin IDs: ";
            // displayDeviceArray<binID_t>(binIDsEachSphereTouches, *pNumBinSphereTouchPairs);
            // std::cout << "Corresponding sphere IDs: ";
            // displayDeviceArray<bodyID_t>(sphereIDsEachBinTouches, *pNumBinSphereTouchPairs);

            // 4th step: allocate and populate SORTED binIDsEachSphereTouches and sphereIDsEachBinTouches. Note
            // numBinsSphereTouchesScan can retire now (analytical contacts have been processed).
            scratchPad.finishUsingTempVector("numBinsSphereTouchesScan");
            scratchPad.finishUsingTempVector("numAnalGeoSphereTouchesScan");
            CD_temp_arr_bytes = (*pNumBinSphereTouchPairs) * sizeof(bodyID_t);
            sphereIDsEachBinTouches_sorted =
                (bodyID_t*)scratchPad.allocateTempVector("sphereIDsEachBinTouches_sorted", CD_temp_arr_bytes);
            CD_temp_arr_bytes = (*pNumBinSphereTouchPairs) * sizeof(binID_t);
            binID_t* binIDsEachSphereTouches_sorted =
                (binID_t*)scratchPad.allocateTempVector("binIDsEachSphereTouches_sorted", CD_temp_arr_bytes);
            // hostSortByKey<binID_t, bodyID_t>(granData->binIDsEachSphereTouches, granData->sphereIDsEachBinTouches,
            //                                  *pNumBinSphereTouchPairs);
            cubDEMSortByKeys<binID_t, bodyID_t>(binIDsEachSphereTouches, binIDsEachSphereTouches_sorted,
                                                sphereIDsEachBinTouches, sphereIDsEachBinTouches_sorted,
                                                *pNumBinSphereTouchPairs, this_stream, scratchPad);
            // std::cout << "Sorted bin IDs: ";
            // displayDeviceArray<binID_t>(binIDsEachSphereTouches_sorted, *pNumBinSphereTouchPairs);
            // std::cout << "Corresponding sphere IDs: ";
            // displayDeviceArray<bodyID_t>(sphereIDsEachBinTouches_sorted, *pNumBinSphereTouchPairs);

            // 5th step: use DeviceRunLengthEncode to identify those active (that have bodies in them) bins.
            // Also, binIDsEachSphereTouches is large enough for a unique scan because total sphere--bin pairs are more
            // than active bins.
            binID_t* binIDsUnique = (binID_t*)binIDsEachSphereTouches;
            pNumActiveBins = scratchPad.getDualStructDevice("numActiveBins");
            cubDEMUnique<binID_t>(binIDsEachSphereTouches_sorted, binIDsUnique, pNumActiveBins,
                                  *pNumBinSphereTouchPairs, this_stream, scratchPad);
            // Allocate space for encoding output, and run it. Note the (unsorted) binIDsEachSphereTouches and
            // sphereIDsEachBinTouches can retire now.
            scratchPad.finishUsingTempVector("binIDsEachSphereTouches");
            scratchPad.finishUsingTempVector("sphereIDsEachBinTouches");
            // Get the unique check result to host
            scratchPad.syncDualStructDeviceToHost("numActiveBins");
            pNumActiveBins = scratchPad.getDualStructHost("numActiveBins");
            CD_temp_arr_bytes = (*pNumActiveBins) * sizeof(binID_t);
            // This activeBinIDs will need some host treatment later on...
            scratchPad.allocateDualArray("activeBinIDs", CD_temp_arr_bytes);
            activeBinIDs = (binID_t*)scratchPad.getDualArrayDevice("activeBinIDs");
            CD_temp_arr_bytes = (*pNumActiveBins) * sizeof(spheresBinTouches_t);
            numSpheresBinTouches =
                (spheresBinTouches_t*)scratchPad.allocateTempVector("numSpheresBinTouches", CD_temp_arr_bytes);
            // Here you don't have to toHost() again as the runlength should give the same numActiveBins as before
            pNumActiveBins = scratchPad.getDualStructDevice("numActiveBins");
            cubDEMRunLengthEncode<binID_t, spheresBinTouches_t>(binIDsEachSphereTouches_sorted, activeBinIDs,
                                                                numSpheresBinTouches, pNumActiveBins,
                                                                *pNumBinSphereTouchPairs, this_stream, scratchPad);
            pNumActiveBins = scratchPad.getDualStructHost("numActiveBins");
            // std::cout << "numActiveBins: " << *pNumActiveBins << std::endl;
            // std::cout << "activeBinIDs: ";
            // displayDeviceArray<binID_t>(activeBinIDs, *pNumActiveBins);
            // std::cout << "numSpheresBinTouches: ";
            // displayDeviceArray<spheresBinTouches_t>(numSpheresBinTouches, *pNumActiveBins);
            // std::cout << "binIDsEachSphereTouches_sorted: ";
            // displayDeviceArray<binID_t>(binIDsEachSphereTouches_sorted, *pNumBinSphereTouchPairs);
            scratchPad.finishUsingDualStruct("numBinSphereTouchPairs");

            // We find the max geo num in a bin for the purpose of adjusting bin size.
            scratchPad.allocateDualStruct("maxGeoInBin");
            spheresBinTouches_t* pMaxGeoInBin = (spheresBinTouches_t*)scratchPad.getDualStructDevice("maxGeoInBin");
            cubDEMMax<spheresBinTouches_t>(numSpheresBinTouches, pMaxGeoInBin, *pNumActiveBins, this_stream,
                                           scratchPad);
            scratchPad.syncDualStructDeviceToHost("maxGeoInBin");
            // Hmm... this only works in little-endian systems... I don't use undefined behavior that often but this
            // one...
            stateParams.maxSphFoundInBin = *((spheresBinTouches_t*)scratchPad.getDualStructHost("maxGeoInBin"));
            scratchPad.finishUsingDualStruct("maxGeoInBin");

            // Then, scan to find the offsets that are used to index into sphereIDsEachBinTouches_sorted to obtain
            // bin-wise spheres. Note binIDsEachSphereTouches_sorted can retire.
            scratchPad.finishUsingTempVector("binIDsEachSphereTouches_sorted");
            CD_temp_arr_bytes = (*pNumActiveBins) * sizeof(binSphereTouchPairs_t);
            sphereIDsLookUpTable =
                (binSphereTouchPairs_t*)scratchPad.allocateTempVector("sphereIDsLookUpTable", CD_temp_arr_bytes);
            cubDEMPrefixScan<spheresBinTouches_t, binSphereTouchPairs_t>(numSpheresBinTouches, sphereIDsLookUpTable,
                                                                         *pNumActiveBins, this_stream, scratchPad);
            // std::cout << "sphereIDsLookUpTable: ";
            // displayDeviceArray<binSphereTouchPairs_t>(sphereIDsLookUpTable, *pNumActiveBins);
        }

        // -----------------------------------------------------------------------------------------------------------
        // Triangle-related discretization and triangle--analytical contact detection
        // -----------------------------------------------------------------------------------------------------------

        // If there are meshes, they need to be processed too
        scratchPad.allocateDualStruct("numActiveBinsForTri");
        *scratchPad.getDualStructHost("numActiveBinsForTri") =
            0;                         // May need a default value on host for no-triangle case
        size_t* pNumActiveBinsForTri;  // Can be host or device pointers
        bodyID_t* triIDsEachBinTouches_sorted;
        trianglesBinTouches_t* numTrianglesBinTouches;
        binsTriangleTouchPairs_t* triIDsLookUpTable;
        float3 *sandwichANode1, *sandwichANode2, *sandwichANode3, *sandwichBNode1, *sandwichBNode2, *sandwichBNode3;
        // The following two pointers are device pointers, as outside this (simParams->nTriGM > 0) block, they are only
        // used as read-only ingredients in kernels. But their data is stored using DualArray, as the data need at some
        // point be processed on host.
        binID_t *mapTriActBinToSphActBin, *activeBinIDsForTri;
        if (simParams->nTriGM > 0) {
            // 0-th step: Make `sandwich' for each triangle (or say, create a prism out of each triangle). This is
            // obviously for our delayed contact detection safety. And finally, if a sphere's distance away from one of
            // the 2 prism surfaces is smaller than its radius, it has contact with this prism, hence potentially with
            // this triangle.
            CD_temp_arr_bytes = simParams->nTriGM * sizeof(float3) * 3;
            sandwichANode1 = (float3*)scratchPad.allocateTempVector("sandwichANode1", CD_temp_arr_bytes);
            sandwichANode2 = sandwichANode1 + simParams->nTriGM;
            sandwichANode3 = sandwichANode2 + simParams->nTriGM;
            sandwichBNode1 = (float3*)scratchPad.allocateTempVector("sandwichBNode1", CD_temp_arr_bytes);
            sandwichBNode2 = sandwichBNode1 + simParams->nTriGM;
            sandwichBNode3 = sandwichBNode2 + simParams->nTriGM;
            size_t blocks_needed_for_tri =
                (simParams->nTriGM + DEME_NUM_TRIANGLE_PER_BLOCK - 1) / DEME_NUM_TRIANGLE_PER_BLOCK;
            bin_triangle_kernels->kernel("makeTriangleSandwich")
                .instantiate()
                .configure(dim3(blocks_needed_for_tri), dim3(DEME_NUM_TRIANGLE_PER_BLOCK), 0, this_stream)
                .launch(&simParams, &granData, sandwichANode1, sandwichANode2, sandwichANode3, sandwichBNode1,
                        sandwichBNode2, sandwichBNode3);
            DEME_GPU_CALL(cudaStreamSynchronize(this_stream));

            // 1st step: register the number of triangle--bin touching pairs for each triangle for further processing.
            // We also use the opportunity to find how many analytical objects each triangle touches.
            CD_temp_arr_bytes = simParams->nTriGM * sizeof(binsTriangleTouches_t);
            binsTriangleTouches_t* numBinsTriTouches =
                (binsTriangleTouches_t*)scratchPad.allocateTempVector("numBinsTriTouches", CD_temp_arr_bytes);
            objID_t* numAnalGeoTriTouches = nullptr;
            if (solverFlags.meshUniversalContact) {
                CD_temp_arr_bytes = simParams->nTriGM * sizeof(objID_t);
                numAnalGeoTriTouches =
                    (objID_t*)scratchPad.allocateTempVector("numAnalGeoTriTouches", CD_temp_arr_bytes);
            }
            bin_triangle_kernels->kernel("getNumberOfBinsEachTriangleTouches")
                .instantiate()
                .configure(dim3(blocks_needed_for_tri), dim3(DEME_NUM_TRIANGLE_PER_BLOCK), 0, this_stream)
                .launch(&simParams, &granData, numBinsTriTouches, numAnalGeoTriTouches, sandwichANode1, sandwichANode2,
                        sandwichANode3, sandwichBNode1, sandwichBNode2, sandwichBNode3,
                        solverFlags.meshUniversalContact);
            DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            // std::cout << "numBinsTriTouches: " << std::endl;
            // displayDeviceArray<binsTriangleTouches_t>(numBinsTriTouches, simParams->nTriGM);
            // std::cout << "numAnalGeoTriTouches: " << std::endl;
            // displayDeviceArray<objID_t>(numAnalGeoTriTouches, simParams->nTriGM);

            // 2nd step: prefix scan tri--bin touching pairs
            // The last element of this scanned array is useful: it can be used to check if the 2 sweeps reach the same
            // conclusion on bin--tri touch pairs
            CD_temp_arr_bytes = (simParams->nTriGM + 1) * sizeof(binsTriangleTouchPairs_t);
            binsTriangleTouchPairs_t* numBinsTriTouchesScan =
                (binsTriangleTouchPairs_t*)scratchPad.allocateTempVector("numBinsTriTouchesScan", CD_temp_arr_bytes);
            cubDEMPrefixScan<binsTriangleTouches_t, binsTriangleTouchPairs_t>(
                numBinsTriTouches, numBinsTriTouchesScan, simParams->nTriGM, this_stream, scratchPad);
            scratchPad.allocateDualStruct("numBinTriTouchPairs");
            size_t* pNumBinTriTouchPairs = scratchPad.getDualStructDevice("numBinTriTouchPairs");
            deviceAdd<size_t, binsTriangleTouchPairs_t, binsTriangleTouches_t>(
                pNumBinTriTouchPairs, &(numBinsTriTouchesScan[simParams->nTriGM - 1]),
                &(numBinsTriTouches[simParams->nTriGM - 1]), this_stream);
            deviceAssign<binsTriangleTouchPairs_t, size_t>(&(numBinsTriTouchesScan[simParams->nTriGM]),
                                                           pNumBinTriTouchPairs, this_stream);
            scratchPad.syncDualStructDeviceToHost("numBinTriTouchPairs");
            pNumBinTriTouchPairs = scratchPad.getDualStructHost("numBinTriTouchPairs");
            // Again, numBinsTriTouchesScan is used in populateBinTriangleTouchingPairs

            binsTriangleTouchPairs_t* numAnalGeoTriTouchesScan = nullptr;
            if (solverFlags.meshUniversalContact) {
                // Now, the same process is done for tri--analytical geometry pairs as well.
                // One extra elem is used for storing the final elem in scan result.
                CD_temp_arr_bytes = (simParams->nTriGM + 1) * sizeof(binsTriangleTouchPairs_t);
                numAnalGeoTriTouchesScan = (binsTriangleTouchPairs_t*)scratchPad.allocateTempVector(
                    "numAnalGeoTriTouchesScan", CD_temp_arr_bytes);
                cubDEMPrefixScan<objID_t, binsTriangleTouchPairs_t>(numAnalGeoTriTouches, numAnalGeoTriTouchesScan,
                                                                    simParams->nTriGM, this_stream, scratchPad);
                // numContacts is for temp use, and it got overwritten in this step
                deviceAdd<binsTriangleTouchPairs_t, objID_t, binsTriangleTouchPairs_t>(
                    &(numAnalGeoTriTouchesScan[simParams->nTriGM]), &(numAnalGeoTriTouches[simParams->nTriGM - 1]),
                    &(numAnalGeoTriTouchesScan[simParams->nTriGM - 1]), this_stream);
                deviceAssign<size_t, binsTriangleTouchPairs_t>(
                    &(scratchPad.numContacts), &(numAnalGeoTriTouchesScan[simParams->nTriGM]), this_stream);
                // numContact is updated (with geo--sphere pair number), get it to host
                scratchPad.numContacts.toHost();
                nTriGeoContact = *scratchPad.numContacts;
                // But we still want numContacts to keep the running length of contacts, so overwrite it; no worry as
                // numContacts is host-major, so no need to update device
                *scratchPad.numContacts = nSphereGeoContact + nTriGeoContact;
                if (*scratchPad.numContacts > idGeometryA.size()) {
                    contactEventArraysResize(*(scratchPad.numContacts), idGeometryA, idGeometryB, contactType,
                                             contactPersistency, granData);
                }
                // std::cout << "numAnalGeoTriTouchesScan: " << std::endl;
                // displayDeviceArray<binsTriangleTouchPairs_t>(numAnalGeoTriTouchesScan, simParams->nTriGM);
            }

            // 3rd step: use a custom kernel to figure out all tri--bin touching pairs. Note numBinsTriTouches can
            // retire now.
            scratchPad.finishUsingTempVector("numBinsTriTouches");
            scratchPad.finishUsingTempVector("numAnalGeoTriTouches");

            CD_temp_arr_bytes = *pNumBinTriTouchPairs * sizeof(binID_t);
            binID_t* binIDsEachTriTouches =
                (binID_t*)scratchPad.allocateTempVector("binIDsEachTriTouches", CD_temp_arr_bytes);
            CD_temp_arr_bytes = *pNumBinTriTouchPairs * sizeof(bodyID_t);
            bodyID_t* triIDsEachBinTouches =
                (bodyID_t*)scratchPad.allocateTempVector("triIDsEachBinTouches", CD_temp_arr_bytes);
            // Tri--geo contact pairs go after sphere--anal-geo contacts
            bodyID_t* idTriA = (granData->idGeometryA + nSphereGeoContact);
            bodyID_t* idGeoB = (granData->idGeometryB + nSphereGeoContact);
            contact_t* dType = (granData->contactType + nSphereGeoContact);
            bin_triangle_kernels->kernel("populateBinTriangleTouchingPairs")
                .instantiate()
                .configure(dim3(blocks_needed_for_tri), dim3(DEME_NUM_TRIANGLE_PER_BLOCK), 0, this_stream)
                .launch(&simParams, &granData, numBinsTriTouchesScan, numAnalGeoTriTouchesScan, binIDsEachTriTouches,
                        triIDsEachBinTouches, sandwichANode1, sandwichANode2, sandwichANode3, sandwichBNode1,
                        sandwichBNode2, sandwichBNode3, idTriA, idGeoB, dType, solverFlags.meshUniversalContact);
            DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            // std::cout << "binIDsEachTriTouches: " << std::endl;
            // displayDeviceArray<binsTriangleTouches_t>(binIDsEachTriTouches, *pNumBinTriTouchPairs);
            // std::cout << "dType: " << std::endl;
            // displayDeviceArray<contact_t>(dType, nTriGeoContact);
            // std::cout << "mesh patch pairs:" << std::endl;
            // displayDeviceArray<patchIDPair_t>(patchPairs, nTriGeoContact);

            // 4th step: allocate and populate SORTED binIDsEachTriTouches and triIDsEachBinTouches. Note
            // numBinsTriTouchesScan can retire now (analytical contacts also processed).
            scratchPad.finishUsingTempVector("numBinsTriTouchesScan");
            scratchPad.finishUsingTempVector("numAnalGeoTriTouchesScan");
            CD_temp_arr_bytes = *pNumBinTriTouchPairs * sizeof(bodyID_t);
            triIDsEachBinTouches_sorted =
                (bodyID_t*)scratchPad.allocateTempVector("triIDsEachBinTouches_sorted", CD_temp_arr_bytes);
            CD_temp_arr_bytes = *pNumBinTriTouchPairs * sizeof(binID_t);
            binID_t* binIDsEachTriTouches_sorted =
                (binID_t*)scratchPad.allocateTempVector("binIDsEachTriTouches_sorted", CD_temp_arr_bytes);
            cubDEMSortByKeys<binID_t, bodyID_t>(binIDsEachTriTouches, binIDsEachTriTouches_sorted, triIDsEachBinTouches,
                                                triIDsEachBinTouches_sorted, *pNumBinTriTouchPairs, this_stream,
                                                scratchPad);

            // 5th step: use DeviceRunLengthEncode to identify those active (that have tris in them) bins.
            // Also, binIDsEachTriTouches is large enough for a unique scan because total sphere--bin pairs are more
            // than active bins.
            binID_t* binIDsUnique = (binID_t*)binIDsEachTriTouches;
            pNumActiveBinsForTri = scratchPad.getDualStructDevice("numActiveBinsForTri");
            cubDEMUnique<binID_t>(binIDsEachTriTouches_sorted, binIDsUnique, pNumActiveBinsForTri,
                                  *pNumBinTriTouchPairs, this_stream, scratchPad);
            // Allocate space for encoding output, and run it. Note the (unsorted) binIDsEachTriTouches and
            // triIDsEachBinTouches can retire now.
            scratchPad.finishUsingTempVector("binIDsEachTriTouches");
            scratchPad.finishUsingTempVector("triIDsEachBinTouches");
            // Bring value to host
            scratchPad.syncDualStructDeviceToHost("numActiveBinsForTri");
            pNumActiveBinsForTri = scratchPad.getDualStructHost("numActiveBinsForTri");
            CD_temp_arr_bytes = (*pNumActiveBinsForTri) * sizeof(binID_t);
            // Again, activeBinIDsForTri has some data processing needed on host, so allocated as DualArray
            scratchPad.allocateDualArray("activeBinIDsForTri", CD_temp_arr_bytes);
            activeBinIDsForTri = (binID_t*)scratchPad.getDualArrayDevice("activeBinIDsForTri");
            CD_temp_arr_bytes = (*pNumActiveBinsForTri) * sizeof(trianglesBinTouches_t);
            numTrianglesBinTouches =
                (trianglesBinTouches_t*)scratchPad.allocateTempVector("numTrianglesBinTouches", CD_temp_arr_bytes);
            // Again, no need to bring numActiveBinsForTri to host again, as values not changed
            pNumActiveBinsForTri = scratchPad.getDualStructDevice("numActiveBinsForTri");
            cubDEMRunLengthEncode<binID_t, trianglesBinTouches_t>(binIDsEachTriTouches_sorted, activeBinIDsForTri,
                                                                  numTrianglesBinTouches, pNumActiveBinsForTri,
                                                                  *pNumBinTriTouchPairs, this_stream, scratchPad);
            pNumActiveBinsForTri = scratchPad.getDualStructHost("numActiveBinsForTri");
            // std::cout << "activeBinIDsForTri: " << std::endl;
            // displayDeviceArray<binID_t>(activeBinIDsForTri, *pNumActiveBinsForTri);
            // std::cout << "NumActiveBinsForTri: " << *pNumActiveBinsForTri << std::endl;
            // std::cout << "NumActiveBins: " << *pNumActiveBins << std::endl;
            scratchPad.finishUsingDualStruct("numBinTriTouchPairs");

            // We find the max tri num in a bin for the purpose of adjusting bin size
            scratchPad.allocateDualStruct("maxGeoInBin");
            trianglesBinTouches_t* pMaxGeoInBin = (trianglesBinTouches_t*)scratchPad.getDualStructDevice("maxGeoInBin");
            cubDEMMax<trianglesBinTouches_t>(numTrianglesBinTouches, pMaxGeoInBin, *pNumActiveBinsForTri, this_stream,
                                             scratchPad);
            scratchPad.syncDualStructDeviceToHost("maxGeoInBin");
            // Find the max tri number. Hmm... this only works in little-endian systems... I don't use undefined
            // behavior that often but this one...
            stateParams.maxTriFoundInBin = *((trianglesBinTouches_t*)scratchPad.getDualStructHost("maxGeoInBin"));
            scratchPad.finishUsingDualStruct("maxGeoInBin");

            // 6th step: map activeBinIDsForTri to activeBinIDs, so that when we are processing the bins in
            // activeBinIDsForTri, we know where to find the corresponding bin that resides in activeBinIDs, to bring
            // spheres into this bin-wise contact detection sweep.
            CD_temp_arr_bytes = (*pNumActiveBinsForTri) * sizeof(binID_t);
            scratchPad.allocateDualArray("mapTriActBinToSphActBin", CD_temp_arr_bytes);
            {
                // This `merge search' task is very unsuitable for GPU... So we create host version of these work arrays
                // temporarily and use them. Don't worry about the temporary part, as it's only in logic, no actual
                // memory allocation, as vectorPool handles it.
                scratchPad.syncDualArrayDeviceToHost("activeBinIDsForTri");
                binID_t* activeBinIDsForTri = (binID_t*)scratchPad.getDualArrayHost("activeBinIDsForTri");
                // There has to be spheres for activeBinIDs to exist, so we have to check
                binID_t* activeBinIDs;
                if (scratchPad.existDualArray("activeBinIDs")) {
                    scratchPad.syncDualArrayDeviceToHost("activeBinIDs");
                    activeBinIDs = (binID_t*)scratchPad.getDualArrayHost("activeBinIDs");
                }
                binID_t* mapTriActBinToSphActBin = (binID_t*)scratchPad.getDualArrayHost("mapTriActBinToSphActBin");
                hostMergeSearchMapGen(activeBinIDsForTri, activeBinIDs, mapTriActBinToSphActBin, *pNumActiveBinsForTri,
                                      *pNumActiveBins, deme::NULL_BINID);
                // activeBinIDsForTri and activeBinIDs are not changed, so no need to sync them back
                scratchPad.syncDualArrayHostToDevice("mapTriActBinToSphActBin");
            }
            // Bind back some of the device pointers is not necessary as they are not changed
            // But it's needed for mapTriActBinToSphActBin as it's its first assignment...
            mapTriActBinToSphActBin = (binID_t*)scratchPad.getDualArrayDevice("mapTriActBinToSphActBin");
            // activeBinIDsForTri = (binID_t*)scratchPad.getDualArrayDevice("activeBinIDsForTri");
            // activeBinIDs = (binID_t*)scratchPad.getDualArrayDevice("activeBinIDs");
            // std::cout << "mapTriActBinToSphActBin: " << std::endl;
            // displayDeviceArray<binID_t>(mapTriActBinToSphActBin, *pNumActiveBinsForTri);

            // 7th step: scan to find the offsets that are used to index into triIDsEachBinTouches_sorted to obtain
            // bin-wise triangles. Note binIDsEachTriTouches_sorted can retire.
            scratchPad.finishUsingTempVector("binIDsEachTriTouches_sorted");
            CD_temp_arr_bytes = (*pNumActiveBinsForTri) * sizeof(binsTriangleTouchPairs_t);
            triIDsLookUpTable =
                (binsTriangleTouchPairs_t*)scratchPad.allocateTempVector("triIDsLookUpTable", CD_temp_arr_bytes);
            cubDEMPrefixScan<trianglesBinTouches_t, binsTriangleTouchPairs_t>(
                numTrianglesBinTouches, triIDsLookUpTable, *pNumActiveBinsForTri, this_stream, scratchPad);
        }
        timers.GetTimer("Discretize domain").stop();

        // -----------------------------------------------------------------------------------------------------------
        // Populating contact pairs
        // -----------------------------------------------------------------------------------------------------------

        timers.GetTimer("Find contact pairs").start();
        // Final step: find the contact pairs. One-two punch: first find num of contacts in each bin, then prescan, then
        // find the actual pair names. A new temp array is needed for this numSphContactsInEachBin. Note that
        // binContactPairs_t also doubles as the type for the number of tri--sph and tri--tri contact pairs.
        binContactPairs_t *numSphContactsInEachBin, *numTriSphContactsInEachBin, *numTriTriContactsInEachBin = nullptr;
        // Figure out how many blocks are needed for the contact detection, with one block per bin.
        size_t blocks_needed_for_bins_sph = 0;
        if (simParams->nSpheresGM > 0) {
            blocks_needed_for_bins_sph = *pNumActiveBins;
            CD_temp_arr_bytes = (*pNumActiveBins) * sizeof(binContactPairs_t);
            numSphContactsInEachBin =
                (binContactPairs_t*)scratchPad.allocateTempVector("numSphContactsInEachBin", CD_temp_arr_bytes);
        }
        size_t blocks_needed_for_bins_tri = 0;
        if (simParams->nTriGM > 0) {
            blocks_needed_for_bins_tri = *pNumActiveBinsForTri;
            CD_temp_arr_bytes = (*pNumActiveBinsForTri) * sizeof(binContactPairs_t);
            numTriSphContactsInEachBin =
                (binContactPairs_t*)scratchPad.allocateTempVector("numTriSphContactsInEachBin", CD_temp_arr_bytes);
            // Naturally, for tri--tri contacts, only active bins for tri are needed
            if (solverFlags.meshUniversalContact) {
                CD_temp_arr_bytes = (*pNumActiveBinsForTri) * sizeof(binContactPairs_t);
                numTriTriContactsInEachBin =
                    (binContactPairs_t*)scratchPad.allocateTempVector("numTriTriContactsInEachBin", CD_temp_arr_bytes);
            }
        }

        // Only needed when there are spheres or triangles
        if (blocks_needed_for_bins_sph > 0 || blocks_needed_for_bins_tri > 0) {
            if (blocks_needed_for_bins_sph > 0) {
                sphere_contact_kernels->kernel("getNumberOfSphereContactsEachBin")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_bins_sph), dim3(DEME_KT_CD_NTHREADS_PER_BLOCK), 0, this_stream)
                    .launch(&simParams, &granData, sphereIDsEachBinTouches_sorted, activeBinIDs, numSpheresBinTouches,
                            sphereIDsLookUpTable, numSphContactsInEachBin, *pNumActiveBins);
                DEME_GPU_CALL_WATCH_BETA(cudaStreamSynchronize(this_stream));
            }

            if (blocks_needed_for_bins_tri > 0) {
                // We got both tri--sph and tri--tri contacts in this kernel
                sphTri_contact_kernels->kernel("getNumberOfTriangleContactsEachBin")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_bins_tri), dim3(DEME_KT_CD_NTHREADS_PER_BLOCK), 0, this_stream)
                    .launch(&simParams, &granData, sphereIDsEachBinTouches_sorted, activeBinIDs, numSpheresBinTouches,
                            sphereIDsLookUpTable, mapTriActBinToSphActBin, triIDsEachBinTouches_sorted,
                            activeBinIDsForTri, numTrianglesBinTouches, triIDsLookUpTable, numTriSphContactsInEachBin,
                            numTriTriContactsInEachBin, sandwichANode1, sandwichANode2, sandwichANode3, sandwichBNode1,
                            sandwichBNode2, sandwichBNode3, *pNumActiveBinsForTri, solverFlags.meshUniversalContact);
                DEME_GPU_CALL_WATCH_BETA(cudaStreamSynchronize(this_stream));
                // std::cout << "numTriSphContactsInEachBin: " << std::endl;
                // displayDeviceArray<binContactPairs_t>(numTriSphContactsInEachBin, *pNumActiveBinsForTri);
                // std::cout << "numTriTriContactsInEachBin: " << std::endl;
                // displayDeviceArray<binContactPairs_t>(numTriTriContactsInEachBin, *pNumActiveBinsForTri);
            }

            // Prescan numSphContactsInEachBin to get the final sphSphContactReportOffsets and
            // triSphContactReportOffsets. New vectors are needed.
            // The extra entry is maybe superfluous and is for extra safety, in case the 2 sweeps do not agree with each
            // other.
            contactPairs_t* sphSphContactReportOffsets;
            if (simParams->nSpheresGM > 0) {
                CD_temp_arr_bytes = (*pNumActiveBins + 1) * sizeof(contactPairs_t);
                sphSphContactReportOffsets =
                    (contactPairs_t*)scratchPad.allocateTempVector("sphSphContactReportOffsets", CD_temp_arr_bytes);
                cubDEMPrefixScan<binContactPairs_t, contactPairs_t>(numSphContactsInEachBin, sphSphContactReportOffsets,
                                                                    *pNumActiveBins, this_stream, scratchPad);
            }
            contactPairs_t *triSphContactReportOffsets, *triTriContactReportOffsets = nullptr;
            if (simParams->nTriGM > 0) {
                CD_temp_arr_bytes = (*pNumActiveBinsForTri + 1) * sizeof(contactPairs_t);
                triSphContactReportOffsets =
                    (contactPairs_t*)scratchPad.allocateTempVector("triSphContactReportOffsets", CD_temp_arr_bytes);
                cubDEMPrefixScan<binContactPairs_t, contactPairs_t>(numTriSphContactsInEachBin,
                                                                    triSphContactReportOffsets, *pNumActiveBinsForTri,
                                                                    this_stream, scratchPad);
                if (solverFlags.meshUniversalContact) {
                    // tri--tri contact report offsets...
                    CD_temp_arr_bytes = (*pNumActiveBinsForTri + 1) * sizeof(contactPairs_t);
                    triTriContactReportOffsets =
                        (contactPairs_t*)scratchPad.allocateTempVector("triTriContactReportOffsets", CD_temp_arr_bytes);
                    cubDEMPrefixScan<binContactPairs_t, contactPairs_t>(numTriTriContactsInEachBin,
                                                                        triTriContactReportOffsets,
                                                                        *pNumActiveBinsForTri, this_stream, scratchPad);
                }
            }
            // DEME_DEBUG_PRINTF("Num contacts each bin:");
            // DEME_DEBUG_EXEC(displayDeviceArray<binContactPairs_t>(numSphContactsInEachBin, *pNumActiveBins));
            // DEME_DEBUG_PRINTF("Tri contact report offsets:");
            // DEME_DEBUG_EXEC(displayDeviceArray<contactPairs_t>(triSphContactReportOffsets, *pNumActiveBinsForTri));
            // DEME_DEBUG_PRINTF("Family number:");
            // DEME_DEBUG_EXEC(displayDeviceArray<family_t>(granData->familyID.device(), simParams->nOwnerBodies));

            // Add sphere--sphere contacts together with sphere--analytical geometry contacts
            if (simParams->nSpheresGM > 0) {
                scratchPad.allocateDualStruct("numSSContact");
                deviceAdd<size_t, binContactPairs_t, contactPairs_t>(
                    scratchPad.getDualStructDevice("numSSContact"), &(numSphContactsInEachBin[*pNumActiveBins - 1]),
                    &(sphSphContactReportOffsets[*pNumActiveBins - 1]), this_stream);
                deviceAssign<contactPairs_t, size_t>(&(sphSphContactReportOffsets[*pNumActiveBins]),
                                                     scratchPad.getDualStructDevice("numSSContact"), this_stream);
                scratchPad.syncDualStructDeviceToHost("numSSContact");
                nSphereSphereContact = *scratchPad.getDualStructHost("numSSContact");
                scratchPad.finishUsingDualStruct("numSSContact");
            }
            if (simParams->nTriGM > 0) {
                scratchPad.allocateDualStruct("numSMContact");
                deviceAdd<size_t, binContactPairs_t, contactPairs_t>(
                    scratchPad.getDualStructDevice("numSMContact"),
                    &(numTriSphContactsInEachBin[*pNumActiveBinsForTri - 1]),
                    &(triSphContactReportOffsets[*pNumActiveBinsForTri - 1]), this_stream);
                deviceAssign<contactPairs_t, size_t>(&(triSphContactReportOffsets[*pNumActiveBinsForTri]),
                                                     scratchPad.getDualStructDevice("numSMContact"), this_stream);
                scratchPad.syncDualStructDeviceToHost("numSMContact");
                nTriSphereContact = *scratchPad.getDualStructHost("numSMContact");
                scratchPad.finishUsingDualStruct("numSMContact");
                // If mesh-universal contact is on, tri--tri contacts are also possible, add it here...
                if (solverFlags.meshUniversalContact) {
                    scratchPad.allocateDualStruct("numMMContact");
                    deviceAdd<size_t, binContactPairs_t, contactPairs_t>(
                        scratchPad.getDualStructDevice("numMMContact"),
                        &(numTriTriContactsInEachBin[*pNumActiveBinsForTri - 1]),
                        &(triTriContactReportOffsets[*pNumActiveBinsForTri - 1]), this_stream);
                    deviceAssign<contactPairs_t, size_t>(&(triTriContactReportOffsets[*pNumActiveBinsForTri]),
                                                         scratchPad.getDualStructDevice("numMMContact"), this_stream);
                    scratchPad.syncDualStructDeviceToHost("numMMContact");
                    nTriTriContact = *scratchPad.getDualStructHost("numMMContact");
                    scratchPad.finishUsingDualStruct("numMMContact");
                }
            }
            // std::cout << "nSphereGeoContact: " << nSphereGeoContact << std::endl;
            // std::cout << "nSphereSphereContact: " << nSphereSphereContact << std::endl;
            // std::cout << "nTriSphereContact: " << nTriSphereContact << std::endl;
            // std::cout << "nTriTriContact: " << nTriTriContact << std::endl;
            // ----------------------------------------------------------------------------------------
            // IMPORTANT NOTE: contacts got here may have duplicates, as we do not rule out duplicates in tri--tri
            // contacts, so the numbers here are not reliable. These duplicates will be filtered out later.
            // ----------------------------------------------------------------------------------------

            *scratchPad.numContacts =
                nSphereSphereContact + nSphereGeoContact + nTriGeoContact + nTriSphereContact + nTriTriContact;
            if (*scratchPad.numContacts > idGeometryA.size()) {
                contactEventArraysResize(*scratchPad.numContacts, idGeometryA, idGeometryB, contactType,
                                         contactPersistency, granData);
            }

            // Sphere--sphere contact pairs go after tri--anal-geo contacts
            bodyID_t* idSphA = (granData->idGeometryA + nSphereGeoContact + nTriGeoContact);
            bodyID_t* idSphB = (granData->idGeometryB + nSphereGeoContact + nTriGeoContact);
            contact_t* dType = (granData->contactType + nSphereGeoContact + nTriGeoContact);
            // Then fill in those contacts
            if (blocks_needed_for_bins_sph > 0) {
                sphere_contact_kernels->kernel("populateSphereContactPairsEachBin")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_bins_sph), dim3(DEME_KT_CD_NTHREADS_PER_BLOCK), 0, this_stream)
                    .launch(&simParams, &granData, sphereIDsEachBinTouches_sorted, activeBinIDs, numSpheresBinTouches,
                            sphereIDsLookUpTable, sphSphContactReportOffsets, idSphA, idSphB, dType, *pNumActiveBins);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            }

            // Triangle--sphere contact pairs go after sphere--sphere contacts. Remember to mark their type.
            if (blocks_needed_for_bins_tri > 0) {
                // The contact in this section can be sph--tri...
                bodyID_t* idSphA_sm =
                    (granData->idGeometryA + nSphereGeoContact + nTriGeoContact + nSphereSphereContact);
                bodyID_t* idTriB_sm =
                    (granData->idGeometryB + nSphereGeoContact + nTriGeoContact + nSphereSphereContact);
                contact_t* dType_sm =
                    (granData->contactType + nSphereGeoContact + nTriGeoContact + nSphereSphereContact);
                // Mesh patch IDs of SM, MM contacts go after MA contacts
                // Or tri--tri... They go after tri--sph contacts
                bodyID_t* idTriA_mm = (granData->idGeometryA + nSphereGeoContact + nTriGeoContact +
                                       nSphereSphereContact + nTriSphereContact);
                bodyID_t* idTriB_mm = (granData->idGeometryB + nSphereGeoContact + nTriGeoContact +
                                       nSphereSphereContact + nTriSphereContact);
                contact_t* dType_mm = (granData->contactType + nSphereGeoContact + nTriGeoContact +
                                       nSphereSphereContact + nTriSphereContact);
                // And two possible types of contact are resolved both in this kernel
                sphTri_contact_kernels->kernel("populateTriangleContactsEachBin")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_bins_tri), dim3(DEME_KT_CD_NTHREADS_PER_BLOCK), 0, this_stream)
                    .launch(&simParams, &granData, sphereIDsEachBinTouches_sorted, activeBinIDs, numSpheresBinTouches,
                            sphereIDsLookUpTable, mapTriActBinToSphActBin, triIDsEachBinTouches_sorted,
                            activeBinIDsForTri, numTrianglesBinTouches, triIDsLookUpTable, triSphContactReportOffsets,
                            triTriContactReportOffsets, idSphA_sm, idTriB_sm, dType_sm, idTriA_mm, idTriB_mm, dType_mm,
                            sandwichANode1, sandwichANode2, sandwichANode3, sandwichBNode1, sandwichBNode2,
                            sandwichBNode3, *pNumActiveBinsForTri, solverFlags.meshUniversalContact);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            }
        }  // End of bin-wise contact detection subroutine

        // The following vectors are used till the end
        scratchPad.finishUsingTempVector("sphereIDsEachBinTouches_sorted");
        scratchPad.finishUsingDualArray("activeBinIDs");
        scratchPad.finishUsingTempVector("numSpheresBinTouches");
        scratchPad.finishUsingTempVector("sphereIDsLookUpTable");
        scratchPad.finishUsingTempVector("sandwichANode1");
        scratchPad.finishUsingTempVector("sandwichBNode1");
        scratchPad.finishUsingTempVector("triIDsEachBinTouches_sorted");
        scratchPad.finishUsingDualArray("activeBinIDsForTri");
        scratchPad.finishUsingTempVector("numTrianglesBinTouches");
        scratchPad.finishUsingDualArray("mapTriActBinToSphActBin");
        scratchPad.finishUsingTempVector("triIDsLookUpTable");
        scratchPad.finishUsingTempVector("numSphContactsInEachBin");
        scratchPad.finishUsingTempVector("numTriSphContactsInEachBin");
        scratchPad.finishUsingTempVector("numTriTriContactsInEachBin");
        scratchPad.finishUsingTempVector("sphSphContactReportOffsets");
        scratchPad.finishUsingTempVector("triSphContactReportOffsets");
        scratchPad.finishUsingTempVector("triTriContactReportOffsets");

        scratchPad.finishUsingDualStruct("numActiveBins");
        scratchPad.finishUsingDualStruct("numActiveBinsForTri");

        // -----------------------------------------------------------------------------------------------------------
        // One more task: If the user specified persistent contacts, we check the previous contact list
        // and see if there are some contacts we need to add to the current list. Even if we detected 0 contacts, we
        // might still have persistent contacts to add to the list.
        // -----------------------------------------------------------------------------------------------------------

        // At user API level, persistent contacts are only supported for history-based models, and this check is totally
        // for redundancy purpose.
        if (solverFlags.hasPersistentContacts && !solverFlags.isHistoryless) {
            // A bool array to help find what persistent contacts from the prev array need to be processed...
            size_t flag_arr_bytes = (*scratchPad.numPrevContacts) * sizeof(notStupidBool_t);
            notStupidBool_t* grab_flags = (notStupidBool_t*)scratchPad.allocateTempVector("grab_flags", flag_arr_bytes);
            size_t blocks_needed_for_flagging =
                (*scratchPad.numPrevContacts + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
            if (blocks_needed_for_flagging > 0) {
                markBoolIf<<<dim3(blocks_needed_for_flagging), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, this_stream>>>(
                    grab_flags, granData->contactPersistency, CONTACT_IS_PERSISTENT, *scratchPad.numPrevContacts);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            }
            // Store the number of persistent contacts
            scratchPad.allocateDualStruct("numPersistCnts");

            // Then extract the persistent array
            // This many elements are sufficient, at very least...
            size_t selected_ids_bytes = (*scratchPad.numPrevContacts) * sizeof(bodyID_t);
            size_t selected_types_bytes = (*scratchPad.numPrevContacts) * sizeof(contact_t);
            bodyID_t* selected_idA = (bodyID_t*)scratchPad.allocateTempVector("selected_idA", selected_ids_bytes);
            bodyID_t* selected_idB = (bodyID_t*)scratchPad.allocateTempVector("selected_idB", selected_ids_bytes);
            contact_t* selected_types =
                (contact_t*)scratchPad.allocateTempVector("selected_types", selected_types_bytes);

            cubDEMSelectFlagged<bodyID_t, notStupidBool_t>(granData->previous_idGeometryA, selected_idA, grab_flags,
                                                           scratchPad.getDualStructDevice("numPersistCnts"),
                                                           *scratchPad.numPrevContacts, this_stream, scratchPad);
            cubDEMSelectFlagged<bodyID_t, notStupidBool_t>(granData->previous_idGeometryB, selected_idB, grab_flags,
                                                           scratchPad.getDualStructDevice("numPersistCnts"),
                                                           *scratchPad.numPrevContacts, this_stream, scratchPad);
            cubDEMSelectFlagged<contact_t, notStupidBool_t>(granData->previous_contactType, selected_types, grab_flags,
                                                            scratchPad.getDualStructDevice("numPersistCnts"),
                                                            *scratchPad.numPrevContacts, this_stream, scratchPad);
            // Those flag selections give the same result. Bring it to host.
            scratchPad.syncDualStructDeviceToHost("numPersistCnts");
            size_t* pNumPersistCnts = scratchPad.getDualStructHost("numPersistCnts");

            // Then concatenate the persistent contacts (goes first) and the newly detected contacts (follows after)
            size_t total_ids_bytes = (*scratchPad.numContacts + *pNumPersistCnts) * sizeof(bodyID_t);
            size_t total_types_bytes = (*scratchPad.numContacts + *pNumPersistCnts) * sizeof(contact_t);
            size_t total_persistency_bytes = (*scratchPad.numContacts + *pNumPersistCnts) * sizeof(notStupidBool_t);
            selected_ids_bytes = (*pNumPersistCnts) * sizeof(bodyID_t);
            selected_types_bytes = (*pNumPersistCnts) * sizeof(contact_t);
            bodyID_t* total_idA = (bodyID_t*)scratchPad.allocateTempVector("total_idA", total_ids_bytes);
            bodyID_t* total_idB = (bodyID_t*)scratchPad.allocateTempVector("total_idB", total_ids_bytes);
            contact_t* total_types = (contact_t*)scratchPad.allocateTempVector("total_types", total_types_bytes);
            notStupidBool_t* total_persistency =
                (notStupidBool_t*)scratchPad.allocateTempVector("total_persistency", total_persistency_bytes);
            DEME_GPU_CALL(cudaMemcpy(total_idA, selected_idA, selected_ids_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(total_idA + *pNumPersistCnts, granData->idGeometryA,
                                     total_ids_bytes - selected_ids_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(total_idB, selected_idB, selected_ids_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(total_idB + *pNumPersistCnts, granData->idGeometryB,
                                     total_ids_bytes - selected_ids_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(total_types, selected_types, selected_types_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(total_types + *pNumPersistCnts, granData->contactType,
                                     total_types_bytes - selected_types_bytes, cudaMemcpyDeviceToDevice));
            // For the selected portion, persistency is all 1, the rest is all 0
            DEME_GPU_CALL(cudaMemset(total_persistency, CONTACT_NOT_PERSISTENT, total_persistency_bytes));
            size_t blocks_needed_for_setting_1 =
                (*pNumPersistCnts + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
            if (blocks_needed_for_setting_1 > 0) {
                setArr<<<dim3(blocks_needed_for_setting_1), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, this_stream>>>(
                    total_persistency, *pNumPersistCnts, CONTACT_IS_PERSISTENT);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            }
            scratchPad.finishUsingTempVector("grab_flags");
            scratchPad.finishUsingTempVector("selected_idA");
            scratchPad.finishUsingTempVector("selected_idB");
            scratchPad.finishUsingTempVector("selected_types");

            // Then remove potential redundency in the current contact array.
            // To do that, we sort by idA...
            size_t numTotalCnts = *scratchPad.numContacts + *pNumPersistCnts;
            contact_t* contactType_sorted =
                (contact_t*)scratchPad.allocateTempVector("contactType_sorted", total_types_bytes);
            bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector("idA_sorted", total_ids_bytes);
            bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector("idB_sorted", total_ids_bytes);
            notStupidBool_t* persistency_sorted =
                (notStupidBool_t*)scratchPad.allocateTempVector("persistency_sorted", total_persistency_bytes);
            //// TODO: But do I have to SortByKey three times?? Can I zip these value arrays together??
            // Although it is stupid, do pay attention to that it does leverage the fact that RadixSort is stable.
            cubDEMSortByKeys<bodyID_t, bodyID_t>(total_idA, idA_sorted, total_idB, idB_sorted, numTotalCnts,
                                                 this_stream, scratchPad);
            cubDEMSortByKeys<bodyID_t, contact_t>(total_idA, idA_sorted, total_types, contactType_sorted, numTotalCnts,
                                                  this_stream, scratchPad);
            cubDEMSortByKeys<bodyID_t, notStupidBool_t>(total_idA, idA_sorted, total_persistency, persistency_sorted,
                                                        numTotalCnts, this_stream, scratchPad);
            // std::cout << "Contacts before duplication check: " << std::endl;
            // displayDeviceArray<bodyID_t>(idA_sorted, numTotalCnts);
            // displayDeviceArray<bodyID_t>(idB_sorted, numTotalCnts);
            // displayDeviceArray<contact_t>(contactType_sorted, numTotalCnts);
            // displayDeviceArray<notStupidBool_t>(persistency_sorted, numTotalCnts);

            scratchPad.finishUsingTempVector("total_idA");
            scratchPad.finishUsingTempVector("total_idB");
            scratchPad.finishUsingTempVector("total_types");
            scratchPad.finishUsingTempVector("total_persistency");
            scratchPad.finishUsingDualStruct("numPersistCnts");

            removeDuplicateContacts(granData, idA_sorted, idB_sorted, contactType_sorted, persistency_sorted,
                                    idGeometryA, idGeometryB, contactType, contactPersistency, true,
                                    DEME_MAX(simParams->nSpheresGM, simParams->nTriGM), numTotalCnts, this_stream,
                                    scratchPad);

            scratchPad.finishUsingTempVector("contactType_sorted");
            scratchPad.finishUsingTempVector("idA_sorted");
            scratchPad.finishUsingTempVector("idB_sorted");
            scratchPad.finishUsingTempVector("persistency_sorted");

            // This step sorts the contact array by idA and stores them in work arrays, which saves effort later
            contactArraysAreSortedByA = true;
        }

        // -----------------------------------------------------------------------------------------------------------
        // One more thing: We need to remove duplicates in the contact list, because tri--tri contacts are not
        // guaranteed to be unique. Of course, if there are persistent contacts, we have done the duplicate removal
        // already.
        // -----------------------------------------------------------------------------------------------------------

        if (solverFlags.meshUniversalContact && !solverFlags.hasPersistentContacts) {
            // To remove duplicates, we sort by idA...
            size_t numTotalCnts = *scratchPad.numContacts;
            size_t total_ids_bytes = numTotalCnts * sizeof(bodyID_t);
            size_t total_types_bytes = numTotalCnts * sizeof(contact_t);
            size_t total_persistency_bytes = numTotalCnts * sizeof(notStupidBool_t);
            contact_t* contactType_sorted =
                (contact_t*)scratchPad.allocateTempVector("contactType_sorted", total_types_bytes);
            bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector("idA_sorted", total_ids_bytes);
            bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector("idB_sorted", total_ids_bytes);
            notStupidBool_t* persistency_sorted =
                (notStupidBool_t*)scratchPad.allocateTempVector("persistency_sorted", total_persistency_bytes);
            //// TODO: But do I have to SortByKey three times?? Can I zip these value arrays together??
            // Although it is stupid, do pay attention to that it does leverage the fact that RadixSort is stable.
            cubDEMSortByKeys<bodyID_t, bodyID_t>(granData->idGeometryA, idA_sorted, granData->idGeometryB, idB_sorted,
                                                 numTotalCnts, this_stream, scratchPad);
            cubDEMSortByKeys<bodyID_t, contact_t>(granData->idGeometryA, idA_sorted, granData->contactType,
                                                  contactType_sorted, numTotalCnts, this_stream, scratchPad);
            cubDEMSortByKeys<bodyID_t, notStupidBool_t>(granData->idGeometryA, idA_sorted, granData->contactPersistency,
                                                        persistency_sorted, numTotalCnts, this_stream, scratchPad);
            // std::cout << "Contacts before duplication check: " << std::endl;
            // displayDeviceArray<bodyID_t>(idA_sorted, numTotalCnts);
            // displayDeviceArray<bodyID_t>(idB_sorted, numTotalCnts);
            // displayDeviceArray<contact_t>(contactType_sorted, numTotalCnts);

            removeDuplicateContacts(granData, idA_sorted, idB_sorted, contactType_sorted, persistency_sorted,
                                    idGeometryA, idGeometryB, contactType, contactPersistency, false,
                                    DEME_MAX(simParams->nSpheresGM, simParams->nTriGM), numTotalCnts, this_stream,
                                    scratchPad);
            // std::cout << "Contacts after duplication check: " << std::endl;
            // displayDeviceArray<bodyID_t>(granData->idGeometryA, *scratchPad.numContacts);
            // displayDeviceArray<bodyID_t>(granData->idGeometryB, *scratchPad.numContacts);
            // displayDeviceArray<contact_t>(granData->contactType, *scratchPad.numContacts);

            scratchPad.finishUsingTempVector("contactType_sorted");
            scratchPad.finishUsingTempVector("idA_sorted");
            scratchPad.finishUsingTempVector("idB_sorted");
            scratchPad.finishUsingTempVector("persistency_sorted");

            // This step sorts the contact array by idA and stores them in work arrays, which saves effort later
            contactArraysAreSortedByA = true;
        }

        timers.GetTimer("Find contact pairs").stop();
        // std::cout << "Contacts: " << std::endl;
        // displayDeviceArray<bodyID_t>(granData->idGeometryA, *scratchPad.numContacts);
        // displayDeviceArray<bodyID_t>(granData->idGeometryB, *scratchPad.numContacts);
        // displayDeviceArray<contact_t>(granData->contactType, *scratchPad.numContacts);

    }  // End of contact pairs construction of this CD step

    // -----------------------------------------------------------------------------------------------------------
    // Constructing contact history
    // -----------------------------------------------------------------------------------------------------------

    timers.GetTimer("Build history map").start();
    // Now, sort idGeometryAB by their owners. Needed for identifying enduring contacts in history-based models.
    if (*scratchPad.numContacts > 0) {
        size_t type_arr_bytes = (*scratchPad.numContacts) * sizeof(contact_t);
        size_t id_arr_bytes = (*scratchPad.numContacts) * sizeof(bodyID_t);
        // There could be a few cases where arrays are sorted by A: we did duplicate removal; we had persistent contacts
        // etc. In those cases, no need to sort again.
        if (!contactArraysAreSortedByA) {
            contact_t* contactType_sorted =
                (contact_t*)scratchPad.allocateTempVector("contactType_sorted", type_arr_bytes);
            bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector("idA_sorted", id_arr_bytes);
            bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector("idB_sorted", id_arr_bytes);

            //// TODO: But do I have to SortByKey two times?? Can I zip these value arrays together??
            // Although it is stupid, do pay attention to that it does leverage the fact that RadixSort is stable.
            cubDEMSortByKeys<bodyID_t, bodyID_t>(granData->idGeometryA, idA_sorted, granData->idGeometryB, idB_sorted,
                                                 *scratchPad.numContacts, this_stream, scratchPad);
            cubDEMSortByKeys<bodyID_t, contact_t>(granData->idGeometryA, idA_sorted, granData->contactType,
                                                  contactType_sorted, *scratchPad.numContacts, this_stream, scratchPad);

            // Copy back to idGeometry arrays
            DEME_GPU_CALL(cudaMemcpy(granData->idGeometryA, idA_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(granData->idGeometryB, idB_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(
                cudaMemcpy(granData->contactType, contactType_sorted, type_arr_bytes, cudaMemcpyDeviceToDevice));
            // Short-lifespan vectors can now be freed
            scratchPad.finishUsingTempVector("contactType_sorted");
            scratchPad.finishUsingTempVector("idA_sorted");
            scratchPad.finishUsingTempVector("idB_sorted");
        }
        // DEME_DEBUG_PRINTF("New contact IDs (A):");
        // DEME_DEBUG_EXEC(displayDeviceArray<bodyID_t>(granData->idGeometryA, *scratchPad.numContacts));
        // DEME_DEBUG_PRINTF("New contact IDs (B):");
        // DEME_DEBUG_EXEC(displayDeviceArray<bodyID_t>(granData->idGeometryB, *scratchPad.numContacts));
        // DEME_DEBUG_PRINTF("New contact types:");
        // DEME_DEBUG_EXEC(displayDeviceArray<contact_t>(granData->contactType, *scratchPad.numContacts));
        // if (!solverFlag.isHistoryless) {
        //     DEME_DEBUG_PRINTF("Old contact IDs (A):");
        //     DEME_DEBUG_EXEC(displayDeviceArray<bodyID_t>(granData->previous_idGeometryA,
        //     *scratchPad.numPrevContacts)); DEME_DEBUG_PRINTF("Old contact IDs (B):");
        //     DEME_DEBUG_EXEC(displayDeviceArray<bodyID_t>(granData->previous_idGeometryB,
        //     *scratchPad.numPrevContacts)); DEME_DEBUG_PRINTF("Old contact types:");
        //     DEME_DEBUG_EXEC(displayDeviceArray<contact_t>(granData->previous_contactType,
        //     *scratchPad.numPrevContacts));
        // }

        // For history-based models, construct the enduring contact map. bodyA can be sphere or triangle.
        // This CD run and previous CD run could have different number of geos in them. We pick the larger
        // number to refer in building the enduring contact map to avoid potential problems.
        size_t nGeoSafe = DEME_MAX(DEME_MAX(simParams->nSpheresGM, *scratchPad.numPrevSpheres),
                                   DEME_MAX(simParams->nTriGM, *scratchPad.numPrevTriangles));
        // (simParams->nSpheresGM > *scratchPad.numPrevSpheres) ? simParams->nSpheresGM : *scratchPad.numPrevSpheres;

        // First, identify the new and old idA run-length
        size_t run_length_bytes = nGeoSafe * sizeof(geoSphereTouches_t);
        geoSphereTouches_t* new_idA_runlength =
            (geoSphereTouches_t*)scratchPad.allocateTempVector("new_idA_runlength", run_length_bytes);
        size_t unique_id_bytes = nGeoSafe * sizeof(bodyID_t);
        bodyID_t* unique_new_idA = (bodyID_t*)scratchPad.allocateTempVector("unique_new_idA", unique_id_bytes);
        scratchPad.allocateDualStruct("numUniqueNewA");
        cubDEMRunLengthEncode<bodyID_t, geoSphereTouches_t>(granData->idGeometryA, unique_new_idA, new_idA_runlength,
                                                            scratchPad.getDualStructDevice("numUniqueNewA"),
                                                            *scratchPad.numContacts, this_stream, scratchPad);
        scratchPad.syncDualStructDeviceToHost("numUniqueNewA");
        size_t* pNumUniqueNewA = scratchPad.getDualStructHost("numUniqueNewA");
        // Now, we do a tab-keeping job: how many contacts on average a sphere has?
        {
            // Figure out how many contacts an item in idA array typically has.
            stateParams.avgCntsPerSphere =
                (*pNumUniqueNewA > 0) ? (float)(*scratchPad.numContacts) / (float)(*pNumUniqueNewA) : 0.0;

            DEME_DEBUG_PRINTF("Average number of contacts for each geometry: %.7g", stateParams.avgCntsPerSphere);
            if (stateParams.avgCntsPerSphere > solverFlags.errOutAvgSphCnts) {
                DEME_ERROR(
                    "On average a sphere has %.7g contacts, more than the max allowance (%.7g).\nIf you believe "
                    "this is not abnormal, set the allowance high using SetErrorOutAvgContacts before "
                    "initialization.\nIf you think this is because dT drifting too much ahead of kT so the contact "
                    "margin added is too big, use SetCDMaxUpdateFreq to limit the max dT future drift.\nOtherwise, the "
                    "simulation may have diverged and relaxing the physics may help, such as decreasing the step size "
                    "and modifying material properties.\nIf this happens at the start of simulation, check if there "
                    "are initial penetrations, a.k.a. elements initialized inside walls.",
                    stateParams.avgCntsPerSphere, solverFlags.errOutAvgSphCnts);
            }
        }

        // Only need to proceed if history-based
        if (!solverFlags.isHistoryless) {
            geoSphereTouches_t* old_idA_runlength =
                (geoSphereTouches_t*)scratchPad.allocateTempVector("old_idA_runlength", run_length_bytes);
            bodyID_t* unique_old_idA = (bodyID_t*)scratchPad.allocateTempVector("unique_old_idA", unique_id_bytes);
            scratchPad.allocateDualStruct("numUniqueOldA");
            cubDEMRunLengthEncode<bodyID_t, geoSphereTouches_t>(granData->previous_idGeometryA, unique_old_idA,
                                                                old_idA_runlength,
                                                                scratchPad.getDualStructDevice("numUniqueOldA"),
                                                                *(scratchPad.numPrevContacts), this_stream, scratchPad);
            scratchPad.syncDualStructDeviceToHost("numUniqueOldA");
            size_t* pNumUniqueOldA = scratchPad.getDualStructHost("numUniqueOldA");
            // Then, add zeros to run-length arrays such that even if a sphereID is not present in idA, it has a
            // place in the run-length arrays that indicates 0 run-length
            geoSphereTouches_t* new_idA_runlength_full =
                (geoSphereTouches_t*)scratchPad.allocateTempVector("new_idA_runlength_full", run_length_bytes);
            geoSphereTouches_t* old_idA_runlength_full =
                (geoSphereTouches_t*)scratchPad.allocateTempVector("old_idA_runlength_full", run_length_bytes);
            DEME_GPU_CALL(cudaMemset((void*)new_idA_runlength_full, 0, run_length_bytes));
            DEME_GPU_CALL(cudaMemset((void*)old_idA_runlength_full, 0, run_length_bytes));
            size_t blocks_needed_for_mapping =
                (*pNumUniqueNewA + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
            if (blocks_needed_for_mapping > 0) {
                fillRunLengthArray<<<dim3(blocks_needed_for_mapping), dim3(DEME_MAX_THREADS_PER_BLOCK), 0,
                                     this_stream>>>(new_idA_runlength_full, unique_new_idA, new_idA_runlength,
                                                    *pNumUniqueNewA);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            }

            blocks_needed_for_mapping = (*pNumUniqueOldA + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
            if (blocks_needed_for_mapping > 0) {
                fillRunLengthArray<<<dim3(blocks_needed_for_mapping), dim3(DEME_MAX_THREADS_PER_BLOCK), 0,
                                     this_stream>>>(old_idA_runlength_full, unique_old_idA, old_idA_runlength,
                                                    *pNumUniqueOldA);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            }
            // DEME_DEBUG_PRINTF("Unique contact IDs (A):");
            // DEME_DEBUG_EXEC(displayDeviceArray<bodyID_t>(unique_new_idA, *pNumUniqueNewA));
            // DEME_DEBUG_PRINTF("Unique contacts run-length:");
            // DEME_DEBUG_EXEC(displayDeviceArray<geoSphereTouches_t>(new_idA_runlength, *pNumUniqueNewA));
            scratchPad.finishUsingTempVector("old_idA_runlength");
            scratchPad.finishUsingTempVector("unique_old_idA");
            scratchPad.finishUsingDualStruct("numUniqueOldA");

            // Then, prescan to find run-length offsets, in preparation for custom kernels
            size_t scanned_runlength_bytes = nGeoSafe * sizeof(contactPairs_t);
            contactPairs_t* new_idA_scanned_runlength =
                (contactPairs_t*)scratchPad.allocateTempVector("new_idA_scanned_runlength", scanned_runlength_bytes);
            contactPairs_t* old_idA_scanned_runlength =
                (contactPairs_t*)scratchPad.allocateTempVector("old_idA_scanned_runlength", scanned_runlength_bytes);
            cubDEMPrefixScan<geoSphereTouches_t, contactPairs_t>(new_idA_runlength_full, new_idA_scanned_runlength,
                                                                 nGeoSafe, this_stream, scratchPad);
            cubDEMPrefixScan<geoSphereTouches_t, contactPairs_t>(old_idA_runlength_full, old_idA_scanned_runlength,
                                                                 nGeoSafe, this_stream, scratchPad);

            // Then, each thread will scan a sphere, if this sphere has non-zero run-length in both new and old idA,
            // manually store the mapping. This mapping's elemental values are the indices of the corresponding
            // contacts in the previous contact array.
            if (*scratchPad.numContacts > contactMapping.size()) {
                DEME_DUAL_ARRAY_RESIZE_NOVAL(contactMapping, *scratchPad.numContacts);
                granData.toDevice();
            }
            blocks_needed_for_mapping = (nGeoSafe + DEME_NUM_BODIES_PER_BLOCK - 1) / DEME_NUM_BODIES_PER_BLOCK;
            if (blocks_needed_for_mapping > 0) {
                buildPersistentMap<<<dim3(blocks_needed_for_mapping), dim3(DEME_NUM_BODIES_PER_BLOCK), 0,
                                     this_stream>>>(new_idA_runlength_full, old_idA_runlength_full,
                                                    new_idA_scanned_runlength, old_idA_scanned_runlength,
                                                    granData->contactMapping, &granData, nGeoSafe);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            }
            // DEME_DEBUG_PRINTF("Contact mapping:");
            // DEME_DEBUG_EXEC(displayDeviceArray<contactPairs_t>(granData->contactMapping, *scratchPad.numContacts));
            scratchPad.finishUsingTempVector("new_idA_runlength_full");
            scratchPad.finishUsingTempVector("old_idA_runlength_full");
            scratchPad.finishUsingTempVector("new_idA_scanned_runlength");
            scratchPad.finishUsingTempVector("old_idA_scanned_runlength");

            // One thing we need to do before storing the old contact pairs: figure out how it is mapped to the actually
            // shipped contact pair array.
            contactPairs_t* old_arr_unsort_to_sort_map;
            // if (solverFlags.should_sort_pairs) {
            {
                size_t map_arr_bytes = (*scratchPad.numPrevContacts) * sizeof(contactPairs_t);
                old_arr_unsort_to_sort_map =
                    (contactPairs_t*)scratchPad.allocateTempVector("old_arr_unsort_to_sort_map", map_arr_bytes);
                contactPairs_t* one_to_n = (contactPairs_t*)scratchPad.allocateTempVector("one_to_n", map_arr_bytes);
                size_t blocks_needed_for_mapping =
                    (*scratchPad.numPrevContacts + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
                if (blocks_needed_for_mapping > 0) {
                    lineNumbers<<<dim3(blocks_needed_for_mapping), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, this_stream>>>(
                        one_to_n, *scratchPad.numPrevContacts);
                    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));

                    contact_t* old_contactType_sorted = (contact_t*)scratchPad.allocateTempVector(
                        "old_contactType_sorted", (*scratchPad.numPrevContacts) * sizeof(contact_t));
                    // Sorted by type is how we shipped the old contact pair info
                    cubDEMSortByKeys<contact_t, contactPairs_t>(granData->previous_contactType, old_contactType_sorted,
                                                                one_to_n, old_arr_unsort_to_sort_map,
                                                                *scratchPad.numPrevContacts, this_stream, scratchPad);
                    // Now, we have `map from' info. But we need `map to' info.
                    convertToAndFrom<<<dim3(blocks_needed_for_mapping), dim3(DEME_MAX_THREADS_PER_BLOCK), 0,
                                       this_stream>>>(old_arr_unsort_to_sort_map, one_to_n,
                                                      *scratchPad.numPrevContacts);
                    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
                }
                // one_to_n used for temp storage; now give it back to the true mapping we wanted.
                // So here, old_arr_unsort_to_sort_map's memory space is not needed anymore, but one_to_n must still
                // live, a little nuance to pay attention to. However, alas, we can always just delay memory freeing
                // and do it all at the very end: this is exactly what I did here.
                old_arr_unsort_to_sort_map = one_to_n;
                scratchPad.finishUsingTempVector("old_contactType_sorted");
            }

            // Finally, copy new contact array to old contact array for the record. Note we register old contact pairs
            // with the array sorted by A, but when supplying dT, it was sorted by contact type.
            if (*scratchPad.numContacts > previous_idGeometryA.size()) {
                // Note these resizing are automatically on kT's device
                DEME_DUAL_ARRAY_RESIZE_NOVAL(previous_idGeometryA, *scratchPad.numContacts);
                DEME_DUAL_ARRAY_RESIZE_NOVAL(previous_idGeometryB, *scratchPad.numContacts);
                DEME_DUAL_ARRAY_RESIZE_NOVAL(previous_contactType, *scratchPad.numContacts);

                // Re-packing pointers now is automatic
                granData.toDevice();
            }
            DEME_GPU_CALL(cudaMemcpy(granData->previous_idGeometryA, granData->idGeometryA, id_arr_bytes,
                                     cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(granData->previous_idGeometryB, granData->idGeometryB, id_arr_bytes,
                                     cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(granData->previous_contactType, granData->contactType, type_arr_bytes,
                                     cudaMemcpyDeviceToDevice));

            // dT benefits from type-sorted contact array
            // if (solverFlags.should_sort_pairs) {
            {
                size_t type_arr_bytes = (*scratchPad.numContacts) * sizeof(contact_t);
                contact_t* contactType_sorted =
                    (contact_t*)scratchPad.allocateTempVector("contactType_sorted", type_arr_bytes);
                size_t id_arr_bytes = (*scratchPad.numContacts) * sizeof(bodyID_t);
                bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector("idA_sorted", id_arr_bytes);
                bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector("idB_sorted", id_arr_bytes);
                size_t cnt_arr_bytes = (*scratchPad.numContacts) * sizeof(contactPairs_t);
                contactPairs_t* map_sorted =
                    (contactPairs_t*)scratchPad.allocateTempVector("map_sorted", cnt_arr_bytes);

                //// TODO: But do I have to SortByKey three times?? Can I zip these value arrays together??
                cubDEMSortByKeys<contact_t, bodyID_t>(granData->contactType, contactType_sorted, granData->idGeometryB,
                                                      idB_sorted, *scratchPad.numContacts, this_stream, scratchPad);
                cubDEMSortByKeys<contact_t, bodyID_t>(granData->contactType, contactType_sorted, granData->idGeometryA,
                                                      idA_sorted, *scratchPad.numContacts, this_stream, scratchPad);
                cubDEMSortByKeys<contact_t, contactPairs_t>(granData->contactType, contactType_sorted,
                                                            granData->contactMapping, map_sorted,
                                                            *scratchPad.numContacts, this_stream, scratchPad);

                // Finally, map the mapping array so it takes into account that arrays are shipped after sorting.
                size_t blocks_needed_for_mapping =
                    (*scratchPad.numContacts + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
                if (blocks_needed_for_mapping > 0) {
                    rearrangeMapping<<<dim3(blocks_needed_for_mapping), dim3(DEME_MAX_THREADS_PER_BLOCK), 0,
                                       this_stream>>>(map_sorted, old_arr_unsort_to_sort_map, *scratchPad.numContacts);
                    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
                }

                // Copy back to idGeometry arrays
                DEME_GPU_CALL(cudaMemcpy(granData->idGeometryA, idA_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
                DEME_GPU_CALL(cudaMemcpy(granData->idGeometryB, idB_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
                DEME_GPU_CALL(
                    cudaMemcpy(granData->contactType, contactType_sorted, type_arr_bytes, cudaMemcpyDeviceToDevice));
                DEME_GPU_CALL(
                    cudaMemcpy(granData->contactMapping, map_sorted, cnt_arr_bytes, cudaMemcpyDeviceToDevice));

                scratchPad.finishUsingTempVector("contactType_sorted");
                scratchPad.finishUsingTempVector("idA_sorted");
                scratchPad.finishUsingTempVector("idB_sorted");
                scratchPad.finishUsingTempVector("map_sorted");
            }
        } else {  // If historyless, we still want to sort based on type
                  // if (solverFlags.should_sort_pairs) {
            size_t type_arr_bytes = (*scratchPad.numContacts) * sizeof(contact_t);
            contact_t* contactType_sorted =
                (contact_t*)scratchPad.allocateTempVector("contactType_sorted", type_arr_bytes);
            size_t id_arr_bytes = (*scratchPad.numContacts) * sizeof(bodyID_t);
            bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector("idA_sorted", id_arr_bytes);
            bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector("idB_sorted", id_arr_bytes);

            cubDEMSortByKeys<contact_t, bodyID_t>(granData->contactType, contactType_sorted, granData->idGeometryB,
                                                  idB_sorted, *scratchPad.numContacts, this_stream, scratchPad);
            cubDEMSortByKeys<contact_t, bodyID_t>(granData->contactType, contactType_sorted, granData->idGeometryA,
                                                  idA_sorted, *scratchPad.numContacts, this_stream, scratchPad);

            // Copy back to idGeometry arrays
            DEME_GPU_CALL(cudaMemcpy(granData->idGeometryA, idA_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(granData->idGeometryB, idB_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(
                cudaMemcpy(granData->contactType, contactType_sorted, type_arr_bytes, cudaMemcpyDeviceToDevice));
            scratchPad.finishUsingTempVector("contactType_sorted");
            scratchPad.finishUsingTempVector("idA_sorted");
            scratchPad.finishUsingTempVector("idB_sorted");
        }
        // This part is light on memory, so we can delay some freeing
        scratchPad.finishUsingTempVector("new_idA_runlength");
        scratchPad.finishUsingTempVector("unique_new_idA");
        scratchPad.finishUsingDualStruct("numUniqueNewA");
        // Note one_to_n corresponds to the pointer name old_arr_unsort_to_sort_map, an exception to general "matched"
        // naming convension. But we delayed their freeing, so you see them both freed here.
        scratchPad.finishUsingTempVector("one_to_n");
        scratchPad.finishUsingTempVector("old_arr_unsort_to_sort_map");

    }  // End of contact sorting--mapping subroutine
    timers.GetTimer("Build history map").stop();

    // Finally, don't forget to store the number of contacts for the next iteration, even if there is 0 contacts (in
    // that case, mapping will not be constructed, but we don't have to worry b/c in the next iteration, simply no work
    // will be done for the old array and every contact will be new)
    *scratchPad.numPrevContacts = *scratchPad.numContacts;
    *scratchPad.numPrevSpheres = simParams->nSpheresGM;
    *scratchPad.numPrevTriangles = simParams->nTriGM;

    // dT kT may send these numbers to each other from device
    scratchPad.numContacts.toDevice();
    scratchPad.numPrevContacts.toDevice();
    scratchPad.numPrevSpheres.toDevice();
    scratchPad.numPrevTriangles.toDevice();

    // -----------------------------------------------------------------------------------------------------------
    // At the very end, we prepare the auxiliary mesh-involved contact patch ID arrays.
    // This auxiliary array does not need to be carried over between CD steps.
    // -----------------------------------------------------------------------------------------------------------
    if (*scratchPad.numContacts > 0) {
        // Sort contactPatchPairs within each contact type segment
        // This ensures that when dT does reduce-by-key on contactPatchPairs, the keys are sorted

        // First, identify the contact type segments using run-length encoding
        // Maximum number of contact types (5 main types: sph-sph, sph-tri, sph-anal, tri-tri, tri-anal
        contact_t* unique_types =
            (contact_t*)scratchPad.allocateTempVector("unique_types", NUM_SUPPORTED_CONTACT_TYPES * sizeof(contact_t));
        size_t* type_counts =
            (size_t*)scratchPad.allocateTempVector("type_counts", NUM_SUPPORTED_CONTACT_TYPES * sizeof(size_t));
        scratchPad.allocateDualStruct("numUniqueTypes");

        cubDEMRunLengthEncode<contact_t, size_t>(granData->contactType, unique_types, type_counts,
                                                 scratchPad.getDualStructDevice("numUniqueTypes"),
                                                 *scratchPad.numContacts, this_stream, scratchPad);
        scratchPad.syncDualStructDeviceToHost("numUniqueTypes");
        size_t numTypes = *scratchPad.getDualStructHost("numUniqueTypes");

        // Now sort within each type segment
        if (numTypes > 0) {
            // Copy type_counts to host to know segment boundaries
            size_t* host_type_counts = new size_t[numTypes];
            DEME_GPU_CALL(cudaMemcpy(host_type_counts, type_counts, numTypes * sizeof(size_t), cudaMemcpyDeviceToHost));

            // Allocate temp arrays for sorting
            size_t patch_arr_bytes = (*scratchPad.numContacts) * sizeof(patchIDPair_t);
            patchIDPair_t* patchPairs_sorted =
                (patchIDPair_t*)scratchPad.allocateTempVector("patchPairs_sorted", patch_arr_bytes);
            size_t id_arr_bytes = (*scratchPad.numContacts) * sizeof(bodyID_t);
            bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector("idA_sorted", id_arr_bytes);
            bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector("idB_sorted", id_arr_bytes);

            // Allocate temp array to preserve original keys for each sort
            patchIDPair_t* patchPairs_keys =
                (patchIDPair_t*)scratchPad.allocateTempVector("patchPairs_keys", patch_arr_bytes);

            // Sort each segment
            size_t offset = 0;
            for (size_t i = 0; i < numTypes; i++) {
                size_t count = host_type_counts[i];
                if (count > 1) {  // Only sort if segment has more than 1 element
                    size_t segment_bytes = count * sizeof(patchIDPair_t);

                    // First sort: get sorted keys and sort idGeometryA
                    DEME_GPU_CALL(cudaMemcpy(patchPairs_keys + offset, granData->contactPatchPairs + offset,
                                             segment_bytes, cudaMemcpyDeviceToDevice));
                    cubDEMSortByKeys<patchIDPair_t, bodyID_t>(patchPairs_keys + offset, patchPairs_sorted + offset,
                                                              granData->idGeometryA + offset, idA_sorted + offset,
                                                              count, this_stream, scratchPad);

                    // Second sort: sort idGeometryB using original keys
                    // Note: This will overwrite patchPairs_sorted but produce the same sorted key order
                    DEME_GPU_CALL(cudaMemcpy(patchPairs_keys + offset, granData->contactPatchPairs + offset,
                                             segment_bytes, cudaMemcpyDeviceToDevice));
                    cubDEMSortByKeys<patchIDPair_t, bodyID_t>(patchPairs_keys + offset, patchPairs_sorted + offset,
                                                              granData->idGeometryB + offset, idB_sorted + offset,
                                                              count, this_stream, scratchPad);
                    // patchPairs_sorted now contains the final sorted keys (same as from first sort due to stable sort)
                } else if (count == 1) {
                    // Just copy single elements (no sorting needed)
                    DEME_GPU_CALL(cudaMemcpy(patchPairs_sorted + offset, granData->contactPatchPairs + offset,
                                             sizeof(patchIDPair_t), cudaMemcpyDeviceToDevice));
                    DEME_GPU_CALL(cudaMemcpy(idA_sorted + offset, granData->idGeometryA + offset, sizeof(bodyID_t),
                                             cudaMemcpyDeviceToDevice));
                    DEME_GPU_CALL(cudaMemcpy(idB_sorted + offset, granData->idGeometryB + offset, sizeof(bodyID_t),
                                             cudaMemcpyDeviceToDevice));
                }
                offset += count;
            }

            scratchPad.finishUsingTempVector("patchPairs_keys");

            // Copy sorted arrays back
            DEME_GPU_CALL(
                cudaMemcpy(granData->contactPatchPairs, patchPairs_sorted, patch_arr_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(granData->idGeometryA, idA_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(granData->idGeometryB, idB_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));

            delete[] host_type_counts;

            scratchPad.finishUsingTempVector("patchPairs_sorted");
            scratchPad.finishUsingTempVector("idA_sorted");
            scratchPad.finishUsingTempVector("idB_sorted");
        }

        scratchPad.finishUsingTempVector("unique_types");
        scratchPad.finishUsingTempVector("type_counts");
        scratchPad.finishUsingDualStruct("numUniqueTypes");

        // Now, generate the contact patch ID pairs for each contact pair
        if ((*scratchPad.numContacts) > contactPatchPairs.size()) {
            meshPatchPairsResize(*scratchPad.numContacts, contactPatchPairs, granData);
        }

        // Based on the ready-to-ship (this CD iteration) contact arrays...
        size_t blocks_needed_for_patch_ids =
            (*scratchPad.numContacts + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
        if (blocks_needed_for_patch_ids > 0) {
            extractMeshInvolvedContactPatchIDPairs<<<dim3(blocks_needed_for_patch_ids),
                                                     dim3(DEME_MAX_THREADS_PER_BLOCK), 0, this_stream>>>(
                granData->contactPatchPairs,
                /*isMeshInvolvedContact,*/
                granData->contactType, granData->idGeometryA, granData->idGeometryB, granData->triPatchID,
                *scratchPad.numContacts);
            DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
        }
    }
    // std::cout << "contactPatchPairs: " << std::endl;
    // displayDeviceArray<patchIDPair_t>(granData->contactPatchPairs, *scratchPad.numContacts);
}

void overwritePrevContactArrays(DualStruct<DEMDataKT>& kT_data,
                                DualStruct<DEMDataDT>& dT_data,
                                DualArray<bodyID_t>& previous_idGeometryA,
                                DualArray<bodyID_t>& previous_idGeometryB,
                                DualArray<contact_t>& previous_contactType,
                                DualStruct<DEMSimParams>& simParams,
                                DualArray<notStupidBool_t>& contactPersistency,
                                DEMSolverScratchData& scratchPad,
                                cudaStream_t& this_stream,
                                size_t nContacts) {
    // Make sure the storage is large enough
    if (nContacts > previous_idGeometryA.size()) {
        contactEventArraysResize(nContacts, previous_idGeometryA, previous_idGeometryB, previous_contactType,
                                 contactPersistency, kT_data);
    }

    // Copy to temp array for easier usage
    bodyID_t* idA = (bodyID_t*)scratchPad.allocateTempVector("idA", nContacts * sizeof(bodyID_t));
    bodyID_t* idB = (bodyID_t*)scratchPad.allocateTempVector("idB", nContacts * sizeof(bodyID_t));
    contact_t* cType = (contact_t*)scratchPad.allocateTempVector("cType", nContacts * sizeof(contact_t));
    DEME_GPU_CALL(cudaMemcpy(idA, dT_data->idGeometryA, nContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(idB, dT_data->idGeometryB, nContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(cType, dT_data->contactType, nContacts * sizeof(contact_t), cudaMemcpyDeviceToDevice));

    // Prev contact arrays actually need to be sorted based on idA
    // bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector("idA_sorted", nContacts * sizeof(bodyID_t));
    // bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector("idB_sorted", nContacts * sizeof(bodyID_t));
    // contact_t* cType_sorted = (contact_t*)scratchPad.allocateTempVector("cType_sorted", nContacts *
    // sizeof(contact_t)); DEME_GPU_CALL(cudaMemcpy(idA_sorted, idA, nContacts * sizeof(bodyID_t),
    // cudaMemcpyDeviceToDevice)); DEME_GPU_CALL(cudaMemcpy(idB_sorted, idB, nContacts * sizeof(bodyID_t),
    // cudaMemcpyDeviceToDevice)); DEME_GPU_CALL(cudaMemcpy(cType_sorted, cType, nContacts * sizeof(contact_t),
    // cudaMemcpyDeviceToDevice)); hostSortByKey(idA, idB_sorted, nContacts); hostSortByKey(idA_sorted, cType_sorted,
    // nContacts); DEME_GPU_CALL(
    //     cudaMemcpy(kT_data->previous_idGeometryA, idA_sorted, nContacts * sizeof(bodyID_t),
    //     cudaMemcpyDeviceToDevice));
    // DEME_GPU_CALL(
    //     cudaMemcpy(kT_data->previous_idGeometryB, idB_sorted, nContacts * sizeof(bodyID_t),
    //     cudaMemcpyDeviceToDevice));
    // DEME_GPU_CALL(cudaMemcpy(kT_data->previous_contactType, cType_sorted, nContacts * sizeof(contact_t),
    //                          cudaMemcpyDeviceToDevice));
    cubDEMSortByKeys<bodyID_t, bodyID_t>(idA, kT_data->previous_idGeometryA, idB, kT_data->previous_idGeometryB,
                                         nContacts, this_stream, scratchPad);
    cubDEMSortByKeys<bodyID_t, contact_t>(idA, kT_data->previous_idGeometryA, cType, kT_data->previous_contactType,
                                          nContacts, this_stream, scratchPad);

    *scratchPad.numPrevContacts = nContacts;
    // If nSpheresGM is updated, then it should have been taken care of in the init/populate array phase and in kT's
    // simParams now
    *scratchPad.numPrevSpheres = simParams->nSpheresGM;
    *scratchPad.numPrevTriangles = simParams->nTriGM;
    // dT kT may send these numbers to each other from device
    scratchPad.numPrevContacts.toDevice();
    scratchPad.numPrevSpheres.toDevice();
    scratchPad.numPrevTriangles.toDevice();

    scratchPad.finishUsingTempVector("idA");
    scratchPad.finishUsingTempVector("idB");
    scratchPad.finishUsingTempVector("cType");
    // scratchPad.finishUsingTempVector("idA_sorted");
    // scratchPad.finishUsingTempVector("idB_sorted");
    // scratchPad.finishUsingTempVector("cType_sorted");
}

}  // namespace deme
