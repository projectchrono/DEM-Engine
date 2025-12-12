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

namespace deme {

// Array of all supported contact types, used for iterating during mapping construction
// Note: If you add a new contact type, you must update this array and NUM_SUPPORTED_CONTACT_TYPES
static const contact_t ALL_CONTACT_TYPES[NUM_SUPPORTED_CONTACT_TYPES] = {
    SPHERE_SPHERE_CONTACT, SPHERE_TRIANGLE_CONTACT, SPHERE_ANALYTICAL_CONTACT, TRIANGLE_TRIANGLE_CONTACT,
    TRIANGLE_ANALYTICAL_CONTACT};
static_assert(NUM_SUPPORTED_CONTACT_TYPES == 5, "ALL_CONTACT_TYPES array size must match NUM_SUPPORTED_CONTACT_TYPES");

inline void primitiveContactArraysResize(size_t nContactPairs,
                                         DualArray<bodyID_t>& idPrimitiveA,
                                         DualArray<bodyID_t>& idPrimitiveB,
                                         DualArray<contact_t>& contactTypePrimitive,
                                         DualArray<notStupidBool_t>& contactPersistency,
                                         DualStruct<DEMDataKT>& granData) {
    // Note these resizing are automatically on kT's device
    DEME_DUAL_ARRAY_RESIZE_NOVAL(idPrimitiveA, nContactPairs);
    DEME_DUAL_ARRAY_RESIZE_NOVAL(idPrimitiveB, nContactPairs);
    DEME_DUAL_ARRAY_RESIZE_NOVAL(contactTypePrimitive, nContactPairs);

    // In the case of user-loaded contacts, if the persistency array is not long enough then we have to manually
    // extend it.
    DEME_DUAL_ARRAY_RESIZE(contactPersistency, nContactPairs, CONTACT_NOT_PERSISTENT);

    // Re-packing pointers now is automatic

    // It's safe to toDevice even though kT is working now and dT may write to its buffer
    // This is because all buffer arrays are not used in kernels so their pointers are only meaningfully stored on host,
    // so writing from host to device won't change the destination where dT writes
    granData.toDevice();
}

inline void patchArraysResize(size_t nPatchInvolvedContacts,
                              DualArray<bodyID_t>& idA,
                              DualArray<bodyID_t>& idB,
                              DualArray<contact_t>& contactTypePatch,
                              DualStruct<DEMDataKT>& granData) {
    // Note these resizing are automatically on kT's device
    DEME_DUAL_ARRAY_RESIZE_NOVAL(idA, nPatchInvolvedContacts);
    DEME_DUAL_ARRAY_RESIZE_NOVAL(idB, nPatchInvolvedContacts);
    DEME_DUAL_ARRAY_RESIZE_NOVAL(contactTypePatch, nPatchInvolvedContacts);

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
                                    DualArray<bodyID_t>& idPrimitiveA,
                                    DualArray<bodyID_t>& idPrimitiveB,
                                    DualArray<contact_t>& contactTypePrimitive,
                                    DualArray<notStupidBool_t>& contactPersistency,
                                    bool process_persistency,
                                    bodyID_t safe_entity_count,
                                    size_t numTotalCnts,
                                    cudaStream_t& this_stream,
                                    DEMSolverScratchData& scratchPad) {
    // First we run-length it based on idA
    size_t run_length_bytes = safe_entity_count * sizeof(primitivesPrimTouches_t);
    primitivesPrimTouches_t* idA_runlength =
        (primitivesPrimTouches_t*)scratchPad.allocateTempVector("idA_runlength", run_length_bytes);
    size_t unique_id_bytes = safe_entity_count * sizeof(bodyID_t);
    bodyID_t* unique_idA = (bodyID_t*)scratchPad.allocateTempVector("unique_idA", unique_id_bytes);
    scratchPad.allocateDualStruct("numUniqueA");
    cubDEMRunLengthEncode<bodyID_t, primitivesPrimTouches_t>(idA_sorted, unique_idA, idA_runlength,
                                                             scratchPad.getDualStructDevice("numUniqueA"), numTotalCnts,
                                                             this_stream, scratchPad);
    scratchPad.syncDualStructDeviceToHost("numUniqueA");
    size_t* pNumUniqueA = scratchPad.getDualStructHost("numUniqueA");
    size_t scanned_runlength_bytes = (*pNumUniqueA) * sizeof(contactPairs_t);
    contactPairs_t* idA_scanned_runlength =
        (contactPairs_t*)scratchPad.allocateTempVector("idA_scanned_runlength", scanned_runlength_bytes);
    cubDEMPrefixScan<primitivesPrimTouches_t, contactPairs_t>(idA_runlength, idA_scanned_runlength, *pNumUniqueA,
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
    if (*pNumRetainedCnts > idPrimitiveA.size()) {
        primitiveContactArraysResize(*pNumRetainedCnts, idPrimitiveA, idPrimitiveB, contactTypePrimitive,
                                     contactPersistency, granData);
    }
    // Then select those needed contacts
    cubDEMSelectFlagged<bodyID_t, notStupidBool_t>(idA_sorted, granData->idPrimitiveA, retain_flags,
                                                   scratchPad.getDualStructDevice("numRetainedCnts"), numTotalCnts,
                                                   this_stream, scratchPad);
    cubDEMSelectFlagged<bodyID_t, notStupidBool_t>(idB_sorted, granData->idPrimitiveB, retain_flags,
                                                   scratchPad.getDualStructDevice("numRetainedCnts"), numTotalCnts,
                                                   this_stream, scratchPad);
    cubDEMSelectFlagged<contact_t, notStupidBool_t>(contactType_sorted, granData->contactTypePrimitive, retain_flags,
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
    // displayDeviceArray<bodyID_t>(granData->idPrimitiveA, *pNumRetainedCnts);
    // displayDeviceArray<bodyID_t>(granData->idPrimitiveB, *pNumRetainedCnts);
    // displayDeviceArray<contact_t>(granData->contactTypePrimitive, *pNumRetainedCnts);
    // displayDeviceArray<notStupidBool_t>(granData->contactPersistency, *pNumRetainedCnts);

    // And update the number of contacts.
    *scratchPad.numPrimitiveContacts = *pNumRetainedCnts;
    scratchPad.finishUsingDualStruct("numRetainedCnts");

    // Unclaim all temp vectors
    scratchPad.finishUsingTempVector("idA_runlength");
    scratchPad.finishUsingTempVector("unique_idA");
    scratchPad.finishUsingTempVector("idA_scanned_runlength");
    scratchPad.finishUsingTempVector("retain_flags");
}

inline void sortABTypePersistencyByA(bodyID_t* idA,
                                     bodyID_t* idB,
                                     contact_t* types,
                                     notStupidBool_t* persistency,
                                     bodyID_t* idA_sorted,
                                     bodyID_t* idB_sorted,
                                     contact_t* type_sorted,
                                     notStupidBool_t* persistency_sorted,
                                     size_t numCnts,
                                     cudaStream_t& stream,
                                     DEMSolverScratchData& scratchPad) {
    cubDEMSortByKeys<bodyID_t, bodyID_t>(idA, idA_sorted, idB, idB_sorted, numCnts, stream, scratchPad);
    cubDEMSortByKeys<bodyID_t, contact_t>(idA, idA_sorted, types, type_sorted, numCnts, stream, scratchPad);
    cubDEMSortByKeys<bodyID_t, notStupidBool_t>(idA, idA_sorted, persistency, persistency_sorted, numCnts, stream,
                                                scratchPad);
}

inline void sortABTypePersistencyByType(bodyID_t* idA,
                                        bodyID_t* idB,
                                        contact_t* types,
                                        notStupidBool_t* persistency,
                                        bodyID_t* idA_sorted,
                                        bodyID_t* idB_sorted,
                                        contact_t* type_sorted,
                                        notStupidBool_t* persistency_sorted,
                                        size_t numCnts,
                                        cudaStream_t& stream,
                                        DEMSolverScratchData& scratchPad) {
    cubDEMSortByKeys<contact_t, bodyID_t>(types, type_sorted, idA, idA_sorted, numCnts, stream, scratchPad);
    cubDEMSortByKeys<contact_t, bodyID_t>(types, type_sorted, idB, idB_sorted, numCnts, stream, scratchPad);
    cubDEMSortByKeys<contact_t, notStupidBool_t>(types, type_sorted, persistency, persistency_sorted, numCnts, stream,
                                                 scratchPad);
}

void contactDetection(std::shared_ptr<jitify::Program>& bin_sphere_kernels,
                      std::shared_ptr<jitify::Program>& bin_triangle_kernels,
                      std::shared_ptr<jitify::Program>& sphere_contact_kernels,
                      std::shared_ptr<jitify::Program>& sphTri_contact_kernels,
                      DualStruct<DEMDataKT>& granData,
                      DualStruct<DEMSimParams>& simParams,
                      SolverFlags& solverFlags,
                      verbosity_t& verbosity,
                      DualArray<bodyID_t>& idPrimitiveA,
                      DualArray<bodyID_t>& idPrimitiveB,
                      DualArray<contact_t>& contactTypePrimitive,
                      DualArray<bodyID_t>& previous_idPrimitiveA,
                      DualArray<bodyID_t>& previous_idPrimitiveB,
                      DualArray<contact_t>& previous_contactTypePrimitive,
                      DualArray<notStupidBool_t>& contactPersistency,
                      DualArray<contactPairs_t>& contactMapping,
                      // NEW: Separate patch ID arrays and mapping
                      DualArray<bodyID_t>& idPatchA,
                      DualArray<bodyID_t>& idPatchB,
                      DualArray<bodyID_t>& previous_idPatchA,
                      DualArray<bodyID_t>& previous_idPatchB,
                      DualArray<contact_t>& contactTypePatch,
                      DualArray<contact_t>& previous_contactTypePatch,
                      ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>>& typeStartCountPatchMap,
                      DualArray<contactPairs_t>& geomToPatchMap,
                      cudaStream_t& this_stream,
                      DEMSolverScratchData& scratchPad,
                      SolverTimers& timers,
                      kTStateParams& stateParams) {
    // A dumb check
    if (simParams->nSpheresGM == 0 && simParams->nTriGM == 0) {
        *scratchPad.numContacts = 0;
        *scratchPad.numPrimitiveContacts = 0;
        *scratchPad.numPrevContacts = 0;
        *scratchPad.numPrevPrimitiveContacts = 0;
        *scratchPad.numPrevSpheres = 0;
        *scratchPad.numPrevTriangles = 0;
        *scratchPad.numPrevMeshPatches = 0;

        scratchPad.numContacts.toDevice();
        scratchPad.numPrimitiveContacts.toDevice();
        scratchPad.numPrevContacts.toDevice();
        scratchPad.numPrevPrimitiveContacts.toDevice();
        scratchPad.numPrevSpheres.toDevice();
        scratchPad.numPrevTriangles.toDevice();
        scratchPad.numPrevMeshPatches.toDevice();
        return;
    }
    // These are needed for the solver to keep tab... But you know, we may have no triangles or no contacts, so
    // initializing them is needed.
    stateParams.maxSphFoundInBin = 0;
    stateParams.maxTriFoundInBin = 0;
    stateParams.avgCntsPerPrimitive = 0;

    // A special flag that marks if the contact arrays are sorted based on type. At the initial generation stage, they
    // are indeed generated type by type
    bool primitiveContactArraysAreSortedByType = true;

    // total bytes needed for temp arrays in contact detection
    size_t CD_temp_arr_bytes = 0;

    // Contact counts
    size_t nSphereSphereContact = 0, nTriSphereContact = 0, nTriTriContact = 0, nSphereGeoContact = 0,
           nTriGeoContact = 0;

    // Track patch contact start/count per type (for the new redesigned mapping approach)
    ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>> typeStartCountPatchMap_thisStep;
    typeStartCountPatchMap_thisStep.SetAll({0, 0});

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
                &(scratchPad.numPrimitiveContacts), &(numAnalGeoSphereTouchesScan[simParams->nSpheresGM]), this_stream);
            // numContact is updated (with geo--sphere pair number), get it to host
            scratchPad.numPrimitiveContacts.toHost();
            nSphereGeoContact = *scratchPad.numPrimitiveContacts;
            if (*scratchPad.numPrimitiveContacts > idPrimitiveA.size()) {
                primitiveContactArraysResize(*(scratchPad.numPrimitiveContacts), idPrimitiveA, idPrimitiveB,
                                             contactTypePrimitive, contactPersistency, granData);
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
                        binIDsEachSphereTouches, sphereIDsEachBinTouches, granData->idPrimitiveA,
                        granData->idPrimitiveB, granData->contactTypePrimitive);
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
                // numPrimitiveContacts is for temp use, and it got overwritten in this step
                deviceAdd<binsTriangleTouchPairs_t, objID_t, binsTriangleTouchPairs_t>(
                    &(numAnalGeoTriTouchesScan[simParams->nTriGM]), &(numAnalGeoTriTouches[simParams->nTriGM - 1]),
                    &(numAnalGeoTriTouchesScan[simParams->nTriGM - 1]), this_stream);
                deviceAssign<size_t, binsTriangleTouchPairs_t>(
                    &(scratchPad.numPrimitiveContacts), &(numAnalGeoTriTouchesScan[simParams->nTriGM]), this_stream);
                // numContact is updated (with geo--sphere pair number), get it to host
                scratchPad.numPrimitiveContacts.toHost();
                nTriGeoContact = *scratchPad.numPrimitiveContacts;
                // But we still want numPrimitiveContacts to keep the running length of contacts, so overwrite it; no
                // worry as numPrimitiveContacts is host-major, so no need to update device
                *scratchPad.numPrimitiveContacts = nSphereGeoContact + nTriGeoContact;
                if (*scratchPad.numPrimitiveContacts > idPrimitiveA.size()) {
                    primitiveContactArraysResize(*(scratchPad.numPrimitiveContacts), idPrimitiveA, idPrimitiveB,
                                                 contactTypePrimitive, contactPersistency, granData);
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
            bodyID_t* idTriA = (granData->idPrimitiveA + nSphereGeoContact);
            bodyID_t* idGeoB = (granData->idPrimitiveB + nSphereGeoContact);
            contact_t* dType = (granData->contactTypePrimitive + nSphereGeoContact);
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

            *scratchPad.numPrimitiveContacts =
                nSphereSphereContact + nSphereGeoContact + nTriGeoContact + nTriSphereContact + nTriTriContact;
            if (*scratchPad.numPrimitiveContacts > idPrimitiveA.size()) {
                primitiveContactArraysResize(*scratchPad.numPrimitiveContacts, idPrimitiveA, idPrimitiveB,
                                             contactTypePrimitive, contactPersistency, granData);
            }

            // Sphere--sphere contact pairs go after tri--anal-geo contacts
            bodyID_t* idSphA = (granData->idPrimitiveA + nSphereGeoContact + nTriGeoContact);
            bodyID_t* idSphB = (granData->idPrimitiveB + nSphereGeoContact + nTriGeoContact);
            contact_t* dType = (granData->contactTypePrimitive + nSphereGeoContact + nTriGeoContact);
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
                    (granData->idPrimitiveA + nSphereGeoContact + nTriGeoContact + nSphereSphereContact);
                bodyID_t* idTriB_sm =
                    (granData->idPrimitiveB + nSphereGeoContact + nTriGeoContact + nSphereSphereContact);
                contact_t* dType_sm =
                    (granData->contactTypePrimitive + nSphereGeoContact + nTriGeoContact + nSphereSphereContact);
                // Mesh patch IDs of SM, MM contacts go after MA contacts
                // Or tri--tri... They go after tri--sph contacts
                bodyID_t* idTriA_mm = (granData->idPrimitiveA + nSphereGeoContact + nTriGeoContact +
                                       nSphereSphereContact + nTriSphereContact);
                bodyID_t* idTriB_mm = (granData->idPrimitiveB + nSphereGeoContact + nTriGeoContact +
                                       nSphereSphereContact + nTriSphereContact);
                contact_t* dType_mm = (granData->contactTypePrimitive + nSphereGeoContact + nTriGeoContact +
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
            // std::cout << "idPrimitiveA: " << std::endl;
            // displayDeviceArray<bodyID_t>(granData->idPrimitiveA, *scratchPad.numPrimitiveContacts);
            // std::cout << "idPrimitiveB: " << std::endl;
            // displayDeviceArray<bodyID_t>(granData->idPrimitiveB, *scratchPad.numPrimitiveContacts);
            // std::cout << "contactTypePrimitive: " << std::endl;
            // displayDeviceArray<contact_t>(granData->contactTypePrimitive, *scratchPad.numPrimitiveContacts);
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

        // At user API level, persistent contacts are only supported for history-based models.
        if (solverFlags.hasPersistentContacts) {
            if (solverFlags.isHistoryless) {
                DEME_ERROR(std::string(
                    "You cannot mark persistent contacts when using a wildcard-less/history-less contact model (since "
                    "persistency is a part of the history).\nYou can use a different force model, and if you have to "
                    "use this one, add a placeholder wildcard."));
            }
            // A bool array to help find what persistent contacts from the prev array need to be processed...
            // Contact persistency array is kT-only, and since the solver does not modify it, it can be seen as the
            // prev-primitive contact array's persistency info.
            size_t flag_arr_bytes = (*scratchPad.numPrevPrimitiveContacts) * sizeof(notStupidBool_t);
            notStupidBool_t* grab_flags = (notStupidBool_t*)scratchPad.allocateTempVector("grab_flags", flag_arr_bytes);
            size_t blocks_needed_for_flagging =
                (*scratchPad.numPrevPrimitiveContacts + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
            if (blocks_needed_for_flagging > 0) {
                markBoolIf<<<dim3(blocks_needed_for_flagging), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, this_stream>>>(
                    grab_flags, granData->contactPersistency, CONTACT_IS_PERSISTENT,
                    *scratchPad.numPrevPrimitiveContacts);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            }
            // Store the number of persistent contacts
            scratchPad.allocateDualStruct("numPersistCnts");

            // Then extract the persistent array
            // This many elements are sufficient, at very least...
            size_t selected_ids_bytes = (*scratchPad.numPrevPrimitiveContacts) * sizeof(bodyID_t);
            size_t selected_types_bytes = (*scratchPad.numPrevPrimitiveContacts) * sizeof(contact_t);
            bodyID_t* selected_idA = (bodyID_t*)scratchPad.allocateTempVector("selected_idA", selected_ids_bytes);
            bodyID_t* selected_idB = (bodyID_t*)scratchPad.allocateTempVector("selected_idB", selected_ids_bytes);
            contact_t* selected_types =
                (contact_t*)scratchPad.allocateTempVector("selected_types", selected_types_bytes);

            cubDEMSelectFlagged<bodyID_t, notStupidBool_t>(granData->previous_idPrimitiveA, selected_idA, grab_flags,
                                                           scratchPad.getDualStructDevice("numPersistCnts"),
                                                           *scratchPad.numPrevPrimitiveContacts, this_stream,
                                                           scratchPad);
            cubDEMSelectFlagged<bodyID_t, notStupidBool_t>(granData->previous_idPrimitiveB, selected_idB, grab_flags,
                                                           scratchPad.getDualStructDevice("numPersistCnts"),
                                                           *scratchPad.numPrevPrimitiveContacts, this_stream,
                                                           scratchPad);
            cubDEMSelectFlagged<contact_t, notStupidBool_t>(
                granData->previous_contactTypePrimitive, selected_types, grab_flags,
                scratchPad.getDualStructDevice("numPersistCnts"), *scratchPad.numPrevPrimitiveContacts, this_stream,
                scratchPad);
            // Those flag selections give the same result. Bring it to host.
            scratchPad.syncDualStructDeviceToHost("numPersistCnts");
            size_t* pNumPersistCnts = scratchPad.getDualStructHost("numPersistCnts");

            // Then concatenate the persistent primitive contacts (goes first) and the newly detected contacts (follows
            // after)
            size_t total_ids_bytes = (*scratchPad.numPrimitiveContacts + *pNumPersistCnts) * sizeof(bodyID_t);
            size_t total_types_bytes = (*scratchPad.numPrimitiveContacts + *pNumPersistCnts) * sizeof(contact_t);
            size_t total_persistency_bytes =
                (*scratchPad.numPrimitiveContacts + *pNumPersistCnts) * sizeof(notStupidBool_t);
            selected_ids_bytes = (*pNumPersistCnts) * sizeof(bodyID_t);
            selected_types_bytes = (*pNumPersistCnts) * sizeof(contact_t);
            bodyID_t* total_idA = (bodyID_t*)scratchPad.allocateTempVector("total_idA", total_ids_bytes);
            bodyID_t* total_idB = (bodyID_t*)scratchPad.allocateTempVector("total_idB", total_ids_bytes);
            contact_t* total_types = (contact_t*)scratchPad.allocateTempVector("total_types", total_types_bytes);
            notStupidBool_t* total_persistency =
                (notStupidBool_t*)scratchPad.allocateTempVector("total_persistency", total_persistency_bytes);
            DEME_GPU_CALL(cudaMemcpy(total_idA, selected_idA, selected_ids_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(total_idA + *pNumPersistCnts, granData->idPrimitiveA,
                                     total_ids_bytes - selected_ids_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(total_idB, selected_idB, selected_ids_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(total_idB + *pNumPersistCnts, granData->idPrimitiveB,
                                     total_ids_bytes - selected_ids_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(total_types, selected_types, selected_types_bytes, cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(total_types + *pNumPersistCnts, granData->contactTypePrimitive,
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
            size_t numTotalCnts = *scratchPad.numPrimitiveContacts + *pNumPersistCnts;
            contact_t* contactType_sorted =
                (contact_t*)scratchPad.allocateTempVector("contactType_sorted", total_types_bytes);
            bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector("idA_sorted", total_ids_bytes);
            bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector("idB_sorted", total_ids_bytes);
            notStupidBool_t* persistency_sorted =
                (notStupidBool_t*)scratchPad.allocateTempVector("persistency_sorted", total_persistency_bytes);
            // Pay attention to that it does leverage the fact that RadixSort is stable.
            sortABTypePersistencyByA(total_idA, total_idB, total_types, total_persistency, idA_sorted, idB_sorted,
                                     contactType_sorted, persistency_sorted, numTotalCnts, this_stream, scratchPad);

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
                                    idPrimitiveA, idPrimitiveB, contactTypePrimitive, contactPersistency, true,
                                    DEME_MAX(simParams->nSpheresGM, simParams->nTriGM), numTotalCnts, this_stream,
                                    scratchPad);

            scratchPad.finishUsingTempVector("contactType_sorted");
            scratchPad.finishUsingTempVector("idA_sorted");
            scratchPad.finishUsingTempVector("idB_sorted");
            scratchPad.finishUsingTempVector("persistency_sorted");

            // Here, not sorted by type actually means sorted by idA
            primitiveContactArraysAreSortedByType = false;
        }

        // -----------------------------------------------------------------------------------------------------------
        // One more thing: We need to remove duplicate primitive contacts in the contact list, because tri--tri contacts
        // are not guaranteed to be unique. But of course, if hasPersistentContacts is on, then the duplication removal
        // has already been done in the previous step.
        // -----------------------------------------------------------------------------------------------------------

        if (solverFlags.meshUniversalContact && !solverFlags.hasPersistentContacts) {
            // To remove duplicates, we sort by idA...
            size_t numTotalCnts = *scratchPad.numPrimitiveContacts;
            size_t total_ids_bytes = numTotalCnts * sizeof(bodyID_t);
            size_t total_types_bytes = numTotalCnts * sizeof(contact_t);
            size_t total_persistency_bytes = numTotalCnts * sizeof(notStupidBool_t);
            contact_t* contactType_sorted =
                (contact_t*)scratchPad.allocateTempVector("contactType_sorted", total_types_bytes);
            bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector("idA_sorted", total_ids_bytes);
            bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector("idB_sorted", total_ids_bytes);
            notStupidBool_t* persistency_sorted =
                (notStupidBool_t*)scratchPad.allocateTempVector("persistency_sorted", total_persistency_bytes);
            // Pay attention to that it does leverage the fact that RadixSort is stable.
            sortABTypePersistencyByA(granData->idPrimitiveA, granData->idPrimitiveB, granData->contactTypePrimitive,
                                     granData->contactPersistency, idA_sorted, idB_sorted, contactType_sorted,
                                     persistency_sorted, numTotalCnts, this_stream, scratchPad);
            // std::cout << "Contacts before duplication check: " << std::endl;
            // displayDeviceArray<bodyID_t>(idA_sorted, numTotalCnts);
            // displayDeviceArray<bodyID_t>(idB_sorted, numTotalCnts);
            // displayDeviceArray<contact_t>(contactType_sorted, numTotalCnts);

            removeDuplicateContacts(granData, idA_sorted, idB_sorted, contactType_sorted, persistency_sorted,
                                    idPrimitiveA, idPrimitiveB, contactTypePrimitive, contactPersistency, false,
                                    DEME_MAX(simParams->nSpheresGM, simParams->nTriGM), numTotalCnts, this_stream,
                                    scratchPad);
            // std::cout << "Contacts after duplication check: " << std::endl;
            // displayDeviceArray<bodyID_t>(granData->idPrimitiveA, *scratchPad.numPrimitiveContacts);
            // displayDeviceArray<bodyID_t>(granData->idPrimitiveB, *scratchPad.numPrimitiveContacts);
            // displayDeviceArray<contact_t>(granData->contactTypePrimitive, *scratchPad.numPrimitiveContacts);

            scratchPad.finishUsingTempVector("contactType_sorted");
            scratchPad.finishUsingTempVector("idA_sorted");
            scratchPad.finishUsingTempVector("idB_sorted");
            scratchPad.finishUsingTempVector("persistency_sorted");

            // Here, not sorted by type actually means sorted by idA
            primitiveContactArraysAreSortedByType = false;
        }
        // std::cout << "Primitive contacts: " << std::endl;
        // displayDeviceArray<bodyID_t>(granData->idPrimitiveA, *scratchPad.numPrimitiveContacts);
        // displayDeviceArray<bodyID_t>(granData->idPrimitiveB, *scratchPad.numPrimitiveContacts);
        // displayDeviceArray<contact_t>(granData->contactTypePrimitive, *scratchPad.numPrimitiveContacts);

        // -----------------------------------------------------------------------------------------------------------
        // We need to now do some sanity checks. If primitive contacts are already sorted by idA, we just use them (they
        // are stored in granData). If sorted by type, then we have to sort by idA but do not change the granData
        // arrays, as they will be used later.
        // -----------------------------------------------------------------------------------------------------------
        if (*scratchPad.numPrimitiveContacts > 0) {
            size_t numTotalCnts = *scratchPad.numPrimitiveContacts;
            size_t total_ids_bytes = numTotalCnts * sizeof(bodyID_t);
            bodyID_t* idA = (bodyID_t*)scratchPad.allocateTempVector("idA_sorted_for_sanity_check", total_ids_bytes);
            if (!primitiveContactArraysAreSortedByType) {
                // A sorted already...
                idA = granData->idPrimitiveA;
            } else {
                // Need to sort A...
                cubDEMSortKeys<bodyID_t>(granData->idPrimitiveA, idA, numTotalCnts, this_stream, scratchPad);
            }

            size_t nGeoSafe = DEME_MAX(simParams->nSpheresGM, simParams->nTriGM);
            // For tab-keeping: how many contacts on average a sphere has? (using primitive contacts for this stat)
            // First, identify unique idA in primitive contacts
            size_t run_length_bytes = nGeoSafe * sizeof(primitivesPrimTouches_t);
            primitivesPrimTouches_t* new_idA_runlength =
                (primitivesPrimTouches_t*)scratchPad.allocateTempVector("new_idA_runlength", run_length_bytes);
            size_t unique_id_bytes = nGeoSafe * sizeof(bodyID_t);
            bodyID_t* unique_new_idA = (bodyID_t*)scratchPad.allocateTempVector("unique_new_idA", unique_id_bytes);
            scratchPad.allocateDualStruct("numUniqueNewA");

            cubDEMRunLengthEncode<bodyID_t, primitivesPrimTouches_t>(idA, unique_new_idA, new_idA_runlength,
                                                                     scratchPad.getDualStructDevice("numUniqueNewA"),
                                                                     numTotalCnts, this_stream, scratchPad);
            scratchPad.syncDualStructDeviceToHost("numUniqueNewA");
            size_t* pNumUniqueNewA = scratchPad.getDualStructHost("numUniqueNewA");

            // Figure out how many contacts an item in idA array typically has.
            stateParams.avgCntsPerPrimitive =
                (*pNumUniqueNewA > 0) ? (float)(*scratchPad.numPrimitiveContacts) / (float)(*pNumUniqueNewA) : 0.0;

            DEME_DEBUG_PRINTF("Average number of contacts for each geometry: %.7g", stateParams.avgCntsPerPrimitive);
            if (stateParams.avgCntsPerPrimitive > solverFlags.errOutAvgSphCnts) {
                DEME_ERROR(
                    "On average a primitive (spheres, triangle facets) has %.7g contacts with other primitives, more "
                    "than the max allowance (%.7g).\nIf you believe this number is expected with the physics you are "
                    "simulating, set the allowance high using SetErrorOutAvgContacts before "
                    "initialization.\nIf you think this is because dT drifting too much "
                    "ahead of kT so the contact margin added is too big, use SetCDMaxUpdateFreq to limit the max dT "
                    "future drift.\nOtherwise, the simulation may have diverged with unreasonable entity velocities "
                    "and relaxing the physics may help, such as decreasing the step size and modifying material "
                    "properties.\nIf this happens at the start of simulation, check if there are initial penetrations, "
                    "a.k.a. elements initialized inside walls.\nIf none works and you are going to discuss this on "
                    "forum https://groups.google.com/g/projectchrono, please include a visual rendering of the "
                    "simulation before crash in the post.\n",
                    stateParams.avgCntsPerPrimitive, solverFlags.errOutAvgSphCnts);
            }

            scratchPad.finishUsingTempVector("new_idA_runlength");
            scratchPad.finishUsingTempVector("unique_new_idA");
            scratchPad.finishUsingDualStruct("numUniqueNewA");
            scratchPad.finishUsingTempVector("idA_sorted_for_sanity_check");
        }

        // -----------------------------------------------------------------------------------------------------------
        // Up to this point, we have been working with primitive contacts (sphere-sphere, sphere-triangle,
        // triangle-triangle, sphere-analytical-geometry, triangle-analytical-geometry). Now we need to convert them to
        // patch/convex shape based contacts for dT consumption.
        // -----------------------------------------------------------------------------------------------------------

        // Generate patch IDs (idPatchA/B) and geomToPatchMap
        if (*scratchPad.numPrimitiveContacts > 0) {
            // If there are processes that made the primitive contact arrays unsorted by idA, we need to sort them by
            // type first
            if (!primitiveContactArraysAreSortedByType) {
                size_t numTotalCnts = *scratchPad.numPrimitiveContacts;
                size_t total_ids_bytes = numTotalCnts * sizeof(bodyID_t);
                size_t total_types_bytes = numTotalCnts * sizeof(contact_t);
                size_t total_persistency_bytes = numTotalCnts * sizeof(notStupidBool_t);
                contact_t* contactType_sorted =
                    (contact_t*)scratchPad.allocateTempVector("contactType_sorted", total_types_bytes);
                bodyID_t* idPrimitiveA_sorted =
                    (bodyID_t*)scratchPad.allocateTempVector("idPrimitiveA_sorted", total_ids_bytes);
                bodyID_t* idPrimitiveB_sorted =
                    (bodyID_t*)scratchPad.allocateTempVector("idPrimitiveB_sorted", total_ids_bytes);
                notStupidBool_t* contactPersistency_sorted = (notStupidBool_t*)scratchPad.allocateTempVector(
                    "contactPersistency_sorted", total_persistency_bytes);
                sortABTypePersistencyByType(granData->idPrimitiveA, granData->idPrimitiveB,
                                            granData->contactTypePrimitive, granData->contactPersistency,
                                            idPrimitiveA_sorted, idPrimitiveB_sorted, contactType_sorted,
                                            contactPersistency_sorted, numTotalCnts, this_stream, scratchPad);
                // Then copy back to granData
                DEME_GPU_CALL(
                    cudaMemcpy(granData->idPrimitiveA, idPrimitiveA_sorted, total_ids_bytes, cudaMemcpyDeviceToDevice));
                DEME_GPU_CALL(
                    cudaMemcpy(granData->idPrimitiveB, idPrimitiveB_sorted, total_ids_bytes, cudaMemcpyDeviceToDevice));
                DEME_GPU_CALL(cudaMemcpy(granData->contactTypePrimitive, contactType_sorted, total_types_bytes,
                                         cudaMemcpyDeviceToDevice));
                DEME_GPU_CALL(cudaMemcpy(granData->contactPersistency, contactPersistency_sorted,
                                         total_persistency_bytes, cudaMemcpyDeviceToDevice));
                scratchPad.finishUsingTempVector("contactType_sorted");
                scratchPad.finishUsingTempVector("idPrimitiveA_sorted");
                scratchPad.finishUsingTempVector("idPrimitiveB_sorted");
                scratchPad.finishUsingTempVector("contactPersistency_sorted");
            }
            // Reset the flag; now sorted by types
            primitiveContactArraysAreSortedByType = true;

            // Generate the contact patch ID pairs for each contact pair
            // Use a hashed pair to store the patch ID pairs...
            patchIDPair_t* contactPatchPairs = (patchIDPair_t*)scratchPad.allocateTempVector(
                "contactPatchPairs", (*scratchPad.numPrimitiveContacts) * sizeof(patchIDPair_t));

            // Based on the complete primitive contact arrays...
            size_t blocks_needed_for_patch_ids =
                (*scratchPad.numPrimitiveContacts + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
            if (blocks_needed_for_patch_ids > 0) {
                extractPatchInvolvedContactPatchIDPairs<<<dim3(blocks_needed_for_patch_ids),
                                                          dim3(DEME_MAX_THREADS_PER_BLOCK), 0, this_stream>>>(
                    contactPatchPairs, granData->contactTypePrimitive, granData->idPrimitiveA, granData->idPrimitiveB,
                    granData->triPatchID, *scratchPad.numPrimitiveContacts);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            }

            // Sort contactPatchPairs within each contact type segment, so we can construct geomToPatchMap

            // First, identify the contact type segments using run-length encoding
            // Maximum number of contact types (5 main types: sph-sph, sph-tri, sph-anal, tri-tri, tri-anal)
            contact_t* unique_types = (contact_t*)scratchPad.allocateTempVector(
                "unique_types", NUM_SUPPORTED_CONTACT_TYPES * sizeof(contact_t));
            size_t* type_counts =
                (size_t*)scratchPad.allocateTempVector("type_counts", NUM_SUPPORTED_CONTACT_TYPES * sizeof(size_t));
            scratchPad.allocateDualStruct("numUniqueTypes");

            cubDEMRunLengthEncode<contact_t, size_t>(granData->contactTypePrimitive, unique_types, type_counts,
                                                     scratchPad.getDualStructDevice("numUniqueTypes"),
                                                     *scratchPad.numPrimitiveContacts, this_stream, scratchPad);
            scratchPad.syncDualStructDeviceToHost("numUniqueTypes");
            size_t numTypes = *scratchPad.getDualStructHost("numUniqueTypes");

            // Now sort within each type segment and construct patch ID lists gradually
            // This approach handles the corner case where two primitive contacts from different types
            // have the same encoded patch ID - they should be considered as separate patch contacts.
            if (numTypes > 0) {
                // Copy type_counts to host to know segment boundaries
                size_t* host_type_counts = new size_t[numTypes];
                DEME_GPU_CALL(
                    cudaMemcpy(host_type_counts, type_counts, numTypes * sizeof(size_t), cudaMemcpyDeviceToHost));

                // Copy unique_types to host for contact type assignment
                contact_t* host_unique_types = new contact_t[numTypes];
                DEME_GPU_CALL(
                    cudaMemcpy(host_unique_types, unique_types, numTypes * sizeof(contact_t), cudaMemcpyDeviceToHost));

                // Allocate temp arrays for sorting (for the entire primitive contact array)
                // Note this sort here is an even deeper sort: within each type segment, we sort by patch ID pairs
                size_t patch_arr_bytes = (*scratchPad.numPrimitiveContacts) * sizeof(patchIDPair_t);
                patchIDPair_t* patchPairs_sorted =
                    (patchIDPair_t*)scratchPad.allocateTempVector("patchPairs_sorted", patch_arr_bytes);
                size_t id_arr_bytes = (*scratchPad.numPrimitiveContacts) * sizeof(bodyID_t);
                bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector("idA_sorted", id_arr_bytes);
                bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector("idB_sorted", id_arr_bytes);
                size_t type_arr_bytes = (*scratchPad.numPrimitiveContacts) * sizeof(contact_t);
                contact_t* contactType_sorted =
                    (contact_t*)scratchPad.allocateTempVector("contactType_sorted_patch", type_arr_bytes);

                // Allocate temp arrays for per-type run-length encoding
                size_t max_unique_per_segment = *scratchPad.numPrimitiveContacts;  // Upper bound
                patchIDPair_t* unique_patch_pairs = (patchIDPair_t*)scratchPad.allocateTempVector(
                    "unique_patch_pairs", max_unique_per_segment * sizeof(patchIDPair_t));
                contactPairs_t* patch_pair_counts = (contactPairs_t*)scratchPad.allocateTempVector(
                    "patch_pair_counts", max_unique_per_segment * sizeof(contactPairs_t));
                scratchPad.allocateDualStruct("numUniquePatchPairs");

                // Ensure geomToPatchMap is sized to numPrimitiveContacts before the loop
                if (*scratchPad.numPrimitiveContacts > geomToPatchMap.size()) {
                    DEME_DUAL_ARRAY_RESIZE_NOVAL(geomToPatchMap, *scratchPad.numPrimitiveContacts);
                    granData.toDevice();
                }

                // Track total unique patch pairs across all types
                size_t totalUniquePatchPairs = 0;
                size_t prim_offset = 0;  // Offset in primitive contact array

                // Process each type segment separately
                for (size_t i = 0; i < numTypes; i++) {
                    contact_t thisType = host_unique_types[i];
                    size_t count = host_type_counts[i];

                    if (count == 0)
                        continue;

                    // Step 1a: Sort this type segment by patch ID pairs
                    if (count > 1) {
                        // Sort idPrimitiveA with contactPatchPairs
                        cubDEMSortByKeys<patchIDPair_t, bodyID_t>(
                            contactPatchPairs + prim_offset, patchPairs_sorted + prim_offset,
                            granData->idPrimitiveA + prim_offset, idA_sorted + prim_offset, count, this_stream,
                            scratchPad);

                        // Sort idPrimitiveB with contactPatchPairs
                        cubDEMSortByKeys<patchIDPair_t, bodyID_t>(
                            contactPatchPairs + prim_offset, patchPairs_sorted + prim_offset,
                            granData->idPrimitiveB + prim_offset, idB_sorted + prim_offset, count, this_stream,
                            scratchPad);

                        // Sort contactTypePrimitive with contactPatchPairs
                        cubDEMSortByKeys<patchIDPair_t, contact_t>(
                            contactPatchPairs + prim_offset, patchPairs_sorted + prim_offset,
                            granData->contactTypePrimitive + prim_offset, contactType_sorted + prim_offset, count,
                            this_stream, scratchPad);
                    } else {
                        // Just copy single elements (no sorting needed)
                        DEME_GPU_CALL(cudaMemcpy(patchPairs_sorted + prim_offset, contactPatchPairs + prim_offset,
                                                 sizeof(patchIDPair_t), cudaMemcpyDeviceToDevice));
                        DEME_GPU_CALL(cudaMemcpy(idA_sorted + prim_offset, granData->idPrimitiveA + prim_offset,
                                                 sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
                        DEME_GPU_CALL(cudaMemcpy(idB_sorted + prim_offset, granData->idPrimitiveB + prim_offset,
                                                 sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
                        DEME_GPU_CALL(cudaMemcpy(contactType_sorted + prim_offset,
                                                 granData->contactTypePrimitive + prim_offset, sizeof(contact_t),
                                                 cudaMemcpyDeviceToDevice));
                    }

                    // Step 1b: Run-length encode to find unique patch pairs WITHIN this type segment
                    cubDEMRunLengthEncode<patchIDPair_t, contactPairs_t>(
                        patchPairs_sorted + prim_offset, unique_patch_pairs, patch_pair_counts,
                        scratchPad.getDualStructDevice("numUniquePatchPairs"), count, this_stream, scratchPad);
                    scratchPad.syncDualStructDeviceToHost("numUniquePatchPairs");
                    size_t numUniqueInSegment = *scratchPad.getDualStructHost("numUniquePatchPairs");

                    // Step 2: Ensure idPatchA/B and contactTypePatch are large enough
                    size_t newTotalSize = totalUniquePatchPairs + numUniqueInSegment;
                    if (newTotalSize > idPatchA.size()) {
                        DEME_DUAL_ARRAY_RESIZE_NOVAL(idPatchA, newTotalSize);
                        DEME_DUAL_ARRAY_RESIZE_NOVAL(idPatchB, newTotalSize);
                        DEME_DUAL_ARRAY_RESIZE_NOVAL(contactTypePatch, newTotalSize);
                        granData.toDevice();
                    }

                    // Step 3: Decode unique patch pairs into idPatchA/B at the appropriate offset
                    if (numUniqueInSegment > 0) {
                        // Store the start/count for this type in the patch contact map
                        typeStartCountPatchMap_thisStep[thisType] = {totalUniquePatchPairs, numUniqueInSegment};

                        size_t blocks_needed_for_decode =
                            (numUniqueInSegment + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
                        decodePatchPairsToSeparateArrays<<<dim3(blocks_needed_for_decode),
                                                           dim3(DEME_MAX_THREADS_PER_BLOCK), 0, this_stream>>>(
                            unique_patch_pairs, granData->idPatchA + totalUniquePatchPairs,
                            granData->idPatchB + totalUniquePatchPairs, numUniqueInSegment);
                        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));

                        // Set contactTypePatch for this segment - all have the same type (using GPU kernel)
                        fillContactTypeArray<<<dim3(blocks_needed_for_decode), dim3(DEME_MAX_THREADS_PER_BLOCK), 0,
                                               this_stream>>>(granData->contactTypePatch + totalUniquePatchPairs,
                                                              thisType, numUniqueInSegment);
                        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
                    }

                    // Step 4: Build geomToPatchMap for this segment
                    // Allocate temp array for "is new group" markers for this segment
                    contactPairs_t* isNewGroup =
                        (contactPairs_t*)scratchPad.allocateTempVector("isNewGroup", count * sizeof(contactPairs_t));

                    size_t blocks_needed_for_mark =
                        (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
                    if (blocks_needed_for_mark > 0) {
                        markNewPatchPairGroups<<<dim3(blocks_needed_for_mark), dim3(DEME_MAX_THREADS_PER_BLOCK), 0,
                                                 this_stream>>>(patchPairs_sorted + prim_offset, isNewGroup, count);
                        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
                    }
                    // std::cout << "patchPairs_sorted segment " << i << ": " << std::endl;
                    // displayDeviceArray<patchIDPair_t>(patchPairs_sorted + prim_offset, count);
                    // std::cout << "isNewGroup segment " << i << ": " << std::endl;
                    // displayDeviceArray<contactPairs_t>(isNewGroup, count);

                    // Prefix scan on isNewGroup and write result directly to the geomToPatchMap location. Note in
                    // isNewGroup, first element is 0, then with a jump, element is marked 1, so inclusive scan is
                    // needed to count the number of uniques.
                    cubDEMInclusiveScan<contactPairs_t, contactPairs_t>(
                        isNewGroup, granData->geomToPatchMap + prim_offset, count, this_stream, scratchPad);

                    // Add the global offset (totalUniquePatchPairs) to get the final geomToPatchMap values. Also note
                    // this add offset operation will take care of the +1 needed at the jump of each new type.
                    if (count > 0 && totalUniquePatchPairs > 0) {
                        addOffsetToArray<<<dim3(blocks_needed_for_mark), dim3(DEME_MAX_THREADS_PER_BLOCK), 0,
                                           this_stream>>>(granData->geomToPatchMap + prim_offset,
                                                          (contactPairs_t)totalUniquePatchPairs, count);
                        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
                    }

                    scratchPad.finishUsingTempVector("isNewGroup");

                    // Update offsets
                    totalUniquePatchPairs += numUniqueInSegment;
                    prim_offset += count;
                }

                // Set the total number of patch contacts
                *scratchPad.numContacts = totalUniquePatchPairs;

                scratchPad.finishUsingTempVector("unique_patch_pairs");
                scratchPad.finishUsingTempVector("patch_pair_counts");
                scratchPad.finishUsingDualStruct("numUniquePatchPairs");

                scratchPad.finishUsingTempVector("patchPairs_sorted");
                scratchPad.finishUsingTempVector("idA_sorted");
                scratchPad.finishUsingTempVector("idB_sorted");
                scratchPad.finishUsingTempVector("contactType_sorted_patch");

                delete[] host_type_counts;
                delete[] host_unique_types;
            }

            scratchPad.finishUsingTempVector("unique_types");
            scratchPad.finishUsingTempVector("type_counts");
            scratchPad.finishUsingTempVector("contactPatchPairs");
            scratchPad.finishUsingDualStruct("numUniqueTypes");
            // std::cout << "Patch contacts:" << std::endl;
            // displayDeviceArray<bodyID_t>(granData->idPatchA, *scratchPad.numContacts);
            // displayDeviceArray<bodyID_t>(granData->idPatchB, *scratchPad.numContacts);
            // displayDeviceArray<contact_t>(granData->contactTypePatch, *scratchPad.numContacts);
            // displayDeviceArray<contactPairs_t>(granData->geomToPatchMap, *scratchPad.numPrimitiveContacts);
        }

        timers.GetTimer("Find contact pairs").stop();

    }  // End of contact pairs construction of this CD step

    // -----------------------------------------------------------------------------------------------------------
    // Constructing contact history (patch-based)
    // The contact mapping is now being built between previous_idPatchA/B and current idPatchA/B.
    // We have numContacts elements to work with (patch-based contacts), not numPrimitiveContacts.
    // Both current and previous patch arrays are sorted by contact type, and within each type,
    // they are sorted by the combined contact patch ID pair.
    // -----------------------------------------------------------------------------------------------------------

    timers.GetTimer("Build history map").start();
    // Now, identify enduring contacts in history-based models.
    if (*scratchPad.numContacts > 0) {
        // For history-based models, construct the enduring contact map at the patch level.
        // Only need to actually create the mapping if the force model has history
        if (!solverFlags.isHistoryless) {
            // Resize contactMapping to hold numContacts elements (patch-based)
            if (*scratchPad.numContacts > contactMapping.size()) {
                DEME_DUAL_ARRAY_RESIZE_NOVAL(contactMapping, *scratchPad.numContacts);
                granData.toDevice();
            }

            // Build patch-based contact mapping using per-type kernels
            // Iterate over all supported contact types and launch a kernel for each type that has contacts
            for (size_t type_idx = 0; type_idx < NUM_SUPPORTED_CONTACT_TYPES; type_idx++) {
                contact_t thisType = ALL_CONTACT_TYPES[type_idx];

                // Get start/count for this type in the current and previous steps
                // Using operator[] is safe here since ContactTypeMap initializes all types in constructor
                const auto& curr_info = typeStartCountPatchMap_thisStep[thisType];
                const auto& prev_info = typeStartCountPatchMap[thisType];

                contactPairs_t curr_start = curr_info.first;
                contactPairs_t curr_count = curr_info.second;
                contactPairs_t prev_start = prev_info.first;
                contactPairs_t prev_count = prev_info.second;

                // Skip if no contacts of this type in current step
                if (curr_count == 0) {
                    continue;
                }

                // Launch appropriate kernel based on whether previous step had this type
                size_t blocks_needed = (curr_count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;

                if (prev_count == 0) {
                    // Previous step has no contacts of this type - set all to NULL_MAPPING_PARTNER
                    setNullMappingForType<<<dim3(blocks_needed), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, this_stream>>>(
                        granData->contactMapping, curr_start, curr_count);
                } else {
                    // Both steps have contacts of this type - perform mapping
                    buildPatchContactMappingForType<<<dim3(blocks_needed), dim3(DEME_MAX_THREADS_PER_BLOCK), 0,
                                                      this_stream>>>(
                        granData->idPatchA, granData->idPatchB, granData->previous_idPatchA,
                        granData->previous_idPatchB, granData->contactMapping, curr_start, curr_count, prev_start,
                        prev_count);
                }
            }
            // Synchronize once after all mapping kernels are launched
            DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            // std::cout << "Patch contact mapping:" << std::endl;
            // displayDeviceArray<contactPairs_t>(granData->contactMapping, *scratchPad.numContacts);

            // Copy current patch arrays to previous arrays for the next iteration
            size_t patch_id_arr_bytes = (*scratchPad.numContacts) * sizeof(bodyID_t);
            size_t patch_type_arr_bytes = (*scratchPad.numContacts) * sizeof(contact_t);
            if (*scratchPad.numContacts > previous_idPatchA.size()) {
                patchArraysResize(*scratchPad.numContacts, previous_idPatchA, previous_idPatchB,
                                  previous_contactTypePatch, granData);
            }
            DEME_GPU_CALL(cudaMemcpy(granData->previous_idPatchA, granData->idPatchA, patch_id_arr_bytes,
                                     cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(granData->previous_idPatchB, granData->idPatchB, patch_id_arr_bytes,
                                     cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(granData->previous_contactTypePatch, granData->contactTypePatch,
                                     patch_type_arr_bytes, cudaMemcpyDeviceToDevice));

            // Currently only when using persistent contacts we need to store enduring primitive contact info
            if (solverFlags.hasPersistentContacts) {
                // Resize previous primitive contact arrays if needed
                if (*scratchPad.numPrimitiveContacts > previous_idPrimitiveA.size()) {
                    // Note persistency should not change size here even if called
                    primitiveContactArraysResize(*scratchPad.numPrimitiveContacts, previous_idPrimitiveA,
                                                 previous_idPrimitiveB, previous_contactTypePrimitive,
                                                 contactPersistency, granData);
                }
                // Copy current primitive contact arrays to previous arrays for the next iteration
                size_t primitive_id_arr_bytes = (*scratchPad.numPrimitiveContacts) * sizeof(bodyID_t);
                size_t primitive_type_arr_bytes = (*scratchPad.numPrimitiveContacts) * sizeof(contact_t);
                DEME_GPU_CALL(cudaMemcpy(granData->previous_idPrimitiveA, granData->idPrimitiveA,
                                         primitive_id_arr_bytes, cudaMemcpyDeviceToDevice));
                DEME_GPU_CALL(cudaMemcpy(granData->previous_idPrimitiveB, granData->idPrimitiveB,
                                         primitive_id_arr_bytes, cudaMemcpyDeviceToDevice));
                DEME_GPU_CALL(cudaMemcpy(granData->previous_contactTypePrimitive, granData->contactTypePrimitive,
                                         primitive_type_arr_bytes, cudaMemcpyDeviceToDevice));
            }

            // Update the patch contact map for the next iteration
            typeStartCountPatchMap = typeStartCountPatchMap_thisStep;

        }  // End of history-based model mapping

    }  // End of contact sorting--mapping subroutine
    timers.GetTimer("Build history map").stop();

    // std::cout << "Patch Contacts: " << std::endl;
    // displayDeviceArray<bodyID_t>(granData->idPatchA, *scratchPad.numContacts);
    // displayDeviceArray<bodyID_t>(granData->idPatchB, *scratchPad.numContacts);
    // displayDeviceArray<contact_t>(granData->contactTypePatch, *scratchPad.numContacts);

    // Finally, don't forget to store the number of contacts for the next iteration, even if there is 0 contacts (in
    // that case, mapping will not be constructed, but we don't have to worry b/c in the next iteration, simply no work
    // will be done for the old array and every contact will be new)
    *scratchPad.numPrevContacts = *scratchPad.numContacts;
    *scratchPad.numPrevPrimitiveContacts = *scratchPad.numPrimitiveContacts;
    *scratchPad.numPrevSpheres = simParams->nSpheresGM;
    *scratchPad.numPrevTriangles = simParams->nTriGM;
    *scratchPad.numPrevMeshPatches = simParams->nMeshPatches;

    // dT kT may send these numbers to each other from device
    scratchPad.numContacts.toDevice();
    scratchPad.numPrevContacts.toDevice();
    scratchPad.numPrimitiveContacts.toDevice();
    scratchPad.numPrevPrimitiveContacts.toDevice();
    scratchPad.numPrevSpheres.toDevice();
    scratchPad.numPrevTriangles.toDevice();
    scratchPad.numPrevMeshPatches.toDevice();
}

void overwritePrevContactArrays(DualStruct<DEMDataKT>& kT_data,
                                DualStruct<DEMDataDT>& dT_data,
                                DualArray<bodyID_t>& previous_idPatchA,
                                DualArray<bodyID_t>& previous_idPatchB,
                                DualArray<contact_t>& previous_contactTypePatch,
                                ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>>& typeStartCountPatchMap,
                                DualStruct<DEMSimParams>& simParams,
                                DEMSolverScratchData& scratchPad,
                                cudaStream_t& this_stream,
                                size_t nContacts) {
    // Make sure the storage is large enough
    if (nContacts > previous_idPatchA.size()) {
        patchArraysResize(nContacts, previous_idPatchA, previous_idPatchB, previous_contactTypePatch, kT_data);
    }

    // No sort, copy over
    DEME_GPU_CALL(cudaMemcpy(kT_data->previous_idPatchA, dT_data->idPatchA, nContacts * sizeof(bodyID_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(kT_data->previous_idPatchB, dT_data->idPatchB, nContacts * sizeof(bodyID_t),
                             cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(kT_data->previous_contactTypePatch, dT_data->contactTypePatch,
                             nContacts * sizeof(contact_t), cudaMemcpyDeviceToDevice));

    // Derive typeStartCountPatchMap from the loaded contact arrays
    // This is necessary for the persistent mapping process in the next contact detection step
    if (nContacts > 0) {
        // Use run-length encoding to identify contact type boundaries
        size_t typeCounts_bytes = NUM_SUPPORTED_CONTACT_TYPES * sizeof(contactPairs_t);
        contactPairs_t* typeCounts = (contactPairs_t*)scratchPad.allocateTempVector("typeCounts", typeCounts_bytes);
        
        size_t existingTypes_bytes = NUM_SUPPORTED_CONTACT_TYPES * sizeof(contact_t);
        contact_t* existingTypes = (contact_t*)scratchPad.allocateTempVector("existingTypes", existingTypes_bytes);
        
        scratchPad.allocateDualStruct("numExistingTypes");
        
        // Run-length encode the contact types to find boundaries
        cubDEMRunLengthEncode<contact_t, contactPairs_t>(
            kT_data->previous_contactTypePatch, existingTypes, typeCounts,
            scratchPad.getDualStructDevice("numExistingTypes"), nContacts, this_stream, scratchPad);
        
        scratchPad.syncDualStructDeviceToHost("numExistingTypes");
        size_t numExistingTypes = *scratchPad.getDualStructHost("numExistingTypes");
        
        // Prefix scan to get start offsets
        size_t typeOffsets_bytes = NUM_SUPPORTED_CONTACT_TYPES * sizeof(contactPairs_t);
        contactPairs_t* typeOffsets = (contactPairs_t*)scratchPad.allocateTempVector("typeOffsets", typeOffsets_bytes);
        
        cubDEMPrefixScan<contactPairs_t, contactPairs_t>(typeCounts, typeOffsets, numExistingTypes, this_stream,
                                                         scratchPad);
        
        // Copy results to host to populate the map
        contact_t* host_existingTypes = new contact_t[numExistingTypes];
        contactPairs_t* host_typeOffsets = new contactPairs_t[numExistingTypes];
        
        DEME_GPU_CALL(cudaMemcpy(host_existingTypes, existingTypes, numExistingTypes * sizeof(contact_t),
                                 cudaMemcpyDeviceToHost));
        DEME_GPU_CALL(cudaMemcpy(host_typeOffsets, typeOffsets, numExistingTypes * sizeof(contactPairs_t),
                                 cudaMemcpyDeviceToHost));
        
        // Build the typeStartCountPatchMap
        typeStartCountPatchMap.SetAll({0, 0});
        for (size_t i = 0; i < numExistingTypes; i++) {
            contactPairs_t startOffset = host_typeOffsets[i];
            contactPairs_t count = (i + 1 < numExistingTypes ? host_typeOffsets[i + 1] : (contactPairs_t)nContacts) -
                                   startOffset;
            typeStartCountPatchMap[host_existingTypes[i]] = {startOffset, count};
        }
        
        delete[] host_existingTypes;
        delete[] host_typeOffsets;
        
        scratchPad.finishUsingTempVector("typeCounts");
        scratchPad.finishUsingTempVector("existingTypes");
        scratchPad.finishUsingTempVector("typeOffsets");
        scratchPad.finishUsingDualStruct("numExistingTypes");
    } else {
        // No contacts, reset the map
        typeStartCountPatchMap.SetAll({0, 0});
    }

    *scratchPad.numPrevContacts = nContacts;
    // If nSpheresGM is updated, then it should have been taken care of in the init/populate array phase and in kT's
    // simParams now
    *scratchPad.numPrevSpheres = simParams->nSpheresGM;
    *scratchPad.numPrevTriangles = simParams->nTriGM;
    // dT kT may send these numbers to each other from device
    scratchPad.numPrevContacts.toDevice();
    scratchPad.numPrevSpheres.toDevice();
    scratchPad.numPrevTriangles.toDevice();
}

}  // namespace deme
