//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cub/cub.cuh>
// #include <thrust/sort.h>
#include <core/utils/JitHelper.h>
#include <helper_math.cuh>
#include <core/utils/Macros.h>

#include <algorithms/DEMCubBasedSubroutines.h>
#include <DEM/HostSideHelpers.cpp>

#include <algorithms/DEMCubWrappers.cu>

#include <core/utils/GpuError.h>

namespace sgps {

inline void contactEventArraysResize(size_t nContactPairs,
                                     std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& idGeometryA,
                                     std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& idGeometryB,
                                     std::vector<contact_t, ManagedAllocator<contact_t>>& contactType,
                                     DEMDataKT* granData) {
    // TODO: not tracked? Gotta do something on it
    // TRACKED_QUICK_VECTOR_RESIZE(idGeometryA, nContactPairs);
    // TRACKED_QUICK_VECTOR_RESIZE(idGeometryB, nContactPairs);
    // TRACKED_QUICK_VECTOR_RESIZE(contactType, nContactPairs);
    idGeometryA.resize(nContactPairs);
    idGeometryB.resize(nContactPairs);
    contactType.resize(nContactPairs);

    // Re-pack pointers in case the arrays got reallocated
    granData->idGeometryA = idGeometryA.data();
    granData->idGeometryB = idGeometryB.data();
    granData->contactType = contactType.data();
}

void contactDetection(std::shared_ptr<jitify::Program>& bin_occupation_kernels,
                      std::shared_ptr<jitify::Program>& contact_detection_kernels,
                      std::shared_ptr<jitify::Program>& history_kernels,
                      DEMDataKT* granData,
                      DEMSimParams* simParams,
                      SolverFlags& solverFlags,
                      DEM_VERBOSITY& verbosity,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& idGeometryA,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& idGeometryB,
                      std::vector<contact_t, ManagedAllocator<contact_t>>& contactType,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& previous_idGeometryA,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& previous_idGeometryB,
                      std::vector<contact_t, ManagedAllocator<contact_t>>& previous_contactType,
                      std::vector<contactPairs_t, ManagedAllocator<contactPairs_t>>& contactMapping,
                      cudaStream_t& this_stream,
                      DEMSolverStateDataKT& scratchPad) {
    // total bytes needed for temp arrays in contact detection
    size_t CD_temp_arr_bytes = 0;

    // 1st step: register the number of sphere--bin touching pairs for each sphere for further processing
    CD_temp_arr_bytes = simParams->nSpheresGM * sizeof(binsSphereTouches_t);
    binsSphereTouches_t* numBinsSphereTouches = (binsSphereTouches_t*)scratchPad.allocateTempVector1(CD_temp_arr_bytes);
    // This kernel is also tasked to find how many analytical objects each sphere touches
    // We'll use a new vector 3 to store this
    CD_temp_arr_bytes = simParams->nSpheresGM * sizeof(objID_t);
    objID_t* numAnalGeoSphereTouches = (objID_t*)scratchPad.allocateTempVector3(CD_temp_arr_bytes);
    size_t blocks_needed_for_bodies =
        (simParams->nSpheresGM + SGPS_DEM_NUM_BODIES_PER_BLOCK - 1) / SGPS_DEM_NUM_BODIES_PER_BLOCK;

    bin_occupation_kernels->kernel("getNumberOfBinsEachSphereTouches")
        .instantiate()
        .configure(dim3(blocks_needed_for_bodies), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(granData, numBinsSphereTouches, numAnalGeoSphereTouches);
    GPU_CALL(cudaStreamSynchronize(this_stream));

    // 2nd step: prefix scan sphere--bin touching pairs
    CD_temp_arr_bytes = simParams->nSpheresGM * sizeof(binSphereTouchPairs_t);
    binSphereTouchPairs_t* numBinsSphereTouchesScan =
        (binSphereTouchPairs_t*)scratchPad.allocateTempVector2(CD_temp_arr_bytes);
    cubDEMPrefixScan<binsSphereTouches_t, binSphereTouchPairs_t, DEMSolverStateDataKT>(
        numBinsSphereTouches, numBinsSphereTouchesScan, simParams->nSpheresGM, this_stream, scratchPad);
    size_t* pNumBinSphereTouchPairs = scratchPad.pTempSizeVar1;
    *pNumBinSphereTouchPairs = (size_t)numBinsSphereTouchesScan[simParams->nSpheresGM - 1] +
                               (size_t)numBinsSphereTouches[simParams->nSpheresGM - 1];
    // The same process is done for sphere--analytical geometry pairs as well. Use vector 4 for this.
    CD_temp_arr_bytes = simParams->nSpheresGM * sizeof(binSphereTouchPairs_t);
    binSphereTouchPairs_t* numAnalGeoSphereTouchesScan =
        (binSphereTouchPairs_t*)scratchPad.allocateTempVector4(CD_temp_arr_bytes);
    cubDEMPrefixScan<objID_t, binSphereTouchPairs_t, DEMSolverStateDataKT>(
        numAnalGeoSphereTouches, numAnalGeoSphereTouchesScan, simParams->nSpheresGM, this_stream, scratchPad);
    *(scratchPad.pNumContacts) = (size_t)numAnalGeoSphereTouches[simParams->nSpheresGM - 1] +
                                 (size_t)numAnalGeoSphereTouchesScan[simParams->nSpheresGM - 1];
    if (*scratchPad.pNumContacts > idGeometryA.size()) {
        contactEventArraysResize(*scratchPad.pNumContacts, idGeometryA, idGeometryB, contactType, granData);
    }
    // std::cout << *pNumBinSphereTouchPairs << std::endl;
    // displayArray<binsSphereTouches_t>(numBinsSphereTouches, simParams->nSpheresGM);
    // displayArray<binSphereTouchPairs_t>(numBinsSphereTouchesScan, simParams->nSpheresGM);

    // 3rd step: use a custom kernel to figure out all sphere--bin touching pairs. Note numBinsSphereTouches can retire
    // now so we allocate on temp vector 1 and re-use vector 3.
    CD_temp_arr_bytes = (*pNumBinSphereTouchPairs) * sizeof(binID_t);
    binID_t* binIDsEachSphereTouches = (binID_t*)scratchPad.allocateTempVector1(CD_temp_arr_bytes);
    CD_temp_arr_bytes = (*pNumBinSphereTouchPairs) * sizeof(bodyID_t);
    bodyID_t* sphereIDsEachBinTouches = (bodyID_t*)scratchPad.allocateTempVector3(CD_temp_arr_bytes);
    // This kernel is also responsible of figuring out sphere--analytical geometry pairs
    bin_occupation_kernels->kernel("populateBinSphereTouchingPairs")
        .instantiate()
        .configure(dim3(blocks_needed_for_bodies), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(granData, numBinsSphereTouchesScan, numAnalGeoSphereTouchesScan, binIDsEachSphereTouches,
                sphereIDsEachBinTouches, granData->idGeometryA, granData->idGeometryB, granData->contactType);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    // std::cout << "Unsorted bin IDs: ";
    // displayArray<binID_t>(binIDsEachSphereTouches, *pNumBinSphereTouchPairs);
    // std::cout << "Corresponding sphere IDs: ";
    // displayArray<bodyID_t>(sphereIDsEachBinTouches, *pNumBinSphereTouchPairs);

    // 4th step: allocate and populate SORTED binIDsEachSphereTouches and sphereIDsEachBinTouches. Note
    // numBinsSphereTouchesScan can retire now so we allocate on vector 2 and re-use vector 4.
    CD_temp_arr_bytes = (*pNumBinSphereTouchPairs) * sizeof(bodyID_t);
    bodyID_t* sphereIDsEachBinTouches_sorted = (bodyID_t*)scratchPad.allocateTempVector2(CD_temp_arr_bytes);
    CD_temp_arr_bytes = (*pNumBinSphereTouchPairs) * sizeof(binID_t);
    binID_t* binIDsEachSphereTouches_sorted = (binID_t*)scratchPad.allocateTempVector4(CD_temp_arr_bytes);
    // hostSortByKey<binID_t, bodyID_t>(granData->binIDsEachSphereTouches, granData->sphereIDsEachBinTouches,
    //                                  *pNumBinSphereTouchPairs);
    cubDEMSortByKeys<binID_t, bodyID_t, DEMSolverStateDataKT>(binIDsEachSphereTouches, binIDsEachSphereTouches_sorted,
                                                              sphereIDsEachBinTouches, sphereIDsEachBinTouches_sorted,
                                                              *pNumBinSphereTouchPairs, this_stream, scratchPad);
    // std::cout << "Sorted bin IDs: ";
    // displayArray<binID_t>(binIDsEachSphereTouches_sorted, *pNumBinSphereTouchPairs);
    // std::cout << "Corresponding sphere IDs: ";
    // displayArray<bodyID_t>(sphereIDsEachBinTouches_sorted, *pNumBinSphereTouchPairs);

    // 5th step: use DeviceRunLengthEncode to identify those active (that have bodies in them) bins.
    // Also, binIDsEachSphereTouches is large enough for a unique scan because total sphere--bin pairs are more than
    // active bins.
    binID_t* binIDsUnique = (binID_t*)binIDsEachSphereTouches;
    size_t* pNumActiveBins = scratchPad.pTempSizeVar2;
    cubDEMUnique<binID_t, DEMSolverStateDataKT>(binIDsEachSphereTouches_sorted, binIDsUnique, pNumActiveBins,
                                                *pNumBinSphereTouchPairs, this_stream, scratchPad);
    // Allocate space for encoding output, and run it. Note the (unsorted) binIDsEachSphereTouches and
    // sphereIDsEachBinTouches can retire now, so we allocate on temp vectors 1 and 3.
    CD_temp_arr_bytes = (*pNumActiveBins) * sizeof(binID_t);
    binID_t* activeBinIDs = (binID_t*)scratchPad.allocateTempVector1(CD_temp_arr_bytes);
    CD_temp_arr_bytes = (*pNumActiveBins) * sizeof(spheresBinTouches_t);
    spheresBinTouches_t* numSpheresBinTouches = (spheresBinTouches_t*)scratchPad.allocateTempVector3(CD_temp_arr_bytes);
    cubDEMRunLengthEncode<binID_t, spheresBinTouches_t, DEMSolverStateDataKT>(
        binIDsEachSphereTouches_sorted, activeBinIDs, numSpheresBinTouches, pNumActiveBins, *pNumBinSphereTouchPairs,
        this_stream, scratchPad);
    // std::cout << "numActiveBins: " << *pNumActiveBins << std::endl;
    // std::cout << "activeBinIDs: ";
    // displayArray<binID_t>(activeBinIDs, *pNumActiveBins);
    // std::cout << "numSpheresBinTouches: ";
    // displayArray<spheresBinTouches_t>(numSpheresBinTouches, *pNumActiveBins);
    // std::cout << "binIDsEachSphereTouches_sorted: ";
    // displayArray<binID_t>(binIDsEachSphereTouches_sorted, *pNumBinSphereTouchPairs);

    // Then, scan to find the offsets that are used to index into sphereIDsEachBinTouches_sorted to obtain bin-wise
    // spheres. Note binIDsEachSphereTouches_sorted can retire so we allocate on temp vector 4.
    CD_temp_arr_bytes = (*pNumActiveBins) * sizeof(binSphereTouchPairs_t);
    binSphereTouchPairs_t* sphereIDsLookUpTable =
        (binSphereTouchPairs_t*)scratchPad.allocateTempVector4(CD_temp_arr_bytes);
    cubDEMPrefixScan<binsSphereTouches_t, binSphereTouchPairs_t, DEMSolverStateDataKT>(
        numSpheresBinTouches, sphereIDsLookUpTable, *pNumActiveBins, this_stream, scratchPad);
    // std::cout << "sphereIDsLookUpTable: ";
    // displayArray<binSphereTouchPairs_t>(sphereIDsLookUpTable, *pNumActiveBins);

    // 6th step: find the contact pairs. One-two punch: first find num of contacts in each bin, then prescan, then find
    // the actual pair names. A new temp array is needed for this numContactsInEachBin. Note we assume the number of
    // contact in each bin is the same level as the number of spheres in each bin (capped by the same data type).
    CD_temp_arr_bytes = (*pNumActiveBins) * sizeof(spheresBinTouches_t);
    spheresBinTouches_t* numContactsInEachBin = (spheresBinTouches_t*)scratchPad.allocateTempVector5(CD_temp_arr_bytes);
    size_t blocks_needed_for_bins = (*pNumActiveBins + SGPS_DEM_NUM_BINS_PER_BLOCK - 1) / SGPS_DEM_NUM_BINS_PER_BLOCK;
    if (blocks_needed_for_bins > 0) {
        contact_detection_kernels->kernel("getNumberOfContactsEachBin")
            .instantiate()
            .configure(dim3(blocks_needed_for_bins), dim3(SGPS_DEM_NUM_BINS_PER_BLOCK), 0, this_stream)
            .launch(granData, sphereIDsEachBinTouches_sorted, activeBinIDs, numSpheresBinTouches, sphereIDsLookUpTable,
                    numContactsInEachBin, *pNumActiveBins);
        GPU_CALL(cudaStreamSynchronize(this_stream));

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
        CD_temp_arr_bytes = (*pNumActiveBins) * sizeof(contactPairs_t);
        contactPairs_t* contactReportOffsets = (contactPairs_t*)scratchPad.allocateTempVector6(CD_temp_arr_bytes);
        cubDEMPrefixScan<spheresBinTouches_t, contactPairs_t, DEMSolverStateDataKT>(
            numContactsInEachBin, contactReportOffsets, *pNumActiveBins, this_stream, scratchPad);
        // SGPS_DEM_DEBUG_PRINTF("Contact report offsets:");
        // SGPS_DEM_DEBUG_EXEC(displayArray<contactPairs_t>(contactReportOffsets, *pNumActiveBins));
        // SGPS_DEM_DEBUG_PRINTF("Family number:");
        // SGPS_DEM_DEBUG_EXEC(displayArray<family_t>(granData->familyID, simParams->nOwnerBodies));

        // Add sphere--sphere contacts together with sphere--analytical geometry contacts
        size_t nSphereGeoContact = *scratchPad.pNumContacts;
        size_t nSphereSphereContact =
            (size_t)numContactsInEachBin[*pNumActiveBins - 1] + (size_t)contactReportOffsets[*pNumActiveBins - 1];
        *scratchPad.pNumContacts = nSphereSphereContact + nSphereGeoContact;
        if (*scratchPad.pNumContacts > idGeometryA.size()) {
            contactEventArraysResize(*scratchPad.pNumContacts, idGeometryA, idGeometryB, contactType, granData);
        }

        // Sphere--sphere contact pairs go after sphere--anal-geo contacts
        bodyID_t* idSphA = (granData->idGeometryA + nSphereGeoContact);
        bodyID_t* idSphB = (granData->idGeometryB + nSphereGeoContact);
        // In next kernel call, all contacts registered there will be sphere--sphere contacts
        GPU_CALL(cudaMemset((void*)(granData->contactType + nSphereGeoContact), DEM_SPHERE_SPHERE_CONTACT,
                            nSphereSphereContact * sizeof(contact_t)));
        // Then fill in those contacts
        contact_detection_kernels->kernel("populateContactPairsEachBin")
            .instantiate()
            .configure(dim3(blocks_needed_for_bins), dim3(SGPS_DEM_NUM_BINS_PER_BLOCK), 0, this_stream)
            .launch(granData, sphereIDsEachBinTouches_sorted, activeBinIDs, numSpheresBinTouches, sphereIDsLookUpTable,
                    contactReportOffsets, idSphA, idSphB, *pNumActiveBins);
        GPU_CALL(cudaStreamSynchronize(this_stream));

    }  // End of bin-wise contact detection subroutine

    // Now, sort idGeometryAB by their owners. Needed for identifying persistent contacts in history-based models.
    if (*scratchPad.pNumContacts > 0) {
        if ((!solverFlags.isHistoryless) || solverFlags.should_sort_pairs) {
            // All temp vectors are free now, and all of them are fairly long...
            size_t type_arr_bytes = (*scratchPad.pNumContacts) * sizeof(contact_t);
            contact_t* contactType_sorted = (contact_t*)scratchPad.allocateTempVector1(type_arr_bytes);
            size_t id_arr_bytes = (*scratchPad.pNumContacts) * sizeof(bodyID_t);
            bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector2(id_arr_bytes);
            bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector3(id_arr_bytes);

            // TODO: But do I have to SortByKey twice?? Can I zip these value arrays together??
            cubDEMSortByKeys<bodyID_t, bodyID_t, DEMSolverStateDataKT>(
                granData->idGeometryA, idA_sorted, granData->idGeometryB, idB_sorted, *scratchPad.pNumContacts,
                this_stream, scratchPad);
            cubDEMSortByKeys<bodyID_t, contact_t, DEMSolverStateDataKT>(
                granData->idGeometryA, idA_sorted, granData->contactType, contactType_sorted, *scratchPad.pNumContacts,
                this_stream, scratchPad);

            // Copy back to idGeometry arrays
            GPU_CALL(cudaMemcpy(granData->idGeometryA, idA_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
            GPU_CALL(cudaMemcpy(granData->idGeometryB, idB_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
            GPU_CALL(cudaMemcpy(granData->contactType, contactType_sorted, type_arr_bytes, cudaMemcpyDeviceToDevice));
            // SGPS_DEM_DEBUG_PRINTF("New contact IDs (A):");
            // SGPS_DEM_DEBUG_EXEC(displayArray<bodyID_t>(granData->idGeometryA, *scratchPad.pNumContacts));
            // SGPS_DEM_DEBUG_PRINTF("New contact IDs (B):");
            // SGPS_DEM_DEBUG_EXEC(displayArray<bodyID_t>(granData->idGeometryB, *scratchPad.pNumContacts));
            // SGPS_DEM_DEBUG_PRINTF("New contact types:");
            // SGPS_DEM_DEBUG_EXEC(displayArray<contact_t>(granData->contactType, *scratchPad.pNumContacts));

            // For history-based models, construct the persistent contact map
            if (!solverFlags.isHistoryless) {
                // This CD run and previous CD run could have different number of spheres in them. We pick the larger
                // number to refer in building the persistent contact map to avoid potential problems.
                size_t nSpheresSafe = (simParams->nSpheresGM > *scratchPad.pNumPrevSpheres)
                                          ? simParams->nSpheresGM
                                          : *scratchPad.pNumPrevSpheres;

                // First, identify the new and old idA run-length
                size_t run_length_bytes = nSpheresSafe * sizeof(geoSphereTouches_t);
                geoSphereTouches_t* new_idA_runlength =
                    (geoSphereTouches_t*)scratchPad.allocateTempVector1(run_length_bytes);
                size_t unique_id_bytes = nSpheresSafe * sizeof(bodyID_t);
                bodyID_t* unique_new_idA = (bodyID_t*)scratchPad.allocateTempVector2(unique_id_bytes);
                size_t* pNumUniqueNewA = scratchPad.pTempSizeVar1;
                cubDEMRunLengthEncode<bodyID_t, geoSphereTouches_t, DEMSolverStateDataKT>(
                    granData->idGeometryA, unique_new_idA, new_idA_runlength, pNumUniqueNewA, *scratchPad.pNumContacts,
                    this_stream, scratchPad);

                geoSphereTouches_t* old_idA_runlength =
                    (geoSphereTouches_t*)scratchPad.allocateTempVector3(run_length_bytes);
                bodyID_t* unique_old_idA = (bodyID_t*)scratchPad.allocateTempVector4(unique_id_bytes);
                size_t* pNumUniqueOldA = scratchPad.pTempSizeVar2;
                cubDEMRunLengthEncode<bodyID_t, geoSphereTouches_t, DEMSolverStateDataKT>(
                    granData->previous_idGeometryA, unique_old_idA, old_idA_runlength, pNumUniqueOldA,
                    *(scratchPad.pNumPrevContacts), this_stream, scratchPad);

                // Then, add zeros to run-length arrays such that even if a sphereID is not present in idA, it has a
                // place in the run-length arrays that indicates 0 run-length
                geoSphereTouches_t* new_idA_runlength_full =
                    (geoSphereTouches_t*)scratchPad.allocateTempVector5(run_length_bytes);
                geoSphereTouches_t* old_idA_runlength_full =
                    (geoSphereTouches_t*)scratchPad.allocateTempVector6(run_length_bytes);
                GPU_CALL(cudaMemset((void*)new_idA_runlength_full, 0, run_length_bytes));
                GPU_CALL(cudaMemset((void*)old_idA_runlength_full, 0, run_length_bytes));
                size_t blocks_needed_for_mapping =
                    (*pNumUniqueNewA + SGPS_DEM_MAX_THREADS_PER_BLOCK - 1) / SGPS_DEM_MAX_THREADS_PER_BLOCK;
                if (blocks_needed_for_mapping > 0) {
                    history_kernels->kernel("fillRunLengthArray")
                        .instantiate()
                        .configure(dim3(blocks_needed_for_mapping), dim3(SGPS_DEM_MAX_THREADS_PER_BLOCK), 0,
                                   this_stream)
                        .launch(new_idA_runlength_full, unique_new_idA, new_idA_runlength, *pNumUniqueNewA);
                }

                blocks_needed_for_mapping =
                    (*pNumUniqueOldA + SGPS_DEM_MAX_THREADS_PER_BLOCK - 1) / SGPS_DEM_MAX_THREADS_PER_BLOCK;
                if (blocks_needed_for_mapping > 0) {
                    history_kernels->kernel("fillRunLengthArray")
                        .instantiate()
                        .configure(dim3(blocks_needed_for_mapping), dim3(SGPS_DEM_MAX_THREADS_PER_BLOCK), 0,
                                   this_stream)
                        .launch(old_idA_runlength_full, unique_old_idA, old_idA_runlength, *pNumUniqueOldA);
                }
                // SGPS_DEM_DEBUG_PRINTF("Unique contact IDs (A):");
                // SGPS_DEM_DEBUG_EXEC(displayArray<bodyID_t>(unique_new_idA, *pNumUniqueNewA));
                // SGPS_DEM_DEBUG_PRINTF("Unique contacts run-length:");
                // SGPS_DEM_DEBUG_EXEC(displayArray<geoSphereTouches_t>(new_idA_runlength, *pNumUniqueNewA));

                // Then, prescan to find run-length offsets, in preparation for custom kernels
                size_t scanned_runlength_bytes = nSpheresSafe * sizeof(contactPairs_t);
                contactPairs_t* new_idA_scanned_runlength =
                    (contactPairs_t*)scratchPad.allocateTempVector1(scanned_runlength_bytes);
                contactPairs_t* old_idA_scanned_runlength =
                    (contactPairs_t*)scratchPad.allocateTempVector2(scanned_runlength_bytes);
                cubDEMPrefixScan<geoSphereTouches_t, contactPairs_t, DEMSolverStateDataKT>(
                    new_idA_runlength_full, new_idA_scanned_runlength, nSpheresSafe, this_stream, scratchPad);
                cubDEMPrefixScan<geoSphereTouches_t, contactPairs_t, DEMSolverStateDataKT>(
                    old_idA_runlength_full, old_idA_scanned_runlength, nSpheresSafe, this_stream, scratchPad);

                // Then, each thread will scan a sphere, if this sphere has non-zero run-length in both new and old idA,
                // manually store the mapping. This mapping's elemental values are the indices of the corresponding
                // contacts in the previous contact array.
                if (*scratchPad.pNumContacts > contactMapping.size()) {
                    contactMapping.resize(*scratchPad.pNumContacts);
                    granData->contactMapping = contactMapping.data();
                }
                blocks_needed_for_mapping =
                    (nSpheresSafe + SGPS_DEM_NUM_BODIES_PER_BLOCK - 1) / SGPS_DEM_NUM_BODIES_PER_BLOCK;
                if (blocks_needed_for_mapping > 0) {
                    history_kernels->kernel("buildPersistentMap")
                        .instantiate()
                        .configure(dim3(blocks_needed_for_mapping), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, this_stream)
                        .launch(new_idA_runlength_full, old_idA_runlength_full, new_idA_scanned_runlength,
                                old_idA_scanned_runlength, granData->contactMapping, granData, nSpheresSafe);
                }
                // SGPS_DEM_DEBUG_PRINTF("Contact mapping:");
                // SGPS_DEM_DEBUG_EXEC(displayArray<contactPairs_t>(granData->contactMapping,
                // *scratchPad.pNumContacts));

                // Finally, copy new contact array to old contact array for the record
                if (*scratchPad.pNumContacts > previous_idGeometryA.size()) {
                    previous_idGeometryA.resize(*scratchPad.pNumContacts);
                    previous_idGeometryB.resize(*scratchPad.pNumContacts);
                    previous_contactType.resize(*scratchPad.pNumContacts);

                    granData->previous_idGeometryA = previous_idGeometryA.data();
                    granData->previous_idGeometryB = previous_idGeometryB.data();
                    granData->previous_contactType = previous_contactType.data();
                }
                GPU_CALL(cudaMemcpy(granData->previous_idGeometryA, granData->idGeometryA, id_arr_bytes,
                                    cudaMemcpyDeviceToDevice));
                GPU_CALL(cudaMemcpy(granData->previous_idGeometryB, granData->idGeometryB, id_arr_bytes,
                                    cudaMemcpyDeviceToDevice));
                GPU_CALL(cudaMemcpy(granData->previous_contactType, granData->contactType, type_arr_bytes,
                                    cudaMemcpyDeviceToDevice));
            }
        }
    }  // End of contact sorting--mapping subroutine

    // Now, given the dT force kernel size, how many contacts should each thread takes care of so idA can be resonably
    // cached in shared memory?
    if (solverFlags.use_compact_force_kernel && solverFlags.should_sort_pairs) {
        // Figure out how many contacts an item in idA array typically has
        size_t unique_arr_bytes = (size_t)simParams->nSpheresGM * sizeof(bodyID_t);
        bodyID_t* unique_arr = (bodyID_t*)scratchPad.allocateTempVector1(unique_arr_bytes);
        size_t* num_unique_idA = (size_t*)scratchPad.allocateTempVector2(sizeof(size_t));
        cubDEMUnique<bodyID_t, DEMSolverStateDataKT>(granData->idGeometryA, unique_arr, num_unique_idA,
                                                     *scratchPad.pNumContacts, this_stream, scratchPad);
        double avg_cnts_per_geo =
            (*num_unique_idA > 0) ? (double)(*scratchPad.pNumContacts) / (double)(*num_unique_idA) : 0.0;

        SGPS_DEM_DEBUG_PRINTF("Average number of contacts for each geometry: %.9g", avg_cnts_per_geo);
    }

    // Finally, don't forget to store the number of contacts for the next iteration, even if there is 0 contacts (in
    // that case, mapping will not be constructed, but we don't have to worry b/c in the next iteration, simply no work
    // will be done for the old array and every contact will be new)
    *scratchPad.pNumPrevContacts = *scratchPad.pNumContacts;
    *scratchPad.pNumPrevSpheres = simParams->nSpheresGM;
}

}  // namespace sgps
