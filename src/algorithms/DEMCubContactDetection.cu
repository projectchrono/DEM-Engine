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

void contactDetection(std::shared_ptr<jitify::Program>& bin_occupation,
                      std::shared_ptr<jitify::Program>& contact_detection,
                      DEMDataKT* granData,
                      DEMSimParams* simParams,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& idGeometryA,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& idGeometryB,
                      std::vector<contact_t, ManagedAllocator<contact_t>>& contactType,
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

    bin_occupation->kernel("getNumberOfBinsEachSphereTouches")
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
    scratchPad.setNumBinSphereTouchPairs((size_t)numBinsSphereTouchesScan[simParams->nSpheresGM - 1] +
                                         (size_t)numBinsSphereTouches[simParams->nSpheresGM - 1]);
    // The same process is done for sphere--analytical geometry pairs as well. Use vector 4 for this.
    CD_temp_arr_bytes = simParams->nSpheresGM * sizeof(binSphereTouchPairs_t);
    binSphereTouchPairs_t* numAnalGeoSphereTouchesScan =
        (binSphereTouchPairs_t*)scratchPad.allocateTempVector4(CD_temp_arr_bytes);
    cubDEMPrefixScan<objID_t, binSphereTouchPairs_t, DEMSolverStateDataKT>(
        numAnalGeoSphereTouches, numAnalGeoSphereTouchesScan, simParams->nSpheresGM, this_stream, scratchPad);
    scratchPad.setNumContacts((size_t)numAnalGeoSphereTouches[simParams->nSpheresGM - 1] +
                              (size_t)numAnalGeoSphereTouchesScan[simParams->nSpheresGM - 1]);
    if (scratchPad.getNumContacts() > idGeometryA.size()) {
        contactEventArraysResize(scratchPad.getNumContacts(), idGeometryA, idGeometryB, contactType, granData);
    }
    // std::cout << scratchPad.getNumBinSphereTouchPairs() << std::endl;
    // displayArray<binsSphereTouches_t>(numBinsSphereTouches, simParams->nSpheresGM);
    // displayArray<binSphereTouchPairs_t>(numBinsSphereTouchesScan, simParams->nSpheresGM);

    // 3rd step: use a custom kernel to figure out all sphere--bin touching pairs. Note numBinsSphereTouches can retire
    // now so we allocate on temp vector 1 and re-use vector 3.
    CD_temp_arr_bytes = scratchPad.getNumBinSphereTouchPairs() * sizeof(binID_t);
    binID_t* binIDsEachSphereTouches = (binID_t*)scratchPad.allocateTempVector1(CD_temp_arr_bytes);
    CD_temp_arr_bytes = scratchPad.getNumBinSphereTouchPairs() * sizeof(bodyID_t);
    bodyID_t* sphereIDsEachBinTouches = (bodyID_t*)scratchPad.allocateTempVector3(CD_temp_arr_bytes);
    // This kernel is also responsible of figuring out sphere--analytical geometry pairs
    bin_occupation->kernel("populateBinSphereTouchingPairs")
        .instantiate()
        .configure(dim3(blocks_needed_for_bodies), dim3(SGPS_DEM_NUM_BODIES_PER_BLOCK), 0, this_stream)
        .launch(granData, numBinsSphereTouchesScan, numAnalGeoSphereTouchesScan, binIDsEachSphereTouches,
                sphereIDsEachBinTouches, granData->idGeometryA, granData->idGeometryB, granData->contactType);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    // std::cout << "idGeometryB: ";
    // displayArray<bodyID_t>(granData->idGeometryB, scratchPad.getNumContacts());
    // std::cout << "contactType: ";
    // displayArray<contact_t>(granData->contactType, scratchPad.getNumContacts());
    // std::cout << "Unsorted bin IDs: ";
    // displayArray<binID_t>(binIDsEachSphereTouches, scratchPad.getNumBinSphereTouchPairs());
    // std::cout << "Corresponding sphere IDs: ";
    // displayArray<bodyID_t>(sphereIDsEachBinTouches, scratchPad.getNumBinSphereTouchPairs());

    // 4th step: allocate and populate SORTED binIDsEachSphereTouches and sphereIDsEachBinTouches. Note
    // numBinsSphereTouchesScan can retire now so we allocate on vector 2 and re-use vector 4.
    CD_temp_arr_bytes = scratchPad.getNumBinSphereTouchPairs() * sizeof(bodyID_t);
    bodyID_t* sphereIDsEachBinTouches_sorted = (bodyID_t*)scratchPad.allocateTempVector2(CD_temp_arr_bytes);
    CD_temp_arr_bytes = scratchPad.getNumBinSphereTouchPairs() * sizeof(binID_t);
    binID_t* binIDsEachSphereTouches_sorted = (binID_t*)scratchPad.allocateTempVector4(CD_temp_arr_bytes);
    // hostSortByKey<binID_t, bodyID_t>(granData->binIDsEachSphereTouches, granData->sphereIDsEachBinTouches,
    //                                  scratchPad.getNumBinSphereTouchPairs());
    cubDEMSortByKeys<binID_t, bodyID_t, DEMSolverStateDataKT>(
        binIDsEachSphereTouches, binIDsEachSphereTouches_sorted, sphereIDsEachBinTouches,
        sphereIDsEachBinTouches_sorted, scratchPad.getNumBinSphereTouchPairs(), this_stream, scratchPad);
    // std::cout << "Sorted bin IDs: ";
    // displayArray<binID_t>(binIDsEachSphereTouches_sorted, scratchPad.getNumBinSphereTouchPairs());
    // std::cout << "Corresponding sphere IDs: ";
    // displayArray<bodyID_t>(sphereIDsEachBinTouches_sorted, scratchPad.getNumBinSphereTouchPairs());

    // 5th step: use DeviceRunLengthEncode to identify those active (that have bodies in them) bins.
    // Also, binIDsEachSphereTouches is large enough for a unique scan because total sphere--bin pairs are more than
    // active bins.
    binID_t* binIDsUnique = (binID_t*)binIDsEachSphereTouches;
    cubDEMUnique<binID_t, DEMSolverStateDataKT>(binIDsEachSphereTouches_sorted, binIDsUnique,
                                                scratchPad.getNumActiveBinsPointer(),
                                                scratchPad.getNumBinSphereTouchPairs(), this_stream, scratchPad);
    // Allocate space for encoding output, and run it. Note the (unsorted) binIDsEachSphereTouches and
    // sphereIDsEachBinTouches can retire now, so we allocate on temp vectors 1 and 3.
    CD_temp_arr_bytes = scratchPad.getNumActiveBins() * sizeof(binID_t);
    binID_t* activeBinIDs = (binID_t*)scratchPad.allocateTempVector1(CD_temp_arr_bytes);
    CD_temp_arr_bytes = scratchPad.getNumActiveBins() * sizeof(spheresBinTouches_t);
    spheresBinTouches_t* numSpheresBinTouches = (spheresBinTouches_t*)scratchPad.allocateTempVector3(CD_temp_arr_bytes);
    cubDEMRunLengthEncode<binID_t, spheresBinTouches_t, DEMSolverStateDataKT>(
        binIDsEachSphereTouches_sorted, activeBinIDs, numSpheresBinTouches, scratchPad.getNumActiveBinsPointer(),
        scratchPad.getNumBinSphereTouchPairs(), this_stream, scratchPad);
    // std::cout << "numActiveBins: " << scratchPad.getNumActiveBins() << std::endl;
    // std::cout << "activeBinIDs: ";
    // displayArray<binID_t>(activeBinIDs, scratchPad.getNumActiveBins());
    // std::cout << "numSpheresBinTouches: ";
    // displayArray<spheresBinTouches_t>(numSpheresBinTouches, scratchPad.getNumActiveBins());
    // std::cout << "binIDsEachSphereTouches_sorted: ";
    // displayArray<binID_t>(binIDsEachSphereTouches_sorted, scratchPad.getNumBinSphereTouchPairs());

    // Then, scan to find the offsets that are used to index into sphereIDsEachBinTouches_sorted to obtain bin-wise
    // spheres. Note binIDsEachSphereTouches_sorted can retire so we allocate on temp vector 4.
    CD_temp_arr_bytes = scratchPad.getNumActiveBins() * sizeof(binSphereTouchPairs_t);
    binSphereTouchPairs_t* sphereIDsLookUpTable =
        (binSphereTouchPairs_t*)scratchPad.allocateTempVector4(CD_temp_arr_bytes);
    cubDEMPrefixScan<binsSphereTouches_t, binSphereTouchPairs_t, DEMSolverStateDataKT>(
        numSpheresBinTouches, sphereIDsLookUpTable, scratchPad.getNumActiveBins(), this_stream, scratchPad);
    // std::cout << "sphereIDsLookUpTable: ";
    // displayArray<binSphereTouchPairs_t>(sphereIDsLookUpTable, scratchPad.getNumActiveBins());

    // 6th step: find the contact pairs. One-two punch: first find num of contacts in each bin, then prescan, then find
    // the actual pair names. A new temp array is needed for this numContactsInEachBin. Note we assume the number of
    // contact in each bin is the same level as the number of spheres in each bin (capped by the same data type).
    CD_temp_arr_bytes = scratchPad.getNumActiveBins() * sizeof(spheresBinTouches_t);
    spheresBinTouches_t* numContactsInEachBin = (spheresBinTouches_t*)scratchPad.allocateTempVector5(CD_temp_arr_bytes);
    size_t blocks_needed_for_bins =
        (scratchPad.getNumActiveBins() + SGPS_DEM_NUM_BINS_PER_BLOCK - 1) / SGPS_DEM_NUM_BINS_PER_BLOCK;
    if (blocks_needed_for_bins > 0) {
        contact_detection->kernel("getNumberOfContactsEachBin")
            .instantiate()
            .configure(dim3(blocks_needed_for_bins), dim3(SGPS_DEM_NUM_BINS_PER_BLOCK), 0, this_stream)
            .launch(granData, sphereIDsEachBinTouches_sorted, activeBinIDs, numSpheresBinTouches, sphereIDsLookUpTable,
                    numContactsInEachBin, scratchPad.getNumActiveBins());
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
        CD_temp_arr_bytes = scratchPad.getNumActiveBins() * sizeof(contactPairs_t);
        contactPairs_t* contactReportOffsets = (contactPairs_t*)scratchPad.allocateTempVector6(CD_temp_arr_bytes);
        cubDEMPrefixScan<spheresBinTouches_t, contactPairs_t, DEMSolverStateDataKT>(
            numContactsInEachBin, contactReportOffsets, scratchPad.getNumActiveBins(), this_stream, scratchPad);
        // displayArray<contactPairs_t>(contactReportOffsets, scratchPad.getNumActiveBins());

        // Add sphere--sphere contacts together with sphere--analytical geometry contacts
        size_t nSphereGeoContact = scratchPad.getNumContacts();
        size_t nSphereSphereContact = (size_t)numContactsInEachBin[scratchPad.getNumActiveBins() - 1] +
                                      (size_t)contactReportOffsets[scratchPad.getNumActiveBins() - 1];
        scratchPad.setNumContacts(nSphereSphereContact + nSphereGeoContact);
        if (scratchPad.getNumContacts() > idGeometryA.size()) {
            contactEventArraysResize(scratchPad.getNumContacts(), idGeometryA, idGeometryB, contactType, granData);
        }
        // std::cout << "NumContacts: " << scratchPad.getNumContacts() << std::endl;

        // Sphere--sphere contact pairs go after sphere--anal-geo contacts
        bodyID_t* idSphA = (granData->idGeometryA + nSphereGeoContact);
        bodyID_t* idSphB = (granData->idGeometryB + nSphereGeoContact);
        // In next kernel call, all contacts registered there will be sphere--sphere contacts
        GPU_CALL(cudaMemset((void*)(granData->contactType + nSphereGeoContact), DEM_SPHERE_SPHERE_CONTACT,
                            nSphereSphereContact * sizeof(contact_t)));
        // Then fill in those contacts
        contact_detection->kernel("populateContactPairsEachBin")
            .instantiate()
            .configure(dim3(blocks_needed_for_bins), dim3(SGPS_DEM_NUM_BINS_PER_BLOCK), 0, this_stream)
            .launch(granData, sphereIDsEachBinTouches_sorted, activeBinIDs, numSpheresBinTouches, sphereIDsLookUpTable,
                    contactReportOffsets, idSphA, idSphB, scratchPad.getNumActiveBins());
        GPU_CALL(cudaStreamSynchronize(this_stream));
        // displayArray<bodyID_t>(granData->idGeometryA, scratchPad.getNumContacts());
        // displayArray<bodyID_t>(granData->idGeometryB, scratchPad.getNumContacts());
    }

    // Now, sort idGeometryAB by their owners. This is to increase dT shmem use rate.
    {
        // All temp vectors are free now. But vector 2 and 4 tend to be long anyway, so we use them in this step
        // Sort based on idGeometryA. TODO: But do I have to SortByKey twice?? Can I zip these value arrays together??
        CD_temp_arr_bytes = scratchPad.getNumContacts() * sizeof(bodyID_t);
        bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector2(CD_temp_arr_bytes);
        bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector4(CD_temp_arr_bytes);
    }
}

}  // namespace sgps
