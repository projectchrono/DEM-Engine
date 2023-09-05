//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <cub/cub.cuh>
// #include <thrust/sort.h>
#include <core/utils/JitHelper.h>
#include <nvmath/helper_math.cuh>

#include <algorithms/DEMCubBasedSubroutines.h>
#include <DEM/HostSideHelpers.hpp>

#include <algorithms/DEMCubWrappers.cu>

#include <core/utils/GpuError.h>

namespace deme {

inline void contactEventArraysResize(size_t nContactPairs,
                                     std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& idGeometryA,
                                     std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& idGeometryB,
                                     std::vector<contact_t, ManagedAllocator<contact_t>>& contactType,
                                     DEMDataKT* granData) {
    //// TODO: not tracked? Gotta do something on it
    // DEME_TRACKED_RESIZE(idGeometryA, nContactPairs);
    // DEME_TRACKED_RESIZE(idGeometryB, nContactPairs);
    // DEME_TRACKED_RESIZE(contactType, nContactPairs);
    idGeometryA.resize(nContactPairs);
    idGeometryB.resize(nContactPairs);
    contactType.resize(nContactPairs);

    // Re-pack pointers in case the arrays got reallocated
    granData->idGeometryA = idGeometryA.data();
    granData->idGeometryB = idGeometryB.data();
    granData->contactType = contactType.data();
}

void contactDetection(std::shared_ptr<jitify::Program>& bin_sphere_kernels,
                      std::shared_ptr<jitify::Program>& bin_triangle_kernels,
                      std::shared_ptr<jitify::Program>& sphere_contact_kernels,
                      std::shared_ptr<jitify::Program>& sphTri_contact_kernels,
                      std::shared_ptr<jitify::Program>& history_kernels,
                      DEMDataKT* granData,
                      DEMSimParams* simParams,
                      SolverFlags& solverFlags,
                      VERBOSITY& verbosity,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& idGeometryA,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& idGeometryB,
                      std::vector<contact_t, ManagedAllocator<contact_t>>& contactType,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& previous_idGeometryA,
                      std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& previous_idGeometryB,
                      std::vector<contact_t, ManagedAllocator<contact_t>>& previous_contactType,
                      std::vector<contactPairs_t, ManagedAllocator<contactPairs_t>>& contactMapping,
                      cudaStream_t& this_stream,
                      DEMSolverStateData& scratchPad,
                      SolverTimers& timers,
                      kTStateParams& stateParams) {
    // A dumb check
    if (simParams->nSpheresGM == 0) {
        *scratchPad.pNumContacts = 0;
        *scratchPad.pNumPrevContacts = 0;
        *scratchPad.pNumPrevSpheres = 0;
        return;
    }
    // These are needed for the solver to keep tab... But you know, we may have no triangles or no contacts, so
    // initializing them is needed.
    stateParams.maxSphFoundInBin = 0;
    stateParams.maxTriFoundInBin = 0;
    stateParams.avgCntsPerSphere = 0;

    // total bytes needed for temp arrays in contact detection
    size_t CD_temp_arr_bytes = 0;

    {
        timers.GetTimer("Discretize domain").start();
        ////////////////////////////////////////////////////////////////////////////////
        // Sphere-related discretization & sphere--analytical contact detection
        ////////////////////////////////////////////////////////////////////////////////

        // 1st step: register the number of sphere--bin touching pairs for each sphere for further processing
        CD_temp_arr_bytes = simParams->nSpheresGM * sizeof(binsSphereTouches_t);
        binsSphereTouches_t* numBinsSphereTouches =
            (binsSphereTouches_t*)scratchPad.allocateTempVector(0, CD_temp_arr_bytes);
        // This kernel is also tasked to find how many analytical objects each sphere touches
        // We'll use a new vector 2 to store this
        CD_temp_arr_bytes = simParams->nSpheresGM * sizeof(objID_t);
        objID_t* numAnalGeoSphereTouches = (objID_t*)scratchPad.allocateTempVector(2, CD_temp_arr_bytes);
        size_t blocks_needed_for_bodies =
            (simParams->nSpheresGM + DEME_NUM_BODIES_PER_BLOCK - 1) / DEME_NUM_BODIES_PER_BLOCK;

        bin_sphere_kernels->kernel("getNumberOfBinsEachSphereTouches")
            .instantiate()
            .configure(dim3(blocks_needed_for_bodies), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, this_stream)
            .launch(simParams, granData, numBinsSphereTouches, numAnalGeoSphereTouches);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));

        // 2nd step: prefix scan sphere--bin touching pairs
        // The last element of this scanned array is useful: it can be used to check if the 2 sweeps reach the same
        // conclusion on bin--sph touch pairs
        CD_temp_arr_bytes = (simParams->nSpheresGM + 1) * sizeof(binSphereTouchPairs_t);
        binSphereTouchPairs_t* numBinsSphereTouchesScan =
            (binSphereTouchPairs_t*)scratchPad.allocateTempVector(1, CD_temp_arr_bytes);
        cubDEMPrefixScan<binsSphereTouches_t, binSphereTouchPairs_t, DEMSolverStateData>(
            numBinsSphereTouches, numBinsSphereTouchesScan, simParams->nSpheresGM, this_stream, scratchPad);
        size_t* pNumBinSphereTouchPairs = scratchPad.pTempSizeVar1;
        *pNumBinSphereTouchPairs = (size_t)numBinsSphereTouchesScan[simParams->nSpheresGM - 1] +
                                   (size_t)numBinsSphereTouches[simParams->nSpheresGM - 1];
        numBinsSphereTouchesScan[simParams->nSpheresGM] = *pNumBinSphereTouchPairs;
        // The same process is done for sphere--analytical geometry pairs as well. Use vector 3 for this.
        // One extra elem is used for storing the final elem in scan result.
        CD_temp_arr_bytes = (simParams->nSpheresGM + 1) * sizeof(binSphereTouchPairs_t);
        binSphereTouchPairs_t* numAnalGeoSphereTouchesScan =
            (binSphereTouchPairs_t*)scratchPad.allocateTempVector(3, CD_temp_arr_bytes);
        cubDEMPrefixScan<objID_t, binSphereTouchPairs_t, DEMSolverStateData>(
            numAnalGeoSphereTouches, numAnalGeoSphereTouchesScan, simParams->nSpheresGM, this_stream, scratchPad);
        *(scratchPad.pNumContacts) = (size_t)numAnalGeoSphereTouches[simParams->nSpheresGM - 1] +
                                     (size_t)numAnalGeoSphereTouchesScan[simParams->nSpheresGM - 1];
        numAnalGeoSphereTouchesScan[simParams->nSpheresGM] = *(scratchPad.pNumContacts);
        if (*scratchPad.pNumContacts > idGeometryA.size()) {
            contactEventArraysResize(*scratchPad.pNumContacts, idGeometryA, idGeometryB, contactType, granData);
        }
        // std::cout << *pNumBinSphereTouchPairs << std::endl;
        // displayArray<binsSphereTouches_t>(numBinsSphereTouches, simParams->nSpheresGM);
        // displayArray<binSphereTouchPairs_t>(numBinsSphereTouchesScan, simParams->nSpheresGM);

        // 3rd step: use a custom kernel to figure out all sphere--bin touching pairs. Note numBinsSphereTouches can
        // retire now so we allocate on temp vector 0 and re-use vector 2.
        CD_temp_arr_bytes = (*pNumBinSphereTouchPairs) * sizeof(binID_t);
        binID_t* binIDsEachSphereTouches = (binID_t*)scratchPad.allocateTempVector(0, CD_temp_arr_bytes);
        CD_temp_arr_bytes = (*pNumBinSphereTouchPairs) * sizeof(bodyID_t);
        bodyID_t* sphereIDsEachBinTouches = (bodyID_t*)scratchPad.allocateTempVector(2, CD_temp_arr_bytes);
        // This kernel is also responsible of figuring out sphere--analytical geometry pairs
        bin_sphere_kernels->kernel("populateBinSphereTouchingPairs")
            .instantiate()
            .configure(dim3(blocks_needed_for_bodies), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, this_stream)
            .launch(simParams, granData, numBinsSphereTouchesScan, numAnalGeoSphereTouchesScan, binIDsEachSphereTouches,
                    sphereIDsEachBinTouches, granData->idGeometryA, granData->idGeometryB, granData->contactType);
        DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
        // std::cout << "Unsorted bin IDs: ";
        // displayArray<binID_t>(binIDsEachSphereTouches, *pNumBinSphereTouchPairs);
        // std::cout << "Corresponding sphere IDs: ";
        // displayArray<bodyID_t>(sphereIDsEachBinTouches, *pNumBinSphereTouchPairs);

        // 4th step: allocate and populate SORTED binIDsEachSphereTouches and sphereIDsEachBinTouches. Note
        // numBinsSphereTouchesScan can retire now so we re-use vector 1 and 3 (analytical contacts have been
        // processed).
        CD_temp_arr_bytes = (*pNumBinSphereTouchPairs) * sizeof(bodyID_t);
        bodyID_t* sphereIDsEachBinTouches_sorted = (bodyID_t*)scratchPad.allocateTempVector(1, CD_temp_arr_bytes);
        CD_temp_arr_bytes = (*pNumBinSphereTouchPairs) * sizeof(binID_t);
        binID_t* binIDsEachSphereTouches_sorted = (binID_t*)scratchPad.allocateTempVector(3, CD_temp_arr_bytes);
        // hostSortByKey<binID_t, bodyID_t>(granData->binIDsEachSphereTouches, granData->sphereIDsEachBinTouches,
        //                                  *pNumBinSphereTouchPairs);
        cubDEMSortByKeys<binID_t, bodyID_t, DEMSolverStateData>(binIDsEachSphereTouches, binIDsEachSphereTouches_sorted,
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
        cubDEMUnique<binID_t, DEMSolverStateData>(binIDsEachSphereTouches_sorted, binIDsUnique, pNumActiveBins,
                                                  *pNumBinSphereTouchPairs, this_stream, scratchPad);
        // Allocate space for encoding output, and run it. Note the (unsorted) binIDsEachSphereTouches and
        // sphereIDsEachBinTouches can retire now, so we allocate on temp vectors 0 and 2.
        CD_temp_arr_bytes = (*pNumActiveBins) * sizeof(binID_t);
        binID_t* activeBinIDs = (binID_t*)scratchPad.allocateTempVector(0, CD_temp_arr_bytes);
        CD_temp_arr_bytes = (*pNumActiveBins) * sizeof(spheresBinTouches_t);
        spheresBinTouches_t* numSpheresBinTouches =
            (spheresBinTouches_t*)scratchPad.allocateTempVector(2, CD_temp_arr_bytes);
        cubDEMRunLengthEncode<binID_t, spheresBinTouches_t, DEMSolverStateData>(
            binIDsEachSphereTouches_sorted, activeBinIDs, numSpheresBinTouches, pNumActiveBins,
            *pNumBinSphereTouchPairs, this_stream, scratchPad);
        // std::cout << "numActiveBins: " << *pNumActiveBins << std::endl;
        // std::cout << "activeBinIDs: ";
        // displayArray<binID_t>(activeBinIDs, *pNumActiveBins);
        // std::cout << "numSpheresBinTouches: ";
        // displayArray<spheresBinTouches_t>(numSpheresBinTouches, *pNumActiveBins);
        // std::cout << "binIDsEachSphereTouches_sorted: ";
        // displayArray<binID_t>(binIDsEachSphereTouches_sorted, *pNumBinSphereTouchPairs);

        // We find the max geo num in a bin for the purpose of adjusting bin size.
        spheresBinTouches_t* pMaxGeoInBin = (spheresBinTouches_t*)scratchPad.pTempSizeVar3;
        cubDEMMax<spheresBinTouches_t, DEMSolverStateData>(numSpheresBinTouches, pMaxGeoInBin, *pNumActiveBins,
                                                           this_stream, scratchPad);
        stateParams.maxSphFoundInBin = (size_t)(*pMaxGeoInBin);

        // Then, scan to find the offsets that are used to index into sphereIDsEachBinTouches_sorted to obtain bin-wise
        // spheres. Note binIDsEachSphereTouches_sorted can retire so we allocate on temp vector 3.
        CD_temp_arr_bytes = (*pNumActiveBins) * sizeof(binSphereTouchPairs_t);
        binSphereTouchPairs_t* sphereIDsLookUpTable =
            (binSphereTouchPairs_t*)scratchPad.allocateTempVector(3, CD_temp_arr_bytes);
        cubDEMPrefixScan<spheresBinTouches_t, binSphereTouchPairs_t, DEMSolverStateData>(
            numSpheresBinTouches, sphereIDsLookUpTable, *pNumActiveBins, this_stream, scratchPad);
        // std::cout << "sphereIDsLookUpTable: ";
        // displayArray<binSphereTouchPairs_t>(sphereIDsLookUpTable, *pNumActiveBins);

        ////////////////////////////////////////////////////////////////////////////////
        // Triangle-related discretization
        ////////////////////////////////////////////////////////////////////////////////

        // If there are meshes, they need to be processed too. All sphere--related temp arrays are in use, so we have to
        // start from 6.
        size_t* pNumActiveBinsForTri = scratchPad.pTempSizeVar1;  // TempVar1 is now free (Temp2 is not tho)
        binID_t *mapTriActBinToSphActBin, *activeBinIDsForTri;
        bodyID_t* triIDsEachBinTouches_sorted;
        trianglesBinTouches_t* numTrianglesBinTouches;
        binsTriangleTouchPairs_t* triIDsLookUpTable;
        float3 *sandwichANode1, *sandwichANode2, *sandwichANode3, *sandwichBNode1, *sandwichBNode2, *sandwichBNode3;
        if (simParams->nTriGM > 0) {
            // 0-th step: Make `sandwich' for each triangle (or say, create a prism out of each triangle). This is
            // obviously for our delayed contact detection safety. And finally, if a sphere's distance away from one of
            // the 2 prism surfaces is smaller than its radius, it has contact with this prism, hence potentially with
            // this triangle.
            CD_temp_arr_bytes = simParams->nTriGM * sizeof(float3) * 3;
            sandwichANode1 = (float3*)scratchPad.allocateTempVector(6, CD_temp_arr_bytes);
            sandwichANode2 = sandwichANode1 + simParams->nTriGM;
            sandwichANode3 = sandwichANode2 + simParams->nTriGM;
            sandwichBNode1 = (float3*)scratchPad.allocateTempVector(7, CD_temp_arr_bytes);
            sandwichBNode2 = sandwichBNode1 + simParams->nTriGM;
            sandwichBNode3 = sandwichBNode2 + simParams->nTriGM;
            size_t blocks_needed_for_tri =
                (simParams->nTriGM + DEME_NUM_TRIANGLE_PER_BLOCK - 1) / DEME_NUM_TRIANGLE_PER_BLOCK;
            bin_triangle_kernels->kernel("makeTriangleSandwich")
                .instantiate()
                .configure(dim3(blocks_needed_for_tri), dim3(DEME_NUM_TRIANGLE_PER_BLOCK), 0, this_stream)
                .launch(simParams, granData, sandwichANode1, sandwichANode2, sandwichANode3, sandwichBNode1,
                        sandwichBNode2, sandwichBNode3);
            DEME_GPU_CALL(cudaStreamSynchronize(this_stream));

            // 1st step: register the number of triangle--bin touching pairs for each triangle for further processing.
            // Because we do a `sandwich' contact detection, we are
            CD_temp_arr_bytes = simParams->nTriGM * sizeof(binsTriangleTouches_t);
            binsTriangleTouches_t* numBinsTriTouches =
                (binsTriangleTouches_t*)scratchPad.allocateTempVector(8, CD_temp_arr_bytes);
            {
                bin_triangle_kernels->kernel("getNumberOfBinsEachTriangleTouches")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_tri), dim3(DEME_NUM_TRIANGLE_PER_BLOCK), 0, this_stream)
                    .launch(simParams, granData, numBinsTriTouches, sandwichANode1, sandwichANode2, sandwichANode3,
                            sandwichBNode1, sandwichBNode2, sandwichBNode3);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            }
            // std::cout << "numBinsTriTouches: " << std::endl;
            // displayArray<binsTriangleTouches_t>(numBinsTriTouches, simParams->nTriGM);
            // displayArray<binsTriangleTouches_t>(numBinsTriTouches + simParams->nTriGM, simParams->nTriGM);

            // 2nd step: prefix scan sphere--bin touching pairs
            // The last element of this scanned array is useful: it can be used to check if the 2 sweeps reach the same
            // conclusion on bin--tri touch pairs
            CD_temp_arr_bytes = (simParams->nTriGM + 1) * sizeof(binsTriangleTouchPairs_t);
            binsTriangleTouchPairs_t* numBinsTriTouchesScan =
                (binsTriangleTouchPairs_t*)scratchPad.allocateTempVector(9, CD_temp_arr_bytes);
            cubDEMPrefixScan<binsTriangleTouches_t, binsTriangleTouchPairs_t, DEMSolverStateData>(
                numBinsTriTouches, numBinsTriTouchesScan, simParams->nTriGM, this_stream, scratchPad);
            size_t numBinTriTouchPairs =
                (size_t)numBinsTriTouchesScan[simParams->nTriGM - 1] + (size_t)numBinsTriTouches[simParams->nTriGM - 1];
            numBinsTriTouchesScan[simParams->nTriGM] =
                numBinTriTouchPairs;  // Again, this is used in populateBinTriangleTouchingPairs

            // 3rd step: use a custom kernel to figure out all sphere--bin touching pairs. Note numBinsTriTouches can
            // retire now so we allocate on temp vector 8.
            CD_temp_arr_bytes = numBinTriTouchPairs * sizeof(binID_t);
            binID_t* binIDsEachTriTouches = (binID_t*)scratchPad.allocateTempVector(8, CD_temp_arr_bytes);
            CD_temp_arr_bytes = numBinTriTouchPairs * sizeof(bodyID_t);
            bodyID_t* triIDsEachBinTouches = (bodyID_t*)scratchPad.allocateTempVector(10, CD_temp_arr_bytes);
            {
                bin_triangle_kernels->kernel("populateBinTriangleTouchingPairs")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_tri), dim3(DEME_NUM_TRIANGLE_PER_BLOCK), 0, this_stream)
                    .launch(simParams, granData, numBinsTriTouchesScan, binIDsEachTriTouches, triIDsEachBinTouches,
                            sandwichANode1, sandwichANode2, sandwichANode3, sandwichBNode1, sandwichBNode2,
                            sandwichBNode3);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            }
            // std::cout << "binIDsEachTriTouches: " << std::endl;
            // displayArray<binsTriangleTouches_t>(binIDsEachTriTouches, numBinTriTouchPairs);

            // 4th step: allocate and populate SORTED binIDsEachTriTouches and triIDsEachBinTouches. Note
            // numBinsTriTouchesScan can retire now so we re-use vector 9 and allocate 11.
            CD_temp_arr_bytes = numBinTriTouchPairs * sizeof(bodyID_t);
            triIDsEachBinTouches_sorted = (bodyID_t*)scratchPad.allocateTempVector(9, CD_temp_arr_bytes);
            CD_temp_arr_bytes = numBinTriTouchPairs * sizeof(binID_t);
            binID_t* binIDsEachTriTouches_sorted = (binID_t*)scratchPad.allocateTempVector(11, CD_temp_arr_bytes);
            cubDEMSortByKeys<binID_t, bodyID_t, DEMSolverStateData>(binIDsEachTriTouches, binIDsEachTriTouches_sorted,
                                                                    triIDsEachBinTouches, triIDsEachBinTouches_sorted,
                                                                    numBinTriTouchPairs, this_stream, scratchPad);

            // 5th step: use DeviceRunLengthEncode to identify those active (that have tris in them) bins.
            // Also, binIDsEachTriTouches is large enough for a unique scan because total sphere--bin pairs are more
            // than active bins.
            binID_t* binIDsUnique = (binID_t*)binIDsEachTriTouches;
            cubDEMUnique<binID_t, DEMSolverStateData>(binIDsEachTriTouches_sorted, binIDsUnique, pNumActiveBinsForTri,
                                                      numBinTriTouchPairs, this_stream, scratchPad);
            // Allocate space for encoding output, and run it. Note the (unsorted) binIDsEachTriTouches and
            // triIDsEachBinTouches can retire now, so we allocate on temp vectors 8 and 10.
            CD_temp_arr_bytes = (*pNumActiveBinsForTri) * sizeof(binID_t);
            activeBinIDsForTri = (binID_t*)scratchPad.allocateTempVector(8, CD_temp_arr_bytes);
            CD_temp_arr_bytes = (*pNumActiveBinsForTri) * sizeof(trianglesBinTouches_t);
            numTrianglesBinTouches = (trianglesBinTouches_t*)scratchPad.allocateTempVector(10, CD_temp_arr_bytes);
            cubDEMRunLengthEncode<binID_t, trianglesBinTouches_t, DEMSolverStateData>(
                binIDsEachTriTouches_sorted, activeBinIDsForTri, numTrianglesBinTouches, pNumActiveBinsForTri,
                numBinTriTouchPairs, this_stream, scratchPad);
            // std::cout << "activeBinIDsForTri: " << std::endl;
            // displayArray<binID_t>(activeBinIDsForTri, *pNumActiveBinsForTri);
            // std::cout << "activeBinIDsForSph: " << std::endl;
            // displayArray<binID_t>(activeBinIDs, *pNumActiveBins);
            // std::cout << "NumActiveBinsForTri: " << *pNumActiveBinsForTri << std::endl;
            // std::cout << "NumActiveBins: " << *pNumActiveBins << std::endl;

            // We find the max geo num in a bin for the purpose of adjusting bin size.
            // TempVar1 fulfilled its purpose at this point, and now it is used as a temp var.
            trianglesBinTouches_t* pMaxGeoInBin = (trianglesBinTouches_t*)scratchPad.pTempSizeVar3;
            cubDEMMax<trianglesBinTouches_t, DEMSolverStateData>(numTrianglesBinTouches, pMaxGeoInBin,
                                                                 *pNumActiveBinsForTri, this_stream, scratchPad);
            stateParams.maxTriFoundInBin = (size_t)(*pMaxGeoInBin);

            // 6th step: map activeBinIDsForTri to activeBinIDs, so that when we are processing the bins in
            // activeBinIDsForTri, we know where to find the corresponding bin that resides in activeBinIDs, to bring
            // spheres into this bin-wise contact detection sweep.
            CD_temp_arr_bytes = (*pNumActiveBinsForTri) * sizeof(binID_t);
            mapTriActBinToSphActBin = (binID_t*)scratchPad.allocateTempVector(12, CD_temp_arr_bytes);
            {
                // size_t blocks_needed_for_map =
                //     (*pNumActiveBinsForTri + DEME_KT_CD_NTHREADS_PER_BLOCK - 1) / DEME_KT_CD_NTHREADS_PER_BLOCK;
                // if (*pNumActiveBins > 0) {
                //     bin_triangle_kernels->kernel("mapTriActiveBinsToSphActiveBins")
                //         .instantiate()
                //         .configure(dim3(blocks_needed_for_map), dim3(DEME_KT_CD_NTHREADS_PER_BLOCK), 0, this_stream)
                //         .launch(activeBinIDsForTri, activeBinIDs, mapTriActBinToSphActBin, *pNumActiveBinsForTri,
                //                 *pNumActiveBins);
                //     DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
                // }
                // This `merge search' task is very unsuitable for GPU...
                hostMergeSearchMapGen(activeBinIDsForTri, activeBinIDs, mapTriActBinToSphActBin, *pNumActiveBinsForTri,
                                      *pNumActiveBins, deme::NULL_BINID);
            }
            // std::cout << "mapTriActBinToSphActBin: " << std::endl;
            // displayArray<binID_t>(mapTriActBinToSphActBin, *pNumActiveBinsForTri);

            // 7th step: scan to find the offsets that are used to index into triIDsEachBinTouches_sorted to obtain
            // bin-wise triangles. Note binIDsEachTriTouches_sorted can retire so we allocate on temp vector 11.
            CD_temp_arr_bytes = (*pNumActiveBinsForTri) * sizeof(binsTriangleTouchPairs_t);
            triIDsLookUpTable = (binsTriangleTouchPairs_t*)scratchPad.allocateTempVector(11, CD_temp_arr_bytes);
            cubDEMPrefixScan<trianglesBinTouches_t, binsTriangleTouchPairs_t, DEMSolverStateData>(
                numTrianglesBinTouches, triIDsLookUpTable, *pNumActiveBinsForTri, this_stream, scratchPad);
        }
        timers.GetTimer("Discretize domain").stop();

        ////////////////////////////////////////////////////////////////////////////////
        // Populating contact pairs
        ////////////////////////////////////////////////////////////////////////////////

        timers.GetTimer("Find contact pairs").start();
        // Final step: find the contact pairs. One-two punch: first find num of contacts in each bin, then prescan, then
        // find the actual pair names. A new temp array is needed for this numSphContactsInEachBin. Note we assume the
        // number of contact in each bin is the same level as the number of spheres in each bin (capped by the same data
        // type).
        CD_temp_arr_bytes = (*pNumActiveBins) * sizeof(binContactPairs_t);
        binContactPairs_t* numSphContactsInEachBin =
            (binContactPairs_t*)scratchPad.allocateTempVector(4, CD_temp_arr_bytes);
        size_t blocks_needed_for_bins_sph = *pNumActiveBins;
        // Some quantities and arrays for triangles as well, should we need them
        size_t blocks_needed_for_bins_tri = 0;
        // binContactPairs_t also doubles as the type for the number of tri--sph contact pairs
        binContactPairs_t* numTriSphContactsInEachBin;
        if (simParams->nTriGM > 0) {
            blocks_needed_for_bins_tri = *pNumActiveBinsForTri;
            CD_temp_arr_bytes = (*pNumActiveBinsForTri) * sizeof(binContactPairs_t);
            numTriSphContactsInEachBin = (binContactPairs_t*)scratchPad.allocateTempVector(13, CD_temp_arr_bytes);
        }

        if (blocks_needed_for_bins_sph > 0) {
            sphere_contact_kernels->kernel("getNumberOfSphereContactsEachBin")
                .instantiate()
                .configure(dim3(blocks_needed_for_bins_sph), dim3(DEME_KT_CD_NTHREADS_PER_BLOCK), 0, this_stream)
                .launch(simParams, granData, sphereIDsEachBinTouches_sorted, activeBinIDs, numSpheresBinTouches,
                        sphereIDsLookUpTable, numSphContactsInEachBin, *pNumActiveBins);
            DEME_GPU_CALL_WATCH_BETA(cudaStreamSynchronize(this_stream));

            if (blocks_needed_for_bins_tri > 0) {
                sphTri_contact_kernels->kernel("getNumberOfSphTriContactsEachBin")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_bins_tri), dim3(DEME_KT_CD_NTHREADS_PER_BLOCK), 0, this_stream)
                    .launch(simParams, granData, sphereIDsEachBinTouches_sorted, activeBinIDs, numSpheresBinTouches,
                            sphereIDsLookUpTable, mapTriActBinToSphActBin, triIDsEachBinTouches_sorted,
                            activeBinIDsForTri, numTrianglesBinTouches, triIDsLookUpTable, numTriSphContactsInEachBin,
                            sandwichANode1, sandwichANode2, sandwichANode3, sandwichBNode1, sandwichBNode2,
                            sandwichBNode3, *pNumActiveBinsForTri);
                DEME_GPU_CALL_WATCH_BETA(cudaStreamSynchronize(this_stream));
                // std::cout << "numTriSphContactsInEachBin: " << std::endl;
                // displayArray<binContactPairs_t>(numTriSphContactsInEachBin, *pNumActiveBinsForTri);
            }

            //// TODO: sphere should have jitified and non-jitified part. Use a component ID > max_comp_id to signal
            /// bringing data from global memory. / TODO: Add tri--sphere CD kernel (if mesh support is to be added).
            /// This kernel integrates tri--boundary CD. Note triangle facets can have jitified (many bodies of the same
            /// type) and non-jitified (a big meshed body) part. Use a component ID > max_comp_id to signal bringing
            /// data from global memory. / TODO: Add tri--tri CD kernel (in the far future, should mesh-rerpesented
            /// geometry to be supported). This kernel integrates tri--boundary CD. / TODO: remember that boundary types
            /// are either all jitified or non-jitified. In principal, they should be all jitified.

            // Prescan numSphContactsInEachBin to get the final sphSphContactReportOffsets and
            // triSphContactReportOffsets. New vectors are needed.
            // The extra entry is maybe superfluous and is for extra safety, in case the 2 sweeps do not agree with each
            // other.
            CD_temp_arr_bytes = (*pNumActiveBins + 1) * sizeof(contactPairs_t);
            contactPairs_t* sphSphContactReportOffsets =
                (contactPairs_t*)scratchPad.allocateTempVector(5, CD_temp_arr_bytes);
            cubDEMPrefixScan<binContactPairs_t, contactPairs_t, DEMSolverStateData>(
                numSphContactsInEachBin, sphSphContactReportOffsets, *pNumActiveBins, this_stream, scratchPad);
            contactPairs_t* triSphContactReportOffsets;
            if (simParams->nTriGM > 0) {
                CD_temp_arr_bytes = (*pNumActiveBinsForTri + 1) * sizeof(contactPairs_t);
                triSphContactReportOffsets = (contactPairs_t*)scratchPad.allocateTempVector(14, CD_temp_arr_bytes);
                cubDEMPrefixScan<binContactPairs_t, contactPairs_t, DEMSolverStateData>(
                    numTriSphContactsInEachBin, triSphContactReportOffsets, *pNumActiveBinsForTri, this_stream,
                    scratchPad);
            }
            // DEME_DEBUG_PRINTF("Num contacts each bin:");
            // DEME_DEBUG_EXEC(displayArray<binContactPairs_t>(numSphContactsInEachBin, *pNumActiveBins));
            // DEME_DEBUG_PRINTF("Tri contact report offsets:");
            // DEME_DEBUG_EXEC(displayArray<contactPairs_t>(triSphContactReportOffsets, *pNumActiveBinsForTri));
            // DEME_DEBUG_PRINTF("Family number:");
            // DEME_DEBUG_EXEC(displayArray<family_t>(granData->familyID, simParams->nOwnerBodies));

            // Add sphere--sphere contacts together with sphere--analytical geometry contacts
            size_t nSphereGeoContact = *scratchPad.pNumContacts;
            size_t nSphereSphereContact = (size_t)numSphContactsInEachBin[*pNumActiveBins - 1] +
                                          (size_t)sphSphContactReportOffsets[*pNumActiveBins - 1];
            sphSphContactReportOffsets[*pNumActiveBins] = nSphereSphereContact;

            size_t nTriSphereContact = 0;
            if (simParams->nTriGM > 0) {
                nTriSphereContact = (size_t)numTriSphContactsInEachBin[*pNumActiveBinsForTri - 1] +
                                    (size_t)triSphContactReportOffsets[*pNumActiveBinsForTri - 1];
                triSphContactReportOffsets[*pNumActiveBinsForTri] = nTriSphereContact;
            }
            // std::cout << "nSphereGeoContact: " << nSphereGeoContact << std::endl;
            // std::cout << "nSphereSphereContact: " << nSphereSphereContact << std::endl;

            *scratchPad.pNumContacts = nSphereSphereContact + nSphereGeoContact + nTriSphereContact;
            if (*scratchPad.pNumContacts > idGeometryA.size()) {
                contactEventArraysResize(*scratchPad.pNumContacts, idGeometryA, idGeometryB, contactType, granData);
            }

            // Sphere--sphere contact pairs go after sphere--anal-geo contacts
            bodyID_t* idSphA = (granData->idGeometryA + nSphereGeoContact);
            bodyID_t* idSphB = (granData->idGeometryB + nSphereGeoContact);
            contact_t* dType = (granData->contactType + nSphereGeoContact);
            // Then fill in those contacts
            sphere_contact_kernels->kernel("populateSphSphContactPairsEachBin")
                .instantiate()
                .configure(dim3(blocks_needed_for_bins_sph), dim3(DEME_KT_CD_NTHREADS_PER_BLOCK), 0, this_stream)
                .launch(simParams, granData, sphereIDsEachBinTouches_sorted, activeBinIDs, numSpheresBinTouches,
                        sphereIDsLookUpTable, sphSphContactReportOffsets, idSphA, idSphB, dType, *pNumActiveBins);
            DEME_GPU_CALL(cudaStreamSynchronize(this_stream));

            // Triangle--sphere contact pairs go after sphere--sphere contacts. Remember to mark their type.
            if (blocks_needed_for_bins_tri > 0) {
                idSphA = (granData->idGeometryA + nSphereGeoContact + nSphereSphereContact);
                bodyID_t* idTriB = (granData->idGeometryB + nSphereGeoContact + nSphereSphereContact);
                dType = (granData->contactType + nSphereGeoContact + nSphereSphereContact);
                sphTri_contact_kernels->kernel("populateTriSphContactsEachBin")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_bins_tri), dim3(DEME_KT_CD_NTHREADS_PER_BLOCK), 0, this_stream)
                    .launch(simParams, granData, sphereIDsEachBinTouches_sorted, activeBinIDs, numSpheresBinTouches,
                            sphereIDsLookUpTable, mapTriActBinToSphActBin, triIDsEachBinTouches_sorted,
                            activeBinIDsForTri, numTrianglesBinTouches, triIDsLookUpTable, triSphContactReportOffsets,
                            idSphA, idTriB, dType, sandwichANode1, sandwichANode2, sandwichANode3, sandwichBNode1,
                            sandwichBNode2, sandwichBNode3, *pNumActiveBinsForTri);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
                // std::cout << "Contacts: " << std::endl;
                // displayArray<bodyID_t>(granData->idGeometryA, *scratchPad.pNumContacts);
                // displayArray<bodyID_t>(granData->idGeometryB, *scratchPad.pNumContacts);
                // displayArray<contact_t>(granData->contactType, *scratchPad.pNumContacts);
            }
        }  // End of bin-wise contact detection subroutine
        timers.GetTimer("Find contact pairs").stop();
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Constructing contact history
    ////////////////////////////////////////////////////////////////////////////////

    timers.GetTimer("Build history map").start();
    // Now, sort idGeometryAB by their owners. Needed for identifying persistent contacts in history-based models.
    if (*scratchPad.pNumContacts > 0) {
        // All temp vectors are free now, and all of them are fairly long...
        size_t type_arr_bytes = (*scratchPad.pNumContacts) * sizeof(contact_t);
        contact_t* contactType_sorted = (contact_t*)scratchPad.allocateTempVector(0, type_arr_bytes);
        size_t id_arr_bytes = (*scratchPad.pNumContacts) * sizeof(bodyID_t);
        bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector(1, id_arr_bytes);
        bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector(2, id_arr_bytes);

        //// TODO: But do I have to SortByKey twice?? Can I zip these value arrays together??
        // Although it is stupid, do pay attention to that it does leverage the fact that RadixSort is stable.
        cubDEMSortByKeys<bodyID_t, bodyID_t, DEMSolverStateData>(granData->idGeometryA, idA_sorted,
                                                                 granData->idGeometryB, idB_sorted,
                                                                 *scratchPad.pNumContacts, this_stream, scratchPad);
        cubDEMSortByKeys<bodyID_t, contact_t, DEMSolverStateData>(granData->idGeometryA, idA_sorted,
                                                                  granData->contactType, contactType_sorted,
                                                                  *scratchPad.pNumContacts, this_stream, scratchPad);

        // Copy back to idGeometry arrays
        DEME_GPU_CALL(cudaMemcpy(granData->idGeometryA, idA_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
        DEME_GPU_CALL(cudaMemcpy(granData->idGeometryB, idB_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
        DEME_GPU_CALL(cudaMemcpy(granData->contactType, contactType_sorted, type_arr_bytes, cudaMemcpyDeviceToDevice));
        // DEME_DEBUG_PRINTF("New contact IDs (A):");
        // DEME_DEBUG_EXEC(displayArray<bodyID_t>(granData->idGeometryA, *scratchPad.pNumContacts));
        // DEME_DEBUG_PRINTF("New contact IDs (B):");
        // DEME_DEBUG_EXEC(displayArray<bodyID_t>(granData->idGeometryB, *scratchPad.pNumContacts));
        // DEME_DEBUG_PRINTF("New contact types:");
        // DEME_DEBUG_EXEC(displayArray<contact_t>(granData->contactType, *scratchPad.pNumContacts));
        // DEME_DEBUG_PRINTF("Old contact IDs (A):");
        // DEME_DEBUG_EXEC(displayArray<bodyID_t>(granData->previous_idGeometryA, *scratchPad.pNumPrevContacts));
        // DEME_DEBUG_PRINTF("Old contact IDs (B):");
        // DEME_DEBUG_EXEC(displayArray<bodyID_t>(granData->previous_idGeometryB, *scratchPad.pNumPrevContacts));
        // DEME_DEBUG_PRINTF("Old contact types:");
        // DEME_DEBUG_EXEC(displayArray<contact_t>(granData->previous_contactType, *scratchPad.pNumPrevContacts));

        // For history-based models, construct the persistent contact map. We dwell on the fact that idA is always
        // for a sphere.
        // This CD run and previous CD run could have different number of spheres in them. We pick the larger
        // number to refer in building the persistent contact map to avoid potential problems.
        size_t nSpheresSafe =
            (simParams->nSpheresGM > *scratchPad.pNumPrevSpheres) ? simParams->nSpheresGM : *scratchPad.pNumPrevSpheres;

        // First, identify the new and old idA run-length
        size_t run_length_bytes = nSpheresSafe * sizeof(geoSphereTouches_t);
        geoSphereTouches_t* new_idA_runlength = (geoSphereTouches_t*)scratchPad.allocateTempVector(0, run_length_bytes);
        size_t unique_id_bytes = nSpheresSafe * sizeof(bodyID_t);
        bodyID_t* unique_new_idA = (bodyID_t*)scratchPad.allocateTempVector(1, unique_id_bytes);
        size_t* pNumUniqueNewA = scratchPad.pTempSizeVar1;
        cubDEMRunLengthEncode<bodyID_t, geoSphereTouches_t, DEMSolverStateData>(
            granData->idGeometryA, unique_new_idA, new_idA_runlength, pNumUniqueNewA, *scratchPad.pNumContacts,
            this_stream, scratchPad);
        // Now, we do a tab-keeping job: how many contacts on average a sphere has?
        {
            // Figure out how many contacts an item in idA array typically has.
            stateParams.avgCntsPerSphere =
                (*pNumUniqueNewA > 0) ? (float)(*scratchPad.pNumContacts) / (float)(*pNumUniqueNewA) : 0.0;

            DEME_STEP_DEBUG_PRINTF("Average number of contacts for each geometry: %.7g", stateParams.avgCntsPerSphere);
            if (stateParams.avgCntsPerSphere > solverFlags.errOutAvgSphCnts) {
                DEME_ERROR(
                    "On average a sphere has %.7g contacts, more than the max allowance (%.7g).\nIf you believe "
                    "this is not abnormal, set the allowance high using SetErrorOutAvgContacts before "
                    "initialization.\nIf you think this is because dT drifting too much ahead of kT so the contact "
                    "margin added is too big, use SetCDMaxUpdateFreq to limit the max dT future drift.\nOtherwise, the "
                    "simulation may have diverged and relaxing the physics may help, such as decreasing the step size "
                    "and modifying material properties.",
                    stateParams.avgCntsPerSphere, solverFlags.errOutAvgSphCnts);
            }
        }

        // Only need to proceed if history-based
        if (!solverFlags.isHistoryless) {
            geoSphereTouches_t* old_idA_runlength =
                (geoSphereTouches_t*)scratchPad.allocateTempVector(2, run_length_bytes);
            bodyID_t* unique_old_idA = (bodyID_t*)scratchPad.allocateTempVector(3, unique_id_bytes);
            size_t* pNumUniqueOldA = scratchPad.pTempSizeVar2;
            cubDEMRunLengthEncode<bodyID_t, geoSphereTouches_t, DEMSolverStateData>(
                granData->previous_idGeometryA, unique_old_idA, old_idA_runlength, pNumUniqueOldA,
                *(scratchPad.pNumPrevContacts), this_stream, scratchPad);

            // Then, add zeros to run-length arrays such that even if a sphereID is not present in idA, it has a
            // place in the run-length arrays that indicates 0 run-length
            geoSphereTouches_t* new_idA_runlength_full =
                (geoSphereTouches_t*)scratchPad.allocateTempVector(4, run_length_bytes);
            geoSphereTouches_t* old_idA_runlength_full =
                (geoSphereTouches_t*)scratchPad.allocateTempVector(5, run_length_bytes);
            DEME_GPU_CALL(cudaMemset((void*)new_idA_runlength_full, 0, run_length_bytes));
            DEME_GPU_CALL(cudaMemset((void*)old_idA_runlength_full, 0, run_length_bytes));
            size_t blocks_needed_for_mapping =
                (*pNumUniqueNewA + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
            if (blocks_needed_for_mapping > 0) {
                history_kernels->kernel("fillRunLengthArray")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_mapping), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, this_stream)
                    .launch(new_idA_runlength_full, unique_new_idA, new_idA_runlength, *pNumUniqueNewA);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            }

            blocks_needed_for_mapping = (*pNumUniqueOldA + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
            if (blocks_needed_for_mapping > 0) {
                history_kernels->kernel("fillRunLengthArray")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_mapping), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, this_stream)
                    .launch(old_idA_runlength_full, unique_old_idA, old_idA_runlength, *pNumUniqueOldA);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            }
            // DEME_DEBUG_PRINTF("Unique contact IDs (A):");
            // DEME_DEBUG_EXEC(displayArray<bodyID_t>(unique_new_idA, *pNumUniqueNewA));
            // DEME_DEBUG_PRINTF("Unique contacts run-length:");
            // DEME_DEBUG_EXEC(displayArray<geoSphereTouches_t>(new_idA_runlength, *pNumUniqueNewA));

            // Then, prescan to find run-length offsets, in preparation for custom kernels
            size_t scanned_runlength_bytes = nSpheresSafe * sizeof(contactPairs_t);
            contactPairs_t* new_idA_scanned_runlength =
                (contactPairs_t*)scratchPad.allocateTempVector(0, scanned_runlength_bytes);
            contactPairs_t* old_idA_scanned_runlength =
                (contactPairs_t*)scratchPad.allocateTempVector(1, scanned_runlength_bytes);
            cubDEMPrefixScan<geoSphereTouches_t, contactPairs_t, DEMSolverStateData>(
                new_idA_runlength_full, new_idA_scanned_runlength, nSpheresSafe, this_stream, scratchPad);
            cubDEMPrefixScan<geoSphereTouches_t, contactPairs_t, DEMSolverStateData>(
                old_idA_runlength_full, old_idA_scanned_runlength, nSpheresSafe, this_stream, scratchPad);

            // Then, each thread will scan a sphere, if this sphere has non-zero run-length in both new and old idA,
            // manually store the mapping. This mapping's elemental values are the indices of the corresponding
            // contacts in the previous contact array.
            if (*scratchPad.pNumContacts > contactMapping.size()) {
                contactMapping.resize(*scratchPad.pNumContacts);
                granData->contactMapping = contactMapping.data();
            }
            blocks_needed_for_mapping = (nSpheresSafe + DEME_NUM_BODIES_PER_BLOCK - 1) / DEME_NUM_BODIES_PER_BLOCK;
            if (blocks_needed_for_mapping > 0) {
                history_kernels->kernel("buildPersistentMap")
                    .instantiate()
                    .configure(dim3(blocks_needed_for_mapping), dim3(DEME_NUM_BODIES_PER_BLOCK), 0, this_stream)
                    .launch(new_idA_runlength_full, old_idA_runlength_full, new_idA_scanned_runlength,
                            old_idA_scanned_runlength, granData->contactMapping, granData, nSpheresSafe);
                DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
            }
            // DEME_DEBUG_PRINTF("Contact mapping:");
            // DEME_DEBUG_EXEC(displayArray<contactPairs_t>(granData->contactMapping,
            // *scratchPad.pNumContacts));

            // One thing we need to do before storing the old contact pairs: figure out how it is mapped to the actually
            // shipped contact pair array.
            contactPairs_t* old_arr_unsort_to_sort_map;
            if (solverFlags.should_sort_pairs) {
                size_t map_arr_bytes = (*scratchPad.pNumPrevContacts) * sizeof(contactPairs_t);
                old_arr_unsort_to_sort_map = (contactPairs_t*)scratchPad.allocateTempVector(1, map_arr_bytes);
                contactPairs_t* one_to_n = (contactPairs_t*)scratchPad.allocateTempVector(0, map_arr_bytes);
                size_t blocks_needed_for_mapping =
                    (*scratchPad.pNumPrevContacts + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
                if (blocks_needed_for_mapping > 0) {
                    history_kernels->kernel("lineNumbers")
                        .instantiate()
                        .configure(dim3(blocks_needed_for_mapping), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, this_stream)
                        .launch(one_to_n, *scratchPad.pNumPrevContacts);
                    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));

                    contact_t* old_contactType_sorted = (contact_t*)scratchPad.allocateTempVector(
                        2, (*scratchPad.pNumPrevContacts) * sizeof(contact_t));
                    cubDEMSortByKeys<contact_t, contactPairs_t, DEMSolverStateData>(
                        granData->previous_contactType, old_contactType_sorted, one_to_n, old_arr_unsort_to_sort_map,
                        *scratchPad.pNumPrevContacts, this_stream, scratchPad);
                    // Now, we have `map from' info. But we need `map to' info.
                    history_kernels->kernel("convertToAndFrom")
                        .instantiate()
                        .configure(dim3(blocks_needed_for_mapping), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, this_stream)
                        .launch(old_arr_unsort_to_sort_map, one_to_n, *scratchPad.pNumPrevContacts);
                    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
                }
                // one_to_n used for temp storage; now give it back to the true mapping we wanted. And now, vector 0 is
                // in use.
                old_arr_unsort_to_sort_map = one_to_n;
            }

            // Finally, copy new contact array to old contact array for the record. Note we register old contact pairs
            // with the array sorted by A, but when supplying dT, it was sorted by contact type.
            if (*scratchPad.pNumContacts > previous_idGeometryA.size()) {
                previous_idGeometryA.resize(*scratchPad.pNumContacts);
                previous_idGeometryB.resize(*scratchPad.pNumContacts);
                previous_contactType.resize(*scratchPad.pNumContacts);

                granData->previous_idGeometryA = previous_idGeometryA.data();
                granData->previous_idGeometryB = previous_idGeometryB.data();
                granData->previous_contactType = previous_contactType.data();
            }
            DEME_GPU_CALL(cudaMemcpy(granData->previous_idGeometryA, granData->idGeometryA, id_arr_bytes,
                                     cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(granData->previous_idGeometryB, granData->idGeometryB, id_arr_bytes,
                                     cudaMemcpyDeviceToDevice));
            DEME_GPU_CALL(cudaMemcpy(granData->previous_contactType, granData->contactType, type_arr_bytes,
                                     cudaMemcpyDeviceToDevice));

            // dT potentially benefits from type-sorted contact array
            if (solverFlags.should_sort_pairs) {
                size_t type_arr_bytes = (*scratchPad.pNumContacts) * sizeof(contact_t);
                contact_t* contactType_sorted = (contact_t*)scratchPad.allocateTempVector(1, type_arr_bytes);
                size_t id_arr_bytes = (*scratchPad.pNumContacts) * sizeof(bodyID_t);
                bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector(2, id_arr_bytes);
                bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector(3, id_arr_bytes);
                size_t cnt_arr_bytes = (*scratchPad.pNumContacts) * sizeof(contactPairs_t);
                contactPairs_t* map_sorted = (contactPairs_t*)scratchPad.allocateTempVector(4, cnt_arr_bytes);

                cubDEMSortByKeys<contact_t, bodyID_t, DEMSolverStateData>(
                    granData->contactType, contactType_sorted, granData->idGeometryB, idB_sorted,
                    *scratchPad.pNumContacts, this_stream, scratchPad);
                cubDEMSortByKeys<contact_t, bodyID_t, DEMSolverStateData>(
                    granData->contactType, contactType_sorted, granData->idGeometryA, idA_sorted,
                    *scratchPad.pNumContacts, this_stream, scratchPad);
                cubDEMSortByKeys<contact_t, contactPairs_t, DEMSolverStateData>(
                    granData->contactType, contactType_sorted, granData->contactMapping, map_sorted,
                    *scratchPad.pNumContacts, this_stream, scratchPad);

                // Finally, map the mapping array so it takes into account that arrays are shipped after sorting.
                size_t blocks_needed_for_mapping =
                    (*scratchPad.pNumContacts + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
                if (blocks_needed_for_mapping > 0) {
                    history_kernels->kernel("rearrangeMapping")
                        .instantiate()
                        .configure(dim3(blocks_needed_for_mapping), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, this_stream)
                        .launch(map_sorted, old_arr_unsort_to_sort_map, *scratchPad.pNumContacts);
                    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
                }

                // Copy back to idGeometry arrays
                DEME_GPU_CALL(cudaMemcpy(granData->idGeometryA, idA_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
                DEME_GPU_CALL(cudaMemcpy(granData->idGeometryB, idB_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
                DEME_GPU_CALL(
                    cudaMemcpy(granData->contactType, contactType_sorted, type_arr_bytes, cudaMemcpyDeviceToDevice));
                DEME_GPU_CALL(
                    cudaMemcpy(granData->contactMapping, map_sorted, cnt_arr_bytes, cudaMemcpyDeviceToDevice));
            }
        } else {  // If historyless, might still want to sort based on type
            if (solverFlags.should_sort_pairs) {
                size_t type_arr_bytes = (*scratchPad.pNumContacts) * sizeof(contact_t);
                contact_t* contactType_sorted = (contact_t*)scratchPad.allocateTempVector(1, type_arr_bytes);
                size_t id_arr_bytes = (*scratchPad.pNumContacts) * sizeof(bodyID_t);
                bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector(2, id_arr_bytes);
                bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector(3, id_arr_bytes);

                cubDEMSortByKeys<contact_t, bodyID_t, DEMSolverStateData>(
                    granData->contactType, contactType_sorted, granData->idGeometryB, idB_sorted,
                    *scratchPad.pNumContacts, this_stream, scratchPad);
                cubDEMSortByKeys<contact_t, bodyID_t, DEMSolverStateData>(
                    granData->contactType, contactType_sorted, granData->idGeometryA, idA_sorted,
                    *scratchPad.pNumContacts, this_stream, scratchPad);

                // Copy back to idGeometry arrays
                DEME_GPU_CALL(cudaMemcpy(granData->idGeometryA, idA_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
                DEME_GPU_CALL(cudaMemcpy(granData->idGeometryB, idB_sorted, id_arr_bytes, cudaMemcpyDeviceToDevice));
                DEME_GPU_CALL(
                    cudaMemcpy(granData->contactType, contactType_sorted, type_arr_bytes, cudaMemcpyDeviceToDevice));
            }
        }
    }  // End of contact sorting--mapping subroutine
    timers.GetTimer("Build history map").stop();

    // Finally, don't forget to store the number of contacts for the next iteration, even if there is 0 contacts (in
    // that case, mapping will not be constructed, but we don't have to worry b/c in the next iteration, simply no work
    // will be done for the old array and every contact will be new)
    *scratchPad.pNumPrevContacts = *scratchPad.pNumContacts;
    *scratchPad.pNumPrevSpheres = simParams->nSpheresGM;
}

void overwritePrevContactArrays(DEMDataKT* kT_data,
                                DEMDataDT* dT_data,
                                std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& previous_idGeometryA,
                                std::vector<bodyID_t, ManagedAllocator<bodyID_t>>& previous_idGeometryB,
                                std::vector<contact_t, ManagedAllocator<contact_t>>& previous_contactType,
                                DEMSimParams* simParams,
                                DEMSolverStateData& scratchPad,
                                cudaStream_t& this_stream,
                                size_t nContacts) {
    // Copy to temp array for easier usage
    bodyID_t* idA = (bodyID_t*)scratchPad.allocateTempVector(0, nContacts * sizeof(bodyID_t));
    bodyID_t* idB = (bodyID_t*)scratchPad.allocateTempVector(1, nContacts * sizeof(bodyID_t));
    contact_t* cType = (contact_t*)scratchPad.allocateTempVector(2, nContacts * sizeof(contact_t));
    DEME_GPU_CALL(cudaMemcpy(idA, dT_data->idGeometryA, nContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(idB, dT_data->idGeometryB, nContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(cType, dT_data->contactType, nContacts * sizeof(contact_t), cudaMemcpyDeviceToDevice));

    // Prev contact arrays actually need to be sorted based on idA
    bodyID_t* idA_sorted = (bodyID_t*)scratchPad.allocateTempVector(3, nContacts * sizeof(bodyID_t));
    bodyID_t* idB_sorted = (bodyID_t*)scratchPad.allocateTempVector(4, nContacts * sizeof(bodyID_t));
    contact_t* cType_sorted = (contact_t*)scratchPad.allocateTempVector(5, nContacts * sizeof(contact_t));
    //// TODO: Why the CUB-based routine will just not run here? Is it related to when and where this method is called?
    /// I have to for now use the host to do the sorting.
    DEME_GPU_CALL(cudaMemcpy(idA_sorted, idA, nContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(idB_sorted, idB, nContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(cType_sorted, cType, nContacts * sizeof(contact_t), cudaMemcpyDeviceToDevice));
    hostSortByKey(idA, idB_sorted, nContacts);
    hostSortByKey(idA_sorted, cType_sorted, nContacts);
    // cubDEMSortByKeys<bodyID_t, bodyID_t, DEMSolverStateData>(idA, idA_sorted, idB, idB_sorted, nContacts,
    //                                                          this_stream, scratchPad);
    // cubDEMSortByKeys<bodyID_t, contact_t, DEMSolverStateData>(idA, idA_sorted, cType, cType_sorted, nContacts,
    //                                                           this_stream, scratchPad);

    // Finally, copy sorted user contact array to the storage
    if (nContacts > previous_idGeometryA.size()) {
        previous_idGeometryA.resize(nContacts);
        previous_idGeometryB.resize(nContacts);
        previous_contactType.resize(nContacts);

        kT_data->previous_idGeometryA = previous_idGeometryA.data();
        kT_data->previous_idGeometryB = previous_idGeometryB.data();
        kT_data->previous_contactType = previous_contactType.data();
    }
    DEME_GPU_CALL(
        cudaMemcpy(kT_data->previous_idGeometryA, idA_sorted, nContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(
        cudaMemcpy(kT_data->previous_idGeometryB, idB_sorted, nContacts * sizeof(bodyID_t), cudaMemcpyDeviceToDevice));
    DEME_GPU_CALL(cudaMemcpy(kT_data->previous_contactType, cType_sorted, nContacts * sizeof(contact_t),
                             cudaMemcpyDeviceToDevice));

    // printf("Old contact IDs (A):\n");
    // displayArray<bodyID_t>(idA_sorted, nContacts);
    // printf("Old contact IDs (B):\n");
    // displayArray<bodyID_t>(idB_sorted, nContacts);
    // printf("Old contact types:\n");
    // displayArray<contact_t>(cType_sorted, nContacts);

    *scratchPad.pNumPrevContacts = nContacts;
    // If nSpheresGM is updated, then it should have been taken care of in the init/populate array phase and in kT's
    // simParams now
    *scratchPad.pNumPrevSpheres = simParams->nSpheresGM;
}

}  // namespace deme
