// DEM contact detection-related custom kernels
#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>
#include <DEMCollisionKernels.cu>
_kernelIncludes_;

// #include <cub/block/block_load.cuh>
// #include <cub/block/block_store.cuh>
// #include <cub/block/block_reduce.cuh>
// #include <cub/block/block_scan.cuh>
// #include <cub/util_ptx.cuh>

// If clump templates are jitified, they will be below
_clumpTemplateDefs_;

inline __device__ void fillSharedMemTriangles(deme::DEMSimParams* simParams,
                                              deme::DEMDataKT* granData,
                                              const deme::spheresBinTouches_t& myThreadID,
                                              const deme::bodyID_t& triID,
                                              deme::bodyID_t* triOwnerIDs,
                                              deme::bodyID_t* triIDs,
                                              deme::family_t* triOwnerFamilies,
                                              float3* sandwichANode1,
                                              float3* sandwichANode2,
                                              float3* sandwichANode3,
                                              float3* sandwichBNode1,
                                              float3* sandwichBNode2,
                                              float3* sandwichBNode3,
                                              float3* triANode1,
                                              float3* triANode2,
                                              float3* triANode3,
                                              float3* triBNode1,
                                              float3* triBNode2,
                                              float3* triBNode3) {
    deme::bodyID_t ownerID = granData->ownerMesh[triID];
    triIDs[myThreadID] = triID;
    triOwnerIDs[myThreadID] = ownerID;
    triOwnerFamilies[myThreadID] = granData->familyID[ownerID];
    double3 ownerXYZ;
    float3 node1, node2, node3;

    voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
        ownerXYZ.x, ownerXYZ.y, ownerXYZ.z, granData->voxelID[ownerID], granData->locX[ownerID],
        granData->locY[ownerID], granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
    float myOriQw = granData->oriQw[ownerID];
    float myOriQx = granData->oriQx[ownerID];
    float myOriQy = granData->oriQy[ownerID];
    float myOriQz = granData->oriQz[ownerID];

    {
        node1 = sandwichANode1[triID];
        node2 = sandwichANode2[triID];
        node3 = sandwichANode3[triID];
        applyOriQToVector3<float, deme::oriQ_t>(node1.x, node1.y, node1.z, myOriQw, myOriQx, myOriQy, myOriQz);
        applyOriQToVector3<float, deme::oriQ_t>(node2.x, node2.y, node2.z, myOriQw, myOriQx, myOriQy, myOriQz);
        applyOriQToVector3<float, deme::oriQ_t>(node3.x, node3.y, node3.z, myOriQw, myOriQx, myOriQy, myOriQz);
        triANode1[myThreadID] = ownerXYZ + node1;
        triANode2[myThreadID] = ownerXYZ + node2;
        triANode3[myThreadID] = ownerXYZ + node3;
    }
    {
        node1 = sandwichBNode1[triID];
        node2 = sandwichBNode2[triID];
        node3 = sandwichBNode3[triID];
        applyOriQToVector3<float, deme::oriQ_t>(node1.x, node1.y, node1.z, myOriQw, myOriQx, myOriQy, myOriQz);
        applyOriQToVector3<float, deme::oriQ_t>(node2.x, node2.y, node2.z, myOriQw, myOriQx, myOriQy, myOriQz);
        applyOriQToVector3<float, deme::oriQ_t>(node3.x, node3.y, node3.z, myOriQw, myOriQx, myOriQy, myOriQz);
        triBNode1[myThreadID] = ownerXYZ + node1;
        triBNode2[myThreadID] = ownerXYZ + node2;
        triBNode3[myThreadID] = ownerXYZ + node3;
    }
}

template <typename T1, typename T2>
inline __device__ void fillSharedMemSpheres(deme::DEMSimParams* simParams,
                                            deme::DEMDataKT* granData,
                                            const deme::spheresBinTouches_t& myThreadID,
                                            const deme::bodyID_t& sphereID,
                                            deme::bodyID_t* ownerIDs,
                                            deme::bodyID_t* bodyIDs,
                                            deme::family_t* ownerFamilies,
                                            T1* radii,
                                            T2* bodyX,
                                            T2* bodyY,
                                            T2* bodyZ) {
    deme::bodyID_t ownerID = granData->ownerClumpBody[sphereID];
    bodyIDs[myThreadID] = sphereID;
    ownerIDs[myThreadID] = ownerID;
    ownerFamilies[myThreadID] = granData->familyID[ownerID];
    double ownerX, ownerY, ownerZ;
    float myRadius;
    float3 myRelPos;

    // Get my component offset info from either jitified arrays or global memory
    // Outputs myRelPos, myRadius (in CD kernels, radius needs to be expanded)
    // Use an input named exactly `sphereID' which is the id of this sphere component
    {
        _componentAcqStrat_;
        myRadius += granData->marginSize[ownerID];
    }

    voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
        ownerX, ownerY, ownerZ, granData->voxelID[ownerID], granData->locX[ownerID], granData->locY[ownerID],
        granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
    float myOriQw = granData->oriQw[ownerID];
    float myOriQx = granData->oriQx[ownerID];
    float myOriQy = granData->oriQy[ownerID];
    float myOriQz = granData->oriQz[ownerID];
    applyOriQToVector3<float, deme::oriQ_t>(myRelPos.x, myRelPos.y, myRelPos.z, myOriQw, myOriQx, myOriQy, myOriQz);
    bodyX[myThreadID] = ownerX + (double)myRelPos.x;
    bodyY[myThreadID] = ownerY + (double)myRelPos.y;
    bodyZ[myThreadID] = ownerZ + (double)myRelPos.z;
    radii[myThreadID] = myRadius;
}

__global__ void getNumberOfSphTriContactsEachBin(deme::DEMSimParams* simParams,
                                                 deme::DEMDataKT* granData,
                                                 deme::bodyID_t* sphereIDsEachBinTouches_sorted,
                                                 deme::binID_t* activeBinIDs,
                                                 deme::spheresBinTouches_t* numSpheresBinTouches,
                                                 deme::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                                 deme::binID_t* mapTriActBinToSphActBin,
                                                 deme::bodyID_t* triIDsEachBinTouches_sorted,
                                                 deme::binID_t* activeBinIDsForTri,
                                                 deme::trianglesBinTouches_t* numTrianglesBinTouches,
                                                 deme::binsTriangleTouchPairs_t* triIDsLookUpTable,
                                                 deme::binContactPairs_t* numTriSphContactsInEachBin,
                                                 float3* sandwichANode1,
                                                 float3* sandwichANode2,
                                                 float3* sandwichANode3,
                                                 float3* sandwichBNode1,
                                                 float3* sandwichBNode2,
                                                 float3* sandwichBNode3,
                                                 size_t nActiveBinsForTri) {
    // Shared storage for bodies involved in this bin. Pre-allocated so that each threads can easily use.
    __shared__ deme::bodyID_t triOwnerIDs[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ deme::bodyID_t triIDs[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ float3 triANode1[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ float3 triANode2[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ float3 triANode3[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ float3 triBNode1[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ float3 triBNode2[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ float3 triBNode3[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ deme::family_t triOwnerFamilies[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ deme::binContactPairs_t blockPairCnt;

    // typedef cub::BlockReduce<deme::binContactPairs_t, DEME_KT_CD_NTHREADS_PER_BLOCK> BlockReduceT;
    // __shared__ typename BlockReduceT::TempStorage temp_storage;

    const deme::trianglesBinTouches_t nTriInBin = numTrianglesBinTouches[blockIdx.x];
    const deme::binID_t binID = activeBinIDsForTri[blockIdx.x];
    if (threadIdx.x == 0 && nTriInBin > simParams->errOutBinTriNum) {
        DEME_ABORT_KERNEL(
            "Bin %u contains %u triangular mesh facets, exceeding maximum allowance (%u).\nIf you want the solver to "
            "run despite this, set allowance higher via SetMaxTriangleInBin before simulation starts.",
            blockIdx.x, nTriInBin, simParams->errOutBinTriNum);
    }
    const deme::spheresBinTouches_t myThreadID = threadIdx.x;
    // But what is the index of the same binID in array activeBinIDs? Well, mapTriActBinToSphActBin comes to rescure.
    const deme::binID_t indForAcqSphInfo = mapTriActBinToSphActBin[blockIdx.x];
    // If it is not an active bin from the perspective of the spheres, then we can move on
    if (indForAcqSphInfo == deme::NULL_BINID) {
        if (threadIdx.x == 0) {
            numTriSphContactsInEachBin[blockIdx.x] = 0;
        }
        return;
    }
    // Have to allocate my sphere info first (each thread handles a sphere)
    const deme::binSphereTouchPairs_t thisSphereTableEntry = sphereIDsLookUpTable[indForAcqSphInfo];
    const deme::binsTriangleTouchPairs_t thisTriTableEntry = triIDsLookUpTable[blockIdx.x];
    // No need to check if there are too many spheres, since we did in another kernel
    const deme::spheresBinTouches_t nSphInBin = numSpheresBinTouches[indForAcqSphInfo];
    if (myThreadID == 0)
        blockPairCnt = 0;
    __syncthreads();

    // This bin may have more than 128 (default) triangles, so we process it by 128-triangle batch
    for (deme::trianglesBinTouches_t processed_count = 0; processed_count < nTriInBin;
         processed_count += DEME_NUM_TRIANGLES_PER_CD_BATCH) {
        // How many threads are still active in this batch
        const deme::trianglesBinTouches_t this_batch_active_count =
            (nTriInBin - processed_count > DEME_NUM_TRIANGLES_PER_CD_BATCH) ? DEME_NUM_TRIANGLES_PER_CD_BATCH
                                                                            : nTriInBin - processed_count;
        // If I need to work on shared memory allocation
        if (myThreadID < this_batch_active_count) {
            deme::bodyID_t triID = triIDsEachBinTouches_sorted[thisTriTableEntry + processed_count + myThreadID];
            fillSharedMemTriangles(simParams, granData, myThreadID, triID, triOwnerIDs, triIDs, triOwnerFamilies,
                                   sandwichANode1, sandwichANode2, sandwichANode3, sandwichBNode1, sandwichBNode2,
                                   sandwichBNode3, triANode1, triANode2, triANode3, triBNode1, triBNode2, triBNode3);
        }
        __syncthreads();

        // We may have more spheres than threads, so we have to process by batch too
        for (deme::spheresBinTouches_t proessed_sph = 0; proessed_sph < nSphInBin;
             proessed_sph += DEME_KT_CD_NTHREADS_PER_BLOCK) {
            // Thread will go idle if no more sphere is for it
            if (proessed_sph + myThreadID < nSphInBin) {
                deme::bodyID_t sphereID =
                    sphereIDsEachBinTouches_sorted[thisSphereTableEntry + proessed_sph + myThreadID];
                deme::bodyID_t ownerID;
                deme::family_t ownerFamily;
                float myRadius;
                float3 sphXYZ;

                // Borrow it from another kernel file...
                fillSharedMemSpheres<float, float>(simParams, granData, 0, sphereID, &ownerID, &sphereID, &ownerFamily,
                                                   &myRadius, &sphXYZ.x, &sphXYZ.y, &sphXYZ.z);

                // Test contact with each triangle in shared memory
                for (deme::trianglesBinTouches_t ind = 0; ind < this_batch_active_count; ind++) {
                    // A mesh facet and a sphere may have the same owner... although it is not possible with the current
                    // implementation...
                    if (ownerID == triOwnerIDs[ind])
                        continue;

                    // Grab family number from memory
                    unsigned int maskMatID = locateMaskPair<unsigned int>(ownerFamily, triOwnerFamilies[ind]);
                    // If marked no contact, skip ths iteration
                    if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                        continue;
                    }
                    // The smaller of the two added margins is recorded...
                    float artificialMargin = (granData->familyExtraMarginSize[ownerFamily] <
                                              granData->familyExtraMarginSize[triOwnerFamilies[ind]])
                                                 ? granData->familyExtraMarginSize[ownerFamily]
                                                 : granData->familyExtraMarginSize[triOwnerFamilies[ind]];

                    float3 cntPnt, normal;
                    float depth;
                    bool in_contact_A, in_contact_B;
                    // NOTE: triangle_sphere_CD_directional, instead of triangle_sphere_CD, is in use here. This is
                    // because if the later is in use, then if a sphere is between 2 sandwiching triangles, then its
                    // potential contact with the original triangle will not be registered. At the same time, we don't
                    // want to use triangle_sphere_CD_directional for the real force calculation, and the concern is
                    // mainly "sphere near needle tip" scenario. Think about it.
                    in_contact_A = triangle_sphere_CD_directional<float3, float>(
                        triANode1[ind], triANode2[ind], triANode3[ind], sphXYZ, myRadius, normal, depth, cntPnt);
                    // If the contact is too shallow (smaller than the smaller of artificial margin, then it can be
                    // dropped to reduce the overall number of contact pairs). Note triangle_sphere_CD_directional gives
                    // negative for contacts.
                    in_contact_A = in_contact_A && (-depth > artificialMargin);

                    // And triangle B...
                    in_contact_B = triangle_sphere_CD_directional<float3, float>(
                        triBNode1[ind], triBNode2[ind], triBNode3[ind], sphXYZ, myRadius, normal, depth, cntPnt);
                    // Same treatment for B...
                    in_contact_B = in_contact_B && (-depth > artificialMargin);

                    // Note the contact point must be calculated through the original triangle, not the 2 phantom
                    // triangles; or we will have double count problems. Use the first triangle as standard.
                    if (in_contact_A || in_contact_B) {
                        snap_to_face(triANode1[ind], triANode2[ind], triANode3[ind], sphXYZ, cntPnt);
                        deme::binID_t contactPntBin = getPointBinID<deme::binID_t>(
                            cntPnt.x, cntPnt.y, cntPnt.z, simParams->binSize, simParams->nbX, simParams->nbY);
                        if (contactPntBin == binID) {
                            atomicAdd(&blockPairCnt, 1);
                        }
                    }
                }  // End of a 256-sphere sweep
            }

        }  // End of sweeping through all spheres in bin
        __syncthreads();
    }  // End of batch-wise triangle processing

    // deme::binContactPairs_t total_count = BlockReduceT(temp_storage).Sum(contact_count);
    if (myThreadID == 0) {
        numTriSphContactsInEachBin[blockIdx.x] = blockPairCnt;
    }
}

__global__ void populateTriSphContactsEachBin(deme::DEMSimParams* simParams,
                                              deme::DEMDataKT* granData,
                                              deme::bodyID_t* sphereIDsEachBinTouches_sorted,
                                              deme::binID_t* activeBinIDs,
                                              deme::spheresBinTouches_t* numSpheresBinTouches,
                                              deme::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                              deme::binID_t* mapTriActBinToSphActBin,
                                              deme::bodyID_t* triIDsEachBinTouches_sorted,
                                              deme::binID_t* activeBinIDsForTri,
                                              deme::trianglesBinTouches_t* numTrianglesBinTouches,
                                              deme::binsTriangleTouchPairs_t* triIDsLookUpTable,
                                              deme::contactPairs_t* triSphContactReportOffsets,
                                              deme::bodyID_t* idSphA,
                                              deme::bodyID_t* idTriB,
                                              deme::contact_t* dType,
                                              float3* sandwichANode1,
                                              float3* sandwichANode2,
                                              float3* sandwichANode3,
                                              float3* sandwichBNode1,
                                              float3* sandwichBNode2,
                                              float3* sandwichBNode3,
                                              size_t nActiveBinsForTri) {
    // Shared storage for bodies involved in this bin. Pre-allocated so that each threads can easily use.
    __shared__ deme::bodyID_t triOwnerIDs[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ deme::bodyID_t triIDs[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ float3 triANode1[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ float3 triANode2[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ float3 triANode3[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ float3 triBNode1[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ float3 triBNode2[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ float3 triBNode3[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ deme::family_t triOwnerFamilies[DEME_NUM_TRIANGLES_PER_CD_BATCH];
    __shared__ deme::binContactPairs_t blockPairCnt;

    const deme::trianglesBinTouches_t nTriInBin = numTrianglesBinTouches[blockIdx.x];
    const deme::spheresBinTouches_t myThreadID = threadIdx.x;
    const deme::binID_t binID = activeBinIDsForTri[blockIdx.x];
    // But what is the index of the same binID in array activeBinIDs? Well, mapTriActBinToSphActBin comes to rescure.
    const deme::binID_t indForAcqSphInfo = mapTriActBinToSphActBin[blockIdx.x];
    // If it is not an active bin from the perspective of the spheres, then we can move on
    if (indForAcqSphInfo == deme::NULL_BINID) {
        return;
    }
    const deme::binSphereTouchPairs_t thisSphereTableEntry = sphereIDsLookUpTable[indForAcqSphInfo];
    const deme::spheresBinTouches_t nSphInBin = numSpheresBinTouches[indForAcqSphInfo];
    const deme::binsTriangleTouchPairs_t thisTriTableEntry = triIDsLookUpTable[blockIdx.x];
    // Get my offset for writing back to the global arrays that contain contact pair info
    const deme::contactPairs_t myReportOffset = triSphContactReportOffsets[blockIdx.x];
    const deme::contactPairs_t myReportOffset_end = triSphContactReportOffsets[blockIdx.x + 1];
    if (myThreadID == 0)
        blockPairCnt = 0;
    __syncthreads();

    // This bin may have more than 128 (default) triangles, so we process it by 128-triangle batch
    for (deme::trianglesBinTouches_t processed_count = 0; processed_count < nTriInBin;
         processed_count += DEME_NUM_TRIANGLES_PER_CD_BATCH) {
        // How many threads are still active in this batch
        const deme::trianglesBinTouches_t this_batch_active_count =
            (nTriInBin - processed_count > DEME_NUM_TRIANGLES_PER_CD_BATCH) ? DEME_NUM_TRIANGLES_PER_CD_BATCH
                                                                            : nTriInBin - processed_count;
        // If I need to work on shared memory allocation
        if (myThreadID < this_batch_active_count) {
            deme::bodyID_t triID = triIDsEachBinTouches_sorted[thisTriTableEntry + processed_count + myThreadID];
            fillSharedMemTriangles(simParams, granData, myThreadID, triID, triOwnerIDs, triIDs, triOwnerFamilies,
                                   sandwichANode1, sandwichANode2, sandwichANode3, sandwichBNode1, sandwichBNode2,
                                   sandwichBNode3, triANode1, triANode2, triANode3, triBNode1, triBNode2, triBNode3);
        }
        __syncthreads();

        // We may have more spheres than threads, so we have to process by batch too
        for (deme::spheresBinTouches_t proessed_sph = 0; proessed_sph < nSphInBin;
             proessed_sph += DEME_KT_CD_NTHREADS_PER_BLOCK) {
            // Thread will go idle if no more sphere is for it
            if (proessed_sph + myThreadID < nSphInBin) {
                deme::bodyID_t sphereID =
                    sphereIDsEachBinTouches_sorted[thisSphereTableEntry + proessed_sph + myThreadID];
                deme::bodyID_t ownerID;
                deme::family_t ownerFamily;
                float myRadius;
                float3 sphXYZ;

                // Borrow it from another kernel file...
                fillSharedMemSpheres<float, float>(simParams, granData, 0, sphereID, &ownerID, &sphereID, &ownerFamily,
                                                   &myRadius, &sphXYZ.x, &sphXYZ.y, &sphXYZ.z);

                // Test contact with each triangle in shared memory
                for (deme::trianglesBinTouches_t ind = 0; ind < this_batch_active_count; ind++) {
                    // A mesh facet and a sphere may have the same owner... although it is not possible with the current
                    // implementation...
                    if (ownerID == triOwnerIDs[ind])
                        continue;

                    // Grab family number from memory
                    unsigned int maskMatID = locateMaskPair<unsigned int>(ownerFamily, triOwnerFamilies[ind]);
                    // If marked no contact, skip ths iteration
                    if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                        continue;
                    }
                    // The smaller of the two added margins is recorded...
                    float artificialMargin = (granData->familyExtraMarginSize[ownerFamily] <
                                              granData->familyExtraMarginSize[triOwnerFamilies[ind]])
                                                 ? granData->familyExtraMarginSize[ownerFamily]
                                                 : granData->familyExtraMarginSize[triOwnerFamilies[ind]];

                    float3 cntPnt, normal;
                    float depth;
                    bool in_contact_A, in_contact_B;
                    // NOTE: triangle_sphere_CD_directional, instead of triangle_sphere_CD, is in use here. This is
                    // because if the later is in use, then if a sphere is between 2 sandwiching triangles, then its
                    // potential contact with the original triangle will not be registered. At the same time, we don't
                    // want to use triangle_sphere_CD_directional for the real force calculation, and the concern is
                    // mainly "sphere near needle tip" scenario. Think about it.
                    in_contact_A = triangle_sphere_CD_directional<float3, float>(
                        triANode1[ind], triANode2[ind], triANode3[ind], sphXYZ, myRadius, normal, depth, cntPnt);
                    // If the contact is too shallow (smaller than the smaller of artificial margin, then it can be
                    // dropped to reduce the overall number of contact pairs). Note triangle_sphere_CD_directional gives
                    // negative for contacts.
                    in_contact_A = in_contact_A && (-depth > artificialMargin);

                    // And triangle B...
                    in_contact_B = triangle_sphere_CD_directional<float3, float>(
                        triBNode1[ind], triBNode2[ind], triBNode3[ind], sphXYZ, myRadius, normal, depth, cntPnt);
                    // Same treatment for B...
                    in_contact_B = in_contact_B && (-depth > artificialMargin);

                    // Note the contact point must be calculated through the original triangle, not the 2 phantom
                    // triangles; or we will have double count problems. Use the first triangle as standard.
                    if (in_contact_A || in_contact_B) {
                        snap_to_face(triANode1[ind], triANode2[ind], triANode3[ind], sphXYZ, cntPnt);
                        deme::binID_t contactPntBin = getPointBinID<deme::binID_t>(
                            cntPnt.x, cntPnt.y, cntPnt.z, simParams->binSize, simParams->nbX, simParams->nbY);
                        if (contactPntBin == binID) {
                            deme::contactPairs_t inBlockOffset = myReportOffset + atomicAdd(&blockPairCnt, 1);
                            if (inBlockOffset < myReportOffset_end) {
                                idSphA[inBlockOffset] = sphereID;
                                idTriB[inBlockOffset] = triIDs[ind];
                                dType[inBlockOffset] = deme::SPHERE_MESH_CONTACT;
                            }
                        }
                    }
                }  // End of a 256-sphere sweep
            }
        }  // End of sweeping through all spheres in bin
        __syncthreads();
    }  // End of batch-wise triangle processing
    // __syncthreads();

    // In practice, I've never seen non-illed contact slots that need to be resolved this way. It's purely for ultra
    // safety.
    if (threadIdx.x == 0) {
        for (deme::contactPairs_t inBlockOffset = myReportOffset + blockPairCnt; inBlockOffset < myReportOffset_end;
             inBlockOffset++) {
            dType[inBlockOffset] = deme::NOT_A_CONTACT;
        }
    }
}
