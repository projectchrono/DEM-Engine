// DEM contact detection-related custom kernels
#include <DEM/Defines.h>
#include <DEMCollisionKernels_SphTri_TriTri.cuh>
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
    deme::bodyID_t ownerID = granData->triOwnerMesh[triID];
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

    // These locations does not include the LBF offset, which is fine since we only care about relative positions here
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

    // These locations does not include the LBF offset
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

inline __device__ bool checkPrismPrismContact(deme::DEMSimParams* simParams,
                                              const float3& triANode1,
                                              const float3& triANode2,
                                              const float3& triANode3,
                                              const float3& triBNode1,
                                              const float3& triBNode2,
                                              const float3& triBNode3,
                                              const float3& triANode1_other,
                                              const float3& triANode2_other,
                                              const float3& triANode3_other,
                                              const float3& triBNode1_other,
                                              const float3& triBNode2_other,
                                              const float3& triBNode3_other) {
    // Calculate the contact point between 2 prisms, and return whether they are in contact
    bool in_contact =
        calc_prism_contact(triANode1, triANode2, triANode3, triBNode1, triBNode2, triBNode3, triANode1_other,
                           triANode2_other, triANode3_other, triBNode1_other, triBNode2_other, triBNode3_other);
    return in_contact;
}

__global__ void getNumberOfTriangleContactsEachBin(deme::DEMSimParams* simParams,
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
                                                   deme::binContactPairs_t* numTriTriContactsInEachBin,
                                                   float3* sandwichANode1,
                                                   float3* sandwichANode2,
                                                   float3* sandwichANode3,
                                                   float3* sandwichBNode1,
                                                   float3* sandwichBNode2,
                                                   float3* sandwichBNode3,
                                                   size_t nActiveBinsForTri,
                                                   bool meshUniversalContact) {
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
    __shared__ deme::binContactPairs_t blockSphTriPairCnt, blockTriTriPairCnt;

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
    // If it is not an active bin from the perspective of the spheres, then we can move on, unless mesh universal
    // contact
    if (indForAcqSphInfo == deme::NULL_BINID && !meshUniversalContact) {
        if (threadIdx.x == 0) {
            numTriSphContactsInEachBin[blockIdx.x] = 0;
        }
        return;
    }
    const deme::binsTriangleTouchPairs_t thisTriTableEntry = triIDsLookUpTable[blockIdx.x];
    // Have to allocate my sphere info first (each thread handles a sphere)
    const deme::spheresBinTouches_t nSphInBin =
        (indForAcqSphInfo == deme::NULL_BINID) ? 0 : numSpheresBinTouches[indForAcqSphInfo];
    const deme::binSphereTouchPairs_t thisSphereTableEntry =
        (nSphInBin == 0) ? 0 : sphereIDsLookUpTable[indForAcqSphInfo];

    // No need to check if there are too many spheres, since we did in another kernel
    if (myThreadID == 0) {
        blockSphTriPairCnt = 0;
        blockTriTriPairCnt = 0;
    }
    __syncthreads();

    // This bin may have more than 128 (default) triangles, so we process it by 128-triangle batch
    for (deme::trianglesBinTouches_t processed_count = 0; processed_count < nTriInBin;
         processed_count += DEME_NUM_TRIANGLES_PER_CD_BATCH) {
        // After this batch is processed, how many tris are still left to go in this bin?
        const deme::trianglesBinTouches_t leftover_count =
            (nTriInBin - processed_count > DEME_NUM_TRIANGLES_PER_CD_BATCH)
                ? nTriInBin - processed_count - DEME_NUM_TRIANGLES_PER_CD_BATCH
                : 0;
        // How many threads are still active in this batch
        const deme::trianglesBinTouches_t this_batch_active_count =
            (leftover_count > 0) ? DEME_NUM_TRIANGLES_PER_CD_BATCH : nTriInBin - processed_count;
        // If I need to work on shared memory allocation
        if (myThreadID < this_batch_active_count) {
            deme::bodyID_t triID = triIDsEachBinTouches_sorted[thisTriTableEntry + processed_count + myThreadID];
            fillSharedMemTriangles(simParams, granData, myThreadID, triID, triOwnerIDs, triIDs, triOwnerFamilies,
                                   sandwichANode1, sandwichANode2, sandwichANode3, sandwichBNode1, sandwichBNode2,
                                   sandwichBNode3, triANode1, triANode2, triANode3, triBNode1, triBNode2, triBNode3);
        }
        __syncthreads();

        // We may have more spheres than threads, so we have to process by batch too
        for (deme::spheresBinTouches_t processed_sph = 0; processed_sph < nSphInBin;
             processed_sph += DEME_KT_CD_NTHREADS_PER_BLOCK) {
            // Thread will go idle if no more sphere is for it
            if (processed_sph + myThreadID < nSphInBin) {
                deme::bodyID_t sphereID =
                    sphereIDsEachBinTouches_sorted[thisSphereTableEntry + processed_sph + myThreadID];
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
                    // NOTE: checkTriSphereOverlap_directional, instead of checkTriSphereOverlap, is in use here. This
                    // is because if the later is in use, then if a sphere is between 2 sandwiching triangles, then its
                    // potential contact with the original triangle will not be registered.
                    in_contact_A = checkTriSphereOverlap_directional<float3, float>(
                        triANode1[ind], triANode2[ind], triANode3[ind], sphXYZ, myRadius, normal, depth, cntPnt);
                    // If the contact is too shallow (smaller than the smaller of artificial margin, then it can be
                    // dropped to reduce the overall number of contact pairs, as when making sandwich, the added safety
                    // margin already includes familyExtra). Note checkTriSphereOverlap_directional gives positive
                    // number for contacts.
                    in_contact_A = in_contact_A && (depth > artificialMargin);

                    // And triangle B...
                    in_contact_B = checkTriSphereOverlap_directional<float3, float>(
                        triBNode1[ind], triBNode2[ind], triBNode3[ind], sphXYZ, myRadius, normal, depth, cntPnt);
                    // Same treatment for B...
                    in_contact_B = in_contact_B && (depth > artificialMargin);

                    // Note the contact point must be calculated through one triangle, not the 2 phantom
                    // triangles; or we will have double count problems. Use the first triangle as standard.
                    if (in_contact_A || in_contact_B) {
                        snap_to_face(triANode1[ind], triANode2[ind], triANode3[ind], sphXYZ, cntPnt);
                        deme::binID_t contactPntBin = getPointBinID<deme::binID_t>(
                            cntPnt.x, cntPnt.y, cntPnt.z, simParams->binSize, simParams->nbX, simParams->nbY);
                        if (contactPntBin == binID) {
                            atomicAdd(&blockSphTriPairCnt, 1);
                        }
                    }
                }  // End of a 256-sphere sweep
            }
        }  // End of sweeping through all spheres in bin

        // Now all sph--tri pairs are processed, so we move on to potential tri--tri pairs
        if (meshUniversalContact) {
            // We have n * (n - 1) / 2 pairs to compare. To ensure even workload, these pairs are distributed to all
            // threads.
            const unsigned int nPairsNeedHandling =
                (unsigned int)this_batch_active_count * ((unsigned int)this_batch_active_count - 1) / 2;
            // Note this distribution is not even, but we need all active threads to process the same amount of pairs,
            // so that each thread can easily know its offset
            const unsigned int nPairsEachHandles =
                (nPairsNeedHandling + DEME_KT_CD_NTHREADS_PER_BLOCK - 1) / DEME_KT_CD_NTHREADS_PER_BLOCK;

            // i, j are local tri number in bin
            unsigned int bodyA, bodyB;
            // We can stop if this thread reaches the end of all potential pairs, nPairsNeedHandling
            for (unsigned int ind = nPairsEachHandles * myThreadID;
                 ind < nPairsNeedHandling && ind < nPairsEachHandles * (myThreadID + 1); ind++) {
                recoverCntPair<unsigned int>(bodyA, bodyB, ind, this_batch_active_count);
                // For 2 bodies to be considered in contact, the contact point must be in this bin (to avoid
                // double-counting), and they do not belong to the same clump
                if (triOwnerIDs[bodyA] == triOwnerIDs[bodyB])
                    continue;

                // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
                unsigned int bodyAFamily = triOwnerFamilies[bodyA];
                unsigned int bodyBFamily = triOwnerFamilies[bodyB];
                unsigned int maskMatID = locateMaskPair<unsigned int>(bodyAFamily, bodyBFamily);
                // If marked no contact, skip ths iteration
                if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                    continue;
                }

                // Tri--tri contact does not take into account bins, as duplicates will be removed in the end
                bool in_contact = checkPrismPrismContact(
                    simParams, triANode1[bodyA], triANode2[bodyA], triANode3[bodyA], triBNode1[bodyA], triBNode2[bodyA],
                    triBNode3[bodyA], triANode1[bodyB], triANode2[bodyB], triANode3[bodyB], triBNode1[bodyB],
                    triBNode2[bodyB], triBNode3[bodyB]);

                /*
                if (in_contact && (contactPntBin != binID)) {
                    unsigned int ZZ = binID/(simParams->nbX*simParams->nbY);
                    unsigned int YY = binID%(simParams->nbX*simParams->nbY)/simParams->nbX;
                    unsigned int XX = binID%(simParams->nbX*simParams->nbY)%simParams->nbX;
                    double binLocX = (XX + 0.5) * simParams->binSize;
                    double binLocY = (YY + 0.5) * simParams->binSize;
                    double binLocZ = (ZZ + 0.5) * simParams->binSize;
                    printf("binLoc: %f, %f, %f\n", binLocX, binLocY, binLocZ);
                    printf("triANode1A: %f, %f, %f\n", triANode1[bodyA].x, triANode1[bodyA].y, triANode1[bodyA].z);
                }
                */

                if (in_contact) {
                    atomicAdd(&blockTriTriPairCnt, 1);
                }
            }

            // Take care of the left-over triangles
            if (myThreadID < this_batch_active_count) {
                for (deme::trianglesBinTouches_t i = 0; i < leftover_count; i++) {
                    deme::bodyID_t cur_ownerID, cur_bodyID;
                    float3 cur_triANode1, cur_triANode2, cur_triANode3, cur_triBNode1, cur_triBNode2, cur_triBNode3;
                    deme::family_t cur_ownerFamily;
                    {
                        const deme::trianglesBinTouches_t cur_ind =
                            processed_count + DEME_NUM_TRIANGLES_PER_CD_BATCH + i;

                        // Get the info of this tri in question here. Note this is a broadcast so should be relatively
                        // fast. And it's not really shared mem filling, just using that function to get the info.
                        deme::bodyID_t cur_triID = triIDsEachBinTouches_sorted[thisTriTableEntry + cur_ind];
                        fillSharedMemTriangles(simParams, granData, 0, cur_triID, &cur_ownerID, &cur_bodyID,
                                               &cur_ownerFamily, sandwichANode1, sandwichANode2, sandwichANode3,
                                               sandwichBNode1, sandwichBNode2, sandwichBNode3, &cur_triANode1,
                                               &cur_triANode2, &cur_triANode3, &cur_triBNode1, &cur_triBNode2,
                                               &cur_triBNode3);
                    }
                    // Then each in-shared-mem sphere compares against it. But first, check if same owner...
                    if (triOwnerIDs[myThreadID] == cur_ownerID)
                        continue;

                    // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
                    unsigned int bodyAFamily = triOwnerFamilies[myThreadID];
                    unsigned int maskMatID = locateMaskPair<unsigned int>(bodyAFamily, cur_ownerFamily);
                    // If marked no contact, skip ths iteration
                    if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                        continue;
                    }

                    // Tri--tri contact does not take into account bins, as duplicates will be removed in the end
                    bool in_contact = checkPrismPrismContact(
                        simParams, triANode1[myThreadID], triANode2[myThreadID], triANode3[myThreadID],
                        triBNode1[myThreadID], triBNode2[myThreadID], triBNode3[myThreadID], cur_triANode1,
                        cur_triANode2, cur_triANode3, cur_triBNode1, cur_triBNode2, cur_triBNode3);

                    if (in_contact) {
                        atomicAdd(&blockTriTriPairCnt, 1);
                    }
                }
            }
        }
        __syncthreads();
    }  // End of batch-wise triangle processing

    // deme::binContactPairs_t total_count = BlockReduceT(temp_storage).Sum(contact_count);
    if (myThreadID == 0) {
        numTriSphContactsInEachBin[blockIdx.x] = blockSphTriPairCnt;
        if (meshUniversalContact) {
            numTriTriContactsInEachBin[blockIdx.x] = blockTriTriPairCnt;
        }
    }
}

__global__ void populateTriangleContactsEachBin(deme::DEMSimParams* simParams,
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
                                                deme::contactPairs_t* triTriContactReportOffsets,
                                                deme::bodyID_t* idSphA_sm,
                                                deme::bodyID_t* idTriB_sm,
                                                deme::contact_t* dType_sm,
                                                deme::bodyID_t* idTriA_mm,
                                                deme::bodyID_t* idTriB_mm,
                                                deme::contact_t* dType_mm,
                                                float3* sandwichANode1,
                                                float3* sandwichANode2,
                                                float3* sandwichANode3,
                                                float3* sandwichBNode1,
                                                float3* sandwichBNode2,
                                                float3* sandwichBNode3,
                                                size_t nActiveBinsForTri,
                                                bool meshUniversalContact) {
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
    __shared__ deme::binContactPairs_t blockSphTriPairCnt, blockTriTriPairCnt;

    const deme::trianglesBinTouches_t nTriInBin = numTrianglesBinTouches[blockIdx.x];
    const deme::spheresBinTouches_t myThreadID = threadIdx.x;
    const deme::binID_t binID = activeBinIDsForTri[blockIdx.x];
    // But what is the index of the same binID in array activeBinIDs? Well, mapTriActBinToSphActBin comes to rescure.
    const deme::binID_t indForAcqSphInfo = mapTriActBinToSphActBin[blockIdx.x];
    // If it is not an active bin from the perspective of the spheres, then we can move on, unless mesh universal
    // contact
    if (indForAcqSphInfo == deme::NULL_BINID && !meshUniversalContact) {
        return;
    }
    const deme::binsTriangleTouchPairs_t thisTriTableEntry = triIDsLookUpTable[blockIdx.x];
    // Have to allocate my sphere info first (each thread handles a sphere)
    const deme::spheresBinTouches_t nSphInBin =
        (indForAcqSphInfo == deme::NULL_BINID) ? 0 : numSpheresBinTouches[indForAcqSphInfo];
    const deme::binSphereTouchPairs_t thisSphereTableEntry =
        (nSphInBin == 0) ? 0 : sphereIDsLookUpTable[indForAcqSphInfo];

    // No need to check if there are too many spheres, since we did in another kernel
    if (myThreadID == 0) {
        blockSphTriPairCnt = 0;
        blockTriTriPairCnt = 0;
    }
    const deme::contactPairs_t smReportOffset = triSphContactReportOffsets[blockIdx.x];
    const deme::contactPairs_t smReportOffset_end = triSphContactReportOffsets[blockIdx.x + 1];
    deme::contactPairs_t mmReportOffset, mmReportOffset_end;
    if (meshUniversalContact) {
        mmReportOffset = triTriContactReportOffsets[blockIdx.x];
        mmReportOffset_end = triTriContactReportOffsets[blockIdx.x + 1];
    }
    __syncthreads();

    // This bin may have more than 128 (default) triangles, so we process it by 128-triangle batch
    for (deme::trianglesBinTouches_t processed_count = 0; processed_count < nTriInBin;
         processed_count += DEME_NUM_TRIANGLES_PER_CD_BATCH) {
        // After this batch is processed, how many tris are still left to go in this bin?
        const deme::trianglesBinTouches_t leftover_count =
            (nTriInBin - processed_count > DEME_NUM_TRIANGLES_PER_CD_BATCH)
                ? nTriInBin - processed_count - DEME_NUM_TRIANGLES_PER_CD_BATCH
                : 0;
        // How many threads are still active in this batch
        const deme::trianglesBinTouches_t this_batch_active_count =
            (leftover_count > 0) ? DEME_NUM_TRIANGLES_PER_CD_BATCH : nTriInBin - processed_count;

        // If I need to work on shared memory allocation
        if (myThreadID < this_batch_active_count) {
            deme::bodyID_t triID = triIDsEachBinTouches_sorted[thisTriTableEntry + processed_count + myThreadID];
            fillSharedMemTriangles(simParams, granData, myThreadID, triID, triOwnerIDs, triIDs, triOwnerFamilies,
                                   sandwichANode1, sandwichANode2, sandwichANode3, sandwichBNode1, sandwichBNode2,
                                   sandwichBNode3, triANode1, triANode2, triANode3, triBNode1, triBNode2, triBNode3);
        }
        __syncthreads();

        // We may have more spheres than threads, so we have to process by batch too
        for (deme::spheresBinTouches_t processed_sph = 0; processed_sph < nSphInBin;
             processed_sph += DEME_KT_CD_NTHREADS_PER_BLOCK) {
            // Thread will go idle if no more sphere is for it
            if (processed_sph + myThreadID < nSphInBin) {
                deme::bodyID_t sphereID =
                    sphereIDsEachBinTouches_sorted[thisSphereTableEntry + processed_sph + myThreadID];
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
                    // NOTE: checkTriSphereOverlap_directional, instead of checkTriSphereOverlap, is in use here. This
                    // is because if the later is in use, then if a sphere is between 2 sandwiching triangles, then its
                    // potential contact with the original triangle will not be registered.
                    in_contact_A = checkTriSphereOverlap_directional<float3, float>(
                        triANode1[ind], triANode2[ind], triANode3[ind], sphXYZ, myRadius, normal, depth, cntPnt);
                    // If the contact is too shallow (smaller than the smaller of artificial margin, then it can be
                    // dropped to reduce the overall number of contact pairs, as when making sandwich, the added safety
                    // margin already includes familyExtra). Note checkTriSphereOverlap_directional gives positive
                    // number for contacts.
                    in_contact_A = in_contact_A && (depth > artificialMargin);

                    // And triangle B...
                    in_contact_B = checkTriSphereOverlap_directional<float3, float>(
                        triBNode1[ind], triBNode2[ind], triBNode3[ind], sphXYZ, myRadius, normal, depth, cntPnt);
                    // Same treatment for B...
                    in_contact_B = in_contact_B && (depth > artificialMargin);

                    // Note the contact point must be calculated through one triangle, not the 2 phantom
                    // triangles; or we will have double count problems. Use the first triangle as standard.
                    if (in_contact_A || in_contact_B) {
                        snap_to_face(triANode1[ind], triANode2[ind], triANode3[ind], sphXYZ, cntPnt);
                        deme::binID_t contactPntBin = getPointBinID<deme::binID_t>(
                            cntPnt.x, cntPnt.y, cntPnt.z, simParams->binSize, simParams->nbX, simParams->nbY);
                        if (contactPntBin == binID) {
                            deme::contactPairs_t inBlockOffset = smReportOffset + atomicAdd(&blockSphTriPairCnt, 1);
                            if (inBlockOffset < smReportOffset_end) {
                                idSphA_sm[inBlockOffset] = sphereID;
                                idTriB_sm[inBlockOffset] = triIDs[ind];
                                dType_sm[inBlockOffset] = deme::SPHERE_TRIANGLE_CONTACT;
                            }
                        }
                    }
                }  // End of a 256-sphere sweep
            }
        }  // End of sweeping through all spheres in bin

        // Now all sph--tri pairs are processed, so we move on to potential tri--tri pairs
        if (meshUniversalContact) {
            // We have n * (n - 1) / 2 pairs to compare. To ensure even workload, these pairs are distributed to all
            // threads.
            const unsigned int nPairsNeedHandling =
                (unsigned int)this_batch_active_count * ((unsigned int)this_batch_active_count - 1) / 2;
            // Note this distribution is not even, but we need all active threads to process the same amount of pairs,
            // so that each thread can easily know its offset
            const unsigned int nPairsEachHandles =
                (nPairsNeedHandling + DEME_KT_CD_NTHREADS_PER_BLOCK - 1) / DEME_KT_CD_NTHREADS_PER_BLOCK;

            // i, j are local tri number in bin
            unsigned int bodyA, bodyB;
            // We can stop if this thread reaches the end of all potential pairs, nPairsNeedHandling
            for (unsigned int ind = nPairsEachHandles * myThreadID;
                 ind < nPairsNeedHandling && ind < nPairsEachHandles * (myThreadID + 1); ind++) {
                recoverCntPair<unsigned int>(bodyA, bodyB, ind, this_batch_active_count);
                // For 2 bodies to be considered in contact, the contact point must be in this bin (to avoid
                // double-counting), and they do not belong to the same clump
                if (triOwnerIDs[bodyA] == triOwnerIDs[bodyB])
                    continue;

                // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
                unsigned int bodyAFamily = triOwnerFamilies[bodyA];
                unsigned int bodyBFamily = triOwnerFamilies[bodyB];
                unsigned int maskMatID = locateMaskPair<unsigned int>(bodyAFamily, bodyBFamily);
                // If marked no contact, skip ths iteration
                if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                    continue;
                }

                // Tri--tri contact does not take into account bins, as duplicates will be removed in the end
                bool in_contact = checkPrismPrismContact(
                    simParams, triANode1[bodyA], triANode2[bodyA], triANode3[bodyA], triBNode1[bodyA], triBNode2[bodyA],
                    triBNode3[bodyA], triANode1[bodyB], triANode2[bodyB], triANode3[bodyB], triBNode1[bodyB],
                    triBNode2[bodyB], triBNode3[bodyB]);

                if (in_contact) {
                    deme::contactPairs_t inBlockOffset = mmReportOffset + atomicAdd(&blockTriTriPairCnt, 1);
                    // The chance of offset going out-of-bound is very low, lower than sph--bin CD step, but I put it
                    // here anyway
                    if (inBlockOffset < mmReportOffset_end) {
                        // ----------------------------------------------------------------------------
                        // IMPORTANT NOTE: Here, I did not adjust A and B ids to ensure A < B, but this is automatically
                        // ensured due to 1) The binID--triID pairs were generated with an inherent order of
                        // triID, then processed through (stable) radix sort, which preserved the blockwise order of
                        // triID; 2) Then the ordered triIDs are loaded to shared mem, and the in-kernel contact
                        // detection had threads reconstruct shared mem offsets from a recoverCntPair process, which
                        // also ensures i < j. Therefore, the generated tri contact pair has A < B. Though a change
                        // in these processes could affect the ordering, and adding a if statement here is probably more
                        // robust, I didn't do that, as a reminder of how fragile the things we built can be.
                        // ----------------------------------------------------------------------------
                        idTriA_mm[inBlockOffset] = triIDs[bodyA];
                        idTriB_mm[inBlockOffset] = triIDs[bodyB];
                        dType_mm[inBlockOffset] = deme::TRIANGLE_TRIANGLE_CONTACT;
                    }
                }
            }

            // Take care of the left-over triangles
            if (myThreadID < this_batch_active_count) {
                for (deme::trianglesBinTouches_t i = 0; i < leftover_count; i++) {
                    deme::bodyID_t cur_ownerID, cur_bodyID;
                    float3 cur_triANode1, cur_triANode2, cur_triANode3, cur_triBNode1, cur_triBNode2, cur_triBNode3;
                    deme::family_t cur_ownerFamily;
                    {
                        const deme::trianglesBinTouches_t cur_ind =
                            processed_count + DEME_NUM_TRIANGLES_PER_CD_BATCH + i;

                        // Get the info of this tri in question here. Note this is a broadcast so should be relatively
                        // fast. And it's not really shared mem filling, just using that function to get the info.
                        deme::bodyID_t cur_triID = triIDsEachBinTouches_sorted[thisTriTableEntry + cur_ind];
                        fillSharedMemTriangles(simParams, granData, 0, cur_triID, &cur_ownerID, &cur_bodyID,
                                               &cur_ownerFamily, sandwichANode1, sandwichANode2, sandwichANode3,
                                               sandwichBNode1, sandwichBNode2, sandwichBNode3, &cur_triANode1,
                                               &cur_triANode2, &cur_triANode3, &cur_triBNode1, &cur_triBNode2,
                                               &cur_triBNode3);
                    }
                    // Then each in-shared-mem sphere compares against it. But first, check if same owner...
                    if (triOwnerIDs[myThreadID] == cur_ownerID)
                        continue;

                    // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
                    unsigned int bodyAFamily = triOwnerFamilies[myThreadID];
                    unsigned int maskMatID = locateMaskPair<unsigned int>(bodyAFamily, cur_ownerFamily);
                    // If marked no contact, skip ths iteration
                    if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                        continue;
                    }

                    // Tri--tri contact does not take into account bins, as duplicates will be removed in the end
                    bool in_contact = checkPrismPrismContact(
                        simParams, triANode1[myThreadID], triANode2[myThreadID], triANode3[myThreadID],
                        triBNode1[myThreadID], triBNode2[myThreadID], triBNode3[myThreadID], cur_triANode1,
                        cur_triANode2, cur_triANode3, cur_triBNode1, cur_triBNode2, cur_triBNode3);

                    if (in_contact) {
                        deme::contactPairs_t inBlockOffset = mmReportOffset + atomicAdd(&blockTriTriPairCnt, 1);
                        // The chance of offset going out-of-bound is very low, lower than sph--bin CD step, but I put
                        // it here anyway
                        if (inBlockOffset < mmReportOffset_end) {
                            idTriA_mm[inBlockOffset] = triIDs[myThreadID];
                            idTriB_mm[inBlockOffset] = cur_bodyID;
                            dType_mm[inBlockOffset] = deme::TRIANGLE_TRIANGLE_CONTACT;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }  // End of batch-wise triangle processing
    // __syncthreads();

    // In practice, I've never seen non-illed contact slots that need to be resolved this way. It's purely for ultra
    // safety.
    if (threadIdx.x == 0) {
        for (deme::contactPairs_t inBlockOffset = smReportOffset + blockSphTriPairCnt;
             inBlockOffset < smReportOffset_end; inBlockOffset++) {
            dType_sm[inBlockOffset] = deme::NOT_A_CONTACT;
        }
        if (meshUniversalContact) {
            for (deme::contactPairs_t inBlockOffset = mmReportOffset + blockTriTriPairCnt;
                 inBlockOffset < mmReportOffset_end; inBlockOffset++) {
                dType_mm[inBlockOffset] = deme::NOT_A_CONTACT;
            }
        }
    }
}
