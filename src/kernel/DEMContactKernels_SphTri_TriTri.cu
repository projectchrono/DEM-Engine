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
    deme::bodyID_t ownerID = granData->ownerTriMesh[triID];
    triIDs[myThreadID] = triID;
    triOwnerIDs[myThreadID] = ownerID;
    triOwnerFamilies[myThreadID] = granData->familyID[ownerID];
    float3 ownerXYZ;
    float3 node1, node2, node3;

    voxelIDToPosition<float, deme::voxelID_t, deme::subVoxelPos_t>(
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
    float ownerX, ownerY, ownerZ;
    float myRadius;
    float3 myRelPos;

    // Get my component offset info from either jitified arrays or global memory
    // Outputs myRelPos, myRadius (in CD kernels, radius needs to be expanded)
    // Use an input named exactly `sphereID' which is the id of this sphere component
    {
        _componentAcqStrat_;
        myRadius += granData->marginSizeSphere[sphereID];
    }

    // These locations does not include the LBF offset
    voxelIDToPosition<float, deme::voxelID_t, deme::subVoxelPos_t>(
        ownerX, ownerY, ownerZ, granData->voxelID[ownerID], granData->locX[ownerID], granData->locY[ownerID],
        granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
    float myOriQw = granData->oriQw[ownerID];
    float myOriQx = granData->oriQx[ownerID];
    float myOriQy = granData->oriQy[ownerID];
    float myOriQz = granData->oriQz[ownerID];
    applyOriQToVector3<float, deme::oriQ_t>(myRelPos.x, myRelPos.y, myRelPos.z, myOriQw, myOriQx, myOriQy, myOriQz);
    bodyX[myThreadID] = ownerX + myRelPos.x;
    bodyY[myThreadID] = ownerY + myRelPos.y;
    bodyZ[myThreadID] = ownerZ + myRelPos.z;
    radii[myThreadID] = myRadius;
}

// Combined AABB overlap check and canonical bin assignment for tri-tri contacts.
// This function does TWO things in one pass to avoid redundant AABB computation:
// 1. Checks if the two prisms' AABBs overlap (early rejection if not)
// 2. Determines if this bin is the canonical bin for this triangle pair
//
// Returns: true if AABBs overlap AND this is the canonical bin to process this pair
//          false otherwise (either no overlap or should be processed in another bin)
//
// CANONICAL BIN ASSIGNMENT:
// PROBLEM: Two triangles can be in many bins simultaneously. If we count this pair in
// every bin where both are present, we get massive duplication.
//
// SOLUTION: For each unique triangle pair, assign it to EXACTLY ONE bin using a deterministic
// rule that can be computed locally (without knowing all bins both triangles touch).
//
// APPROACH: Compute the AABB intersection of both prisms. The MINIMUM bin ID that touches
// this intersection is the canonical bin. Since both triangles must touch this intersection
// region, this bin is guaranteed to contain both triangles.
inline __device__ bool shouldProcessTriTriInThisBin(deme::DEMSimParams* simParams,
                                                    deme::binID_t currentBinID,
                                                    deme::bodyID_t triID_A,
                                                    deme::bodyID_t triID_B,
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
    (void)triID_A;
    (void)triID_B;
    
    // Compute AABB of first prism (6 vertices)
    float minX1 = fminf(fminf(fminf(triANode1.x, triANode2.x), fminf(triANode3.x, triBNode1.x)), fminf(triBNode2.x, triBNode3.x));
    float maxX1 = fmaxf(fmaxf(fmaxf(triANode1.x, triANode2.x), fmaxf(triANode3.x, triBNode1.x)), fmaxf(triBNode2.x, triBNode3.x));
    float minY1 = fminf(fminf(fminf(triANode1.y, triANode2.y), fminf(triANode3.y, triBNode1.y)), fminf(triBNode2.y, triBNode3.y));
    float maxY1 = fmaxf(fmaxf(fmaxf(triANode1.y, triANode2.y), fmaxf(triANode3.y, triBNode1.y)), fmaxf(triBNode2.y, triBNode3.y));
    float minZ1 = fminf(fminf(fminf(triANode1.z, triANode2.z), fminf(triANode3.z, triBNode1.z)), fminf(triBNode2.z, triBNode3.z));
    float maxZ1 = fmaxf(fmaxf(fmaxf(triANode1.z, triANode2.z), fmaxf(triANode3.z, triBNode1.z)), fmaxf(triBNode2.z, triBNode3.z));
    
    // Compute AABB of second prism (6 vertices)
    float minX2 = fminf(fminf(fminf(triANode1_other.x, triANode2_other.x), fminf(triANode3_other.x, triBNode1_other.x)), fminf(triBNode2_other.x, triBNode3_other.x));
    float maxX2 = fmaxf(fmaxf(fmaxf(triANode1_other.x, triANode2_other.x), fmaxf(triANode3_other.x, triBNode1_other.x)), fmaxf(triBNode2_other.x, triBNode3_other.x));
    float minY2 = fminf(fminf(fminf(triANode1_other.y, triANode2_other.y), fminf(triANode3_other.y, triBNode1_other.y)), fminf(triBNode2_other.y, triBNode3_other.y));
    float maxY2 = fmaxf(fmaxf(fmaxf(triANode1_other.y, triANode2_other.y), fmaxf(triANode3_other.y, triBNode1_other.y)), fmaxf(triBNode2_other.y, triBNode3_other.y));
    float minZ2 = fminf(fminf(fminf(triANode1_other.z, triANode2_other.z), fminf(triANode3_other.z, triBNode1_other.z)), fminf(triBNode2_other.z, triBNode3_other.z));
    float maxZ2 = fmaxf(fmaxf(fmaxf(triANode1_other.z, triANode2_other.z), fmaxf(triANode3_other.z, triBNode1_other.z)), fmaxf(triBNode2_other.z, triBNode3_other.z));
    
    // EARLY REJECTION: Check AABB overlap first (avoids expensive SAT if no overlap)
    const float margin = 1e-6f;
    if (minX1 > maxX2 + margin || maxX1 < minX2 - margin ||
        minY1 > maxY2 + margin || maxY1 < minY2 - margin ||
        minZ1 > maxZ2 + margin || maxZ1 < minZ2 - margin) {
        return false;  // AABBs don't overlap, no contact possible
    }
    
    // AABBs overlap - now check if this is the canonical bin for this pair
    const float inv_binSize = (float)simParams->dyn.inv_binSize;
    
    // Compute AABB intersection minimum corner
    float intMinX = fmaxf(minX1, minX2);
    float intMinY = fmaxf(minY1, minY2);
    float intMinZ = fmaxf(minZ1, minZ2);
    
    // Find the minimum bin ID that touches this intersection
    // This is the bin containing the minimum corner of the intersection
    int binIdxX = (int)floorf(intMinX * inv_binSize);
    int binIdxY = (int)floorf(intMinY * inv_binSize);
    int binIdxZ = (int)floorf(intMinZ * inv_binSize);
    
    // Clamp to valid range
    binIdxX = (binIdxX >= 0) ? ((binIdxX < (int)simParams->nbX) ? binIdxX : (int)simParams->nbX - 1) : 0;
    binIdxY = (binIdxY >= 0) ? ((binIdxY < (int)simParams->nbY) ? binIdxY : (int)simParams->nbY - 1) : 0;
    binIdxZ = (binIdxZ >= 0) ? ((binIdxZ < (int)simParams->nbZ) ? binIdxZ : (int)simParams->nbZ - 1) : 0;
    
    deme::binID_t canonicalBin = binIDFrom3Indices<deme::binID_t>(
        (deme::binID_t)binIdxX, (deme::binID_t)binIdxY, (deme::binID_t)binIdxZ,
        simParams->nbX, simParams->nbY, simParams->nbZ);
    
    // Process only if current bin is the canonical bin for this pair
    return (currentBinID == canonicalBin);
}

// Full prism-prism contact check using SAT (Separating Axis Theorem).
// NOTE: AABB overlap check is already done in shouldProcessTriTriInThisBin(),
// so we skip it here and go directly to the full SAT test.
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
    (void)simParams;  // simParams not needed since AABB check moved to shouldProcessTriTriInThisBin
    
    // Calculate the contact point between 2 prisms using full SAT check
    // AABB pre-check already done in shouldProcessTriTriInThisBin
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
                            cntPnt.x, cntPnt.y, cntPnt.z, simParams->dyn.inv_binSize, simParams->nbX, simParams->nbY);
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

                // Use canonical bin assignment to avoid duplicate tri-tri contacts across bins.
                // Pass full prism (both triA and triB faces) for correct AABB computation.
                if (!shouldProcessTriTriInThisBin(simParams, binID, triIDs[bodyA], triIDs[bodyB],
                                                   triANode1[bodyA], triANode2[bodyA], triANode3[bodyA],
                                                   triBNode1[bodyA], triBNode2[bodyA], triBNode3[bodyA],
                                                   triANode1[bodyB], triANode2[bodyB], triANode3[bodyB],
                                                   triBNode1[bodyB], triBNode2[bodyB], triBNode3[bodyB]))
                    continue;

                bool in_contact = checkPrismPrismContact(
                    simParams, triANode1[bodyA], triANode2[bodyA], triANode3[bodyA], triBNode1[bodyA], triBNode2[bodyA],
                    triBNode3[bodyA], triANode1[bodyB], triANode2[bodyB], triANode3[bodyB], triBNode1[bodyB],
                    triBNode2[bodyB], triBNode3[bodyB]);

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

                    // Use canonical bin assignment to avoid duplicate tri-tri contacts across bins.
                    // Pass full prism (both triA and triB faces) for correct AABB computation.
                    if (!shouldProcessTriTriInThisBin(simParams, binID, triIDs[myThreadID], cur_bodyID,
                                                       triANode1[myThreadID], triANode2[myThreadID], triANode3[myThreadID],
                                                       triBNode1[myThreadID], triBNode2[myThreadID], triBNode3[myThreadID],
                                                       cur_triANode1, cur_triANode2, cur_triANode3,
                                                       cur_triBNode1, cur_triBNode2, cur_triBNode3))
                        continue;

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
                            cntPnt.x, cntPnt.y, cntPnt.z, simParams->dyn.inv_binSize, simParams->nbX, simParams->nbY);
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

                // Use canonical bin assignment to avoid duplicate tri-tri contacts across bins.
                // Pass full prism (both triA and triB faces) for correct AABB computation.
                if (!shouldProcessTriTriInThisBin(simParams, binID, triIDs[bodyA], triIDs[bodyB],
                                                   triANode1[bodyA], triANode2[bodyA], triANode3[bodyA],
                                                   triBNode1[bodyA], triBNode2[bodyA], triBNode3[bodyA],
                                                   triANode1[bodyB], triANode2[bodyB], triANode3[bodyB],
                                                   triBNode1[bodyB], triBNode2[bodyB], triBNode3[bodyB]))
                    continue;

                bool in_contact = checkPrismPrismContact(
                    simParams, triANode1[bodyA], triANode2[bodyA], triANode3[bodyA], triBNode1[bodyA], triBNode2[bodyA],
                    triBNode3[bodyA], triANode1[bodyB], triANode2[bodyB], triANode3[bodyB], triBNode1[bodyB],
                    triBNode2[bodyB], triBNode3[bodyB]);

                if (in_contact) {
                    deme::contactPairs_t inBlockOffset = mmReportOffset + atomicAdd(&blockTriTriPairCnt, 1);
                    // Respect the budget-limited offset range from the scaled counts
                    if (inBlockOffset < mmReportOffset_end) {
                        // ----------------------------------------------------------------------------
                        // IMPORTANT NOTE: Here, we don't need to adjust A and B ids to ensure A < B, and it's
                        // automatically ensured due to 1) The binID--triID pairs were generated with an inherent order
                        // of triID, then processed through (stable) radix sort, which preserved the blockwise order of
                        // triID; 2) Then the ordered triIDs are loaded to shared mem, and the in-kernel contact
                        // detection had threads reconstruct shared mem offsets from a recoverCntPair process, which
                        // also ensures i < j. Therefore, the generated tri contact pair has A < B. However, a change
                        // in these processes could affect the ordering, so I added this superfluous check to be
                        // future-proof.
                        // ----------------------------------------------------------------------------
                        deme::bodyID_t triA_ID, triB_ID;
                        if (triIDs[bodyA] <= triIDs[bodyB]) {
                            // This branch will be reached, always
                            triA_ID = triIDs[bodyA];
                            triB_ID = triIDs[bodyB];
                            idTriA_mm[inBlockOffset] = triA_ID;
                            idTriB_mm[inBlockOffset] = triB_ID;
                        } else {
                            triA_ID = triIDs[bodyB];
                            triB_ID = triIDs[bodyA];
                            idTriA_mm[inBlockOffset] = triA_ID;
                            idTriB_mm[inBlockOffset] = triB_ID;
                        }
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

                    // Use canonical bin assignment to avoid duplicate tri-tri contacts across bins.
                    // Pass full prism (both triA and triB faces) for correct AABB computation.
                    if (!shouldProcessTriTriInThisBin(simParams, binID, triIDs[myThreadID], cur_bodyID,
                                                       triANode1[myThreadID], triANode2[myThreadID], triANode3[myThreadID],
                                                       triBNode1[myThreadID], triBNode2[myThreadID], triBNode3[myThreadID],
                                                       cur_triANode1, cur_triANode2, cur_triANode3,
                                                       cur_triBNode1, cur_triBNode2, cur_triBNode3))
                        continue;

                    bool in_contact = checkPrismPrismContact(
                        simParams, triANode1[myThreadID], triANode2[myThreadID], triANode3[myThreadID],
                        triBNode1[myThreadID], triBNode2[myThreadID], triBNode3[myThreadID], cur_triANode1,
                        cur_triANode2, cur_triANode3, cur_triBNode1, cur_triBNode2, cur_triBNode3);

                    if (in_contact) {
                        deme::contactPairs_t inBlockOffset = mmReportOffset + atomicAdd(&blockTriTriPairCnt, 1);
                        // Respect the budget-limited offset range from the scaled counts
                        if (inBlockOffset < mmReportOffset_end) {
                            deme::bodyID_t triA_ID, triB_ID;
                            if (triIDs[myThreadID] <= cur_bodyID) {
                                // This branch will be reached, always
                                triA_ID = triIDs[myThreadID];
                                triB_ID = cur_bodyID;
                                idTriA_mm[inBlockOffset] = triA_ID;
                                idTriB_mm[inBlockOffset] = triB_ID;
                            } else {
                                triA_ID = cur_bodyID;
                                triB_ID = triIDs[myThreadID];
                                idTriA_mm[inBlockOffset] = triA_ID;
                                idTriB_mm[inBlockOffset] = triB_ID;
                            }
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
