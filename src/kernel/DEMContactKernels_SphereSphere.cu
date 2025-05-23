// DEM contact detection-related custom kernels
#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>
_kernelIncludes_;

// #include <cub/block/block_load.cuh>
// #include <cub/block/block_store.cuh>
// #include <cub/block/block_reduce.cuh>
// #include <cub/block/block_scan.cuh>
// #include <cub/util_ptx.cuh>

// If clump templates are jitified, they will be below
_clumpTemplateDefs_;

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

inline __device__ bool calcContactPoint(deme::DEMSimParams* simParams,
                                        const double& XA,
                                        const double& YA,
                                        const double& ZA,
                                        const float& rA,
                                        const double& XB,
                                        const double& YB,
                                        const double& ZB,
                                        const float& rB,
                                        deme::binID_t& binID,
                                        float artificialMarginA,
                                        float artificialMarginB) {
    double contactPntX;
    double contactPntY;
    double contactPntZ;
    bool in_contact;
    float normX;  // Normal directions are placeholders here
    float normY;
    float normZ;
    double overlapDepth;  // overlapDepth is needed for making artificial contacts not too loose.

    //// TODO: I guess <float, float> is fine too.
    in_contact = checkSpheresOverlap<double, float>(XA, YA, ZA, rA, XB, YB, ZB, rB, contactPntX, contactPntY,
                                                    contactPntZ, normX, normY, normZ, overlapDepth);

    // The contact needs to be larger than the smaller articifical margin so that we don't double count the artificially
    // added margin. This is a design choice, to avoid having too many contact pairs when adding artificial margins.
    float artificialMargin = (artificialMarginA < artificialMarginB) ? artificialMarginA : artificialMarginB;
    in_contact = in_contact && (overlapDepth > (double)artificialMargin);
    binID = getPointBinID<deme::binID_t>(contactPntX, contactPntY, contactPntZ, simParams->binSize, simParams->nbX,
                                         simParams->nbY);
    return in_contact;
}

__global__ void getNumberOfSphereContactsEachBin(deme::DEMSimParams* simParams,
                                                 deme::DEMDataKT* granData,
                                                 deme::bodyID_t* sphereIDsEachBinTouches_sorted,
                                                 deme::binID_t* activeBinIDs,
                                                 deme::spheresBinTouches_t* numSpheresBinTouches,
                                                 deme::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                                 deme::binContactPairs_t* numContactsInEachBin,
                                                 size_t nActiveBins) {
    // shared storage for bodies involved in this bin. Pre-allocated so that each threads can easily use.
    __shared__ deme::bodyID_t ownerIDs[DEME_NUM_SPHERES_PER_CD_BATCH];
    __shared__ deme::bodyID_t bodyIDs[DEME_NUM_SPHERES_PER_CD_BATCH];  // In this kernel, this is not used
    __shared__ float radii[DEME_NUM_SPHERES_PER_CD_BATCH];
    __shared__ double bodyX[DEME_NUM_SPHERES_PER_CD_BATCH];
    __shared__ double bodyY[DEME_NUM_SPHERES_PER_CD_BATCH];
    __shared__ double bodyZ[DEME_NUM_SPHERES_PER_CD_BATCH];
    __shared__ deme::family_t ownerFamilies[DEME_NUM_SPHERES_PER_CD_BATCH];
    __shared__ deme::binContactPairs_t blockPairCnt;

    // typedef cub::BlockReduce<deme::binContactPairs_t, DEME_KT_CD_NTHREADS_PER_BLOCK> BlockReduceT;
    // __shared__ typename BlockReduceT::TempStorage temp_storage;

    const deme::spheresBinTouches_t nBodiesInBin = numSpheresBinTouches[blockIdx.x];
    const deme::binID_t binID = activeBinIDs[blockIdx.x];
    if (nBodiesInBin <= 1 || binID == deme::NULL_BINID) {
        // Important: mark 0 contacts before exiting
        if (threadIdx.x == 0) {
            numContactsInEachBin[blockIdx.x] = 0;
        }
        return;
    }
    if (threadIdx.x == 0 && nBodiesInBin > simParams->errOutBinSphNum) {
        DEME_ABORT_KERNEL(
            "Bin %u contains %u sphere components, exceeding maximum allowance (%u).\nIf you want the solver to run "
            "despite this, set allowance higher via SetMaxSphereInBin before simulation starts.",
            blockIdx.x, nBodiesInBin, simParams->errOutBinSphNum);
    }
    const deme::spheresBinTouches_t myThreadID = threadIdx.x;
    const deme::binSphereTouchPairs_t thisBodiesTableEntry = sphereIDsLookUpTable[blockIdx.x];
    if (myThreadID == 0)
        blockPairCnt = 0;
    __syncthreads();

    // This bin may have more than 256 (default) spheres, so we process it by 256-sphere batch
    for (deme::spheresBinTouches_t processed_count = 0; processed_count < nBodiesInBin;
         processed_count += DEME_NUM_SPHERES_PER_CD_BATCH) {
        // After this batch is processed, how many spheres are still left to go in this bin?
        const deme::spheresBinTouches_t leftover_count =
            (nBodiesInBin - processed_count > DEME_NUM_SPHERES_PER_CD_BATCH)
                ? nBodiesInBin - processed_count - DEME_NUM_SPHERES_PER_CD_BATCH
                : 0;
        // In case this is not a full block...
        const deme::spheresBinTouches_t this_batch_active_count =
            (leftover_count > 0) ? DEME_NUM_SPHERES_PER_CD_BATCH : nBodiesInBin - processed_count;
        // If I need to work on shared memory allocation
        if (myThreadID < this_batch_active_count) {
            deme::bodyID_t sphereID =
                sphereIDsEachBinTouches_sorted[thisBodiesTableEntry + processed_count + myThreadID];
            fillSharedMemSpheres<float, double>(simParams, granData, myThreadID, sphereID, ownerIDs, bodyIDs,
                                                ownerFamilies, radii, bodyX, bodyY, bodyZ);
        }
        __syncthreads();

        // this_batch_active_count-sized pairwise sweep
        {
            // We have n * (n - 1) / 2 pairs to compare. To ensure even workload, these pairs are distributed to all
            // threads.
            const unsigned int nPairsNeedHandling =
                (unsigned int)this_batch_active_count * ((unsigned int)this_batch_active_count - 1) / 2;
            // Note this distribution is not even, but we need all active threads to process the same amount of pairs,
            // so that each thread can easily know its offset
            const unsigned int nPairsEachHandles =
                (nPairsNeedHandling + DEME_KT_CD_NTHREADS_PER_BLOCK - 1) / DEME_KT_CD_NTHREADS_PER_BLOCK;

            // i, j are local sphere number in bin
            unsigned int bodyA, bodyB;
            // We can stop if this thread reaches the end of all potential pairs, nPairsNeedHandling
            for (unsigned int ind = nPairsEachHandles * myThreadID;
                 ind < nPairsNeedHandling && ind < nPairsEachHandles * (myThreadID + 1); ind++) {
                recoverCntPair<unsigned int>(bodyA, bodyB, ind, this_batch_active_count);
                // For 2 bodies to be considered in contact, the contact point must be in this bin (to avoid
                // double-counting), and they do not belong to the same clump
                if (ownerIDs[bodyA] == ownerIDs[bodyB])
                    continue;

                // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
                unsigned int bodyAFamily = ownerFamilies[bodyA];
                unsigned int bodyBFamily = ownerFamilies[bodyB];
                unsigned int maskMatID = locateMaskPair<unsigned int>(bodyAFamily, bodyBFamily);
                // If marked no contact, skip ths iteration
                if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                    continue;
                }

                deme::binID_t contactPntBin;
                bool in_contact = calcContactPoint(simParams, bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], radii[bodyA],
                                                   bodyX[bodyB], bodyY[bodyB], bodyZ[bodyB], radii[bodyB],
                                                   contactPntBin, granData->familyExtraMarginSize[bodyAFamily],
                                                   granData->familyExtraMarginSize[bodyBFamily]);
                /*
                if (in_contact) {
                    printf("Contact point I see: %e, %e, %e\n", contactPntX, contactPntY, contactPntZ);
                } else {
                    printf("Distance: %e\n", sqrt( (bodyX[bodyA]-bodyX[bodyB])*(bodyX[bodyA]-bodyX[bodyB])
                                                    + (bodyY[bodyA]-bodyY[bodyB])*(bodyY[bodyA]-bodyY[bodyB])
                                                    + (bodyZ[bodyA]-bodyZ[bodyB])*(bodyZ[bodyA]-bodyZ[bodyB])  ));
                    printf("Sum of radii: %e\n", radii[bodyA] + radii[bodyB]);
                }

                printf("contactPntBin: %u, %u, %u\n", (unsigned int)(contactPntX/_binSize_),
                                                        (unsigned int)(contactPntY/_binSize_),
                                                        (unsigned int)(contactPntZ/_binSize_));
                unsigned int ZZ = binID/(_nbX_*_nbY_);
                unsigned int YY = binID%(_nbX_*_nbY_)/_nbX_;
                unsigned int XX = binID%(_nbX_*_nbY_)%_nbX_;
                printf("binID: %u, %u, %u\n", XX,YY,ZZ);
                printf("bodyA: %f, %f, %f\n", bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA]);
                printf("bodyB: %f, %f, %f\n", bodyX[bodyB], bodyY[bodyB], bodyZ[bodyB]);
                printf("contactPnt: %f, %f, %f\n", contactPntX, contactPntY, contactPntZ);
                printf("contactPntBin: %u\n", contactPntBin);
                */

                if (in_contact && (contactPntBin == binID)) {
                    atomicAdd(&blockPairCnt, 1);
                }
            }
        }
        __syncthreads();

        // Take care of the left-overs. If there are left-overs, then this is a full block. But we still need to do the
        // check, because we could have more threads in a block than max_sphere_num.
        if (myThreadID < this_batch_active_count) {
            for (deme::spheresBinTouches_t i = 0; i < leftover_count; i++) {
                deme::bodyID_t cur_ownerID, cur_bodyID;
                float cur_radii;
                double cur_bodyX, cur_bodyY, cur_bodyZ;
                deme::family_t cur_ownerFamily;
                {
                    const deme::spheresBinTouches_t cur_ind = processed_count + DEME_NUM_SPHERES_PER_CD_BATCH + i;
                    deme::bodyID_t cur_sphereID = sphereIDsEachBinTouches_sorted[thisBodiesTableEntry + cur_ind];

                    // Get the info of this sphere in question here. Note this is a broadcast so should be relatively
                    // fast.
                    fillSharedMemSpheres<float, double>(simParams, granData, 0, cur_sphereID, &cur_ownerID, &cur_bodyID,
                                                        &cur_ownerFamily, &cur_radii, &cur_bodyX, &cur_bodyY,
                                                        &cur_bodyZ);
                }
                // Then each in-shared-mem sphere compares against it. But first, check if same owner...
                if (ownerIDs[myThreadID] == cur_ownerID)
                    continue;

                // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
                unsigned int bodyAFamily = ownerFamilies[myThreadID];
                unsigned int maskMatID = locateMaskPair<unsigned int>(bodyAFamily, cur_ownerFamily);
                // If marked no contact, skip ths iteration
                if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                    continue;
                }

                deme::binID_t contactPntBin;
                bool in_contact = calcContactPoint(simParams, bodyX[myThreadID], bodyY[myThreadID], bodyZ[myThreadID],
                                                   radii[myThreadID], cur_bodyX, cur_bodyY, cur_bodyZ, cur_radii,
                                                   contactPntBin, granData->familyExtraMarginSize[bodyAFamily],
                                                   granData->familyExtraMarginSize[cur_ownerFamily]);

                if (in_contact && (contactPntBin == binID)) {
                    atomicAdd(&blockPairCnt, 1);
                }
            }
        }
        __syncthreads();
    }  // End of sphere-batch for loop
    // deme::binContactPairs_t total_count = BlockReduceT(temp_storage).Sum(contact_count);
    if (myThreadID == 0) {
        numContactsInEachBin[blockIdx.x] = blockPairCnt;
    }
}

__global__ void populateSphSphContactPairsEachBin(deme::DEMSimParams* simParams,
                                                  deme::DEMDataKT* granData,
                                                  deme::bodyID_t* sphereIDsEachBinTouches_sorted,
                                                  deme::binID_t* activeBinIDs,
                                                  deme::spheresBinTouches_t* numSpheresBinTouches,
                                                  deme::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                                  deme::contactPairs_t* contactReportOffsets,
                                                  deme::bodyID_t* idSphA,
                                                  deme::bodyID_t* idSphB,
                                                  deme::contact_t* dType,
                                                  size_t nActiveBins) {
    // shared storage for bodies involved in this bin. Pre-allocated so that each threads can easily use.
    __shared__ deme::bodyID_t ownerIDs[DEME_NUM_SPHERES_PER_CD_BATCH];
    __shared__ deme::bodyID_t bodyIDs[DEME_NUM_SPHERES_PER_CD_BATCH];
    __shared__ float radii[DEME_NUM_SPHERES_PER_CD_BATCH];
    __shared__ double bodyX[DEME_NUM_SPHERES_PER_CD_BATCH];
    __shared__ double bodyY[DEME_NUM_SPHERES_PER_CD_BATCH];
    __shared__ double bodyZ[DEME_NUM_SPHERES_PER_CD_BATCH];
    __shared__ deme::family_t ownerFamilies[DEME_NUM_SPHERES_PER_CD_BATCH];
    __shared__ deme::binContactPairs_t blockPairCnt;

    const deme::spheresBinTouches_t nBodiesInBin = numSpheresBinTouches[blockIdx.x];
    const deme::binID_t binID = activeBinIDs[blockIdx.x];
    if (nBodiesInBin <= 1 || binID == deme::NULL_BINID) {
        return;
    }
    // No need to check max spheres one more time

    const deme::spheresBinTouches_t myThreadID = threadIdx.x;
    const deme::binSphereTouchPairs_t thisBodiesTableEntry = sphereIDsLookUpTable[blockIdx.x];
    if (myThreadID == 0)
        blockPairCnt = 0;
    // Get my offset for writing back to the global arrays that contain contact pair info
    const deme::contactPairs_t myReportOffset = contactReportOffsets[blockIdx.x];
    const deme::contactPairs_t myReportOffset_end = contactReportOffsets[blockIdx.x + 1];
    __syncthreads();

    // This bin may have more than 256 (default) spheres, so we process it by 256-sphere batch
    for (deme::spheresBinTouches_t processed_count = 0; processed_count < nBodiesInBin;
         processed_count += DEME_NUM_SPHERES_PER_CD_BATCH) {
        // After this batch is processed, how many spheres are still left to go in this bin?
        const deme::spheresBinTouches_t leftover_count =
            (nBodiesInBin - processed_count > DEME_NUM_SPHERES_PER_CD_BATCH)
                ? nBodiesInBin - processed_count - DEME_NUM_SPHERES_PER_CD_BATCH
                : 0;
        // In case this is not a full block...
        const deme::spheresBinTouches_t this_batch_active_count =
            (leftover_count > 0) ? DEME_NUM_SPHERES_PER_CD_BATCH : nBodiesInBin - processed_count;
        // If I need to work on shared memory allocation
        if (myThreadID < this_batch_active_count) {
            deme::bodyID_t sphereID =
                sphereIDsEachBinTouches_sorted[thisBodiesTableEntry + processed_count + myThreadID];
            fillSharedMemSpheres<float, double>(simParams, granData, myThreadID, sphereID, ownerIDs, bodyIDs,
                                                ownerFamilies, radii, bodyX, bodyY, bodyZ);
        }
        __syncthreads();

        // this_batch_active_count-sized pairwise sweep
        {
            // We have n * (n - 1) / 2 pairs to compare. To ensure even workload, these pairs are distributed to all
            // threads.
            const unsigned int nPairsNeedHandling =
                (unsigned int)this_batch_active_count * ((unsigned int)this_batch_active_count - 1) / 2;
            // Note this distribution is not even, but we need all active threads to process the same amount of pairs,
            // so that each thread can easily know its offset
            const unsigned int nPairsEachHandles =
                (nPairsNeedHandling + DEME_KT_CD_NTHREADS_PER_BLOCK - 1) / DEME_KT_CD_NTHREADS_PER_BLOCK;

            // i, j are local sphere number in bin
            unsigned int bodyA, bodyB;
            // We can stop if this thread reaches the end of all potential pairs, nPairsNeedHandling
            for (unsigned int ind = nPairsEachHandles * myThreadID;
                 ind < nPairsNeedHandling && ind < nPairsEachHandles * (myThreadID + 1); ind++) {
                recoverCntPair<unsigned int>(bodyA, bodyB, ind, this_batch_active_count);
                // For 2 bodies to be considered in contact, the contact point must be in this bin (to avoid
                // double-counting), and they do not belong to the same clump
                if (ownerIDs[bodyA] == ownerIDs[bodyB])
                    continue;

                // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
                unsigned int bodyAFamily = ownerFamilies[bodyA];
                unsigned int bodyBFamily = ownerFamilies[bodyB];
                unsigned int maskMatID = locateMaskPair<unsigned int>(bodyAFamily, bodyBFamily);
                // If marked no contact, skip ths iteration
                if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                    continue;
                }

                deme::binID_t contactPntBin;
                bool in_contact = calcContactPoint(simParams, bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], radii[bodyA],
                                                   bodyX[bodyB], bodyY[bodyB], bodyZ[bodyB], radii[bodyB],
                                                   contactPntBin, granData->familyExtraMarginSize[bodyAFamily],
                                                   granData->familyExtraMarginSize[bodyBFamily]);

                if (in_contact && (contactPntBin == binID)) {
                    deme::contactPairs_t inBlockOffset = myReportOffset + atomicAdd(&blockPairCnt, 1);
                    // The chance of offset going out-of-bound is very low, lower than sph--bin CD step, but I put it
                    // here anyway
                    if (inBlockOffset < myReportOffset_end) {
                        idSphA[inBlockOffset] = bodyIDs[bodyA];
                        idSphB[inBlockOffset] = bodyIDs[bodyB];
                        dType[inBlockOffset] = deme::SPHERE_SPHERE_CONTACT;
                    }
                }
            }
        }  // End of a batch-wise contact pair detection
        __syncthreads();

        // Take care of the left-overs. If there are left-overs, then this is a full block. But we still need to do the
        // check, because we could have more threads in a block than max_sphere_num.
        if (myThreadID < this_batch_active_count) {
            for (deme::spheresBinTouches_t i = 0; i < leftover_count; i++) {
                deme::bodyID_t cur_ownerID, cur_bodyID;
                float cur_radii;
                double cur_bodyX, cur_bodyY, cur_bodyZ;
                deme::family_t cur_ownerFamily;
                {
                    const deme::spheresBinTouches_t cur_ind = processed_count + DEME_NUM_SPHERES_PER_CD_BATCH + i;
                    deme::bodyID_t cur_sphereID = sphereIDsEachBinTouches_sorted[thisBodiesTableEntry + cur_ind];

                    // Get the info of this sphere in question here. Note this is a broadcast so should be relatively
                    // fast.
                    fillSharedMemSpheres<float, double>(simParams, granData, 0, cur_sphereID, &cur_ownerID, &cur_bodyID,
                                                        &cur_ownerFamily, &cur_radii, &cur_bodyX, &cur_bodyY,
                                                        &cur_bodyZ);
                }
                // Then each in-shared-mem sphere compares against it. But first, check if same owner...
                if (ownerIDs[myThreadID] == cur_ownerID)
                    continue;

                // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
                unsigned int bodyAFamily = ownerFamilies[myThreadID];
                unsigned int maskMatID = locateMaskPair<unsigned int>(bodyAFamily, cur_ownerFamily);
                // If marked no contact, skip ths iteration
                if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                    continue;
                }

                deme::binID_t contactPntBin;
                bool in_contact = calcContactPoint(simParams, bodyX[myThreadID], bodyY[myThreadID], bodyZ[myThreadID],
                                                   radii[myThreadID], cur_bodyX, cur_bodyY, cur_bodyZ, cur_radii,
                                                   contactPntBin, granData->familyExtraMarginSize[bodyAFamily],
                                                   granData->familyExtraMarginSize[cur_ownerFamily]);

                if (in_contact && (contactPntBin == binID)) {
                    deme::contactPairs_t inBlockOffset = myReportOffset + atomicAdd(&blockPairCnt, 1);
                    // The chance of offset going out-of-bound is very low, lower than sph--bin CD step, but I put it
                    // here anyway
                    if (inBlockOffset < myReportOffset_end) {
                        idSphA[inBlockOffset] = bodyIDs[myThreadID];
                        idSphB[inBlockOffset] = cur_bodyID;
                        dType[inBlockOffset] = deme::SPHERE_SPHERE_CONTACT;
                    }
                }
            }
        }  // End of a left-over sweep
        __syncthreads();

    }  // End of sphere-batch for loop
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
