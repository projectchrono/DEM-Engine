// DEM contact detection-related custom kernels
#include <DEM/Defines.h>
#include <kernel/DEMHelperKernels.cu>

// #include <cub/block/block_load.cuh>
// #include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

// If clump templates are jitified, they will be below
_clumpTemplateDefs_;
// Family mask, _nFamilyMaskEntries_ elements are in this array
// __constant__ __device__ bool familyMasks[] = {_familyMasks_};

__global__ void getNumberOfSphereContactsEachBin(deme::DEMSimParams* simParams,
                                                 deme::DEMDataKT* granData,
                                                 deme::bodyID_t* sphereIDsEachBinTouches_sorted,
                                                 deme::binID_t* activeBinIDs,
                                                 deme::spheresBinTouches_t* numSpheresBinTouches,
                                                 deme::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                                 deme::spheresBinTouches_t* numContactsInEachBin,
                                                 size_t nActiveBins) {
    // shared storage for bodies involved in this bin. Pre-allocated so that each threads can easily use.
    __shared__ deme::bodyID_t ownerIDs[DEME_MAX_SPHERES_PER_BIN];
    __shared__ float radii[DEME_MAX_SPHERES_PER_BIN];
    __shared__ double bodyX[DEME_MAX_SPHERES_PER_BIN];
    __shared__ double bodyY[DEME_MAX_SPHERES_PER_BIN];
    __shared__ double bodyZ[DEME_MAX_SPHERES_PER_BIN];
    __shared__ deme::family_t ownerFamilies[DEME_MAX_SPHERES_PER_BIN];

    typedef cub::BlockReduce<deme::spheresBinTouches_t, DEME_MAX_SPHERES_PER_BIN> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    const deme::spheresBinTouches_t nBodiesInBin = numSpheresBinTouches[blockIdx.x];
    if (nBodiesInBin <= 1) {
        // Important: mark 0 contacts before exiting
        if (threadIdx.x == 0) {
            numContactsInEachBin[blockIdx.x] = 0;
        }
        return;
    }
    if (threadIdx.x == 0 && nBodiesInBin > DEME_MAX_SPHERES_PER_BIN) {
        DEME_ABORT_KERNEL("Bin %u contains %u sphere components, exceeding maximum allowance (%u)\n", blockIdx.x,
                          nBodiesInBin, DEME_MAX_SPHERES_PER_BIN);
    }
    const deme::binID_t binID = activeBinIDs[blockIdx.x];
    deme::spheresBinTouches_t myThreadID = threadIdx.x;
    const deme::binSphereTouchPairs_t thisBodiesTableEntry = sphereIDsLookUpTable[blockIdx.x];
    // If I need to work on shared memory allocation
    if (myThreadID < nBodiesInBin) {
        deme::bodyID_t sphereID = sphereIDsEachBinTouches_sorted[thisBodiesTableEntry + myThreadID];
        deme::bodyID_t ownerID = granData->ownerClumpBody[sphereID];
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
            myRadius += simParams->beta;
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
    __syncthreads();

    // We have n * (n - 1) / 2 pairs to compare. To ensure even workload, these pairs are distributed to all threads.
    const unsigned int nPairsNeedHandling = (unsigned int)nBodiesInBin * ((unsigned int)nBodiesInBin - 1) / 2;
    // Note this distribution is not even, but we need all active threads to process the same amount of pairs, so that
    // each thread can easily know its offset
    const unsigned int nPairsEachHandles =
        (nPairsNeedHandling + DEME_KT_CD_NTHREADS_PER_BLOCK - 1) / DEME_KT_CD_NTHREADS_PER_BLOCK;
    {
        deme::spheresBinTouches_t contact_count = 0;
        // i, j are local sphere number in bin
        unsigned int bodyA, bodyB;
        // We can stop if this thread reaches the end of all potential pairs, nPairsNeedHandling
        for (unsigned int ind = nPairsEachHandles * myThreadID;
             ind < nPairsNeedHandling && ind < nPairsEachHandles * (myThreadID + 1); ind++) {
            recoverCntPair<unsigned int>(bodyA, bodyB, ind, nBodiesInBin);
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

            double contactPntX;
            double contactPntY;
            double contactPntZ;
            bool in_contact;
            in_contact = checkSpheresOverlap<double>(bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], radii[bodyA],
                                                     bodyX[bodyB], bodyY[bodyB], bodyZ[bodyB], radii[bodyB],
                                                     contactPntX, contactPntY, contactPntZ);
            deme::binID_t contactPntBin = getPointBinID<deme::binID_t>(
                contactPntX, contactPntY, contactPntZ, simParams->binSize, simParams->nbX, simParams->nbY);

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
                contact_count++;
            }
        }
        __syncthreads();

        deme::spheresBinTouches_t total_count = BlockReduceT(temp_storage).Sum(contact_count);
        if (myThreadID == 0) {
            numContactsInEachBin[blockIdx.x] = total_count;
        }
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
    __shared__ deme::bodyID_t ownerIDs[DEME_MAX_SPHERES_PER_BIN];
    __shared__ deme::bodyID_t bodyIDs[DEME_MAX_SPHERES_PER_BIN];
    __shared__ float radii[DEME_MAX_SPHERES_PER_BIN];
    __shared__ double bodyX[DEME_MAX_SPHERES_PER_BIN];
    __shared__ double bodyY[DEME_MAX_SPHERES_PER_BIN];
    __shared__ double bodyZ[DEME_MAX_SPHERES_PER_BIN];
    __shared__ deme::family_t ownerFamilies[DEME_MAX_SPHERES_PER_BIN];
    __shared__ unsigned int blockPairCnt;

    // typedef cub::BlockScan<deme::spheresBinTouches_t, DEME_MAX_SPHERES_PER_BIN> BlockScanT;
    // __shared__ typename BlockScanT::TempStorage temp_storage;

    const deme::spheresBinTouches_t nBodiesInBin = numSpheresBinTouches[blockIdx.x];
    if (nBodiesInBin <= 1) {
        return;
    }
    // No need to check max spheres one more time

    const deme::binID_t binID = activeBinIDs[blockIdx.x];
    deme::spheresBinTouches_t myThreadID = threadIdx.x;
    const deme::binSphereTouchPairs_t thisBodiesTableEntry = sphereIDsLookUpTable[blockIdx.x];
    // If I need to work on shared memory allocation
    if (myThreadID < nBodiesInBin) {
        if (myThreadID == 0)
            blockPairCnt = 0;
        deme::bodyID_t sphereID = sphereIDsEachBinTouches_sorted[thisBodiesTableEntry + myThreadID];
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
            myRadius += simParams->beta;
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
    __syncthreads();

    // Get my offset for writing back to the global arrays that contain contact pair info
    deme::contactPairs_t myReportOffset = contactReportOffsets[blockIdx.x];
    // We have n * (n - 1) / 2 pairs to compare. To ensure even workload, these pairs are distributed to all threads.
    const unsigned int nPairsNeedHandling = (unsigned int)nBodiesInBin * ((unsigned int)nBodiesInBin - 1) / 2;
    // Note this distribution is not even, but we need all active threads to process the same amount of pairs, so that
    // each thread can easily know its offset
    const unsigned int nPairsEachHandles =
        (nPairsNeedHandling + DEME_KT_CD_NTHREADS_PER_BLOCK - 1) / DEME_KT_CD_NTHREADS_PER_BLOCK;

    // First figure out blockwise report offset. Meaning redoing the previous kernel
    // Blockwise report offset
    // deme::spheresBinTouches_t blockwise_offset;
    {
        // blockwise_offset = 0;
        // i, j are local sphere number in bin
        unsigned int bodyA, bodyB;
        // We can stop if this thread reaches the end of all potential pairs, nPairsNeedHandling
        for (unsigned int ind = nPairsEachHandles * myThreadID;
             ind < nPairsNeedHandling && ind < nPairsEachHandles * (myThreadID + 1); ind++) {
            recoverCntPair<unsigned int>(bodyA, bodyB, ind, nBodiesInBin);
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

            double contactPntX;
            double contactPntY;
            double contactPntZ;
            bool in_contact;
            in_contact = checkSpheresOverlap<double>(bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], radii[bodyA],
                                                     bodyX[bodyB], bodyY[bodyB], bodyZ[bodyB], radii[bodyB],
                                                     contactPntX, contactPntY, contactPntZ);
            deme::binID_t contactPntBin = getPointBinID<deme::binID_t>(
                contactPntX, contactPntY, contactPntZ, simParams->binSize, simParams->nbX, simParams->nbY);

            if (in_contact && (contactPntBin == binID)) {
                // blockwise_offset++;
                unsigned int inBlockOffset = atomicAdd_block(&blockPairCnt, 1);
                idSphA[myReportOffset + inBlockOffset] = bodyIDs[bodyA];
                idSphB[myReportOffset + inBlockOffset] = bodyIDs[bodyB];
                dType[myReportOffset + inBlockOffset] = deme::SPHERE_SPHERE_CONTACT;
            }
        }
        // __syncthreads();
        // BlockScanT(temp_storage).ExclusiveSum(blockwise_offset, blockwise_offset);
    }
    // __syncthreads();

    // Next, fill in the contact pairs
    // {
    //     myReportOffset += blockwise_offset;
    //     // i, j are local sphere number in bin
    //     unsigned int bodyA, bodyB;
    //     // We can stop if this thread reaches the end of all potential pairs, nPairsNeedHandling
    //     for (unsigned int ind = nPairsEachHandles * myThreadID;
    //          ind < nPairsNeedHandling && ind < nPairsEachHandles * (myThreadID + 1); ind++) {
    //         recoverCntPair<unsigned int>(bodyA, bodyB, ind, nBodiesInBin);

    //         // For 2 bodies to be considered in contact, the contact point must be in this bin (to avoid
    //         // double-counting), and they do not belong to the same clump
    //         if (ownerIDs[bodyA] == ownerIDs[bodyB])
    //             continue;

    //         // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
    //         unsigned int bodyAFamily = ownerFamilies[bodyA];
    //         unsigned int bodyBFamily = ownerFamilies[bodyB];
    //         unsigned int maskMatID = locateMaskPair<unsigned int>(bodyAFamily, bodyBFamily);
    //         // If marked no contact, skip ths iteration
    //         if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
    //             continue;
    //         }

    //         double contactPntX;
    //         double contactPntY;
    //         double contactPntZ;
    //         bool in_contact;
    //         in_contact = checkSpheresOverlap<double>(
    //             bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], CDRadii[compOffsets[bodyA]], bodyX[bodyB], bodyY[bodyB],
    //             bodyZ[bodyB], CDRadii[compOffsets[bodyB]], contactPntX, contactPntY, contactPntZ);
    //         deme::binID_t contactPntBin = getPointBinID<deme::binID_t>(
    //             contactPntX, contactPntY, contactPntZ, simParams->binSize, simParams->nbX, simParams->nbY);

    //         if (in_contact && (contactPntBin == binID)) {
    //             idSphA[myReportOffset] = bodyIDs[bodyA];
    //             idSphB[myReportOffset] = bodyIDs[bodyB];
    //             myReportOffset++;
    //         }
    //     }
    // }
}
