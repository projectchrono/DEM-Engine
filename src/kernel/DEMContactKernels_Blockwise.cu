// DEM contact detection-related custom kernels
#include <DEM/DEMDefines.h>
#include <kernel/DEMHelperKernels.cu>

// #include <cub/block/block_load.cuh>
// #include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

// If clump templates are jitified, they will be below
_clumpTemplateDefs_;
// Family mask, _nFamilyMaskEntries_ elements are in this array
// __constant__ __device__ bool familyMasks[] = {_familyMasks_};

__global__ void getNumberOfContactsEachBin(smug::DEMSimParams* simParams,
                                           smug::DEMDataKT* granData,
                                           smug::bodyID_t* sphereIDsEachBinTouches_sorted,
                                           smug::binID_t* activeBinIDs,
                                           smug::spheresBinTouches_t* numSpheresBinTouches,
                                           smug::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                           smug::spheresBinTouches_t* numContactsInEachBin,
                                           size_t nActiveBins) {
    // shared storage for bodies involved in this bin. Pre-allocated so that each threads can easily use.
    __shared__ smug::bodyID_t ownerIDs[SMUG_DEM_MAX_SPHERES_PER_BIN];
    __shared__ float radii[SMUG_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyX[SMUG_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyY[SMUG_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyZ[SMUG_DEM_MAX_SPHERES_PER_BIN];
    __shared__ smug::family_t ownerFamilies[SMUG_DEM_MAX_SPHERES_PER_BIN];

    typedef cub::BlockReduce<smug::spheresBinTouches_t, SMUG_DEM_MAX_SPHERES_PER_BIN> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    const smug::spheresBinTouches_t nBodiesInBin = numSpheresBinTouches[blockIdx.x];
    if (nBodiesInBin <= 1) {
        // Important: mark 0 contacts before exiting
        if (threadIdx.x == 0) {
            numContactsInEachBin[blockIdx.x] = 0;
        }
        return;
    }
    if (threadIdx.x == 0 && nBodiesInBin > SMUG_DEM_MAX_SPHERES_PER_BIN) {
        SMUG_DEM_ABORT_KERNEL("Bin %u contains %u sphere components, exceeding maximum allowance (%u)\n", blockIdx.x,
                              nBodiesInBin, SMUG_DEM_MAX_SPHERES_PER_BIN);
    }
    const smug::binID_t binID = activeBinIDs[blockIdx.x];
    smug::spheresBinTouches_t myThreadID = threadIdx.x;
    const smug::binSphereTouchPairs_t thisBodiesTableEntry = sphereIDsLookUpTable[blockIdx.x];
    // If I need to work on shared memory allocation
    if (myThreadID < nBodiesInBin) {
        smug::bodyID_t sphereID = sphereIDsEachBinTouches_sorted[thisBodiesTableEntry + myThreadID];
        smug::bodyID_t ownerID = granData->ownerClumpBody[sphereID];
        ownerIDs[myThreadID] = ownerID;
        ownerFamilies[myThreadID] = granData->familyID[ownerID];
        double ownerX, ownerY, ownerZ;
        float myRelPosX, myRelPosY, myRelPosZ, myRadius;

        // Get my component offset info from either jitified arrays or global memory
        // Outputs myRelPosXYZ, myRadius (in CD kernels, radius needs to be expanded)
        // Use an input named exactly `sphereID' which is the id of this sphere component
        {
            _componentAcqStrat_;
            myRadius += simParams->beta;
        }

        voxelID2Position<double, smug::voxelID_t, smug::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[ownerID], granData->locX[ownerID], granData->locY[ownerID],
            granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        float myOriQw = granData->oriQw[ownerID];
        float myOriQx = granData->oriQx[ownerID];
        float myOriQy = granData->oriQy[ownerID];
        float myOriQz = granData->oriQz[ownerID];
        applyOriQToVector3<float, smug::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQw, myOriQx, myOriQy, myOriQz);
        bodyX[myThreadID] = ownerX + (double)myRelPosX;
        bodyY[myThreadID] = ownerY + (double)myRelPosY;
        bodyZ[myThreadID] = ownerZ + (double)myRelPosZ;
        radii[myThreadID] = myRadius;
    }
    __syncthreads();

    // We have n * (n - 1) / 2 pairs to compare. To ensure even workload, these pairs are distributed to all threads.
    const unsigned int nPairsNeedHandling = (unsigned int)nBodiesInBin * ((unsigned int)nBodiesInBin - 1) / 2;
    // Note this distribution is not even, but we need all active threads to process the same amount of pairs, so that
    // each thread can easily know its offset
    const unsigned int nPairsEachHandles =
        (nPairsNeedHandling + SMUG_DEM_KT_CD_NTHREADS_PER_BLOCK - 1) / SMUG_DEM_KT_CD_NTHREADS_PER_BLOCK;
    {
        smug::spheresBinTouches_t contact_count = 0;
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
            if (granData->familyMasks[maskMatID] != smug::DEM_DONT_PREVENT_CONTACT) {
                continue;
            }

            double contactPntX;
            double contactPntY;
            double contactPntZ;
            bool in_contact;
            in_contact = checkSpheresOverlap<double>(bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], radii[bodyA],
                                                     bodyX[bodyB], bodyY[bodyB], bodyZ[bodyB], radii[bodyB],
                                                     contactPntX, contactPntY, contactPntZ);
            smug::binID_t contactPntBin = getPointBinID<smug::binID_t>(
                contactPntX, contactPntY, contactPntZ, simParams->binSize, simParams->nbX, simParams->nbY);

            /*
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
        smug::spheresBinTouches_t total_count = BlockReduceT(temp_storage).Sum(contact_count);
        if (myThreadID == 0) {
            numContactsInEachBin[blockIdx.x] = total_count;
        }
    }
}

__global__ void populateContactPairsEachBin(smug::DEMSimParams* simParams,
                                            smug::DEMDataKT* granData,
                                            smug::bodyID_t* sphereIDsEachBinTouches_sorted,
                                            smug::binID_t* activeBinIDs,
                                            smug::spheresBinTouches_t* numSpheresBinTouches,
                                            smug::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                            smug::contactPairs_t* contactReportOffsets,
                                            smug::bodyID_t* idSphA,
                                            smug::bodyID_t* idSphB,
                                            size_t nActiveBins) {
    // shared storage for bodies involved in this bin. Pre-allocated so that each threads can easily use.
    __shared__ smug::bodyID_t ownerIDs[SMUG_DEM_MAX_SPHERES_PER_BIN];
    __shared__ smug::bodyID_t bodyIDs[SMUG_DEM_MAX_SPHERES_PER_BIN];
    __shared__ float radii[SMUG_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyX[SMUG_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyY[SMUG_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyZ[SMUG_DEM_MAX_SPHERES_PER_BIN];
    __shared__ smug::family_t ownerFamilies[SMUG_DEM_MAX_SPHERES_PER_BIN];
    __shared__ unsigned int blockPairCnt;

    // typedef cub::BlockScan<smug::spheresBinTouches_t, SMUG_DEM_MAX_SPHERES_PER_BIN> BlockScanT;
    // __shared__ typename BlockScanT::TempStorage temp_storage;

    const smug::spheresBinTouches_t nBodiesInBin = numSpheresBinTouches[blockIdx.x];
    if (nBodiesInBin <= 1) {
        return;
    }
    // No need to check max spheres one more time

    const smug::binID_t binID = activeBinIDs[blockIdx.x];
    smug::spheresBinTouches_t myThreadID = threadIdx.x;
    const smug::binSphereTouchPairs_t thisBodiesTableEntry = sphereIDsLookUpTable[blockIdx.x];
    // If I need to work on shared memory allocation
    if (myThreadID < nBodiesInBin) {
        if (myThreadID == 0)
            blockPairCnt = 0;
        smug::bodyID_t sphereID = sphereIDsEachBinTouches_sorted[thisBodiesTableEntry + myThreadID];
        smug::bodyID_t ownerID = granData->ownerClumpBody[sphereID];
        bodyIDs[myThreadID] = sphereID;
        ownerIDs[myThreadID] = ownerID;
        ownerFamilies[myThreadID] = granData->familyID[ownerID];
        double ownerX, ownerY, ownerZ;
        float myRelPosX, myRelPosY, myRelPosZ, myRadius;

        // Get my component offset info from either jitified arrays or global memory
        // Outputs myRelPosXYZ, myRadius (in CD kernels, radius needs to be expanded)
        // Use an input named exactly `sphereID' which is the id of this sphere component
        {
            _componentAcqStrat_;
            myRadius += simParams->beta;
        }

        voxelID2Position<double, smug::voxelID_t, smug::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[ownerID], granData->locX[ownerID], granData->locY[ownerID],
            granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        float myOriQw = granData->oriQw[ownerID];
        float myOriQx = granData->oriQx[ownerID];
        float myOriQy = granData->oriQy[ownerID];
        float myOriQz = granData->oriQz[ownerID];
        applyOriQToVector3<float, smug::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQw, myOriQx, myOriQy, myOriQz);
        bodyX[myThreadID] = ownerX + (double)myRelPosX;
        bodyY[myThreadID] = ownerY + (double)myRelPosY;
        bodyZ[myThreadID] = ownerZ + (double)myRelPosZ;
        radii[myThreadID] = myRadius;
    }
    __syncthreads();

    // Get my offset for writing back to the global arrays that contain contact pair info
    smug::contactPairs_t myReportOffset = contactReportOffsets[blockIdx.x];
    // We have n * (n - 1) / 2 pairs to compare. To ensure even workload, these pairs are distributed to all threads.
    const unsigned int nPairsNeedHandling = (unsigned int)nBodiesInBin * ((unsigned int)nBodiesInBin - 1) / 2;
    // Note this distribution is not even, but we need all active threads to process the same amount of pairs, so that
    // each thread can easily know its offset
    const unsigned int nPairsEachHandles =
        (nPairsNeedHandling + SMUG_DEM_KT_CD_NTHREADS_PER_BLOCK - 1) / SMUG_DEM_KT_CD_NTHREADS_PER_BLOCK;

    // First figure out blockwise report offset. Meaning redoing the previous kernel
    // Blockwise report offset
    // smug::spheresBinTouches_t blockwise_offset;
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
            if (granData->familyMasks[maskMatID] != smug::DEM_DONT_PREVENT_CONTACT) {
                continue;
            }

            double contactPntX;
            double contactPntY;
            double contactPntZ;
            bool in_contact;
            in_contact = checkSpheresOverlap<double>(bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], radii[bodyA],
                                                     bodyX[bodyB], bodyY[bodyB], bodyZ[bodyB], radii[bodyB],
                                                     contactPntX, contactPntY, contactPntZ);
            smug::binID_t contactPntBin = getPointBinID<smug::binID_t>(
                contactPntX, contactPntY, contactPntZ, simParams->binSize, simParams->nbX, simParams->nbY);

            if (in_contact && (contactPntBin == binID)) {
                // blockwise_offset++;
                unsigned int inBlockOffset = atomicAdd_block(&blockPairCnt, 1);
                idSphA[myReportOffset + inBlockOffset] = bodyIDs[bodyA];
                idSphB[myReportOffset + inBlockOffset] = bodyIDs[bodyB];
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
    //         if (granData->familyMasks[maskMatID] != smug::DEM_DONT_PREVENT_CONTACT) {
    //             continue;
    //         }

    //         double contactPntX;
    //         double contactPntY;
    //         double contactPntZ;
    //         bool in_contact;
    //         in_contact = checkSpheresOverlap<double>(
    //             bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], CDRadii[compOffsets[bodyA]], bodyX[bodyB], bodyY[bodyB],
    //             bodyZ[bodyB], CDRadii[compOffsets[bodyB]], contactPntX, contactPntY, contactPntZ);
    //         smug::binID_t contactPntBin = getPointBinID<smug::binID_t>(
    //             contactPntX, contactPntY, contactPntZ, simParams->binSize, simParams->nbX, simParams->nbY);

    //         if (in_contact && (contactPntBin == binID)) {
    //             idSphA[myReportOffset] = bodyIDs[bodyA];
    //             idSphB[myReportOffset] = bodyIDs[bodyB];
    //             myReportOffset++;
    //         }
    //     }
    // }
}
