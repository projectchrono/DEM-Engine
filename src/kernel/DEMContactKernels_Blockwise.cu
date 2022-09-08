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

__global__ void getNumberOfContactsEachBin(sgps::DEMSimParams* simParams,
                                           sgps::DEMDataKT* granData,
                                           sgps::bodyID_t* sphereIDsEachBinTouches_sorted,
                                           sgps::binID_t* activeBinIDs,
                                           sgps::spheresBinTouches_t* numSpheresBinTouches,
                                           sgps::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                           sgps::spheresBinTouches_t* numContactsInEachBin,
                                           size_t nActiveBins) {
    // shared storage for bodies involved in this bin. Pre-allocated so that each threads can easily use.
    __shared__ sgps::bodyID_t ownerIDs[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ float radii[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyX[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyY[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyZ[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ sgps::family_t ownerFamilies[SGPS_DEM_MAX_SPHERES_PER_BIN];

    typedef cub::BlockReduce<sgps::spheresBinTouches_t, SGPS_DEM_MAX_SPHERES_PER_BIN> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    const sgps::spheresBinTouches_t nBodiesInBin = numSpheresBinTouches[blockIdx.x];
    if (nBodiesInBin <= 1) {
        // Important: mark 0 contacts before exiting
        if (threadIdx.x == 0) {
            numContactsInEachBin[blockIdx.x] = 0;
        }
        return;
    }
    if (threadIdx.x == 0 && nBodiesInBin > SGPS_DEM_MAX_SPHERES_PER_BIN) {
        SGPS_DEM_ABORT_KERNEL("Bin %u contains %u sphere components, exceeding maximum allowance (%u)\n", blockIdx.x,
                              nBodiesInBin, SGPS_DEM_MAX_SPHERES_PER_BIN);
    }
    const sgps::binID_t binID = activeBinIDs[blockIdx.x];
    sgps::spheresBinTouches_t myThreadID = threadIdx.x;
    const sgps::binSphereTouchPairs_t thisBodiesTableEntry = sphereIDsLookUpTable[blockIdx.x];
    // If I need to work on shared memory allocation
    if (myThreadID < nBodiesInBin) {
        sgps::bodyID_t sphereID = sphereIDsEachBinTouches_sorted[thisBodiesTableEntry + myThreadID];
        sgps::bodyID_t ownerID = granData->ownerClumpBody[sphereID];
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

        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[ownerID], granData->locX[ownerID], granData->locY[ownerID],
            granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        float myOriQ0 = granData->oriQw[ownerID];
        float myOriQ1 = granData->oriQx[ownerID];
        float myOriQ2 = granData->oriQy[ownerID];
        float myOriQ3 = granData->oriQz[ownerID];
        applyOriQToVector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQ0, myOriQ1, myOriQ2, myOriQ3);
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
        (nPairsNeedHandling + SGPS_DEM_KT_CD_NTHREADS_PER_BLOCK - 1) / SGPS_DEM_KT_CD_NTHREADS_PER_BLOCK;
    {
        sgps::spheresBinTouches_t contact_count = 0;
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
            if (granData->familyMasks[maskMatID] != sgps::DEM_DONT_PREVENT_CONTACT) {
                continue;
            }

            double contactPntX;
            double contactPntY;
            double contactPntZ;
            bool in_contact;
            in_contact = checkSpheresOverlap<double>(bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], radii[bodyA],
                                                     bodyX[bodyB], bodyY[bodyB], bodyZ[bodyB], radii[bodyB],
                                                     contactPntX, contactPntY, contactPntZ);
            sgps::binID_t contactPntBin = getPointBinID<sgps::binID_t>(
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
        sgps::spheresBinTouches_t total_count = BlockReduceT(temp_storage).Sum(contact_count);
        if (myThreadID == 0) {
            numContactsInEachBin[blockIdx.x] = total_count;
        }
    }
}

__global__ void populateContactPairsEachBin(sgps::DEMSimParams* simParams,
                                            sgps::DEMDataKT* granData,
                                            sgps::bodyID_t* sphereIDsEachBinTouches_sorted,
                                            sgps::binID_t* activeBinIDs,
                                            sgps::spheresBinTouches_t* numSpheresBinTouches,
                                            sgps::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                            sgps::contactPairs_t* contactReportOffsets,
                                            sgps::bodyID_t* idSphA,
                                            sgps::bodyID_t* idSphB,
                                            size_t nActiveBins) {
    // shared storage for bodies involved in this bin. Pre-allocated so that each threads can easily use.
    __shared__ sgps::bodyID_t ownerIDs[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ sgps::bodyID_t bodyIDs[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ float radii[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyX[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyY[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyZ[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ sgps::family_t ownerFamilies[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ unsigned int blockPairCnt;

    // typedef cub::BlockScan<sgps::spheresBinTouches_t, SGPS_DEM_MAX_SPHERES_PER_BIN> BlockScanT;
    // __shared__ typename BlockScanT::TempStorage temp_storage;

    const sgps::spheresBinTouches_t nBodiesInBin = numSpheresBinTouches[blockIdx.x];
    if (nBodiesInBin <= 1) {
        return;
    }
    // No need to check max spheres one more time

    const sgps::binID_t binID = activeBinIDs[blockIdx.x];
    sgps::spheresBinTouches_t myThreadID = threadIdx.x;
    const sgps::binSphereTouchPairs_t thisBodiesTableEntry = sphereIDsLookUpTable[blockIdx.x];
    // If I need to work on shared memory allocation
    if (myThreadID < nBodiesInBin) {
        if (myThreadID == 0)
            blockPairCnt = 0;
        sgps::bodyID_t sphereID = sphereIDsEachBinTouches_sorted[thisBodiesTableEntry + myThreadID];
        sgps::bodyID_t ownerID = granData->ownerClumpBody[sphereID];
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

        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[ownerID], granData->locX[ownerID], granData->locY[ownerID],
            granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        float myOriQ0 = granData->oriQw[ownerID];
        float myOriQ1 = granData->oriQx[ownerID];
        float myOriQ2 = granData->oriQy[ownerID];
        float myOriQ3 = granData->oriQz[ownerID];
        applyOriQToVector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQ0, myOriQ1, myOriQ2, myOriQ3);
        bodyX[myThreadID] = ownerX + (double)myRelPosX;
        bodyY[myThreadID] = ownerY + (double)myRelPosY;
        bodyZ[myThreadID] = ownerZ + (double)myRelPosZ;
        radii[myThreadID] = myRadius;
    }
    __syncthreads();

    // Get my offset for writing back to the global arrays that contain contact pair info
    sgps::contactPairs_t myReportOffset = contactReportOffsets[blockIdx.x];
    // We have n * (n - 1) / 2 pairs to compare. To ensure even workload, these pairs are distributed to all threads.
    const unsigned int nPairsNeedHandling = (unsigned int)nBodiesInBin * ((unsigned int)nBodiesInBin - 1) / 2;
    // Note this distribution is not even, but we need all active threads to process the same amount of pairs, so that
    // each thread can easily know its offset
    const unsigned int nPairsEachHandles =
        (nPairsNeedHandling + SGPS_DEM_KT_CD_NTHREADS_PER_BLOCK - 1) / SGPS_DEM_KT_CD_NTHREADS_PER_BLOCK;

    // First figure out blockwise report offset. Meaning redoing the previous kernel
    // Blockwise report offset
    // sgps::spheresBinTouches_t blockwise_offset;
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
            if (granData->familyMasks[maskMatID] != sgps::DEM_DONT_PREVENT_CONTACT) {
                continue;
            }

            double contactPntX;
            double contactPntY;
            double contactPntZ;
            bool in_contact;
            in_contact = checkSpheresOverlap<double>(bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], radii[bodyA],
                                                     bodyX[bodyB], bodyY[bodyB], bodyZ[bodyB], radii[bodyB],
                                                     contactPntX, contactPntY, contactPntZ);
            sgps::binID_t contactPntBin = getPointBinID<sgps::binID_t>(
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
    //         if (granData->familyMasks[maskMatID] != sgps::DEM_DONT_PREVENT_CONTACT) {
    //             continue;
    //         }

    //         double contactPntX;
    //         double contactPntY;
    //         double contactPntZ;
    //         bool in_contact;
    //         in_contact = checkSpheresOverlap<double>(
    //             bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], CDRadii[compOffsets[bodyA]], bodyX[bodyB], bodyY[bodyB],
    //             bodyZ[bodyB], CDRadii[compOffsets[bodyB]], contactPntX, contactPntY, contactPntZ);
    //         sgps::binID_t contactPntBin = getPointBinID<sgps::binID_t>(
    //             contactPntX, contactPntY, contactPntZ, simParams->binSize, simParams->nbX, simParams->nbY);

    //         if (in_contact && (contactPntBin == binID)) {
    //             idSphA[myReportOffset] = bodyIDs[bodyA];
    //             idSphB[myReportOffset] = bodyIDs[bodyB];
    //             myReportOffset++;
    //         }
    //     }
    // }
}
