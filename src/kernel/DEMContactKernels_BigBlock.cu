// DEM contact detection-related custom kernels
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>

// #include <cub/block/block_load.cuh>
// #include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
// "-I/opt/apps/cuda/x86_64/11.6.0/default/include"

__global__ void getNumberOfContactsEachBin(sgps::DEMDataKT* granData,
                                           sgps::bodyID_t* sphereIDsEachBinTouches_sorted,
                                           sgps::binID_t* activeBinIDs,
                                           sgps::spheresBinTouches_t* numSpheresBinTouches,
                                           sgps::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                           sgps::spheresBinTouches_t* numContactsInEachBin) {
    // shared storage for bodies involved in this bin. Pre-allocated so that each threads can easily use.
    __shared__ sgps::bodyID_t ownerIDs[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ sgps::clumpComponentOffset_t compOffsets[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyX[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyY[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyZ[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ sgps::family_t ownerFamilies[SGPS_DEM_MAX_SPHERES_PER_BIN];

    typedef cub::BlockReduce<sgps::spheresBinTouches_t, SGPS_DEM_MAX_SPHERES_PER_BIN> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    // _nDistinctClumpComponents_ elements are in these arrays
    const float CDRadii[] = {_CDRadii_};
    const float CDRelPosX[] = {_CDRelPosX_};
    const float CDRelPosY[] = {_CDRelPosY_};
    const float CDRelPosZ[] = {_CDRelPosZ_};

    // _nFamilyMaskEntries_ elements are in this array
    const bool familyMasks[] = {_familyMasks_};

    const sgps::spheresBinTouches_t nBodiesInBin = numSpheresBinTouches[blockIdx.x];
    if (nBodiesInBin <= 1) {
        return;
    }
    if (threadIdx.x == 0 && nBodiesInBin > SGPS_DEM_MAX_SPHERES_PER_BIN) {
        SGPS_DEM_ABORT_KERNEL("Bin %zu contains %u sphere components, exceeding maximum allowance (%u)\n", blockIdx.x,
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
        sgps::clumpComponentOffset_t compOffset = granData->clumpComponentOffset[sphereID];
        compOffsets[myThreadID] = compOffset;
        double ownerX, ownerY, ownerZ;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[ownerID], granData->locX[ownerID], granData->locY[ownerID],
            granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        float myRelPosX = CDRelPosX[compOffset];
        float myRelPosY = CDRelPosY[compOffset];
        float myRelPosZ = CDRelPosZ[compOffset];
        float myOriQ0 = granData->oriQ0[ownerID];
        float myOriQ1 = granData->oriQ1[ownerID];
        float myOriQ2 = granData->oriQ2[ownerID];
        float myOriQ3 = granData->oriQ3[ownerID];
        applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQ0, myOriQ1, myOriQ2, myOriQ3);
        bodyX[myThreadID] = ownerX + (double)myRelPosX;
        bodyY[myThreadID] = ownerY + (double)myRelPosY;
        bodyZ[myThreadID] = ownerZ + (double)myRelPosZ;
    }
    __syncthreads();

    // We have n * (n - 1) / 2 pairs to compare. To ensure even workload, these pairs are distributed to all threads.
    const unsigned int nPairsNeedHandling = (unsigned int)nBodiesInBin * ((unsigned int)nBodiesInBin - 1) / 2;
    // Note this distribution is not even, but we need all active threads to process the same amount of pairs, so that
    // each thread can easily know its offset
    const unsigned int nPairsEachHandles =
        (nPairsNeedHandling + SGPS_DEM_MAX_SPHERES_PER_BIN - 1) / SGPS_DEM_MAX_SPHERES_PER_BIN;
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
            if (familyMasks[maskMatID] != sgps::DEM_DONT_PREVENT_CONTACT) {
                continue;
            }

            double contactPntX;
            double contactPntY;
            double contactPntZ;
            bool in_contact;
            in_contact = checkSpheresOverlap<double>(
                bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], CDRadii[compOffsets[bodyA]], bodyX[bodyB], bodyY[bodyB],
                bodyZ[bodyB], CDRadii[compOffsets[bodyB]], contactPntX, contactPntY, contactPntZ);
            sgps::binID_t contactPntBin =
                getPointBinID<sgps::binID_t>(contactPntX, contactPntY, contactPntZ, _binSize_, _nbX_, _nbY_);

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

__global__ void populateContactPairsEachBin(sgps::DEMDataKT* granData,
                                            sgps::bodyID_t* sphereIDsEachBinTouches_sorted,
                                            sgps::binID_t* activeBinIDs,
                                            sgps::spheresBinTouches_t* numSpheresBinTouches,
                                            sgps::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                            sgps::contactPairs_t* contactReportOffsets,
                                            sgps::bodyID_t* idSphA,
                                            sgps::bodyID_t* idSphB) {
    // shared storage for bodies involved in this bin. Pre-allocated so that each threads can easily use.
    __shared__ sgps::bodyID_t ownerIDs[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ sgps::bodyID_t bodyIDs[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ sgps::clumpComponentOffset_t compOffsets[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyX[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyY[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ double bodyZ[SGPS_DEM_MAX_SPHERES_PER_BIN];
    __shared__ sgps::family_t ownerFamilies[SGPS_DEM_MAX_SPHERES_PER_BIN];

    typedef cub::BlockScan<sgps::spheresBinTouches_t, SGPS_DEM_MAX_SPHERES_PER_BIN> BlockScanT;
    __shared__ typename BlockScanT::TempStorage temp_storage;

    // _nDistinctClumpComponents_ elements are in these arrays
    const float CDRadii[] = {_CDRadii_};
    const float CDRelPosX[] = {_CDRelPosX_};
    const float CDRelPosY[] = {_CDRelPosY_};
    const float CDRelPosZ[] = {_CDRelPosZ_};

    // _nFamilyMaskEntries_ elements are in this array
    const bool familyMasks[] = {_familyMasks_};

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
        sgps::bodyID_t sphereID = sphereIDsEachBinTouches_sorted[thisBodiesTableEntry + myThreadID];
        bodyIDs[myThreadID] = sphereID;
        sgps::bodyID_t ownerID = granData->ownerClumpBody[sphereID];
        ownerIDs[myThreadID] = ownerID;
        ownerFamilies[myThreadID] = granData->familyID[ownerID];
        sgps::clumpComponentOffset_t compOffset = granData->clumpComponentOffset[sphereID];
        compOffsets[myThreadID] = compOffset;
        double ownerX, ownerY, ownerZ;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[ownerID], granData->locX[ownerID], granData->locY[ownerID],
            granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        float myRelPosX = CDRelPosX[compOffset];
        float myRelPosY = CDRelPosY[compOffset];
        float myRelPosZ = CDRelPosZ[compOffset];
        float myOriQ0 = granData->oriQ0[ownerID];
        float myOriQ1 = granData->oriQ1[ownerID];
        float myOriQ2 = granData->oriQ2[ownerID];
        float myOriQ3 = granData->oriQ3[ownerID];
        applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQ0, myOriQ1, myOriQ2, myOriQ3);
        bodyX[myThreadID] = ownerX + (double)myRelPosX;
        bodyY[myThreadID] = ownerY + (double)myRelPosY;
        bodyZ[myThreadID] = ownerZ + (double)myRelPosZ;
    }
    __syncthreads();

    // Get my offset for writing back to the global arrays that contain contact pair info
    sgps::contactPairs_t myReportOffset = contactReportOffsets[blockIdx.x];
    // We have n * (n - 1) / 2 pairs to compare. To ensure even workload, these pairs are distributed to all threads.
    const unsigned int nPairsNeedHandling = (unsigned int)nBodiesInBin * ((unsigned int)nBodiesInBin - 1) / 2;
    // Note this distribution is not even, but we need all active threads to process the same amount of pairs, so that
    // each thread can easily know its offset
    const unsigned int nPairsEachHandles =
        (nPairsNeedHandling + SGPS_DEM_MAX_SPHERES_PER_BIN - 1) / SGPS_DEM_MAX_SPHERES_PER_BIN;

    // First figure out blockwise report offset. Meaning redoing the previous kernel
    // Blockwise report offset
    sgps::spheresBinTouches_t blockwise_offset;
    {
        blockwise_offset = 0;
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
            if (familyMasks[maskMatID] != sgps::DEM_DONT_PREVENT_CONTACT) {
                continue;
            }

            double contactPntX;
            double contactPntY;
            double contactPntZ;
            bool in_contact;
            in_contact = checkSpheresOverlap<double>(
                bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], CDRadii[compOffsets[bodyA]], bodyX[bodyB], bodyY[bodyB],
                bodyZ[bodyB], CDRadii[compOffsets[bodyB]], contactPntX, contactPntY, contactPntZ);
            sgps::binID_t contactPntBin =
                getPointBinID<sgps::binID_t>(contactPntX, contactPntY, contactPntZ, _binSize_, _nbX_, _nbY_);

            if (in_contact && (contactPntBin == binID)) {
                blockwise_offset++;
            }
        }
        __syncthreads();
        BlockScanT(temp_storage).ExclusiveSum(blockwise_offset, blockwise_offset);
    }
    __syncthreads();

    // Next, fill in the contact pairs
    {
        myReportOffset += blockwise_offset;
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
            if (familyMasks[maskMatID] != sgps::DEM_DONT_PREVENT_CONTACT) {
                continue;
            }

            double contactPntX;
            double contactPntY;
            double contactPntZ;
            bool in_contact;
            in_contact = checkSpheresOverlap<double>(
                bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], CDRadii[compOffsets[bodyA]], bodyX[bodyB], bodyY[bodyB],
                bodyZ[bodyB], CDRadii[compOffsets[bodyB]], contactPntX, contactPntY, contactPntZ);
            sgps::binID_t contactPntBin =
                getPointBinID<sgps::binID_t>(contactPntX, contactPntY, contactPntZ, _binSize_, _nbX_, _nbY_);

            if (in_contact && (contactPntBin == binID)) {
                idSphA[myReportOffset] = bodyIDs[bodyA];
                idSphB[myReportOffset] = bodyIDs[bodyB];
                myReportOffset++;
            }
        }
    }
}
