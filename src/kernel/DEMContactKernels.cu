// DEM contact detection-related custom kernels
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>

__global__ void getNumberOfContactsEachBin(sgps::DEMDataKT* granData,
                                           sgps::bodyID_t* sphereIDsEachBinTouches_sorted,
                                           sgps::binID_t* activeBinIDs,
                                           sgps::spheresBinTouches_t* numSpheresBinTouches,
                                           sgps::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                           sgps::spheresBinTouches_t* numContactsInEachBin,
                                           size_t nActiveBins) {
    // _nDistinctClumpComponents_ elements are in these arrays
    const float CDRadii[] = {_CDRadii_};
    const float CDRelPosX[] = {_CDRelPosX_};
    const float CDRelPosY[] = {_CDRelPosY_};
    const float CDRelPosZ[] = {_CDRelPosZ_};

    // _nFamilyMaskEntries_ elements are in this array
    const bool familyMasks[] = {_familyMasks_};

    // Only active bins got execute this...
    sgps::binID_t myActiveID = blockIdx.x * blockDim.x + threadIdx.x;
    // I need to store all the sphereIDs that I am supposed to look into
    // A100 has about 164K shMem... these arrays really need to be small, or we can only fit a small number of bins in
    // one block
    sgps::bodyID_t ownerIDs[SGPS_DEM_MAX_SPHERES_PER_BIN];
    sgps::clumpComponentOffset_t compOffsets[SGPS_DEM_MAX_SPHERES_PER_BIN];
    double bodyX[SGPS_DEM_MAX_SPHERES_PER_BIN];
    double bodyY[SGPS_DEM_MAX_SPHERES_PER_BIN];
    double bodyZ[SGPS_DEM_MAX_SPHERES_PER_BIN];
    sgps::family_t ownerFamily[SGPS_DEM_MAX_SPHERES_PER_BIN];
    if (myActiveID < nActiveBins) {
        // I got a true bin ID
        sgps::binID_t binID = activeBinIDs[myActiveID];

        sgps::spheresBinTouches_t contact_count = 0;
        // Grab the bodies that I care, put into local memory
        sgps::spheresBinTouches_t nBodiesMeHandle = numSpheresBinTouches[myActiveID];
        sgps::binSphereTouchPairs_t myBodiesTableEntry = sphereIDsLookUpTable[myActiveID];
        // printf("nBodies: %u\n", nBodiesMeHandle);
        for (sgps::spheresBinTouches_t i = 0; i < nBodiesMeHandle; i++) {
            sgps::bodyID_t bodyID = sphereIDsEachBinTouches_sorted[myBodiesTableEntry + i];
            ownerIDs[i] = granData->ownerClumpBody[bodyID];
            ownerFamily[i] = granData->familyID[ownerIDs[i]];
            compOffsets[i] = granData->clumpComponentOffset[bodyID];
            double ownerX, ownerY, ownerZ;
            voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
                ownerX, ownerY, ownerZ, granData->voxelID[ownerIDs[i]], granData->locX[ownerIDs[i]],
                granData->locY[ownerIDs[i]], granData->locZ[ownerIDs[i]], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            float myRelPosX = CDRelPosX[compOffsets[i]];
            float myRelPosY = CDRelPosY[compOffsets[i]];
            float myRelPosZ = CDRelPosZ[compOffsets[i]];
            float myOriQ0 = granData->oriQ0[ownerIDs[i]];
            float myOriQ1 = granData->oriQ1[ownerIDs[i]];
            float myOriQ2 = granData->oriQ2[ownerIDs[i]];
            float myOriQ3 = granData->oriQ3[ownerIDs[i]];
            applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQ0, myOriQ1, myOriQ2, myOriQ3);
            bodyX[i] = ownerX + (double)myRelPosX;
            bodyY[i] = ownerY + (double)myRelPosY;
            bodyZ[i] = ownerZ + (double)myRelPosZ;
        }

        for (sgps::spheresBinTouches_t bodyA = 0; bodyA < nBodiesMeHandle; bodyA++) {
            for (sgps::spheresBinTouches_t bodyB = bodyA + 1; bodyB < nBodiesMeHandle; bodyB++) {
                // For 2 bodies to be considered in contact, the contact point must be in this bin (to avoid
                // double-counting), and they do not belong to the same clump
                if (ownerIDs[bodyA] == ownerIDs[bodyB])
                    continue;

                // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
                unsigned int bodyAFamily = ownerFamily[bodyA];
                unsigned int bodyBFamily = ownerFamily[bodyB];
                unsigned int maskMatID = locateMatPair<unsigned int>(bodyAFamily, bodyBFamily);
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
        }
        numContactsInEachBin[myActiveID] = contact_count;
    }
}

__global__ void populateContactPairsEachBin(sgps::DEMDataKT* granData,
                                            sgps::bodyID_t* sphereIDsEachBinTouches_sorted,
                                            sgps::binID_t* activeBinIDs,
                                            sgps::spheresBinTouches_t* numSpheresBinTouches,
                                            sgps::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                            sgps::contactPairs_t* contactReportOffsets,
                                            sgps::bodyID_t* idSphA,
                                            sgps::bodyID_t* idSphB,
                                            size_t nActiveBins) {
    // _nDistinctClumpComponents_ elements are in these arrays
    const float CDRadii[] = {_CDRadii_};
    const float CDRelPosX[] = {_CDRelPosX_};
    const float CDRelPosY[] = {_CDRelPosY_};
    const float CDRelPosZ[] = {_CDRelPosZ_};

    // _nFamilyMaskEntries_ elements are in this array
    const bool familyMasks[] = {_familyMasks_};

    // Only active bins got to execute this...
    sgps::binID_t myActiveID = blockIdx.x * blockDim.x + threadIdx.x;
    // I need to store all the sphereIDs that I am supposed to look into
    // A100 has about 164K shMem... these arrays really need to be small, or we can only fit a small number of bins in
    // one block
    sgps::bodyID_t ownerIDs[SGPS_DEM_MAX_SPHERES_PER_BIN];
    sgps::bodyID_t bodyIDs[SGPS_DEM_MAX_SPHERES_PER_BIN];
    sgps::clumpComponentOffset_t compOffsets[SGPS_DEM_MAX_SPHERES_PER_BIN];
    double bodyX[SGPS_DEM_MAX_SPHERES_PER_BIN];
    double bodyY[SGPS_DEM_MAX_SPHERES_PER_BIN];
    double bodyZ[SGPS_DEM_MAX_SPHERES_PER_BIN];
    sgps::family_t ownerFamily[SGPS_DEM_MAX_SPHERES_PER_BIN];
    if (myActiveID < nActiveBins) {
        // But I got a true bin ID
        sgps::binID_t binID = activeBinIDs[myActiveID];

        // Grab the bodies that I care, put into local memory
        sgps::spheresBinTouches_t nBodiesMeHandle = numSpheresBinTouches[myActiveID];
        sgps::binSphereTouchPairs_t myBodiesTableEntry = sphereIDsLookUpTable[myActiveID];
        for (sgps::spheresBinTouches_t i = 0; i < nBodiesMeHandle; i++) {
            sgps::bodyID_t bodyID = sphereIDsEachBinTouches_sorted[myBodiesTableEntry + i];
            ownerIDs[i] = granData->ownerClumpBody[bodyID];
            ownerFamily[i] = granData->familyID[ownerIDs[i]];
            bodyIDs[i] = bodyID;
            compOffsets[i] = granData->clumpComponentOffset[bodyID];
            double ownerX, ownerY, ownerZ;
            voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
                ownerX, ownerY, ownerZ, granData->voxelID[ownerIDs[i]], granData->locX[ownerIDs[i]],
                granData->locY[ownerIDs[i]], granData->locZ[ownerIDs[i]], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            float myRelPosX = CDRelPosX[compOffsets[i]];
            float myRelPosY = CDRelPosY[compOffsets[i]];
            float myRelPosZ = CDRelPosZ[compOffsets[i]];
            float myOriQ0 = granData->oriQ0[ownerIDs[i]];
            float myOriQ1 = granData->oriQ1[ownerIDs[i]];
            float myOriQ2 = granData->oriQ2[ownerIDs[i]];
            float myOriQ3 = granData->oriQ3[ownerIDs[i]];
            applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQ0, myOriQ1, myOriQ2, myOriQ3);
            bodyX[i] = ownerX + (double)myRelPosX;
            bodyY[i] = ownerY + (double)myRelPosY;
            bodyZ[i] = ownerZ + (double)myRelPosZ;
        }

        // Get my offset for writing back to the global arrays that contain contact pair info
        sgps::contactPairs_t myReportOffset = contactReportOffsets[myActiveID];

        for (sgps::spheresBinTouches_t bodyA = 0; bodyA < nBodiesMeHandle; bodyA++) {
            for (sgps::spheresBinTouches_t bodyB = bodyA + 1; bodyB < nBodiesMeHandle; bodyB++) {
                // For 2 bodies to be considered in contact, the contact point must be in this bin (to avoid
                // double-counting), and they do not belong to the same clump
                if (ownerIDs[bodyA] == ownerIDs[bodyB])
                    continue;

                // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
                unsigned int bodyAFamily = ownerFamily[bodyA];
                unsigned int bodyBFamily = ownerFamily[bodyB];
                unsigned int maskMatID = locateMatPair<unsigned int>(bodyAFamily, bodyBFamily);
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
}
