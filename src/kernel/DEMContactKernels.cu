// DEM contact detection-related custom kernels
#include <DEM/DEMDefines.h>
#include <kernel/DEMHelperKernels.cu>

#include <cub/util_ptx.cuh>

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
    // Only active bins got execute this...
    smug::binID_t myActiveID = blockIdx.x * blockDim.x + threadIdx.x;
    // I need to store all the sphereIDs that I am supposed to look into
    // A100 has about 164K shMem... these arrays really need to be small, or we can only fit a small number of bins in
    // one block
    smug::bodyID_t ownerIDs[SMUG_DEM_MAX_SPHERES_PER_BIN];
    float radii[SMUG_DEM_MAX_SPHERES_PER_BIN];
    double bodyX[SMUG_DEM_MAX_SPHERES_PER_BIN];
    double bodyY[SMUG_DEM_MAX_SPHERES_PER_BIN];
    double bodyZ[SMUG_DEM_MAX_SPHERES_PER_BIN];
    smug::family_t ownerFamily[SMUG_DEM_MAX_SPHERES_PER_BIN];
    if (myActiveID < nActiveBins) {
        // I got a true bin ID
        smug::binID_t binID = activeBinIDs[myActiveID];

        smug::spheresBinTouches_t contact_count = 0;
        // Grab the bodies that I care, put into local memory
        smug::spheresBinTouches_t nBodiesMeHandle = numSpheresBinTouches[myActiveID];
        if (nBodiesMeHandle > SMUG_DEM_MAX_SPHERES_PER_BIN) {
            SMUG_DEM_ABORT_KERNEL("Bin %u contains %u sphere components, exceeding maximum allowance (%u)\n",
                                  myActiveID, nBodiesMeHandle, SMUG_DEM_MAX_SPHERES_PER_BIN);
        }

        smug::binSphereTouchPairs_t myBodiesTableEntry = sphereIDsLookUpTable[myActiveID];
        // printf("nBodies: %u\n", nBodiesMeHandle);
        for (smug::spheresBinTouches_t i = 0; i < nBodiesMeHandle; i++) {
            smug::bodyID_t sphereID = sphereIDsEachBinTouches_sorted[myBodiesTableEntry + i];
            ownerIDs[i] = granData->ownerClumpBody[sphereID];
            ownerFamily[i] = granData->familyID[ownerIDs[i]];
            double ownerX, ownerY, ownerZ;
            float myRelPosX, myRelPosY, myRelPosZ, myRadius;

            // Get my component offset info from either jitified arrays or global memory
            // Outputs myRelPosXYZ, myRadius (in CD kernels, radius needs to be expanded)
            // Use an input named exactly `sphereID' which is the id of this sphere component
            {
                _componentAcqStrat_;
                myRadius += simParams->beta;
            }

            voxelIDToPosition<double, smug::voxelID_t, smug::subVoxelPos_t>(
                ownerX, ownerY, ownerZ, granData->voxelID[ownerIDs[i]], granData->locX[ownerIDs[i]],
                granData->locY[ownerIDs[i]], granData->locZ[ownerIDs[i]], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            float myOriQw = granData->oriQw[ownerIDs[i]];
            float myOriQx = granData->oriQx[ownerIDs[i]];
            float myOriQy = granData->oriQy[ownerIDs[i]];
            float myOriQz = granData->oriQz[ownerIDs[i]];
            applyOriQToVector3<float, smug::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQw, myOriQx, myOriQy,
                                                    myOriQz);
            bodyX[i] = ownerX + (double)myRelPosX;
            bodyY[i] = ownerY + (double)myRelPosY;
            bodyZ[i] = ownerZ + (double)myRelPosZ;
            radii[i] = myRadius;
        }

        for (smug::spheresBinTouches_t bodyA = 0; bodyA < nBodiesMeHandle; bodyA++) {
            for (smug::spheresBinTouches_t bodyB = bodyA + 1; bodyB < nBodiesMeHandle; bodyB++) {
                // For 2 bodies to be considered in contact, the contact point must be in this bin (to avoid
                // double-counting), and they do not belong to the same clump
                if (ownerIDs[bodyA] == ownerIDs[bodyB])
                    continue;

                // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
                unsigned int bodyAFamily = ownerFamily[bodyA];
                unsigned int bodyBFamily = ownerFamily[bodyB];
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
        }
        numContactsInEachBin[myActiveID] = contact_count;
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
    // Only active bins got to execute this...
    smug::binID_t myActiveID = blockIdx.x * blockDim.x + threadIdx.x;
    // I need to store all the sphereIDs that I am supposed to look into
    // A100 has about 164K shMem... these arrays really need to be small, or we can only fit a small number of bins in
    // one block
    smug::bodyID_t ownerIDs[SMUG_DEM_MAX_SPHERES_PER_BIN];
    smug::bodyID_t bodyIDs[SMUG_DEM_MAX_SPHERES_PER_BIN];
    float radii[SMUG_DEM_MAX_SPHERES_PER_BIN];
    double bodyX[SMUG_DEM_MAX_SPHERES_PER_BIN];
    double bodyY[SMUG_DEM_MAX_SPHERES_PER_BIN];
    double bodyZ[SMUG_DEM_MAX_SPHERES_PER_BIN];
    smug::family_t ownerFamily[SMUG_DEM_MAX_SPHERES_PER_BIN];
    if (myActiveID < nActiveBins) {
        // But I got a true bin ID
        smug::binID_t binID = activeBinIDs[myActiveID];

        // Grab the bodies that I care, put into local memory
        smug::spheresBinTouches_t nBodiesMeHandle = numSpheresBinTouches[myActiveID];
        smug::binSphereTouchPairs_t myBodiesTableEntry = sphereIDsLookUpTable[myActiveID];
        for (smug::spheresBinTouches_t i = 0; i < nBodiesMeHandle; i++) {
            smug::bodyID_t sphereID = sphereIDsEachBinTouches_sorted[myBodiesTableEntry + i];
            ownerIDs[i] = granData->ownerClumpBody[sphereID];
            ownerFamily[i] = granData->familyID[ownerIDs[i]];
            bodyIDs[i] = sphereID;
            double ownerX, ownerY, ownerZ;
            float myRelPosX, myRelPosY, myRelPosZ, myRadius;

            // Get my component offset info from either jitified arrays or global memory
            // Outputs myRelPosXYZ, myRadius (in CD kernels, radius needs to be expanded)
            // Use an input named exactly `sphereID' which is the id of this sphere component
            {
                _componentAcqStrat_;
                myRadius += simParams->beta;
            }

            voxelIDToPosition<double, smug::voxelID_t, smug::subVoxelPos_t>(
                ownerX, ownerY, ownerZ, granData->voxelID[ownerIDs[i]], granData->locX[ownerIDs[i]],
                granData->locY[ownerIDs[i]], granData->locZ[ownerIDs[i]], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            float myOriQw = granData->oriQw[ownerIDs[i]];
            float myOriQx = granData->oriQx[ownerIDs[i]];
            float myOriQy = granData->oriQy[ownerIDs[i]];
            float myOriQz = granData->oriQz[ownerIDs[i]];
            applyOriQToVector3<float, smug::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQw, myOriQx, myOriQy,
                                                    myOriQz);
            bodyX[i] = ownerX + (double)myRelPosX;
            bodyY[i] = ownerY + (double)myRelPosY;
            bodyZ[i] = ownerZ + (double)myRelPosZ;
            radii[i] = myRadius;
        }

        // Get my offset for writing back to the global arrays that contain contact pair info
        smug::contactPairs_t myReportOffset = contactReportOffsets[myActiveID];

        for (smug::spheresBinTouches_t bodyA = 0; bodyA < nBodiesMeHandle; bodyA++) {
            for (smug::spheresBinTouches_t bodyB = bodyA + 1; bodyB < nBodiesMeHandle; bodyB++) {
                // For 2 bodies to be considered in contact, the contact point must be in this bin (to avoid
                // double-counting), and they do not belong to the same clump
                if (ownerIDs[bodyA] == ownerIDs[bodyB])
                    continue;

                // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
                unsigned int bodyAFamily = ownerFamily[bodyA];
                unsigned int bodyBFamily = ownerFamily[bodyB];
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
                    idSphA[myReportOffset] = bodyIDs[bodyA];
                    idSphB[myReportOffset] = bodyIDs[bodyB];
                    myReportOffset++;
                }
            }
        }
    }
}
