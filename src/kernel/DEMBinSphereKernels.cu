// DEM bin--sphere relations-related custom kernels
#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>
_kernelIncludes_;

// If clump templates are jitified, they will be below
_clumpTemplateDefs_;
// Definitions of analytical entites are below
_analyticalEntityDefs_;

__global__ void getNumberOfBinsEachSphereTouches(deme::DEMSimParams* simParams,
                                                 deme::DEMDataKT* granData,
                                                 deme::binsSphereTouches_t* numBinsSphereTouches,
                                                 deme::objID_t* numAnalGeoSphereTouches) {
    deme::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < simParams->nSpheresGM) {
        // Register sphere--analytical geometry contacts
        deme::objID_t contact_count = 0;
        // Sphere's family ID
        unsigned int sphFamilyNum;
        double3 myPosXYZ;
        double myRadius;
        {
            // My sphere voxel ID and my relPos
            deme::bodyID_t myOwnerID = granData->ownerClumpBody[sphereID];
            sphFamilyNum = granData->familyID[myOwnerID];
            float3 myRelPos;
            double3 ownerXYZ;

            // Get my component offset info from either jitified arrays or global memory
            // Outputs myRelPos, myRadius (in CD kernels, radius needs to be expanded)
            // Use an input named exactly `sphereID' which is the id of this sphere component
            {
                _componentAcqStrat_;
                myRadius += granData->marginSize[myOwnerID];
            }

            {
                voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
                    ownerXYZ.x, ownerXYZ.y, ownerXYZ.z, granData->voxelID[myOwnerID], granData->locX[myOwnerID],
                    granData->locY[myOwnerID], granData->locZ[myOwnerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
                const float myOriQw = granData->oriQw[myOwnerID];
                const float myOriQx = granData->oriQx[myOwnerID];
                const float myOriQy = granData->oriQy[myOwnerID];
                const float myOriQz = granData->oriQz[myOwnerID];
                applyOriQToVector3<float, deme::oriQ_t>(myRelPos.x, myRelPos.y, myRelPos.z, myOriQw, myOriQx, myOriQy,
                                                        myOriQz);
                myPosXYZ = ownerXYZ + to_double3(myRelPos);
            }

            deme::binsSphereTouches_t numX, numY, numZ;
            {
                // The bin number that I live in (with fractions)?
                double myBinX = myPosXYZ.x / simParams->binSize;
                double myBinY = myPosXYZ.y / simParams->binSize;
                double myBinZ = myPosXYZ.z / simParams->binSize;
                // How many bins my radius spans (with fractions)?
                double myRadiusSpan = myRadius / simParams->binSize;
                // printf("myRadius: %f\n", myRadiusSpan);
                // Now, figure out how many bins I touch in each direction
                numX = ((myBinX + myRadiusSpan < (double)simParams->nbX) ? (unsigned int)(myBinX + myRadiusSpan)
                                                                         : (unsigned int)simParams->nbX - 1) -
                       (unsigned int)((myBinX - myRadiusSpan > 0.0) ? myBinX - myRadiusSpan : 0.0) + 1;
                numY = ((myBinY + myRadiusSpan < (double)simParams->nbY) ? (unsigned int)(myBinY + myRadiusSpan)
                                                                         : (unsigned int)simParams->nbY - 1) -
                       (unsigned int)((myBinY - myRadiusSpan > 0.0) ? myBinY - myRadiusSpan : 0.0) + 1;
                numZ = ((myBinZ + myRadiusSpan < (double)simParams->nbZ) ? (unsigned int)(myBinZ + myRadiusSpan)
                                                                         : (unsigned int)simParams->nbZ - 1) -
                       (unsigned int)((myBinZ - myRadiusSpan > 0.0) ? myBinZ - myRadiusSpan : 0.0) + 1;
                //// TODO: Add an error message if numX * numY * numZ > MAX(binsSphereTouches_t)
            }

            // Write the number of bins this sphere touches back to the global array
            numBinsSphereTouches[sphereID] = numX * numY * numZ;
            // printf("This sp takes num of bins: %u\n", numX * numY * numZ);
        }

        // Each sphere entity should also check if it overlaps with an analytical boundary-type geometry
        for (deme::objID_t objB = 0; objB < simParams->nAnalGM; objB++) {
            deme::bodyID_t objBOwner = objOwner[objB];
            // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
            unsigned int objFamilyNum = granData->familyID[objBOwner];
            unsigned int maskMatID =
                locateMaskPair<unsigned int>((unsigned int)sphFamilyNum, (unsigned int)objFamilyNum);
            // If marked no contact, skip ths iteration
            if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                continue;
            }
            double3 ownerXYZ;
            voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
                ownerXYZ.x, ownerXYZ.y, ownerXYZ.z, granData->voxelID[objBOwner], granData->locX[objBOwner],
                granData->locY[objBOwner], granData->locZ[objBOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            const float ownerOriQw = granData->oriQw[objBOwner];
            const float ownerOriQx = granData->oriQx[objBOwner];
            const float ownerOriQy = granData->oriQy[objBOwner];
            const float ownerOriQz = granData->oriQz[objBOwner];
            float objBRelPosX = objRelPosX[objB];
            float objBRelPosY = objRelPosY[objB];
            float objBRelPosZ = objRelPosZ[objB];
            float objBRotX = objRotX[objB];
            float objBRotY = objRotY[objB];
            float objBRotZ = objRotZ[objB];
            applyOriQToVector3<float, deme::oriQ_t>(objBRelPosX, objBRelPosY, objBRelPosZ, ownerOriQw, ownerOriQx,
                                                    ownerOriQy, ownerOriQz);
            applyOriQToVector3<float, deme::oriQ_t>(objBRotX, objBRotY, objBRotZ, ownerOriQw, ownerOriQx, ownerOriQy,
                                                    ownerOriQz);
            double3 objBPosXYZ = ownerXYZ + make_double3(objBRelPosX, objBRelPosY, objBRelPosZ);

            double overlapDepth;
            deme::contact_t contact_type;
            {
                double3 cntPnt;  // cntPnt here is a placeholder
                float3 cntNorm;  // cntNorm is placeholder too
                contact_type = checkSphereEntityOverlap<double3, float, double>(
                    myPosXYZ, myRadius, objType[objB], objBPosXYZ, make_float3(objBRotX, objBRotY, objBRotZ),
                    objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB], granData->marginSize[objBOwner],
                    cntPnt, cntNorm, overlapDepth);
            }
            // overlapDepth (which has both entities' full margins) needs to be larger than the smaller one of the two
            // added margin to be considered in-contact.
            double marginThres =
                (granData->familyExtraMarginSize[sphFamilyNum] < granData->familyExtraMarginSize[objFamilyNum])
                    ? granData->familyExtraMarginSize[sphFamilyNum]
                    : granData->familyExtraMarginSize[objFamilyNum];
            if (contact_type && overlapDepth > marginThres) {
                contact_count++;
            }
        }
        numAnalGeoSphereTouches[sphereID] = contact_count;
    }
}

__global__ void populateBinSphereTouchingPairs(deme::DEMSimParams* simParams,
                                               deme::DEMDataKT* granData,
                                               deme::binSphereTouchPairs_t* numBinsSphereTouchesScan,
                                               deme::binSphereTouchPairs_t* numAnalGeoSphereTouchesScan,
                                               deme::binID_t* binIDsEachSphereTouches,
                                               deme::bodyID_t* sphereIDsEachBinTouches,
                                               deme::bodyID_t* idGeoA,
                                               deme::bodyID_t* idGeoB,
                                               deme::contact_t* contactType) {
    deme::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < simParams->nSpheresGM) {
        double3 myPosXYZ;
        double myRadius;
        unsigned int sphFamilyNum;

        {
            // My sphere voxel ID and my relPos
            deme::bodyID_t myOwnerID = granData->ownerClumpBody[sphereID];
            sphFamilyNum = granData->familyID[myOwnerID];
            float3 myRelPos;
            double3 ownerXYZ;

            // Get my component offset info from either jitified arrays or global memory
            // Outputs myRelPos, myRadius (in CD kernels, radius needs to be expanded)
            // Use an input named exactly `sphereID' which is the id of this sphere component
            {
                _componentAcqStrat_;
                myRadius += granData->marginSize[myOwnerID];
            }

            // Get the offset of my spot where I should start writing back to the global bin--sphere pair registration
            // array
            deme::binSphereTouchPairs_t myReportOffset = numBinsSphereTouchesScan[sphereID];
            const deme::binSphereTouchPairs_t myReportOffset_end = numBinsSphereTouchesScan[sphereID + 1];

            {
                voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
                    ownerXYZ.x, ownerXYZ.y, ownerXYZ.z, granData->voxelID[myOwnerID], granData->locX[myOwnerID],
                    granData->locY[myOwnerID], granData->locZ[myOwnerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
                const float myOriQw = granData->oriQw[myOwnerID];
                const float myOriQx = granData->oriQx[myOwnerID];
                const float myOriQy = granData->oriQy[myOwnerID];
                const float myOriQz = granData->oriQz[myOwnerID];
                applyOriQToVector3<float, deme::oriQ_t>(myRelPos.x, myRelPos.y, myRelPos.z, myOriQw, myOriQx, myOriQy,
                                                        myOriQz);
                myPosXYZ = ownerXYZ + to_double3(myRelPos);
            }

            // The bin number that I live in (with fractions)?
            double myBinX = myPosXYZ.x / simParams->binSize;
            double myBinY = myPosXYZ.y / simParams->binSize;
            double myBinZ = myPosXYZ.z / simParams->binSize;
            // How many bins my radius spans (with fractions)?
            double myRadiusSpan = myRadius / simParams->binSize;
            // Now, write the IDs of those bins that I touch, back to the global memory
            deme::binID_t thisBinID;
            for (deme::binID_t k = (deme::binID_t)((myBinZ - myRadiusSpan > 0.0) ? myBinZ - myRadiusSpan : 0.0);
                 (k <= (deme::binID_t)(myBinZ + myRadiusSpan)) && (k < simParams->nbZ); k++) {
                for (deme::binID_t j = (deme::binID_t)((myBinY - myRadiusSpan > 0.0) ? myBinY - myRadiusSpan : 0.0);
                     (j <= (deme::binID_t)(myBinY + myRadiusSpan)) && (j < simParams->nbY); j++) {
                    for (deme::binID_t i = (deme::binID_t)((myBinX - myRadiusSpan > 0.0) ? myBinX - myRadiusSpan : 0.0);
                         (i <= (deme::binID_t)(myBinX + myRadiusSpan)) && (i < simParams->nbX); i++) {
                        if (myReportOffset >= myReportOffset_end) {
                            continue;  // No stepping on the next one's domain
                        }
                        thisBinID =
                            binIDFrom3Indices<deme::binID_t>(i, j, k, simParams->nbX, simParams->nbY, simParams->nbZ);
                        binIDsEachSphereTouches[myReportOffset] = thisBinID;
                        sphereIDsEachBinTouches[myReportOffset] = sphereID;
                        myReportOffset++;
                    }
                }
            }
            // First found that this `not filled' problem can happen in the triangle bin--tri intersection detections
            // part... Quite peculiar.
            for (; myReportOffset < myReportOffset_end; myReportOffset++) {
                binIDsEachSphereTouches[myReportOffset] = deme::NULL_BINID;
                sphereIDsEachBinTouches[myReportOffset] = sphereID;
            }
        }

        deme::binSphereTouchPairs_t mySphereGeoReportOffset = numAnalGeoSphereTouchesScan[sphereID];
        deme::binSphereTouchPairs_t mySphereGeoReportOffset_end = numAnalGeoSphereTouchesScan[sphereID + 1];
        // Each sphere entity should also check if it overlaps with an analytical boundary-type geometry
        for (deme::objID_t objB = 0; objB < simParams->nAnalGM; objB++) {
            deme::bodyID_t objBOwner = objOwner[objB];
            // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
            unsigned int objFamilyNum = granData->familyID[objBOwner];
            unsigned int maskMatID = locateMaskPair<unsigned int>(sphFamilyNum, objFamilyNum);
            // If marked no contact, skip ths iteration
            if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                continue;
            }
            double3 ownerXYZ;
            voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
                ownerXYZ.x, ownerXYZ.y, ownerXYZ.z, granData->voxelID[objBOwner], granData->locX[objBOwner],
                granData->locY[objBOwner], granData->locZ[objBOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            const float ownerOriQw = granData->oriQw[objBOwner];
            const float ownerOriQx = granData->oriQx[objBOwner];
            const float ownerOriQy = granData->oriQy[objBOwner];
            const float ownerOriQz = granData->oriQz[objBOwner];
            float objBRelPosX = objRelPosX[objB];
            float objBRelPosY = objRelPosY[objB];
            float objBRelPosZ = objRelPosZ[objB];
            float objBRotX = objRotX[objB];
            float objBRotY = objRotY[objB];
            float objBRotZ = objRotZ[objB];
            applyOriQToVector3<float, deme::oriQ_t>(objBRelPosX, objBRelPosY, objBRelPosZ, ownerOriQw, ownerOriQx,
                                                    ownerOriQy, ownerOriQz);
            applyOriQToVector3<float, deme::oriQ_t>(objBRotX, objBRotY, objBRotZ, ownerOriQw, ownerOriQx, ownerOriQy,
                                                    ownerOriQz);
            double3 objBPosXYZ = ownerXYZ + make_double3(objBRelPosX, objBRelPosY, objBRelPosZ);

            double overlapDepth;
            deme::contact_t contact_type;
            {
                double3 cntPnt;  // cntPnt here is a placeholder
                float3 cntNorm;  // cntNorm is placeholder too
                contact_type = checkSphereEntityOverlap<double3, float, double>(
                    myPosXYZ, myRadius, objType[objB], objBPosXYZ, make_float3(objBRotX, objBRotY, objBRotZ),
                    objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB], granData->marginSize[objBOwner],
                    cntPnt, cntNorm, overlapDepth);
            }
            // overlapDepth (which has both entities' full margins) needs to be larger than the smaller one of the two
            // added margin to be considered in-contact.
            double marginThres =
                (granData->familyExtraMarginSize[sphFamilyNum] < granData->familyExtraMarginSize[objFamilyNum])
                    ? granData->familyExtraMarginSize[sphFamilyNum]
                    : granData->familyExtraMarginSize[objFamilyNum];
            if (contact_type && overlapDepth > marginThres) {
                idGeoA[mySphereGeoReportOffset] = sphereID;
                idGeoB[mySphereGeoReportOffset] = (deme::bodyID_t)objB;
                contactType[mySphereGeoReportOffset] = contact_type;
                mySphereGeoReportOffset++;
                if (mySphereGeoReportOffset >= mySphereGeoReportOffset_end) {
                    return;  // Don't step on the next sphere's domain
                }
            }
        }
        // In practice, I've never seen non-illed contact slots that need to be resolved this way. It's purely for ultra
        // safety.
        for (; mySphereGeoReportOffset < mySphereGeoReportOffset_end; mySphereGeoReportOffset++) {
            contactType[mySphereGeoReportOffset] = deme::NOT_A_CONTACT;
        }
    }
}
