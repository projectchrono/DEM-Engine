// DEM bin--sphere relations-related custom kernels
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>

__global__ void getNumberOfBinsEachSphereTouches(sgps::DEMDataKT* granData,
                                                 sgps::binsSphereTouches_t* numBinsSphereTouches,
                                                 sgps::objID_t* numAnalGeoSphereTouches) {
    // _nDistinctClumpComponents_ elements are in these arrays
    const float CDRadii[] = {_CDRadii_};
    const float CDRelPosX[] = {_CDRelPosX_};
    const float CDRelPosY[] = {_CDRelPosY_};
    const float CDRelPosZ[] = {_CDRelPosZ_};

    // _nFamilyMaskEntries_ elements are in this array
    const bool familyMasks[] = {_familyMasks_};

    // _nAnalGM_ elements are in these arrays
    const sgps::objType_t objType[] = {_objType_};
    const sgps::bodyID_t objOwner[] = {_objOwner_};
    const bool objNormal[] = {_objNormal_};
    const float objRelPosX[] = {_objRelPosX_};
    const float objRelPosY[] = {_objRelPosY_};
    const float objRelPosZ[] = {_objRelPosZ_};
    const float objRotX[] = {_objRotX_};
    const float objRotY[] = {_objRotY_};
    const float objRotZ[] = {_objRotZ_};
    const float objSize1[] = {_objSize1_};
    const float objSize2[] = {_objSize2_};
    const float objSize3[] = {_objSize3_};

    sgps::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;

    if (sphereID < _nSpheresGM_) {
        // Register sphere--analytical geometry contacts
        sgps::objID_t contact_count = 0;
        // Sphere's family ID
        unsigned int sphFamilyNum;
        double myPosX, myPosY, myPosZ;
        double myRadius;
        {
            // My sphere voxel ID and my relPos
            sgps::bodyID_t myOwnerID = granData->ownerClumpBody[sphereID];
            sphFamilyNum = granData->familyID[myOwnerID];
            sgps::clumpComponentOffset_t myCompOffset = granData->clumpComponentOffset[sphereID];
            double ownerX, ownerY, ownerZ;
            voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
                ownerX, ownerY, ownerZ, granData->voxelID[myOwnerID], granData->locX[myOwnerID],
                granData->locY[myOwnerID], granData->locZ[myOwnerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            float myRelPosX = CDRelPosX[myCompOffset];
            float myRelPosY = CDRelPosY[myCompOffset];
            float myRelPosZ = CDRelPosZ[myCompOffset];
            const float myOriQ0 = granData->oriQ0[myOwnerID];
            const float myOriQ1 = granData->oriQ1[myOwnerID];
            const float myOriQ2 = granData->oriQ2[myOwnerID];
            const float myOriQ3 = granData->oriQ3[myOwnerID];
            applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQ0, myOriQ1, myOriQ2, myOriQ3);
            // The bin number that I live in (with fractions)?
            myPosX = ownerX + (double)myRelPosX;
            myPosY = ownerY + (double)myRelPosY;
            myPosZ = ownerZ + (double)myRelPosZ;
            double myBinX = myPosX / _binSize_;
            double myBinY = myPosY / _binSize_;
            double myBinZ = myPosZ / _binSize_;
            // printf("myBinXYZ: %f, %f, %f\n", ownerX, ownerY, ownerZ);
            // printf("myVoxel: %lu\n", granData->voxelID[myOwnerID]);
            // How many bins my radius spans (with fractions)?
            myRadius = (double)CDRadii[myCompOffset];
            double myRadiusSpan = myRadius / _binSize_;
            // printf("myRadius: %f\n", myRadiusSpan);
            // Now, figure out how many bins I touch in each direction
            sgps::binsSphereTouches_t numX =
                (unsigned int)(myBinX + myRadiusSpan) - (unsigned int)(myBinX - myRadiusSpan) + 1;
            sgps::binsSphereTouches_t numY =
                (unsigned int)(myBinY + myRadiusSpan) - (unsigned int)(myBinY - myRadiusSpan) + 1;
            sgps::binsSphereTouches_t numZ =
                (unsigned int)(myBinZ + myRadiusSpan) - (unsigned int)(myBinZ - myRadiusSpan) + 1;
            // TODO: Add an error message if numX * numY * numZ > MAX(binsSphereTouches_t)

            // Write the number of bins this sphere touches back to the global array
            numBinsSphereTouches[sphereID] = numX * numY * numZ;
            // printf("This sp takes num of bins: %u\n", numX * numY * numZ);
        }

        // Each sphere entity should also check if it overlaps with an analytical boundary-type geometry
        for (sgps::objID_t objB = 0; objB < _nAnalGM_; objB++) {
            sgps::contact_t contact_type;
            sgps::bodyID_t objBOwner = objOwner[objB];
            // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
            unsigned int objFamilyNum = granData->familyID[objBOwner];
            unsigned int maskMatID =
                locateMatPair<unsigned int>((unsigned int)sphFamilyNum, (unsigned int)objFamilyNum);
            // If marked no contact, skip ths iteration
            if (familyMasks[maskMatID] != sgps::DEM_DONT_PREVENT_CONTACT) {
                continue;
            }
            double ownerX, ownerY, ownerZ;
            voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
                ownerX, ownerY, ownerZ, granData->voxelID[objBOwner], granData->locX[objBOwner],
                granData->locY[objBOwner], granData->locZ[objBOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            const float ownerOriQ0 = granData->oriQ0[objBOwner];
            const float ownerOriQ1 = granData->oriQ1[objBOwner];
            const float ownerOriQ2 = granData->oriQ2[objBOwner];
            const float ownerOriQ3 = granData->oriQ3[objBOwner];
            float objBRelPosX = objRelPosX[objB];
            float objBRelPosY = objRelPosY[objB];
            float objBRelPosZ = objRelPosZ[objB];
            float objBRotX = objRotX[objB];
            float objBRotY = objRotY[objB];
            float objBRotZ = objRotZ[objB];
            applyOriQ2Vector3<float, sgps::oriQ_t>(objBRelPosX, objBRelPosX, objBRelPosX, ownerOriQ0, ownerOriQ1,
                                                   ownerOriQ2, ownerOriQ3);
            applyOriQ2Vector3<float, sgps::oriQ_t>(objBRotX, objBRotY, objBRotZ, ownerOriQ0, ownerOriQ1, ownerOriQ2,
                                                   ownerOriQ3);
            double objBPosX = ownerX + (double)objBRelPosX;
            double objBPosY = ownerY + (double)objBRelPosY;
            double objBPosZ = ownerZ + (double)objBRelPosZ;
            contact_type = checkSphereEntityOverlap<double>(
                myPosX, myPosY, myPosZ, myRadius, objType[objB], objBPosX, objBPosY, objBPosZ, objBRotX, objBRotY,
                objBRotZ, objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB], _beta_);

            if (contact_type) {
                contact_count++;
            }
        }
        numAnalGeoSphereTouches[sphereID] = contact_count;
    }
}

__global__ void populateBinSphereTouchingPairs(sgps::DEMDataKT* granData,
                                               sgps::binSphereTouchPairs_t* numBinsSphereTouchesScan,
                                               sgps::binSphereTouchPairs_t* numAnalGeoSphereTouchesScan,
                                               sgps::binID_t* binIDsEachSphereTouches,
                                               sgps::bodyID_t* sphereIDsEachBinTouches,
                                               sgps::bodyID_t* idGeoA,
                                               sgps::bodyID_t* idGeoB,
                                               sgps::contact_t* contactType) {
    //  elements are in these arrays
    const float CDRadii[] = {_CDRadii_};
    const float CDRelPosX[] = {_CDRelPosX_};
    const float CDRelPosY[] = {_CDRelPosY_};
    const float CDRelPosZ[] = {_CDRelPosZ_};

    // _nFamilyMaskEntries_ elements are in this array
    const bool familyMasks[] = {_familyMasks_};

    // _nAnalGM_ elements are in these arrays
    const sgps::objType_t objType[] = {_objType_};
    const sgps::bodyID_t objOwner[] = {_objOwner_};
    const bool objNormal[] = {_objNormal_};
    const float objRelPosX[] = {_objRelPosX_};
    const float objRelPosY[] = {_objRelPosY_};
    const float objRelPosZ[] = {_objRelPosZ_};
    const float objRotX[] = {_objRotX_};
    const float objRotY[] = {_objRotY_};
    const float objRotZ[] = {_objRotZ_};
    const float objSize1[] = {_objSize1_};
    const float objSize2[] = {_objSize2_};
    const float objSize3[] = {_objSize3_};

    sgps::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < _nSpheresGM_) {
        double myPosX, myPosY, myPosZ;
        double myRadius;
        unsigned int sphFamilyNum;
        sgps::binSphereTouchPairs_t mySphereGeoReportOffset = numAnalGeoSphereTouchesScan[sphereID];
        {
            // My sphere voxel ID and my relPos
            sgps::bodyID_t myOwnerID = granData->ownerClumpBody[sphereID];
            sphFamilyNum = granData->familyID[myOwnerID];
            sgps::clumpComponentOffset_t myCompOffset = granData->clumpComponentOffset[sphereID];
            // Get the offset of my spot where I should start writing back to the global bin--sphere pair registration
            // array
            sgps::binSphereTouchPairs_t myReportOffset = numBinsSphereTouchesScan[sphereID];

            double ownerX, ownerY, ownerZ;
            voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
                ownerX, ownerY, ownerZ, granData->voxelID[myOwnerID], granData->locX[myOwnerID],
                granData->locY[myOwnerID], granData->locZ[myOwnerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            float myRelPosX = CDRelPosX[myCompOffset];
            float myRelPosY = CDRelPosY[myCompOffset];
            float myRelPosZ = CDRelPosZ[myCompOffset];
            const float myOriQ0 = granData->oriQ0[myOwnerID];
            const float myOriQ1 = granData->oriQ1[myOwnerID];
            const float myOriQ2 = granData->oriQ2[myOwnerID];
            const float myOriQ3 = granData->oriQ3[myOwnerID];
            applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQ0, myOriQ1, myOriQ2, myOriQ3);
            // The bin number that I live in (with fractions)?
            myPosX = ownerX + (double)myRelPosX;
            myPosY = ownerY + (double)myRelPosY;
            myPosZ = ownerZ + (double)myRelPosZ;
            double myBinX = myPosX / _binSize_;
            double myBinY = myPosY / _binSize_;
            double myBinZ = myPosZ / _binSize_;
            // How many bins my radius spans (with fractions)?
            myRadius = (double)CDRadii[myCompOffset];
            double myRadiusSpan = myRadius / _binSize_;
            // Now, write the IDs of those bins that I touch, back to the global memory
            sgps::binID_t thisBinID;
            for (unsigned int k = (unsigned int)(myBinZ - myRadiusSpan); k <= (unsigned int)(myBinZ + myRadiusSpan);
                 k++) {
                for (unsigned int j = (unsigned int)(myBinY - myRadiusSpan); j <= (unsigned int)(myBinY + myRadiusSpan);
                     j++) {
                    for (unsigned int i = (unsigned int)(myBinX - myRadiusSpan);
                         i <= (unsigned int)(myBinX + myRadiusSpan); i++) {
                        thisBinID = (sgps::binID_t)i + (sgps::binID_t)j * _nbX_ + (sgps::binID_t)k * _nbX_ * _nbY_;
                        binIDsEachSphereTouches[myReportOffset] = thisBinID;
                        sphereIDsEachBinTouches[myReportOffset] = sphereID;
                        myReportOffset++;
                    }
                }
            }
        }

        // Each sphere entity should also check if it overlaps with an analytical boundary-type geometry
        for (sgps::objID_t objB = 0; objB < _nAnalGM_; objB++) {
            sgps::contact_t contact_type;
            sgps::bodyID_t objBOwner = objOwner[objB];
            // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
            unsigned int objFamilyNum = granData->familyID[objBOwner];
            unsigned int maskMatID = locateMatPair<unsigned int>(sphFamilyNum, objFamilyNum);
            // If marked no contact, skip ths iteration
            if (familyMasks[maskMatID] != sgps::DEM_DONT_PREVENT_CONTACT) {
                continue;
            }
            double ownerX, ownerY, ownerZ;
            voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
                ownerX, ownerY, ownerZ, granData->voxelID[objBOwner], granData->locX[objBOwner],
                granData->locY[objBOwner], granData->locZ[objBOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            const float ownerOriQ0 = granData->oriQ0[objBOwner];
            const float ownerOriQ1 = granData->oriQ1[objBOwner];
            const float ownerOriQ2 = granData->oriQ2[objBOwner];
            const float ownerOriQ3 = granData->oriQ3[objBOwner];
            float objBRelPosX = objRelPosX[objB];
            float objBRelPosY = objRelPosY[objB];
            float objBRelPosZ = objRelPosZ[objB];
            float objBRotX = objRotX[objB];
            float objBRotY = objRotY[objB];
            float objBRotZ = objRotZ[objB];
            applyOriQ2Vector3<float, sgps::oriQ_t>(objBRelPosX, objBRelPosX, objBRelPosX, ownerOriQ0, ownerOriQ1,
                                                   ownerOriQ2, ownerOriQ3);
            applyOriQ2Vector3<float, sgps::oriQ_t>(objBRotX, objBRotY, objBRotZ, ownerOriQ0, ownerOriQ1, ownerOriQ2,
                                                   ownerOriQ3);
            double objBPosX = ownerX + (double)objBRelPosX;
            double objBPosY = ownerY + (double)objBRelPosY;
            double objBPosZ = ownerZ + (double)objBRelPosZ;
            contact_type = checkSphereEntityOverlap<double>(
                myPosX, myPosY, myPosZ, myRadius, objType[objB], objBPosX, objBPosY, objBPosZ, objBRotX, objBRotY,
                objBRotZ, objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB], _beta_);

            if (contact_type) {
                idGeoA[mySphereGeoReportOffset] = sphereID;
                idGeoB[mySphereGeoReportOffset] = (sgps::bodyID_t)objB;
                contactType[mySphereGeoReportOffset] = contact_type;
                mySphereGeoReportOffset++;
            }
        }
    }
}
