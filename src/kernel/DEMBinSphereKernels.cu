// DEM bin--sphere relations-related custom kernels
#include <DEM/DEMDefines.h>
#include <kernel/DEMHelperKernels.cu>

// If clump templates are jitified, they will be below
_clumpTemplateDefs_;
// Definitions of analytical entites are below
_analyticalEntityDefs_;
// Family mask, _nFamilyMaskEntries_ elements are in this array
// __constant__ __device__ bool familyMasks[] = {_familyMasks_};

__global__ void getNumberOfBinsEachSphereTouches(sgps::DEMSimParams* simParams,
                                                 sgps::DEMDataKT* granData,
                                                 sgps::binsSphereTouches_t* numBinsSphereTouches,
                                                 sgps::objID_t* numAnalGeoSphereTouches) {
    sgps::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < simParams->nSpheresGM) {
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
            float myRelPosX, myRelPosY, myRelPosZ;
            double ownerX, ownerY, ownerZ;

            // Get my component offset info from either jitified arrays or global memory
            // Outputs myRelPosXYZ, myRadius (in CD kernels, radius needs to be expanded)
            // Use an input named exactly `sphereID' which is the id of this sphere component
            {
                _componentAcqStrat_;
                myRadius += simParams->beta;
            }

            voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
                ownerX, ownerY, ownerZ, granData->voxelID[myOwnerID], granData->locX[myOwnerID],
                granData->locY[myOwnerID], granData->locZ[myOwnerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            const float myOriQ0 = granData->oriQ0[myOwnerID];
            const float myOriQ1 = granData->oriQ1[myOwnerID];
            const float myOriQ2 = granData->oriQ2[myOwnerID];
            const float myOriQ3 = granData->oriQ3[myOwnerID];
            applyOriQToVector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQ0, myOriQ1, myOriQ2,
                                                    myOriQ3);
            // The bin number that I live in (with fractions)?
            myPosX = ownerX + (double)myRelPosX;
            myPosY = ownerY + (double)myRelPosY;
            myPosZ = ownerZ + (double)myRelPosZ;
            double myBinX = myPosX / simParams->binSize;
            double myBinY = myPosY / simParams->binSize;
            double myBinZ = myPosZ / simParams->binSize;
            // How many bins my radius spans (with fractions)?
            double myRadiusSpan = myRadius / simParams->binSize;
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
        for (sgps::objID_t objB = 0; objB < simParams->nAnalGM; objB++) {
            sgps::contact_t contact_type;
            sgps::bodyID_t objBOwner = objOwner[objB];
            // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
            unsigned int objFamilyNum = granData->familyID[objBOwner];
            unsigned int maskMatID =
                locateMaskPair<unsigned int>((unsigned int)sphFamilyNum, (unsigned int)objFamilyNum);
            // If marked no contact, skip ths iteration
            if (granData->familyMasks[maskMatID] != sgps::DEM_DONT_PREVENT_CONTACT) {
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
            applyOriQToVector3<float, sgps::oriQ_t>(objBRelPosX, objBRelPosY, objBRelPosZ, ownerOriQ0, ownerOriQ1,
                                                    ownerOriQ2, ownerOriQ3);
            applyOriQToVector3<float, sgps::oriQ_t>(objBRotX, objBRotY, objBRotZ, ownerOriQ0, ownerOriQ1, ownerOriQ2,
                                                    ownerOriQ3);
            double objBPosX = ownerX + (double)objBRelPosX;
            double objBPosY = ownerY + (double)objBRelPosY;
            double objBPosZ = ownerZ + (double)objBRelPosZ;
            contact_type = checkSphereEntityOverlap<double>(
                myPosX, myPosY, myPosZ, myRadius, objType[objB], objBPosX, objBPosY, objBPosZ, objBRotX, objBRotY,
                objBRotZ, objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB], simParams->beta);

            if (contact_type) {
                contact_count++;
            }
        }
        numAnalGeoSphereTouches[sphereID] = contact_count;
    }
}

__global__ void populateBinSphereTouchingPairs(sgps::DEMSimParams* simParams,
                                               sgps::DEMDataKT* granData,
                                               sgps::binSphereTouchPairs_t* numBinsSphereTouchesScan,
                                               sgps::binSphereTouchPairs_t* numAnalGeoSphereTouchesScan,
                                               sgps::binID_t* binIDsEachSphereTouches,
                                               sgps::bodyID_t* sphereIDsEachBinTouches,
                                               sgps::bodyID_t* idGeoA,
                                               sgps::bodyID_t* idGeoB,
                                               sgps::contact_t* contactType) {
    sgps::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < simParams->nSpheresGM) {
        double myPosX, myPosY, myPosZ;
        double myRadius;
        unsigned int sphFamilyNum;
        sgps::binSphereTouchPairs_t mySphereGeoReportOffset = numAnalGeoSphereTouchesScan[sphereID];
        {
            // My sphere voxel ID and my relPos
            sgps::bodyID_t myOwnerID = granData->ownerClumpBody[sphereID];
            sphFamilyNum = granData->familyID[myOwnerID];
            float myRelPosX, myRelPosY, myRelPosZ;
            double ownerX, ownerY, ownerZ;

            // Get my component offset info from either jitified arrays or global memory
            // Outputs myRelPosXYZ, myRadius (in CD kernels, radius needs to be expanded)
            // Use an input named exactly `sphereID' which is the id of this sphere component
            {
                _componentAcqStrat_;
                myRadius += simParams->beta;
            }

            // Get the offset of my spot where I should start writing back to the global bin--sphere pair registration
            // array
            sgps::binSphereTouchPairs_t myReportOffset = numBinsSphereTouchesScan[sphereID];
            voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
                ownerX, ownerY, ownerZ, granData->voxelID[myOwnerID], granData->locX[myOwnerID],
                granData->locY[myOwnerID], granData->locZ[myOwnerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            const float myOriQ0 = granData->oriQ0[myOwnerID];
            const float myOriQ1 = granData->oriQ1[myOwnerID];
            const float myOriQ2 = granData->oriQ2[myOwnerID];
            const float myOriQ3 = granData->oriQ3[myOwnerID];
            applyOriQToVector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQ0, myOriQ1, myOriQ2,
                                                    myOriQ3);
            // The bin number that I live in (with fractions)?
            myPosX = ownerX + (double)myRelPosX;
            myPosY = ownerY + (double)myRelPosY;
            myPosZ = ownerZ + (double)myRelPosZ;
            double myBinX = myPosX / simParams->binSize;
            double myBinY = myPosY / simParams->binSize;
            double myBinZ = myPosZ / simParams->binSize;
            // How many bins my radius spans (with fractions)?
            double myRadiusSpan = myRadius / simParams->binSize;
            // Now, write the IDs of those bins that I touch, back to the global memory
            sgps::binID_t thisBinID;
            for (unsigned int k = (unsigned int)(myBinZ - myRadiusSpan); k <= (unsigned int)(myBinZ + myRadiusSpan);
                 k++) {
                for (unsigned int j = (unsigned int)(myBinY - myRadiusSpan); j <= (unsigned int)(myBinY + myRadiusSpan);
                     j++) {
                    for (unsigned int i = (unsigned int)(myBinX - myRadiusSpan);
                         i <= (unsigned int)(myBinX + myRadiusSpan); i++) {
                        thisBinID = (sgps::binID_t)i + (sgps::binID_t)j * simParams->nbX +
                                    (sgps::binID_t)k * simParams->nbX * simParams->nbY;
                        binIDsEachSphereTouches[myReportOffset] = thisBinID;
                        sphereIDsEachBinTouches[myReportOffset] = sphereID;
                        myReportOffset++;
                    }
                }
            }
        }

        // Each sphere entity should also check if it overlaps with an analytical boundary-type geometry
        for (sgps::objID_t objB = 0; objB < simParams->nAnalGM; objB++) {
            sgps::contact_t contact_type;
            sgps::bodyID_t objBOwner = objOwner[objB];
            // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
            unsigned int objFamilyNum = granData->familyID[objBOwner];
            unsigned int maskMatID = locateMaskPair<unsigned int>(sphFamilyNum, objFamilyNum);
            // If marked no contact, skip ths iteration
            if (granData->familyMasks[maskMatID] != sgps::DEM_DONT_PREVENT_CONTACT) {
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
            applyOriQToVector3<float, sgps::oriQ_t>(objBRelPosX, objBRelPosY, objBRelPosZ, ownerOriQ0, ownerOriQ1,
                                                    ownerOriQ2, ownerOriQ3);
            applyOriQToVector3<float, sgps::oriQ_t>(objBRotX, objBRotY, objBRotZ, ownerOriQ0, ownerOriQ1, ownerOriQ2,
                                                    ownerOriQ3);
            double objBPosX = ownerX + (double)objBRelPosX;
            double objBPosY = ownerY + (double)objBRelPosY;
            double objBPosZ = ownerZ + (double)objBRelPosZ;
            contact_type = checkSphereEntityOverlap<double>(
                myPosX, myPosY, myPosZ, myRadius, objType[objB], objBPosX, objBPosY, objBPosZ, objBRotX, objBRotY,
                objBRotZ, objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB], simParams->beta);

            if (contact_type) {
                idGeoA[mySphereGeoReportOffset] = sphereID;
                idGeoB[mySphereGeoReportOffset] = (sgps::bodyID_t)objB;
                contactType[mySphereGeoReportOffset] = contact_type;
                mySphereGeoReportOffset++;
            }
        }
    }
}
