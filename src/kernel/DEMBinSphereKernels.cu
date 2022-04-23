// DEM bin--sphere relations-related custom kernels
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>

__global__ void getNumberOfBinsEachSphereTouches(sgps::DEMDataKT* granData,
                                                 sgps::binsSphereTouches_t* numBinsSphereTouches,
                                                 sgps::objID_t* numAnalGeoSphereTouches) {
    // CUDA does not support initializing shared arrays, so we have to manually load them
    __shared__ float CDRadii[_nDistinctClumpComponents_];
    __shared__ float CDRelPosX[_nDistinctClumpComponents_];
    __shared__ float CDRelPosY[_nDistinctClumpComponents_];
    __shared__ float CDRelPosZ[_nDistinctClumpComponents_];
    if (threadIdx.x < _nActiveLoadingThreads_) {
        const float jitifiedCDRadii[_nDistinctClumpComponents_] = {_CDRadii_};
        const float jitifiedCDRelPosX[_nDistinctClumpComponents_] = {_CDRelPosX_};
        const float jitifiedCDRelPosY[_nDistinctClumpComponents_] = {_CDRelPosY_};
        const float jitifiedCDRelPosZ[_nDistinctClumpComponents_] = {_CDRelPosZ_};
        for (sgps::clumpComponentOffset_t i = threadIdx.x; i < _nDistinctClumpComponents_;
             i += _nActiveLoadingThreads_) {
            CDRadii[i] = jitifiedCDRadii[i];
            CDRelPosX[i] = jitifiedCDRelPosX[i];
            CDRelPosY[i] = jitifiedCDRelPosY[i];
            CDRelPosZ[i] = jitifiedCDRelPosZ[i];
        }
    }
    const sgps::objType_t objType[_nAnalGMSafe_] = {_objType_};
    const bool objNormal[_nAnalGMSafe_] = {_objNormal_};
    const float objRelPosX[_nAnalGMSafe_] = {_objRelPosX_};
    const float objRelPosY[_nAnalGMSafe_] = {_objRelPosY_};
    const float objRelPosZ[_nAnalGMSafe_] = {_objRelPosZ_};
    const float objRotX[_nAnalGMSafe_] = {_objRotX_};
    const float objRotY[_nAnalGMSafe_] = {_objRotY_};
    const float objRotZ[_nAnalGMSafe_] = {_objRotZ_};
    const float objSize1[_nAnalGMSafe_] = {_objSize1_};
    const float objSize2[_nAnalGMSafe_] = {_objSize2_};
    const float objSize3[_nAnalGMSafe_] = {_objSize3_};
    __syncthreads();

    sgps::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;

    if (sphereID < _nSpheresGM_) {
        // Register sphere--analytical geometry contacts
        sgps::objID_t contact_count = 0;
        // My sphere voxel ID and my relPos
        sgps::bodyID_t myOwnerID = granData->ownerClumpBody[sphereID];
        sgps::clumpComponentOffset_t myCompOffset = granData->clumpComponentOffset[sphereID];
        double ownerX, ownerY, ownerZ;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[myOwnerID], granData->locX[myOwnerID], granData->locY[myOwnerID],
            granData->locZ[myOwnerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        float myRelPosX = CDRelPosX[myCompOffset];
        float myRelPosY = CDRelPosY[myCompOffset];
        float myRelPosZ = CDRelPosZ[myCompOffset];
        const float myOriQ0 = granData->oriQ0[myOwnerID];
        const float myOriQ1 = granData->oriQ1[myOwnerID];
        const float myOriQ2 = granData->oriQ2[myOwnerID];
        const float myOriQ3 = granData->oriQ3[myOwnerID];
        applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQ0, myOriQ1, myOriQ2, myOriQ3);
        // The bin number that I live in (with fractions)?
        double myPosX = ownerX + (double)myRelPosX;
        double myPosY = ownerY + (double)myRelPosY;
        double myPosZ = ownerZ + (double)myRelPosZ;
        double myBinX = myPosX / _binSize_;
        double myBinY = myPosY / _binSize_;
        double myBinZ = myPosZ / _binSize_;
        // printf("myBinXYZ: %f, %f, %f\n", ownerX, ownerY, ownerZ);
        // printf("myVoxel: %lu\n", granData->voxelID[myOwnerID]);
        // How many bins my radius spans (with fractions)?
        double myRadiusSpan = (double)(CDRadii[myCompOffset]) / _binSize_;
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

        // Each sphere entity should also check if it overlaps with an analytical boundary-type geometry
        for (sgps::objID_t objB = 0; objB < _nAnalGM_; objB++) {
            sgps::contact_t contact_type;
            contact_type = checkSphereEntityOverlap<double>(
                myPosX, myPosY, myPosZ, CDRadii[myCompOffset], objType[objB], objRelPosX[objB], objRelPosY[objB],
                objRelPosZ[objB], objRotX[objB], objRotY[objB], objRotZ[objB], objSize1[objB], objSize2[objB],
                objSize3[objB], objNormal[objB], _beta_);

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
    // CUDA does not support initializing shared arrays, so we have to manually load them
    __shared__ float CDRadii[_nDistinctClumpComponents_];
    __shared__ float CDRelPosX[_nDistinctClumpComponents_];
    __shared__ float CDRelPosY[_nDistinctClumpComponents_];
    __shared__ float CDRelPosZ[_nDistinctClumpComponents_];
    if (threadIdx.x < _nActiveLoadingThreads_) {
        const float jitifiedCDRadii[_nDistinctClumpComponents_] = {_CDRadii_};
        const float jitifiedCDRelPosX[_nDistinctClumpComponents_] = {_CDRelPosX_};
        const float jitifiedCDRelPosY[_nDistinctClumpComponents_] = {_CDRelPosY_};
        const float jitifiedCDRelPosZ[_nDistinctClumpComponents_] = {_CDRelPosZ_};
        for (sgps::clumpComponentOffset_t i = threadIdx.x; i < _nDistinctClumpComponents_;
             i += _nActiveLoadingThreads_) {
            CDRadii[i] = jitifiedCDRadii[i];
            CDRelPosX[i] = jitifiedCDRelPosX[i];
            CDRelPosY[i] = jitifiedCDRelPosY[i];
            CDRelPosZ[i] = jitifiedCDRelPosZ[i];
        }
    }
    const sgps::objType_t objType[_nAnalGMSafe_] = {_objType_};
    const bool objNormal[_nAnalGMSafe_] = {_objNormal_};
    const float objRelPosX[_nAnalGMSafe_] = {_objRelPosX_};
    const float objRelPosY[_nAnalGMSafe_] = {_objRelPosY_};
    const float objRelPosZ[_nAnalGMSafe_] = {_objRelPosZ_};
    const float objRotX[_nAnalGMSafe_] = {_objRotX_};
    const float objRotY[_nAnalGMSafe_] = {_objRotY_};
    const float objRotZ[_nAnalGMSafe_] = {_objRotZ_};
    const float objSize1[_nAnalGMSafe_] = {_objSize1_};
    const float objSize2[_nAnalGMSafe_] = {_objSize2_};
    const float objSize3[_nAnalGMSafe_] = {_objSize3_};
    __syncthreads();

    sgps::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < _nSpheresGM_) {
        // My sphere voxel ID and my relPos
        sgps::bodyID_t myOwnerID = granData->ownerClumpBody[sphereID];
        sgps::clumpComponentOffset_t myCompOffset = granData->clumpComponentOffset[sphereID];
        // Get the offset of my spot where I should start writing back to the global bin--sphere pair registration array
        sgps::binSphereTouchPairs_t myReportOffset = numBinsSphereTouchesScan[sphereID];
        sgps::binSphereTouchPairs_t mySphereGeoReportOffset = numAnalGeoSphereTouchesScan[sphereID];
        double ownerX, ownerY, ownerZ;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[myOwnerID], granData->locX[myOwnerID], granData->locY[myOwnerID],
            granData->locZ[myOwnerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        float myRelPosX = CDRelPosX[myCompOffset];
        float myRelPosY = CDRelPosY[myCompOffset];
        float myRelPosZ = CDRelPosZ[myCompOffset];
        const float myOriQ0 = granData->oriQ0[myOwnerID];
        const float myOriQ1 = granData->oriQ1[myOwnerID];
        const float myOriQ2 = granData->oriQ2[myOwnerID];
        const float myOriQ3 = granData->oriQ3[myOwnerID];
        applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQ0, myOriQ1, myOriQ2, myOriQ3);
        // The bin number that I live in (with fractions)?
        double myPosX = ownerX + (double)myRelPosX;
        double myPosY = ownerY + (double)myRelPosY;
        double myPosZ = ownerZ + (double)myRelPosZ;
        double myBinX = myPosX / _binSize_;
        double myBinY = myPosY / _binSize_;
        double myBinZ = myPosZ / _binSize_;
        // How many bins my radius spans (with fractions)?
        double myRadiusSpan = (double)(CDRadii[myCompOffset]) / _binSize_;
        // Now, write the IDs of those bins that I touch, back to the global memory
        sgps::binID_t thisBinID;
        for (unsigned int k = (unsigned int)(myBinZ - myRadiusSpan); k <= (unsigned int)(myBinZ + myRadiusSpan); k++) {
            for (unsigned int j = (unsigned int)(myBinY - myRadiusSpan); j <= (unsigned int)(myBinY + myRadiusSpan);
                 j++) {
                for (unsigned int i = (unsigned int)(myBinX - myRadiusSpan); i <= (unsigned int)(myBinX + myRadiusSpan);
                     i++) {
                    thisBinID = (sgps::binID_t)i + (sgps::binID_t)j * _nbX_ + (sgps::binID_t)k * _nbX_ * _nbY_;
                    binIDsEachSphereTouches[myReportOffset] = thisBinID;
                    sphereIDsEachBinTouches[myReportOffset] = sphereID;
                    myReportOffset++;
                }
            }
        }

        // Each sphere entity should also check if it overlaps with an analytical boundary-type geometry
        for (sgps::objID_t objB = 0; objB < _nAnalGM_; objB++) {
            sgps::contact_t contact_type;
            contact_type = checkSphereEntityOverlap<double>(
                myPosX, myPosY, myPosZ, CDRadii[myCompOffset], objType[objB], objRelPosX[objB], objRelPosY[objB],
                objRelPosZ[objB], objRotX[objB], objRotY[objB], objRotZ[objB], objSize1[objB], objSize2[objB],
                objSize3[objB], objNormal[objB], _beta_);

            if (contact_type) {
                idGeoA[mySphereGeoReportOffset] = sphereID;
                idGeoB[mySphereGeoReportOffset] = objB;
                contactType[mySphereGeoReportOffset] = contact_type;
                mySphereGeoReportOffset++;
            }
        }
    }
}
