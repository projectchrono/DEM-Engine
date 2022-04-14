// DEM bin--sphere relations-related custom kernels
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>

__global__ void getNumberOfBinsEachSphereTouches(sgps::DEMDataKT* granData,
                                                 sgps::binsSphereTouches_t* numBinsSphereTouches) {
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
    __syncthreads();

    sgps::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < _nSpheresGM_) {
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
        float myOriQ0 = granData->oriQ0[myOwnerID];
        float myOriQ1 = granData->oriQ1[myOwnerID];
        float myOriQ2 = granData->oriQ2[myOwnerID];
        float myOriQ3 = granData->oriQ3[myOwnerID];
        applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQ0, myOriQ1, myOriQ2, myOriQ3);
        // The bin number that I live in (with fractions)?
        double myBinX = (ownerX + (double)myRelPosX) / _binSize_;
        double myBinY = (ownerY + (double)myRelPosY) / _binSize_;
        double myBinZ = (ownerZ + (double)myRelPosZ) / _binSize_;
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
    }
}

__global__ void populateBinSphereTouchingPairs(sgps::DEMDataKT* granData,
                                               sgps::binSphereTouchPairs_t* numBinsSphereTouchesScan,
                                               sgps::binID_t* binIDsEachSphereTouches,
                                               sgps::bodyID_t* sphereIDsEachBinTouches) {
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
    __syncthreads();

    sgps::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < _nSpheresGM_) {
        // My sphere voxel ID and my relPos
        sgps::bodyID_t myOwnerID = granData->ownerClumpBody[sphereID];
        sgps::clumpComponentOffset_t myCompOffset = granData->clumpComponentOffset[sphereID];
        // Get the offset of my spot where I should start writing back to the global bin--sphere pair registration array
        sgps::binSphereTouchPairs_t myReportOffset = numBinsSphereTouchesScan[sphereID];
        double ownerX, ownerY, ownerZ;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[myOwnerID], granData->locX[myOwnerID], granData->locY[myOwnerID],
            granData->locZ[myOwnerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        float myRelPosX = CDRelPosX[myCompOffset];
        float myRelPosY = CDRelPosY[myCompOffset];
        float myRelPosZ = CDRelPosZ[myCompOffset];
        float myOriQ0 = granData->oriQ0[myOwnerID];
        float myOriQ1 = granData->oriQ1[myOwnerID];
        float myOriQ2 = granData->oriQ2[myOwnerID];
        float myOriQ3 = granData->oriQ3[myOwnerID];
        applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, myOriQ0, myOriQ1, myOriQ2, myOriQ3);
        // The bin number that I live in (with fractions)?
        double myBinX = (ownerX + (double)myRelPosX) / _binSize_;
        double myBinY = (ownerY + (double)myRelPosY) / _binSize_;
        double myBinZ = (ownerZ + (double)myRelPosZ) / _binSize_;
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
    }
}
