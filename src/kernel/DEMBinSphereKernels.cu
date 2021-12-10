// DEM bin--sphere relations-related custom kernels
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>

__global__ void getNumberOfBinsEachSphereTouches(sgps::DEMSimParams* simParams,
                                                 sgps::DEMDataKT* granData,
                                                 sgps::DEMTemplate* granTemplates) {
    // __shared__ const distinctSphereRadii[@NUM_OF_THAT_ARR@] = {@THAT_ARR@};
    // TODO: These info should be jitfied not brought from global mem
    __shared__ float CDRadii[TEST_SHARED_SIZE];
    __shared__ float CDRelPosX[TEST_SHARED_SIZE];
    __shared__ float CDRelPosY[TEST_SHARED_SIZE];
    __shared__ float CDRelPosZ[TEST_SHARED_SIZE];
    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < simParams->nDistinctClumpComponents; i++) {
            CDRadii[i] = granTemplates->radiiSphere[i] * simParams->beta;
            CDRelPosX[i] = granTemplates->relPosSphereX[i];
            CDRelPosY[i] = granTemplates->relPosSphereY[i];
            CDRelPosZ[i] = granTemplates->relPosSphereZ[i];
        }
    }
    __syncthreads();

    sgps::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < simParams->nSpheresGM) {
        // My sphere voxel ID and my relPos
        sgps::bodyID_t myOwnerID = granData->ownerClumpBody[sphereID];
        sgps::clumpComponentOffset_t myCompOffset = granData->clumpComponentOffset[sphereID];
        double ownerX, ownerY, ownerZ;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[myOwnerID], granData->locX[myOwnerID], granData->locY[myOwnerID],
            granData->locZ[myOwnerID], simParams->nvXp2, simParams->nvYp2, simParams->voxelSize, simParams->l);
        float myRelPosX = CDRelPosX[myCompOffset];
        float myRelPosY = CDRelPosY[myCompOffset];
        float myRelPosZ = CDRelPosZ[myCompOffset];
        applyOriQToVector3<float, float>(myRelPosX, myRelPosY, myRelPosZ);
        // The bin number that I live in (with fractions)?
        double myBinX = (ownerX + (double)myRelPosX) / simParams->binSize;
        double myBinY = (ownerY + (double)myRelPosY) / simParams->binSize;
        double myBinZ = (ownerZ + (double)myRelPosZ) / simParams->binSize;
        // printf("myBinXYZ: %f, %f, %f\n", ownerX, ownerY, ownerZ);
        // printf("myVoxel: %lu\n", granData->voxelID[myOwnerID]);
        // How many bins my radius spans (with fractions)?
        double myRadiusSpan = (double)(CDRadii[myCompOffset]) / simParams->binSize;
        // printf("myRadius: %f\n", myRadiusSpan);
        // Now, figure out how many bins I touch in each direction
        sgps::binsSphereTouches_t numX =
            (unsigned int)(myBinX + myRadiusSpan) - (unsigned int)(myBinX - myRadiusSpan) + 1;
        sgps::binsSphereTouches_t numY =
            (unsigned int)(myBinY + myRadiusSpan) - (unsigned int)(myBinY - myRadiusSpan) + 1;
        sgps::binsSphereTouches_t numZ =
            (unsigned int)(myBinZ + myRadiusSpan) - (unsigned int)(myBinZ - myRadiusSpan) + 1;
        // Write the number of bins this sphere touches back to the global array
        granData->numBinsSphereTouches[sphereID] = numX * numY * numZ;
        // printf("This sp takes num of bins: %u\n", numX * numY * numZ);
    }
}

__global__ void populateBinSphereTouchingPairs(sgps::DEMSimParams* simParams,
                                               sgps::DEMDataKT* granData,
                                               sgps::DEMTemplate* granTemplates) {
    // __shared__ const distinctSphereRadii[@NUM_OF_THAT_ARR@] = {@THAT_ARR@};
    // TODO: These info should be jitfied not brought from global mem
    __shared__ float CDRadii[TEST_SHARED_SIZE];
    __shared__ float CDRelPosX[TEST_SHARED_SIZE];
    __shared__ float CDRelPosY[TEST_SHARED_SIZE];
    __shared__ float CDRelPosZ[TEST_SHARED_SIZE];
    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < simParams->nDistinctClumpComponents; i++) {
            CDRadii[i] = granTemplates->radiiSphere[i] * simParams->beta;
            CDRelPosX[i] = granTemplates->relPosSphereX[i];
            CDRelPosY[i] = granTemplates->relPosSphereY[i];
            CDRelPosZ[i] = granTemplates->relPosSphereZ[i];
        }
    }
    __syncthreads();

    sgps::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < simParams->nSpheresGM) {
        // My sphere voxel ID and my relPos
        sgps::bodyID_t myOwnerID = granData->ownerClumpBody[sphereID];
        sgps::clumpComponentOffset_t myCompOffset = granData->clumpComponentOffset[sphereID];
        // Get the offset of my spot where I should start writing back to the global bin--sphere pair registration array
        sgps::binsSphereTouches_t myReportOffset = granData->numBinsSphereTouches[sphereID];
        double ownerX, ownerY, ownerZ;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[myOwnerID], granData->locX[myOwnerID], granData->locY[myOwnerID],
            granData->locZ[myOwnerID], simParams->nvXp2, simParams->nvYp2, simParams->voxelSize, simParams->l);
        float myRelPosX = CDRelPosX[myCompOffset];
        float myRelPosY = CDRelPosY[myCompOffset];
        float myRelPosZ = CDRelPosZ[myCompOffset];
        applyOriQToVector3<float, float>(myRelPosX, myRelPosY, myRelPosZ);
        // The bin number that I live in (with fractions)?
        double myBinX = (ownerX + (double)myRelPosX) / simParams->binSize;
        double myBinY = (ownerY + (double)myRelPosY) / simParams->binSize;
        double myBinZ = (ownerZ + (double)myRelPosZ) / simParams->binSize;
        // How many bins my radius spans (with fractions)?
        double myRadiusSpan = (double)(CDRadii[myCompOffset]) / simParams->binSize;
        // Now, write the IDs of those bins that I touch, back to the global memory
        sgps::binID_t thisBinID;
        for (unsigned int k = (unsigned int)(myBinZ - myRadiusSpan); k <= (unsigned int)(myBinZ + myRadiusSpan); k++) {
            for (unsigned int j = (unsigned int)(myBinY - myRadiusSpan); j <= (unsigned int)(myBinY + myRadiusSpan);
                 j++) {
                for (unsigned int i = (unsigned int)(myBinX - myRadiusSpan); i <= (unsigned int)(myBinX + myRadiusSpan);
                     i++) {
                    thisBinID = (sgps::binID_t)i + (sgps::binID_t)j * simParams->nbX +
                                (sgps::binID_t)k * (sgps::binID_t)simParams->nbX * simParams->nbY;
                    granData->binIDsEachSphereTouches[myReportOffset] = thisBinID;
                    granData->sphereIDsEachBinTouches[myReportOffset] = sphereID;
                    myReportOffset++;
                }
            }
        }
    }
}
