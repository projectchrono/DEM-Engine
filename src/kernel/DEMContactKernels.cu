// DEM contact detection related custom kernels
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
        sgps::voxelID_t ownerVoxelX;
        sgps::voxelID_t ownerVoxelY;
        sgps::voxelID_t ownerVoxelZ;
        IDChopper<sgps::voxelID_t, sgps::voxelID_t>(ownerVoxelX, ownerVoxelY, ownerVoxelZ, granData->voxelID[myOwnerID],
                                                    simParams->nvXp2, simParams->nvYp2);
        float myRelPosX = CDRelPosX[myCompOffset];
        float myRelPosY = CDRelPosY[myCompOffset];
        float myRelPosZ = CDRelPosZ[myCompOffset];
        applyOriQToVector3<float, float>(myRelPosX, myRelPosY, myRelPosZ);
        // The bin number that I live in (with fractions)?
        double myBinX = ((double)ownerVoxelX * simParams->voxelSize + (double)myRelPosX) / simParams->binSize;
        double myBinY = ((double)ownerVoxelY * simParams->voxelSize + (double)myRelPosY) / simParams->binSize;
        double myBinZ = ((double)ownerVoxelZ * simParams->voxelSize + (double)myRelPosZ) / simParams->binSize;
        // How many bins my radius spans (with fractions)?
        double myRadiusSpan = (double)(CDRadii[myCompOffset]) / simParams->binSize;
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

__global__ void populateBinSphereTouchingPairs(sgps::DEMSimParams* simParams, sgps::DEMDataKT* granData) {
    // __shared__ const distinctSphereRadii[???];
    sgps::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < simParams->nSpheresGM) {
    }
}
