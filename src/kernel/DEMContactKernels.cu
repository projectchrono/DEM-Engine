// DEM contact detection related custom kernels
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>

__global__ void getNumberOfBinsEachSphereTouches(sgps::DEMSimParams* simParams,
                                                 sgps::DEMDataKT* granData,
                                                 sgps::DEMTemplate* granTemplates) {
    // __shared__ const distinctSphereRadii[@NUM_OF_THAT_ARR@] = {@THAT_ARR@};
    // TODO: These info should be jitfied not brought from global mem
    __shared__ unsigned int CDRadii[128];
    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < simParams->nDistinctClumpComponents; i++) {
            CDRadii[i] = granTemplates->inflatedRadiiVoxelRatio[i];
        }
    }
    __syncthreads();

    sgps::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < simParams->nSpheresGM) {
        printf("This sp takes bins: %u\n", CDRadii[granData->clumpComponentOffset[sphereID]] / simParams->binSize);
    }
}

__global__ void populateBinSphereTouchingPairs(sgps::DEMSimParams* simParams, sgps::DEMDataKT* granData) {
    // __shared__ const distinctSphereRadii[???];
    sgps::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < simParams->nSpheresGM) {
    }
}
