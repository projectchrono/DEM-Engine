// DEM contact detection-related custom kernels
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>

inline __device__ bool ifTwoSpheresOverlap(double XA,
                                           double YA,
                                           double ZA,
                                           float radA,
                                           sgps::bodyID_t ownerA,
                                           double XB,
                                           double YB,
                                           double ZB,
                                           float radB,
                                           sgps::bodyID_t ownerB) {
    if (ownerA == ownerB) {
        return false;
    }
    if (distSquared<double>(XA, YA, ZA, XB, YB, ZB) > (radA + radB) * (radA + radB)) {
        return false;
    }
    // TODO: Check if the contact point is inside the bin
    return true;
}

__global__ void getNumberOfContactsEachBin(sgps::DEMSimParams* simParams,
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

    sgps::binID_t binID = blockIdx.x * blockDim.x + threadIdx.x;
    // I need to store all the sphereIDs that I am supposed to look into
    // A100 has about 164K shMem... these arrays really need to be small, or we can only fit a small number of bins in
    // one block
    sgps::bodyID_t ownerIDs[MAX_SPHERES_PER_BIN];
    sgps::clumpComponentOffset_t compOffsets[MAX_SPHERES_PER_BIN];
    double bodyX[MAX_SPHERES_PER_BIN];
    double bodyY[MAX_SPHERES_PER_BIN];
    double bodyZ[MAX_SPHERES_PER_BIN];
    if (binID < simParams->nActiveBins) {
        sgps::contactPairs_t contact_count = 0;
        // Grab the bodies that I care, put into local memory
        sgps::spheresBinTouches_t nBodiesMeHandle = granData->numSpheresBinTouches[binID];
        sgps::binsSphereTouches_t myBodiesTableEntry = granData->sphereIDsLookUpTable[binID];
        for (sgps::spheresBinTouches_t i = 0; i < nBodiesMeHandle; i++) {
            sgps::bodyID_t bodyID = granData->sphereIDsLookUpTable[myBodiesTableEntry + i];
            ownerIDs[i] = granData->ownerClumpBody[bodyID];
            sgps::clumpComponentOffset_t thisCompOffset = granData->clumpComponentOffset[bodyID];
            sgps::voxelID_t ownerVoxelX;
            sgps::voxelID_t ownerVoxelY;
            sgps::voxelID_t ownerVoxelZ;
            IDChopper<sgps::voxelID_t, sgps::voxelID_t>(ownerVoxelX, ownerVoxelY, ownerVoxelZ,
                                                        granData->voxelID[ownerIDs[i]], simParams->nvXp2,
                                                        simParams->nvYp2);
            float myRelPosX = CDRelPosX[thisCompOffset];
            float myRelPosY = CDRelPosY[thisCompOffset];
            float myRelPosZ = CDRelPosZ[thisCompOffset];
            applyOriQToVector3<float, float>(myRelPosX, myRelPosY, myRelPosZ);
            bodyX[i] = (double)ownerVoxelX * simParams->voxelSize + (double)myRelPosX;
            bodyY[i] = (double)ownerVoxelY * simParams->voxelSize + (double)myRelPosY;
            bodyZ[i] = (double)ownerVoxelZ * simParams->voxelSize + (double)myRelPosZ;
        }
        for (sgps::spheresBinTouches_t bodyA = 0; bodyA < nBodiesMeHandle - 1; bodyA++) {
            for (sgps::spheresBinTouches_t bodyB = bodyA + 1; bodyB < nBodiesMeHandle; bodyB++) {
                // For 2 bodies to be considered in contact, the contact point must be in this bin, and they do not
                // belong to the same clump
                bool in_contact = ifTwoSpheresOverlap(
                    bodyX[bodyA], bodyY[bodyA], bodyZ[bodyA], CDRadii[compOffsets[bodyA]], ownerIDs[bodyA],
                    bodyX[bodyB], bodyY[bodyB], bodyZ[bodyB], CDRadii[compOffsets[bodyB]], ownerIDs[bodyB]);
                if (in_contact) {
                    contact_count++;
                }
            }
        }
        granData->numContactsInEachBin[binID] = contact_count;
    }
}
