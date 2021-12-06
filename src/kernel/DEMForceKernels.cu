// DEM force computation related custom kernels
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>

inline __device__ void cleanUpContactForces(sgps::bodyID_t thisBody,
                                            sgps::DEMSimParams* simParams,
                                            sgps::DEMDataDT* granData) {
    // Actually, h should be JITCed into the kernel itself
    granData->contactForces[thisBody].x = 0;
    granData->contactForces[thisBody].y = 0;
    granData->contactForces[thisBody].z = 0;
}

inline __device__ void cleanUpAcc(sgps::bodyID_t thisClump, sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData) {
    // Actually, h should be JITCed into the kernel itself
    granData->h2aX[thisClump] = 0;
    granData->h2aY[thisClump] = 0;
    granData->h2aZ[thisClump] = 0;
}

inline __device__ void applyGravity(sgps::bodyID_t thisClump,
                                    sgps::DEMSimParams* simParams,
                                    sgps::DEMDataDT* granData) {
    // Actually, l should be JITCed into the kernel itself
    granData->h2aX[thisClump] += simParams->h * simParams->h * simParams->Gx / simParams->l;
    granData->h2aY[thisClump] += simParams->h * simParams->h * simParams->Gy / simParams->l;
    granData->h2aZ[thisClump] += simParams->h * simParams->h * simParams->Gz / simParams->l;
}

__global__ void prepareForceArrays(sgps::DEMSimParams* simParams,
                                   sgps::DEMDataDT* granData,
                                   sgps::DEMTemplate* granTemplates) {
    sgps::bodyID_t thisBody = blockIdx.x * blockDim.x + threadIdx.x;
    if (thisBody < simParams->nSpheresGM) {
        cleanUpContactForces(thisBody, simParams, granData);
        // Exploiting that nClumps <= nSpheres
        if (thisBody < simParams->nClumpBodies) {
            cleanUpAcc(thisBody, simParams, granData);
            applyGravity(thisBody, simParams, granData);
        }
    }
}

__global__ void calculateNormalContactForces(sgps::DEMSimParams* simParams,
                                             sgps::DEMDataDT* granData,
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

    sgps::contactPairs_t myContactID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myContactID < simParams->nContactPairs) {
        // From a contact ID, grab relevant info on 2 contact bodies
        sgps::bodyID_t bodyA = granData->idGeometryA[myContactID];
        sgps::bodyID_t bodyB = granData->idGeometryB[myContactID];
        sgps::bodyID_t bodyAOwner = granData->ownerClumpBody[bodyA];
        sgps::bodyID_t bodyBOwner = granData->ownerClumpBody[bodyB];
        sgps::clumpComponentOffset_t bodyACompOffset = granData->clumpComponentOffset[bodyA];
        sgps::clumpComponentOffset_t bodyBCompOffset = granData->clumpComponentOffset[bodyB];

        // Take care of 2 bodies in order, bodyA first
        double ownerX, ownerY, ownerZ;
        voxelID2Position<double, sgps::bodyID_t, sgps::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[bodyAOwner], granData->locX[bodyAOwner],
            granData->locY[bodyAOwner], granData->locZ[bodyAOwner], simParams->nvXp2, simParams->nvYp2,
            simParams->voxelSize, simParams->l);
        float myRelPosX = CDRelPosX[bodyACompOffset];
        float myRelPosY = CDRelPosY[bodyACompOffset];
        float myRelPosZ = CDRelPosZ[bodyACompOffset];
        applyOriQToVector3<float, float>(myRelPosX, myRelPosY, myRelPosZ);
        double bodyAX = ownerX + (double)myRelPosX;
        double bodyAY = ownerY + (double)myRelPosY;
        double bodyAZ = ownerZ + (double)myRelPosZ;
        // Then bodyB
        voxelID2Position<double, sgps::bodyID_t, sgps::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[bodyBOwner], granData->locX[bodyBOwner],
            granData->locY[bodyBOwner], granData->locZ[bodyBOwner], simParams->nvXp2, simParams->nvYp2,
            simParams->voxelSize, simParams->l);
        myRelPosX = CDRelPosX[bodyBCompOffset];
        myRelPosY = CDRelPosY[bodyBCompOffset];
        myRelPosZ = CDRelPosZ[bodyBCompOffset];
        applyOriQToVector3<float, float>(myRelPosX, myRelPosY, myRelPosZ);
        double bodyBX = ownerX + (double)myRelPosX;
        double bodyBY = ownerY + (double)myRelPosY;
        double bodyBZ = ownerZ + (double)myRelPosZ;

        // Now compute the contact point to see if they are truely still in contact
        double contactPntX, contactPntY, contactPntZ;
        double A2BX, A2BY, A2BZ;
        double overlapDepth;
        bool in_contact;
        checkSpheresOverlap<double>(bodyAX, bodyAY, bodyAZ, CDRadii[bodyACompOffset], bodyBX, bodyBY, bodyBZ,
                                    CDRadii[bodyBCompOffset], contactPntX, contactPntY, contactPntZ, A2BX, A2BY, A2BZ,
                                    overlapDepth, in_contact);
        if (in_contact) {
            // Instead of add the force to a body-based register, we store it in event-based register, and use CUB to
            // reduce them afterwards atomicAdd(granData->bodyForceX + bodyA, -force * A2BX);
            // atomicAdd(granData->bodyForceX + bodyB, force * A2BX);
        }
    }
}
