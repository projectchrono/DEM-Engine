// DEM force computation related custom kernels
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>
#include <helper_math.cuh>

/**
 * Template arguments:
 *   - T1: the floating point accuracy level for contact point location/penetration depth
 *   - T2: the floating point accuracy level for the relative position of 2 bodies involved
 *
 * Basic idea: calculate the frictionless force between 2 bodies, return as a float3
 *
 * Assumptions:
 *  - The A2B vector is normalized
 *
 */
template <typename T1, typename T2>
inline __device__ float3 calcNormalForce(T1 overlapDepth, T2 A2BX, T2 A2BY, T2 A2BZ, sgps::contact_t contact_type) {
    // A placeholder calculation
    float F_mag = 1e18 * overlapDepth;
    return make_float3(-F_mag * A2BX, -F_mag * A2BY, -F_mag * A2BZ);
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
        double AownerX, AownerY, AownerZ;
        sgps::oriQ_t AoriQ0, AoriQ1, AoriQ2, AoriQ3;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            AownerX, AownerY, AownerZ, granData->voxelID[bodyAOwner], granData->locX[bodyAOwner],
            granData->locY[bodyAOwner], granData->locZ[bodyAOwner], simParams->nvXp2, simParams->nvYp2,
            simParams->voxelSize, simParams->l);
        float myRelPosX = CDRelPosX[bodyACompOffset];
        float myRelPosY = CDRelPosY[bodyACompOffset];
        float myRelPosZ = CDRelPosZ[bodyACompOffset];
        applyOriQToVector3<float, float>(myRelPosX, myRelPosY, myRelPosZ);
        double bodyAX = AownerX + (double)myRelPosX;
        double bodyAY = AownerY + (double)myRelPosY;
        double bodyAZ = AownerZ + (double)myRelPosZ;
        // Then bodyB
        double BownerX, BownerY, BownerZ;
        sgps::oriQ_t BoriQ0, BoriQ1, BoriQ2, BoriQ3;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            BownerX, BownerY, BownerZ, granData->voxelID[bodyBOwner], granData->locX[bodyBOwner],
            granData->locY[bodyBOwner], granData->locZ[bodyBOwner], simParams->nvXp2, simParams->nvYp2,
            simParams->voxelSize, simParams->l);
        myRelPosX = CDRelPosX[bodyBCompOffset];
        myRelPosY = CDRelPosY[bodyBCompOffset];
        myRelPosZ = CDRelPosZ[bodyBCompOffset];
        applyOriQToVector3<float, float>(myRelPosX, myRelPosY, myRelPosZ);
        double bodyBX = BownerX + (double)myRelPosX;
        double bodyBY = BownerY + (double)myRelPosY;
        double bodyBZ = BownerZ + (double)myRelPosZ;

        // Now compute the contact point to see if they are truely still in contact
        double contactPntX, contactPntY, contactPntZ;
        float A2BX, A2BY, A2BZ;
        double overlapDepth;
        bool in_contact;
        checkSpheresOverlap<double, float>(bodyAX, bodyAY, bodyAZ, CDRadii[bodyACompOffset], bodyBX, bodyBY, bodyBZ,
                                           CDRadii[bodyBCompOffset], contactPntX, contactPntY, contactPntZ, A2BX, A2BY,
                                           A2BZ, overlapDepth, in_contact);
        if (in_contact) {
            // Instead of add the force to a body-based register, we store it in event-based register, and use CUB to
            // reduce them afterwards
            // atomicAdd(granData->bodyForceX + bodyA, -force * A2BX);
            // atomicAdd(granData->bodyForceX + bodyB, force * A2BX);

            // Right now, only handles contact type 0, sphere--sphere
            sgps::contact_t contact_type = 0;
            granData->contactForces[myContactID] =
                calcNormalForce<double, float>(overlapDepth, A2BX, A2BY, A2BZ, contact_type);
            granData->contactPointGeometryA[myContactID] = findLocalCoord<double>(
                contactPntX, contactPntY, contactPntZ, AownerX, AownerY, AownerZ, AoriQ0, AoriQ1, AoriQ2, AoriQ3);
            granData->contactPointGeometryA[myContactID] = findLocalCoord<double>(
                contactPntX, contactPntY, contactPntZ, BownerX, BownerY, BownerZ, BoriQ0, BoriQ1, BoriQ2, BoriQ3);
        } else {
            granData->contactForces[myContactID] = make_float3(0, 0, 0);
        }
    }
}
