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
inline __device__ float3 calcNormalForce(const T1& overlapDepth,
                                         const T2& A2BX,
                                         const T2& A2BY,
                                         const T2& A2BZ,
                                         const sgps::contact_t& contact_type,
                                         const float& k,
                                         const float& g) {
    // Note this ad-hoc ``force'' is actually a fake acceleration written in terms of multiples of l
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

    // First, find relevant bodyIDs, then locate their owners... (how??)

    // But, we will keep everything as is, and test in the end (when cub and jit are in place) how this treatment
    // improves efficiency

    sgps::contactPairs_t myContactID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myContactID < simParams->nContactPairs) {
        // From a contact ID, grab relevant info on 2 contact bodies
        sgps::bodyID_t bodyA = granData->idGeometryA[myContactID];
        sgps::bodyID_t bodyB = granData->idGeometryB[myContactID];
        sgps::bodyID_t bodyAOwner = granData->ownerClumpBody[bodyA];
        sgps::bodyID_t bodyBOwner = granData->ownerClumpBody[bodyB];
        sgps::clumpComponentOffset_t bodyACompOffset = granData->clumpComponentOffset[bodyA];
        sgps::clumpComponentOffset_t bodyBCompOffset = granData->clumpComponentOffset[bodyB];
        sgps::materialsOffset_t bodyAMatType = granData->materialTupleOffset[bodyA];
        sgps::materialsOffset_t bodyBMatType = granData->materialTupleOffset[bodyB];

        // Take care of 2 bodies in order, bodyA first, location and velocity
        double AownerX, AownerY, AownerZ;
        sgps::oriQ_t AoriQ0, AoriQ1, AoriQ2, AoriQ3;
        float3 ALinVel;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            AownerX, AownerY, AownerZ, granData->voxelID[bodyAOwner], granData->locX[bodyAOwner],
            granData->locY[bodyAOwner], granData->locZ[bodyAOwner], simParams->nvXp2, simParams->nvYp2,
            simParams->voxelSize, simParams->l);
        float myRelPosX = CDRelPosX[bodyACompOffset];
        float myRelPosY = CDRelPosY[bodyACompOffset];
        float myRelPosZ = CDRelPosZ[bodyACompOffset];
        AoriQ0 = granData->oriQ0[bodyAOwner];
        AoriQ1 = granData->oriQ1[bodyAOwner];
        AoriQ2 = granData->oriQ2[bodyAOwner];
        AoriQ3 = granData->oriQ3[bodyAOwner];
        applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, AoriQ0, AoriQ1, AoriQ2, AoriQ3);
        double bodyAX = AownerX + (double)myRelPosX;
        double bodyAY = AownerY + (double)myRelPosY;
        double bodyAZ = AownerZ + (double)myRelPosZ;
        ALinVel.x = granData->hvX[bodyAOwner] * simParams->l / simParams->h;

        // Then bodyB, location and velocity
        double BownerX, BownerY, BownerZ;
        sgps::oriQ_t BoriQ0, BoriQ1, BoriQ2, BoriQ3;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            BownerX, BownerY, BownerZ, granData->voxelID[bodyBOwner], granData->locX[bodyBOwner],
            granData->locY[bodyBOwner], granData->locZ[bodyBOwner], simParams->nvXp2, simParams->nvYp2,
            simParams->voxelSize, simParams->l);
        myRelPosX = CDRelPosX[bodyBCompOffset];
        myRelPosY = CDRelPosY[bodyBCompOffset];
        myRelPosZ = CDRelPosZ[bodyBCompOffset];
        BoriQ0 = granData->oriQ0[bodyBOwner];
        BoriQ1 = granData->oriQ1[bodyBOwner];
        BoriQ2 = granData->oriQ2[bodyBOwner];
        BoriQ3 = granData->oriQ3[bodyBOwner];
        applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, BoriQ0, BoriQ1, BoriQ2, BoriQ3);
        double bodyBX = BownerX + (double)myRelPosX;
        double bodyBY = BownerY + (double)myRelPosY;
        double bodyBZ = BownerZ + (double)myRelPosZ;

        // Now compute the contact point to see if they are truly still in contact
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
            // Get k, g, etc. from the (upper-triangle) material property matrix
            unsigned int matEntry = locateMatPair<unsigned int>(bodyAMatType, bodyBMatType);
            float k = granTemplates->kProxy[matEntry];
            float g = granTemplates->gProxy[matEntry];
            // Find the contact point in the local (body), but global-axes-aligned frame
            // float3 locCPA = findLocalCoord<double>(contactPntX, contactPntY, contactPntZ, AownerX, AownerY, AownerZ,
            // AoriQ0, AoriQ1, AoriQ2, AoriQ3); float3 locCPB = findLocalCoord<double>(contactPntX, contactPntY,
            // contactPntZ, BownerX, BownerY, BownerZ, BoriQ0, BoriQ1, BoriQ2, BoriQ3);
            float3 locCPA = vectorAB<float>(contactPntX, contactPntY, contactPntZ, AownerX, AownerY, AownerZ);
            float3 locCPB = vectorAB<float>(contactPntX, contactPntY, contactPntZ, BownerX, BownerY, BownerZ);
            // We also need velocity (linear + rotational) in global frame to be used in the damping terms

            // Calculate contact force
            granData->contactForces[myContactID] =
                calcNormalForce<double, float>(overlapDepth, A2BX, A2BY, A2BZ, contact_type, k, g);
            // Write hard-earned results back to global arrays
            granData->contactPointGeometryA[myContactID] = locCPA;
            granData->contactPointGeometryB[myContactID] = locCPB;
        } else {
            granData->contactForces[myContactID] = make_float3(0, 0, 0);
        }
    }
}
