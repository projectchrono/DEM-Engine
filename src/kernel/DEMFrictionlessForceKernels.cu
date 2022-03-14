// DEM force computation related custom kernels
//#include <thirdparty/nvidia_helper_math/helper_math.cuh>
//#include <kernel/cuda_kernel_helper_math.cu>
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>

// Calculate the frictionless force between 2 bodies, return as a float3
// Assumes A2B vector is normalized
inline __device__ float3 calcNormalForce(const double& overlapDepth,
                                         const float3& A2B,
                                         const float3& velA2B,
                                         const sgps::contact_t& contact_type,
                                         const float& E,
                                         const float& G) {
    // Note this ad-hoc ``force'' is actually a fake acceleration written in terms of multiples of l
    float F_mag = 1e18 * overlapDepth;
    return make_float3(-F_mag * A2B.x, -F_mag * A2B.y, -F_mag * A2B.z);
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
    __shared__ float ClumpMasses[TEST_SHARED_SIZE];
    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < simParams->nDistinctClumpComponents; i++) {
            CDRadii[i] = granTemplates->radiiSphere[i] * simParams->beta;
            CDRelPosX[i] = granTemplates->relPosSphereX[i];
            CDRelPosY[i] = granTemplates->relPosSphereY[i];
            CDRelPosZ[i] = granTemplates->relPosSphereZ[i];
        }
        for (unsigned int i = 0; i < simParams->nDistinctClumpBodyTopologies; i++) {
            ClumpMasses[i] = granTemplates->massClumpBody[i];
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
        float AOwnerMass = ClumpMasses[granData->inertiaPropOffsets[bodyAOwner]];
        float BOwnerMass = ClumpMasses[granData->inertiaPropOffsets[bodyBOwner]];

        // Take care of 2 bodies in order, bodyA first, grab location and velocity to local cache
        float3 ALinVel, ARotVel, myRelPos;
        double3 Aowner, bodyAPos;
        sgps::oriQ_t AoriQ0, AoriQ1, AoriQ2, AoriQ3;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            Aowner.x, Aowner.y, Aowner.z, granData->voxelID[bodyAOwner], granData->locX[bodyAOwner],
            granData->locY[bodyAOwner], granData->locZ[bodyAOwner], simParams->nvXp2, simParams->nvYp2,
            simParams->voxelSize, simParams->l);
        myRelPos.x = CDRelPosX[bodyACompOffset];
        myRelPos.y = CDRelPosY[bodyACompOffset];
        myRelPos.z = CDRelPosZ[bodyACompOffset];
        AoriQ0 = granData->oriQ0[bodyAOwner];
        AoriQ1 = granData->oriQ1[bodyAOwner];
        AoriQ2 = granData->oriQ2[bodyAOwner];
        AoriQ3 = granData->oriQ3[bodyAOwner];
        applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPos.x, myRelPos.y, myRelPos.z, AoriQ0, AoriQ1, AoriQ2, AoriQ3);
        bodyAPos.x = Aowner.x + (double)myRelPos.x;
        bodyAPos.y = Aowner.y + (double)myRelPos.y;
        bodyAPos.z = Aowner.z + (double)myRelPos.z;
        ALinVel.x = granData->hvX[bodyAOwner] * simParams->l / simParams->h;
        ALinVel.y = granData->hvY[bodyAOwner] * simParams->l / simParams->h;
        ALinVel.z = granData->hvZ[bodyAOwner] * simParams->l / simParams->h;
        ARotVel.x = granData->hOmgBarX[bodyAOwner] / simParams->h;
        ARotVel.y = granData->hOmgBarY[bodyAOwner] / simParams->h;
        ARotVel.z = granData->hOmgBarZ[bodyAOwner] / simParams->h;

        // Then bodyB, location and velocity
        float3 BLinVel, BRotVel;
        double3 Bowner, bodyBPos;
        sgps::oriQ_t BoriQ0, BoriQ1, BoriQ2, BoriQ3;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            Bowner.x, Bowner.y, Bowner.z, granData->voxelID[bodyBOwner], granData->locX[bodyBOwner],
            granData->locY[bodyBOwner], granData->locZ[bodyBOwner], simParams->nvXp2, simParams->nvYp2,
            simParams->voxelSize, simParams->l);
        myRelPos.x = CDRelPosX[bodyBCompOffset];
        myRelPos.y = CDRelPosY[bodyBCompOffset];
        myRelPos.z = CDRelPosZ[bodyBCompOffset];
        BoriQ0 = granData->oriQ0[bodyBOwner];
        BoriQ1 = granData->oriQ1[bodyBOwner];
        BoriQ2 = granData->oriQ2[bodyBOwner];
        BoriQ3 = granData->oriQ3[bodyBOwner];
        applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPos.x, myRelPos.y, myRelPos.z, BoriQ0, BoriQ1, BoriQ2, BoriQ3);
        bodyBPos.x = Bowner.x + (double)myRelPos.x;
        bodyBPos.y = Bowner.y + (double)myRelPos.y;
        bodyBPos.z = Bowner.z + (double)myRelPos.z;
        BLinVel.x = granData->hvX[bodyBOwner] * simParams->l / simParams->h;
        BLinVel.y = granData->hvY[bodyBOwner] * simParams->l / simParams->h;
        BLinVel.z = granData->hvZ[bodyBOwner] * simParams->l / simParams->h;
        BRotVel.x = granData->hOmgBarX[bodyBOwner] / simParams->h;
        BRotVel.y = granData->hOmgBarY[bodyBOwner] / simParams->h;
        BRotVel.z = granData->hOmgBarZ[bodyBOwner] / simParams->h;

        // Now compute the contact point to see if they are truly still in contact
        double3 contactPnt;
        float3 A2B;  // Unit vector pointing from the center of body (sphere) A to body B
        double overlapDepth;
        bool in_contact;
        checkSpheresOverlap<double, float>(bodyAPos.x, bodyAPos.y, bodyAPos.z, CDRadii[bodyACompOffset], bodyBPos.x,
                                           bodyBPos.y, bodyBPos.z, CDRadii[bodyBCompOffset], contactPnt.x, contactPnt.y,
                                           contactPnt.z, A2B.x, A2B.y, A2B.z, overlapDepth, in_contact);
        if (in_contact) {
            // Instead of add the force to a body-based register, we store it in event-based register, and use CUB to
            // reduce them afterwards
            // atomicAdd(granData->bodyForceX + bodyA, -force * A2BX);
            // atomicAdd(granData->bodyForceX + bodyB, force * A2BX);

            // Right now, only handles contact type 0, sphere--sphere
            sgps::contact_t contact_type = 0;
            // Get k, g, etc. from the (upper-triangle) material property matrix
            unsigned int matEntry = locateMatPair<unsigned int>(bodyAMatType, bodyBMatType);
            float E = granTemplates->kProxy[matEntry];
            float G = granTemplates->gProxy[matEntry];
            // Find the contact point in the local (body), but global-axes-aligned frame
            // float3 locCPA = findLocalCoord<double>(contactPntX, contactPntY, contactPntZ, AownerX, AownerY, AownerZ,
            // AoriQ0, AoriQ1, AoriQ2, AoriQ3); float3 locCPB = findLocalCoord<double>(contactPntX, contactPntY,
            // contactPntZ, BownerX, BownerY, BownerZ, BoriQ0, BoriQ1, BoriQ2, BoriQ3);
            float3 locCPA = contactPnt - Aowner;
            float3 locCPB = contactPnt - Bowner;
            // We also need the relative velocity between A and B in global frame to use in the damping terms
            float3 velA2B = (ALinVel + cross(ARotVel, locCPA)) - (BLinVel + cross(BRotVel, locCPB));

            // Calculate contact force
            granData->contactForces[myContactID] = calcNormalForce(overlapDepth, A2B, velA2B, contact_type, E, G);
            // Write hard-earned results back to global arrays
            granData->contactPointGeometryA[myContactID] = locCPA;
            granData->contactPointGeometryB[myContactID] = locCPB;
        } else {
            granData->contactForces[myContactID] = make_float3(0, 0, 0);
        }
    }
}
