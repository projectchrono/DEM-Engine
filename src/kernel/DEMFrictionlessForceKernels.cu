// DEM force computation related custom kernels
//#include <thirdparty/nvidia_helper_math/helper_math.cuh>
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>

// Calculate the frictionless force between 2 bodies, return as a float3
// Assumes B2A vector is normalized
inline __device__ float3 calcNormalForce(const double& overlapDepth,
                                         const float3& B2A,
                                         const float3& velB2A,
                                         const float& ARadius,
                                         const float& BRadius,
                                         const float& AOwnerMass,
                                         const float& BOwnerMass,
                                         const sgps::contact_t& contact_type,
                                         const float& E,
                                         const float& CoR) {
    // Note this ad-hoc ``force'' is actually a fake acceleration written in terms of multiples of l
    // float F_mag = 1e18 * overlapDepth;
    // return make_float3(F_mag * B2A.x, F_mag * B2A.y, F_mag * B2A.z);

    // normal component of relative velocity
    const float projection = dot(velB2A, B2A);
    float3 vrel_tan = velB2A - projection * B2A;  // May want to report this for tangent force calculation

    const float mass_eff = (AOwnerMass * BOwnerMass) / (AOwnerMass + BOwnerMass);
    float sqrt_Rd = sqrt(overlapDepth * (ARadius * BRadius) / (ARadius + BRadius));
    const float Sn = 2. * E * sqrt_Rd;

    const float loge = (CoR < SGPS_DEM_TINY_FLOAT) ? log(SGPS_DEM_TINY_FLOAT) : log(CoR);
    float beta = loge / sqrt(loge * loge + SGPS_PI_SQUARED);

    const float k_n = SGPS_TWO_OVER_THREE * Sn;
    const float gamma_n = SGPS_TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(Sn * mass_eff);

    // normal force (that A feels)
    // printf("overlapDepth: %f\n", overlapDepth);
    // printf("kn * overlapDepth: %f\n", k_n * overlapDepth);
    // printf("gn * projection: %f\n", gamma_n * projection);
    float3 force = (k_n * overlapDepth + gamma_n * projection) * B2A;
    return force;
}

__global__ void calculateNormalContactForces(sgps::DEMSimParams* simParams,
                                             sgps::DEMDataDT* granData,
                                             size_t nContactPairs,
                                             sgps::DEMTemplate* granTemplates) {
    // CUDA does not support initializing shared arrays, so we have to manually load them
    __shared__ float Radii[_nDistinctClumpComponents_];
    __shared__ float CDRelPosX[_nDistinctClumpComponents_];
    __shared__ float CDRelPosY[_nDistinctClumpComponents_];
    __shared__ float CDRelPosZ[_nDistinctClumpComponents_];
    __shared__ float ClumpMasses[_nDistinctClumpBodyTopologies_];
    if (threadIdx.x < _nActiveLoadingThreads_) {
        const float jitifiedRadii[_nDistinctClumpComponents_] = {_Radii_};
        const float jitifiedCDRelPosX[_nDistinctClumpComponents_] = {_CDRelPosX_};
        const float jitifiedCDRelPosY[_nDistinctClumpComponents_] = {_CDRelPosY_};
        const float jitifiedCDRelPosZ[_nDistinctClumpComponents_] = {_CDRelPosZ_};
        const float jitifiedMass[_nDistinctClumpBodyTopologies_] = {_ClumpMasses_};
        for (sgps::clumpBodyInertiaOffset_t i = threadIdx.x; i < _nDistinctClumpBodyTopologies_;
             i += _nActiveLoadingThreads_) {
            ClumpMasses[i] = jitifiedMass[i];
        }
        for (sgps::clumpComponentOffset_t i = threadIdx.x; i < _nDistinctClumpComponents_;
             i += _nActiveLoadingThreads_) {
            Radii[i] = jitifiedRadii[i];
            CDRelPosX[i] = jitifiedCDRelPosX[i];
            CDRelPosY[i] = jitifiedCDRelPosY[i];
            CDRelPosZ[i] = jitifiedCDRelPosZ[i];
        }
    }
    __syncthreads();

    // First, find relevant bodyIDs, then locate their owners... (how??)

    // But, we will keep everything as is, and test in the end (when cub and jit are in place) how this treatment
    // improves efficiency

    sgps::contactPairs_t myContactID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myContactID < nContactPairs) {
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
        float ARadius = Radii[bodyACompOffset];
        float BRadius = Radii[bodyBCompOffset];

        // Take care of 2 bodies in order, bodyA first, grab location and velocity to local cache
        float3 ALinVel, ARotVel, myRelPos;
        double3 AOwnerPos, bodyAPos;
        sgps::oriQ_t AoriQ0, AoriQ1, AoriQ2, AoriQ3;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            AOwnerPos.x, AOwnerPos.y, AOwnerPos.z, granData->voxelID[bodyAOwner], granData->locX[bodyAOwner],
            granData->locY[bodyAOwner], granData->locZ[bodyAOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        myRelPos.x = CDRelPosX[bodyACompOffset];
        myRelPos.y = CDRelPosY[bodyACompOffset];
        myRelPos.z = CDRelPosZ[bodyACompOffset];
        AoriQ0 = granData->oriQ0[bodyAOwner];
        AoriQ1 = granData->oriQ1[bodyAOwner];
        AoriQ2 = granData->oriQ2[bodyAOwner];
        AoriQ3 = granData->oriQ3[bodyAOwner];
        applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPos.x, myRelPos.y, myRelPos.z, AoriQ0, AoriQ1, AoriQ2, AoriQ3);
        bodyAPos.x = AOwnerPos.x + (double)myRelPos.x;
        bodyAPos.y = AOwnerPos.y + (double)myRelPos.y;
        bodyAPos.z = AOwnerPos.z + (double)myRelPos.z;
        ALinVel.x = granData->vX[bodyAOwner];
        ALinVel.y = granData->vY[bodyAOwner];
        ALinVel.z = granData->vZ[bodyAOwner];
        ARotVel.x = granData->omgBarX[bodyAOwner];
        ARotVel.y = granData->omgBarY[bodyAOwner];
        ARotVel.z = granData->omgBarZ[bodyAOwner];

        // Then bodyB, location and velocity
        float3 BLinVel, BRotVel;
        double3 BOwnerPos, bodyBPos;
        sgps::oriQ_t BoriQ0, BoriQ1, BoriQ2, BoriQ3;
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            BOwnerPos.x, BOwnerPos.y, BOwnerPos.z, granData->voxelID[bodyBOwner], granData->locX[bodyBOwner],
            granData->locY[bodyBOwner], granData->locZ[bodyBOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        myRelPos.x = CDRelPosX[bodyBCompOffset];
        myRelPos.y = CDRelPosY[bodyBCompOffset];
        myRelPos.z = CDRelPosZ[bodyBCompOffset];
        BoriQ0 = granData->oriQ0[bodyBOwner];
        BoriQ1 = granData->oriQ1[bodyBOwner];
        BoriQ2 = granData->oriQ2[bodyBOwner];
        BoriQ3 = granData->oriQ3[bodyBOwner];
        applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPos.x, myRelPos.y, myRelPos.z, BoriQ0, BoriQ1, BoriQ2, BoriQ3);
        bodyBPos.x = BOwnerPos.x + (double)myRelPos.x;
        bodyBPos.y = BOwnerPos.y + (double)myRelPos.y;
        bodyBPos.z = BOwnerPos.z + (double)myRelPos.z;
        BLinVel.x = granData->vX[bodyBOwner];
        BLinVel.y = granData->vY[bodyBOwner];
        BLinVel.z = granData->vZ[bodyBOwner];
        BRotVel.x = granData->omgBarX[bodyBOwner];
        BRotVel.y = granData->omgBarY[bodyBOwner];
        BRotVel.z = granData->omgBarZ[bodyBOwner];

        // Now compute the contact point to see if they are truly still in contact
        double3 contactPnt;
        float3 B2A;  // Unit vector pointing from the center of body (sphere) B to body A
        double overlapDepth;
        bool in_contact;
        checkSpheresOverlap<double, float>(bodyAPos.x, bodyAPos.y, bodyAPos.z, ARadius, bodyBPos.x, bodyBPos.y,
                                           bodyBPos.z, BRadius, contactPnt.x, contactPnt.y, contactPnt.z, B2A.x, B2A.y,
                                           B2A.z, overlapDepth, in_contact);
        if (in_contact) {
            // Instead of add the force to a body-based register, we store it in event-based register, and use CUB to
            // reduce them afterwards
            // atomicAdd(granData->bodyForceX + bodyA, -force * B2AX);
            // atomicAdd(granData->bodyForceX + bodyB, force * B2AX);

            // Right now, only handles contact type 0, sphere--sphere
            sgps::contact_t contact_type = 0;
            // Get k, g, etc. from the (upper-triangle) material property matrix
            unsigned int matEntry = locateMatPair<unsigned int>(bodyAMatType, bodyBMatType);
            float E = granTemplates->EProxy[matEntry];
            // float G = granTemplates->GProxy[matEntry];
            float CoR = granTemplates->CoRProxy[matEntry];

            // Find the contact point in the local (body), but global-axes-aligned frame
            // float3 locCPA = findLocalCoord<double>(contactPntX, contactPntY, contactPntZ, AOwnerX, AOwnerY, AOwnerZ,
            // AoriQ0, AoriQ1, AoriQ2, AoriQ3); float3 locCPB = findLocalCoord<double>(contactPntX, contactPntY,
            // contactPntZ, BOwnerX, BOwnerY, BOwnerZ, BoriQ0, BoriQ1, BoriQ2, BoriQ3);
            float3 locCPA = contactPnt - AOwnerPos;
            float3 locCPB = contactPnt - BOwnerPos;
            // We also need the relative velocity between A and B in global frame to use in the damping terms
            float3 velB2A = (ALinVel + cross(ARotVel, locCPA)) - (BLinVel + cross(BRotVel, locCPB));

            // Calculate contact force
            granData->contactForces[myContactID] = calcNormalForce(overlapDepth, B2A, velB2A, ARadius, BRadius,
                                                                   AOwnerMass, BOwnerMass, contact_type, E, CoR);
            // Write hard-earned results back to global arrays
            granData->contactPointGeometryA[myContactID] = locCPA;
            granData->contactPointGeometryB[myContactID] = locCPB;
        } else {
            granData->contactForces[myContactID] = make_float3(0, 0, 0);
        }
    }
}
