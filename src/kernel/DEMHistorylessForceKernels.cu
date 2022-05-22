// DEM force computation related custom kernels
//#include <thirdparty/nvidia_helper_math/helper_math.cuh>
#include <DEM/DEMDefines.h>
#include <kernel/DEMHelperKernels.cu>

__global__ void calculateContactForces(sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData, size_t nContactPairs) {
    // _nTotalBodyTopologies_ or _nDistinctClumpComponents_ elements are in these arrays
    const float Radii[] = {_Radii_};
    const float CDRelPosX[] = {_CDRelPosX_};
    const float CDRelPosY[] = {_CDRelPosY_};
    const float CDRelPosZ[] = {_CDRelPosZ_};
    const float ClumpMasses[] = {_ClumpMasses_};

    // _nMatTuples_ elements are in these arrays
    const float EProxy[] = {_EProxy_};
    const float nuProxy[] = {_nuProxy_};
    const float CoRProxy[] = {_CoRProxy_};

    // _nAnalGM_ elements are in these arrays
    const sgps::objType_t objType[_nAnalGMSafe_] = {_objType_};
    const sgps::bodyID_t objOwner[_nAnalGMSafe_] = {_objOwner_};
    const bool objNormal[_nAnalGMSafe_] = {_objNormal_};
    const sgps::materialsOffset_t objMaterial[_nAnalGMSafe_] = {_objMaterial_};
    const float objRelPosX[_nAnalGMSafe_] = {_objRelPosX_};
    const float objRelPosY[_nAnalGMSafe_] = {_objRelPosY_};
    const float objRelPosZ[_nAnalGMSafe_] = {_objRelPosZ_};
    const float objRotX[_nAnalGMSafe_] = {_objRotX_};
    const float objRotY[_nAnalGMSafe_] = {_objRotY_};
    const float objRotZ[_nAnalGMSafe_] = {_objRotZ_};
    const float objSize1[_nAnalGMSafe_] = {_objSize1_};
    const float objSize2[_nAnalGMSafe_] = {_objSize2_};
    const float objSize3[_nAnalGMSafe_] = {_objSize3_};

    sgps::contactPairs_t myContactID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myContactID < nContactPairs) {
        // Identify contact type first
        sgps::contact_t myContactType = granData->contactType[myContactID];
        // Allocate the registers needed
        double3 contactPnt;
        float3 B2A;  // Unit vector pointing from body B to body A
        double overlapDepth;
        double3 AOwnerPos, bodyAPos, BOwnerPos, bodyBPos;
        float3 ALinVel, ARotVel, BLinVel, BRotVel;
        float AOwnerMass, ARadius, BOwnerMass, BRadius;
        sgps::materialsOffset_t bodyAMatType, bodyBMatType;
        sgps::family_t AOwnerFamily, BOwnerFamily;
        // Take care of 2 bodies in order, bodyA first, grab location and velocity to local cache
        // We know in this kernel, bodyA will be a sphere; bodyB can be something else
        {
            sgps::bodyID_t bodyA = granData->idGeometryA[myContactID];
            sgps::bodyID_t bodyAOwner = granData->ownerClumpBody[bodyA];
            sgps::clumpComponentOffset_t bodyACompOffset = granData->clumpComponentOffset[bodyA];
            bodyAMatType = granData->materialTupleOffset[bodyA];
            AOwnerMass = ClumpMasses[granData->inertiaPropOffsets[bodyAOwner]];
            ARadius = Radii[bodyACompOffset];
            AOwnerFamily = granData->familyID[bodyAOwner];
            float3 myRelPos;
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
        }

        // Then bodyB, location and velocity
        if (myContactType == sgps::DEM_SPHERE_SPHERE_CONTACT) {
            sgps::bodyID_t bodyB = granData->idGeometryB[myContactID];
            sgps::bodyID_t bodyBOwner = granData->ownerClumpBody[bodyB];
            sgps::clumpComponentOffset_t bodyBCompOffset = granData->clumpComponentOffset[bodyB];
            bodyBMatType = granData->materialTupleOffset[bodyB];
            BOwnerMass = ClumpMasses[granData->inertiaPropOffsets[bodyBOwner]];
            BRadius = Radii[bodyBCompOffset];
            BOwnerFamily = granData->familyID[bodyBOwner];
            float3 myRelPos;
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
            myContactType = checkSpheresOverlap<double, float>(
                bodyAPos.x, bodyAPos.y, bodyAPos.z, ARadius, bodyBPos.x, bodyBPos.y, bodyBPos.z, BRadius, contactPnt.x,
                contactPnt.y, contactPnt.z, B2A.x, B2A.y, B2A.z, overlapDepth);
        } else {
            // If B is analytical entity, its owner, relative location, material info is jitified
            sgps::objID_t bodyB = granData->idGeometryB[myContactID];
            sgps::bodyID_t bodyBOwner = objOwner[bodyB];
            bodyBMatType = objMaterial[bodyB];
            BOwnerMass = ClumpMasses[granData->inertiaPropOffsets[bodyBOwner]];
            // TODO: fix these...
            BRadius = 10000.f;
            float3 myRelPos, bodyBRot;
            sgps::oriQ_t BoriQ0, BoriQ1, BoriQ2, BoriQ3;

            voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
                BOwnerPos.x, BOwnerPos.y, BOwnerPos.z, granData->voxelID[bodyBOwner], granData->locX[bodyBOwner],
                granData->locY[bodyBOwner], granData->locZ[bodyBOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            myRelPos.x = objRelPosX[bodyB];
            myRelPos.y = objRelPosY[bodyB];
            myRelPos.z = objRelPosZ[bodyB];
            BoriQ0 = granData->oriQ0[bodyBOwner];
            BoriQ1 = granData->oriQ1[bodyBOwner];
            BoriQ2 = granData->oriQ2[bodyBOwner];
            BoriQ3 = granData->oriQ3[bodyBOwner];
            applyOriQ2Vector3<float, sgps::oriQ_t>(myRelPos.x, myRelPos.y, myRelPos.z, BoriQ0, BoriQ1, BoriQ2, BoriQ3);
            bodyBPos.x = BOwnerPos.x + (double)myRelPos.x;
            bodyBPos.y = BOwnerPos.y + (double)myRelPos.y;
            bodyBPos.z = BOwnerPos.z + (double)myRelPos.z;

            // B's orientation (such as plane normal) is rotated with its owner too
            bodyBRot.x = objRotX[bodyB];
            bodyBRot.y = objRotY[bodyB];
            bodyBRot.z = objRotZ[bodyB];
            applyOriQ2Vector3<float, sgps::oriQ_t>(bodyBRot.x, bodyBRot.y, bodyBRot.z, BoriQ0, BoriQ1, BoriQ2, BoriQ3);

            BLinVel.x = granData->vX[bodyBOwner];
            BLinVel.y = granData->vY[bodyBOwner];
            BLinVel.z = granData->vZ[bodyBOwner];
            BRotVel.x = granData->omgBarX[bodyBOwner];
            BRotVel.y = granData->omgBarY[bodyBOwner];
            BRotVel.z = granData->omgBarZ[bodyBOwner];

            // Note for this test on dT side we don't enlarge entities
            myContactType = checkSphereEntityOverlap<double>(
                bodyAPos.x, bodyAPos.y, bodyAPos.z, ARadius, objType[bodyB], bodyBPos.x, bodyBPos.y, bodyBPos.z,
                bodyBRot.x, bodyBRot.y, bodyBRot.z, objSize1[bodyB], objSize2[bodyB], objSize3[bodyB], objNormal[bodyB],
                0.0, contactPnt.x, contactPnt.y, contactPnt.z, B2A.x, B2A.y, B2A.z, overlapDepth);
        }

        if (myContactType != sgps::DEM_NOT_A_CONTACT) {
            // Material properties and time (user referrable)
            float E, CoR, h;
            {
                h = simParams->h;
                matProxy2ContactParam<float>(E, CoR, EProxy[bodyAMatType], nuProxy[bodyAMatType],
                                             CoRProxy[bodyAMatType], EProxy[bodyBMatType], nuProxy[bodyBMatType],
                                             CoRProxy[bodyBMatType]);
            }
            // Variables that we need to report back (user referrable)
            float3 velB2A, delta_tan, force;
            {
                // Find the contact point in the local (body), but global-axes-aligned frame
                // float3 locCPA = findLocalCoord<double>(contactPntX, contactPntY, contactPntZ, AOwnerX, AOwnerY,
                // AOwnerZ, AoriQ0, AoriQ1, AoriQ2, AoriQ3); float3 locCPB = findLocalCoord<double>(contactPntX,
                // contactPntY, contactPntZ, BOwnerX, BOwnerY, BOwnerZ, BoriQ0, BoriQ1, BoriQ2, BoriQ3);
                float3 locCPA = contactPnt - AOwnerPos;
                float3 locCPB = contactPnt - BOwnerPos;
                granData->contactPointGeometryA[myContactID] = locCPA;
                granData->contactPointGeometryB[myContactID] = locCPB;
                // We also need the relative velocity between A and B in global frame to use in the damping terms
                velB2A = (ALinVel + cross(ARotVel, locCPA)) - (BLinVel + cross(BRotVel, locCPB));
                delta_tan = granData->contactHistory[myContactID];
            }

            // The following part, the force model, is user-specifiable
            // NOTE!! "force" must be properly set by this piece of code
            { _DEMForceModel_; }

            // Write hard-earned values back to global memory
            granData->contactForces[myContactID] = force;
        } else {
            granData->contactForces[myContactID] = make_float3(0, 0, 0);
        }
    }
}
