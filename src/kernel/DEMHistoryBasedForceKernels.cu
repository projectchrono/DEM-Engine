// DEM force computation related custom kernels
#include <DEM/DEMDefines.h>
#include <kernel/DEMHelperKernels.cu>

// If clump templates are jitified, they will be below
_clumpTemplateDefs_;
// Definitions of analytical entites are below
_analyticalEntityDefs_;
// Material properties are below
_materialDefs_;
// If mass properties are jitified, then they are below
_massDefs_;

__global__ void calculateContactForces(sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData, size_t nContactPairs) {
    sgps::contactPairs_t myContactID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myContactID < nContactPairs) {
        // Identify contact type first
        sgps::contact_t myContactType = granData->contactType[myContactID];
        // The following quantities are always calculated, regardless of force model
        double3 contactPnt;
        float3 B2A;  // Unit vector pointing from body B to body A (contact normal)
        double overlapDepth;
        double3 AOwnerPos, bodyAPos, BOwnerPos, bodyBPos;
        float AOwnerMass, ARadius, BOwnerMass, BRadius;
        sgps::oriQ_t AoriQ0, AoriQ1, AoriQ2, AoriQ3;
        sgps::oriQ_t BoriQ0, BoriQ1, BoriQ2, BoriQ3;
        sgps::materialsOffset_t bodyAMatType, bodyBMatType;
        // Then allocate the optional quantities that will be needed in the force model (note: this one can't be in a
        // curly bracket, obviously...)
        _forceModelIngredientDefinition_;
        // Take care of 2 bodies in order, bodyA first, grab location and velocity to local cache
        // We know in this kernel, bodyA will be a sphere; bodyB can be something else
        {
            sgps::bodyID_t sphereID = granData->idGeometryA[myContactID];
            sgps::bodyID_t myOwner = granData->ownerClumpBody[sphereID];

            float myRelPosX, myRelPosY, myRelPosZ, myRadius;
            // Get my component offset info from either jitified arrays or global memory
            // Outputs myRelPosXYZ, myRadius
            // Use an input named exactly `sphereID' which is the id of this sphere component
            { _componentAcqStrat_; }

            // Get my mass info from either jitified arrays or global memory
            // Outputs myMass
            // Use an input named exactly `myOwner' which is the id of this owner
            {
                float myMass;
                _massAcqStrat_;
                AOwnerMass = myMass;
            }

            voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
                AOwnerPos.x, AOwnerPos.y, AOwnerPos.z, granData->voxelID[myOwner], granData->locX[myOwner],
                granData->locY[myOwner], granData->locZ[myOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);

            AoriQ0 = granData->oriQ0[myOwner];
            AoriQ1 = granData->oriQ1[myOwner];
            AoriQ2 = granData->oriQ2[myOwner];
            AoriQ3 = granData->oriQ3[myOwner];
            applyOriQToVector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, AoriQ0, AoriQ1, AoriQ2, AoriQ3);
            bodyAPos.x = AOwnerPos.x + (double)myRelPosX;
            bodyAPos.y = AOwnerPos.y + (double)myRelPosY;
            bodyAPos.z = AOwnerPos.z + (double)myRelPosZ;

            ARadius = myRadius;
            bodyAMatType = granData->sphereMaterialOffset[sphereID];

            // Optional force model ingredients are loaded here...
            _forceModelIngredientAcqForA_;
        }

        // Then bodyB, location and velocity
        if (myContactType == sgps::DEM_SPHERE_SPHERE_CONTACT) {
            sgps::bodyID_t sphereID = granData->idGeometryB[myContactID];
            sgps::bodyID_t myOwner = granData->ownerClumpBody[sphereID];

            float myRelPosX, myRelPosY, myRelPosZ, myRadius;
            // Get my component offset info from either jitified arrays or global memory
            // Outputs myRelPosXYZ, myRadius
            // Use an input named exactly `sphereID' which is the id of this sphere component
            { _componentAcqStrat_; }

            // Get my mass info from either jitified arrays or global memory
            // Outputs myMass
            // Use an input named exactly `myOwner' which is the id of this owner
            {
                float myMass;
                _massAcqStrat_;
                BOwnerMass = myMass;
            }

            voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
                BOwnerPos.x, BOwnerPos.y, BOwnerPos.z, granData->voxelID[myOwner], granData->locX[myOwner],
                granData->locY[myOwner], granData->locZ[myOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            BoriQ0 = granData->oriQ0[myOwner];
            BoriQ1 = granData->oriQ1[myOwner];
            BoriQ2 = granData->oriQ2[myOwner];
            BoriQ3 = granData->oriQ3[myOwner];
            applyOriQToVector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, BoriQ0, BoriQ1, BoriQ2, BoriQ3);
            bodyBPos.x = BOwnerPos.x + (double)myRelPosX;
            bodyBPos.y = BOwnerPos.y + (double)myRelPosY;
            bodyBPos.z = BOwnerPos.z + (double)myRelPosZ;

            BRadius = myRadius;
            bodyBMatType = granData->sphereMaterialOffset[sphereID];

            _forceModelIngredientAcqForB_;

            myContactType = checkSpheresOverlap<double, float>(
                bodyAPos.x, bodyAPos.y, bodyAPos.z, ARadius, bodyBPos.x, bodyBPos.y, bodyBPos.z, BRadius, contactPnt.x,
                contactPnt.y, contactPnt.z, B2A.x, B2A.y, B2A.z, overlapDepth);
        } else {
            // If B is analytical entity, its owner, relative location, material info is jitified
            sgps::objID_t bodyB = granData->idGeometryB[myContactID];
            sgps::bodyID_t myOwner = objOwner[bodyB];
            bodyBMatType = objMaterial[bodyB];
            BOwnerMass = objMass[bodyB];
            //// TODO: Is this OK?
            BRadius = SGPS_DEM_HUGE_FLOAT;
            float myRelPosX, myRelPosY, myRelPosZ;
            float3 bodyBRot;

            voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
                BOwnerPos.x, BOwnerPos.y, BOwnerPos.z, granData->voxelID[myOwner], granData->locX[myOwner],
                granData->locY[myOwner], granData->locZ[myOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            myRelPosX = objRelPosX[bodyB];
            myRelPosY = objRelPosY[bodyB];
            myRelPosZ = objRelPosZ[bodyB];
            BoriQ0 = granData->oriQ0[myOwner];
            BoriQ1 = granData->oriQ1[myOwner];
            BoriQ2 = granData->oriQ2[myOwner];
            BoriQ3 = granData->oriQ3[myOwner];
            applyOriQToVector3<float, sgps::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, BoriQ0, BoriQ1, BoriQ2, BoriQ3);
            bodyBPos.x = BOwnerPos.x + (double)myRelPosX;
            bodyBPos.y = BOwnerPos.y + (double)myRelPosY;
            bodyBPos.z = BOwnerPos.z + (double)myRelPosZ;

            // B's orientation (such as plane normal) is rotated with its owner too
            bodyBRot.x = objRotX[bodyB];
            bodyBRot.y = objRotY[bodyB];
            bodyBRot.z = objRotZ[bodyB];
            applyOriQToVector3<float, sgps::oriQ_t>(bodyBRot.x, bodyBRot.y, bodyBRot.z, BoriQ0, BoriQ1, BoriQ2, BoriQ3);

            _forceModelIngredientAcqForB_;

            // Note for this test on dT side we don't enlarge entities
            myContactType = checkSphereEntityOverlap<double>(
                bodyAPos.x, bodyAPos.y, bodyAPos.z, ARadius, objType[bodyB], bodyBPos.x, bodyBPos.y, bodyBPos.z,
                bodyBRot.x, bodyBRot.y, bodyBRot.z, objSize1[bodyB], objSize2[bodyB], objSize3[bodyB], objNormal[bodyB],
                0.0, contactPnt.x, contactPnt.y, contactPnt.z, B2A.x, B2A.y, B2A.z, overlapDepth);
        }

        if (myContactType != sgps::DEM_NOT_A_CONTACT) {
            float3 delta_tan;
            float delta_time;
            float3 force = make_float3(0, 0, 0);
            float3 torque_only_force = make_float3(0, 0, 0);
            // Local position of the contact point is always a piece of info we require... regardless of force model
            float3 locCPA = contactPnt - AOwnerPos;
            float3 locCPB = contactPnt - BOwnerPos;
            // Now map this contact point location to bodies' local ref
            applyOriQToVector3<float, sgps::oriQ_t>(locCPA.x, locCPA.y, locCPA.z, AoriQ0, -AoriQ1, -AoriQ2, -AoriQ3);
            applyOriQToVector3<float, sgps::oriQ_t>(locCPB.x, locCPB.y, locCPB.z, BoriQ0, -BoriQ1, -BoriQ2, -BoriQ3);
            // The following part, the force model, is user-specifiable
            // NOTE!! "force" and "delta_tan" and "delta_time" must be properly set by this piece of code
            { _DEMForceModel_; }

            // Write hard-earned values back to global memory
            granData->contactPointGeometryA[myContactID] = locCPA;
            granData->contactPointGeometryB[myContactID] = locCPB;
            granData->contactForces[myContactID] = force;
            granData->contactTorque_convToForce[myContactID] = torque_only_force;
            granData->contactHistory[myContactID] = delta_tan;
            granData->contactDuration[myContactID] = delta_time;
        } else {
            granData->contactForces[myContactID] = make_float3(0, 0, 0);
            granData->contactTorque_convToForce[myContactID] = make_float3(0, 0, 0);
            // The contact is no longer active, so we need to destroy its contact history recording
            granData->contactHistory[myContactID] = make_float3(0, 0, 0);
            granData->contactDuration[myContactID] = 0;
        }
    }
}
