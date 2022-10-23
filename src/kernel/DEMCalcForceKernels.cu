// DEM force computation related custom kernels
#include <DEM/Defines.h>
#include <kernel/DEMHelperKernels.cu>
#include <kernel/DEMCollisionKernels.cu>

// If clump templates are jitified, they will be below
_clumpTemplateDefs_;
// Definitions of analytical entites are below
_analyticalEntityDefs_;
// Material properties are below
_materialDefs_;
// If mass properties are jitified, then they are below
_massDefs_;

template <typename T1>
inline __device__ void equipOwnerPosRot(deme::DEMDataDT* granData,
                                        const deme::bodyID_t& myOwner,
                                        T1& relPos,
                                        double3& ownerPos,
                                        double3& bodyPos,
                                        float4& oriQ) {
    voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
        ownerPos.x, ownerPos.y, ownerPos.z, granData->voxelID[myOwner], granData->locX[myOwner],
        granData->locY[myOwner], granData->locZ[myOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
    oriQ.w = granData->oriQw[myOwner];
    oriQ.x = granData->oriQx[myOwner];
    oriQ.y = granData->oriQy[myOwner];
    oriQ.z = granData->oriQz[myOwner];
    applyOriQToVector3(relPos.x, relPos.y, relPos.z, oriQ.w, oriQ.x, oriQ.y, oriQ.z);
    bodyPos.x = ownerPos.x + (double)relPos.x;
    bodyPos.y = ownerPos.y + (double)relPos.y;
    bodyPos.z = ownerPos.z + (double)relPos.z;
}

__global__ void calculateContactForces(deme::DEMSimParams* simParams, deme::DEMDataDT* granData, size_t nContactPairs) {
    deme::contactPairs_t myContactID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myContactID < nContactPairs) {
        // Identify contact type first
        deme::contact_t myContactType = granData->contactType[myContactID];
        // The following quantities are always calculated, regardless of force model
        double3 contactPnt;
        float3 B2A;  // Unit vector pointing from body B to body A (contact normal)
        double overlapDepth;
        double3 AOwnerPos, bodyAPos, BOwnerPos, bodyBPos;
        float AOwnerMass, ARadius, BOwnerMass, BRadius;
        float4 AOriQ, BOriQ;
        deme::materialsOffset_t bodyAMatType, bodyBMatType;
        // Then allocate the optional quantities that will be needed in the force model (note: this one can't be in a
        // curly bracket, obviously...)
        _forceModelIngredientDefinition_;
        // Take care of 2 bodies in order, bodyA first, grab location and velocity to local cache
        // We know in this kernel, bodyA will be a sphere; bodyB can be something else
        {
            deme::bodyID_t sphereID = granData->idGeometryA[myContactID];
            deme::bodyID_t myOwner = granData->ownerClumpBody[sphereID];

            float3 myRelPos;
            float myRadius;
            // Get my component offset info from either jitified arrays or global memory
            // Outputs myRelPos, myRadius
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

            equipOwnerPosRot(granData, myOwner, myRelPos, AOwnerPos, bodyAPos, AOriQ);

            ARadius = myRadius;
            bodyAMatType = granData->sphereMaterialOffset[sphereID];

            // Optional force model ingredients are loaded here...
            _forceModelIngredientAcqForA_;
        }

        // Then bodyB, location and velocity
        if (myContactType == deme::SPHERE_SPHERE_CONTACT) {
            deme::bodyID_t sphereID = granData->idGeometryB[myContactID];
            deme::bodyID_t myOwner = granData->ownerClumpBody[sphereID];

            float3 myRelPos;
            float myRadius;
            // Get my component offset info from either jitified arrays or global memory
            // Outputs myRelPos, myRadius
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

            equipOwnerPosRot(granData, myOwner, myRelPos, BOwnerPos, bodyBPos, BOriQ);

            BRadius = myRadius;
            bodyBMatType = granData->sphereMaterialOffset[sphereID];

            _forceModelIngredientAcqForB_;

            myContactType = checkSpheresOverlap<double, float>(
                bodyAPos.x, bodyAPos.y, bodyAPos.z, ARadius, bodyBPos.x, bodyBPos.y, bodyBPos.z, BRadius, contactPnt.x,
                contactPnt.y, contactPnt.z, B2A.x, B2A.y, B2A.z, overlapDepth);
        } else if (myContactType == deme::SPHERE_MESH_CONTACT) {
            deme::bodyID_t triB = granData->idGeometryB[myContactID];
            deme::bodyID_t myOwner = granData->ownerMesh[triB];
            //// TODO: Is this OK?
            BRadius = DEME_HUGE_FLOAT;
            bodyBMatType = granData->triMaterialOffset[triB];

            double3 triNode1 = to_double3(granData->relPosNode1[triB]);
            double3 triNode2 = to_double3(granData->relPosNode2[triB]);
            double3 triNode3 = to_double3(granData->relPosNode3[triB]);

            // Get my mass info from either jitified arrays or global memory
            // Outputs myMass
            // Use an input named exactly `myOwner' which is the id of this owner
            {
                float myMass;
                _massAcqStrat_;
                BOwnerMass = myMass;
            }

            // bodyBPos is for a place holder for the outcome triNode1 position
            equipOwnerPosRot(granData, myOwner, triNode1, BOwnerPos, bodyBPos, BOriQ);
            triNode1 = bodyBPos;
            // Do this to node 2 and 3 as well
            applyOriQToVector3(triNode2.x, triNode2.y, triNode2.z, BOriQ.w, BOriQ.x, BOriQ.y, BOriQ.z);
            triNode2 += BOwnerPos;
            applyOriQToVector3(triNode3.x, triNode3.y, triNode3.z, BOriQ.w, BOriQ.x, BOriQ.y, BOriQ.z);
            triNode3 += BOwnerPos;
            // Assign the correct bodyBPos
            bodyBPos = triangleCentroid<double3>(triNode1, triNode2, triNode3);

            _forceModelIngredientAcqForB_;

            double3 contact_normal;
            bool in_contact = triangle_sphere_CD<double3, double>(triNode1, triNode2, triNode3, bodyAPos, ARadius,
                                                                  contact_normal, overlapDepth, contactPnt);
            B2A = to_float3(contact_normal);
            overlapDepth = -overlapDepth;  // triangle_sphere_CD gives neg. number for overlapping cases

            // If not in contact, correct myContactType
            if (!in_contact) {
                myContactType = deme::NOT_A_CONTACT;
            }
        } else {
            // If B is analytical entity, its owner, relative location, material info is jitified
            deme::objID_t bodyB = granData->idGeometryB[myContactID];
            deme::bodyID_t myOwner = objOwner[bodyB];
            bodyBMatType = objMaterial[bodyB];
            BOwnerMass = objMass[bodyB];
            //// TODO: Is this OK?
            BRadius = DEME_HUGE_FLOAT;
            float3 myRelPos;
            float3 bodyBRot;
            myRelPos.x = objRelPosX[bodyB];
            myRelPos.y = objRelPosY[bodyB];
            myRelPos.z = objRelPosZ[bodyB];

            equipOwnerPosRot(granData, myOwner, myRelPos, BOwnerPos, bodyBPos, BOriQ);

            // B's orientation (such as plane normal) is rotated with its owner too
            bodyBRot.x = objRotX[bodyB];
            bodyBRot.y = objRotY[bodyB];
            bodyBRot.z = objRotZ[bodyB];
            applyOriQToVector3<float, deme::oriQ_t>(bodyBRot.x, bodyBRot.y, bodyBRot.z, BOriQ.w, BOriQ.x, BOriQ.y,
                                                    BOriQ.z);

            _forceModelIngredientAcqForB_;

            // Note for this test on dT side we don't enlarge entities
            myContactType = checkSphereEntityOverlap<double3, float, double>(
                bodyAPos, ARadius, objType[bodyB], bodyBPos, bodyBRot, objSize1[bodyB], objSize2[bodyB],
                objSize3[bodyB], objNormal[bodyB], 0.0, contactPnt, B2A, overlapDepth);
        }

        _forceModelContactWildcardAcq_;
        if (myContactType != deme::NOT_A_CONTACT) {
            float3 force = make_float3(0, 0, 0);
            float3 torque_only_force = make_float3(0, 0, 0);
            // Local position of the contact point is always a piece of info we require... regardless of force model
            float3 locCPA = to_float3(contactPnt - AOwnerPos);
            float3 locCPB = to_float3(contactPnt - BOwnerPos);
            // Now map this contact point location to bodies' local ref
            applyOriQToVector3<float, deme::oriQ_t>(locCPA.x, locCPA.y, locCPA.z, AOriQ.w, -AOriQ.x, -AOriQ.y,
                                                    -AOriQ.z);
            applyOriQToVector3<float, deme::oriQ_t>(locCPB.x, locCPB.y, locCPB.z, BOriQ.w, -BOriQ.x, -BOriQ.y,
                                                    -BOriQ.z);
            // The following part, the force model, is user-specifiable
            // NOTE!! "force" and "delta_tan" and "delta_time" must be properly set by this piece of code
            { _DEMForceModel_; }

            // Write contact location values back to global memory
            _contactInfoWrite_;

            // If force model modifies owner wildcards, write them back here
            _forceModelOwnerWildcardWrite_;

            // Optionally, the forces can be reduced to acc right here (may be faster for polydisperse spheres)
            _forceCollectInPlaceStrat_;
        } else {
            // The contact is no longer active, so we need to destroy its contact history recording
            _forceModelContactWildcardDestroy_;
        }

        // Updated contact wildcards need to be write back to global mem. It is here because contact wildcard may need
        // to be destroyed for non-contact, so it has to go last.
        _forceModelContactWildcardWrite_;
    }
}
