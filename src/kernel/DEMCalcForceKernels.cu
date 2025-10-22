// DEM force computation related custom kernels
#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>
#include <DEMCollisionKernels.cuh>
_kernelIncludes_;

// If clump templates are jitified, they will be below
_clumpTemplateDefs_;
// Definitions of analytical entites are below
_analyticalEntityDefs_;
// Material properties are below
_materialDefs_;
// If mass properties are jitified, then they are below
_massDefs_;
_moiDefs_;
// If the user has some utility functions, they will be included here
_forceModelPrerequisites_;

template <typename T1>
inline __device__ void equipOwnerPosRot(deme::DEMSimParams* simParams,
                                        deme::DEMDataDT* granData,
                                        const deme::bodyID_t& myOwner,
                                        T1& relPos,
                                        double3& ownerPos,
                                        double3& bodyPos,
                                        float4& oriQ) {
    voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
        ownerPos.x, ownerPos.y, ownerPos.z, granData->voxelID[myOwner], granData->locX[myOwner],
        granData->locY[myOwner], granData->locZ[myOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
    // Do this and we get the `true' pos...
    ownerPos.x += simParams->LBFX;
    ownerPos.y += simParams->LBFY;
    ownerPos.z += simParams->LBFZ;
    oriQ.w = granData->oriQw[myOwner];
    oriQ.x = granData->oriQx[myOwner];
    oriQ.y = granData->oriQy[myOwner];
    oriQ.z = granData->oriQz[myOwner];
    applyOriQToVector3(relPos.x, relPos.y, relPos.z, oriQ.w, oriQ.x, oriQ.y, oriQ.z);
    bodyPos.x = ownerPos.x + (double)relPos.x;
    bodyPos.y = ownerPos.y + (double)relPos.y;
    bodyPos.z = ownerPos.z + (double)relPos.z;
}

// Template device function for contact force calculation - will be called by 5 specialized kernels
template <deme::contact_t CONTACT_TYPE>
__device__ __forceinline__ void calculateContactForcesImpl(deme::DEMSimParams* simParams,
                                                           deme::DEMDataDT* granData,
                                                           deme::contactPairs_t myContactID) {
    // Contact type is known at compile time
    deme::contact_t ContactType = CONTACT_TYPE;
    // The following quantities are always calculated, regardless of force model
    double3 contactPnt;
    float3 B2A;  // Unit vector pointing from body B to body A (contact normal)
    double overlapDepth;
    double3 AOwnerPos, bodyAPos, BOwnerPos, bodyBPos;
    float AOwnerMass, ARadius, BOwnerMass, BRadius;
    float4 AOriQ, BOriQ;
    deme::materialsOffset_t bodyAMatType, bodyBMatType;
    // The user-specified extra margin size (how much we should be lenient in determining `in-contact')
    float extraMarginSize = 0.;
    // Triangle A's three points are defined outside, as may be reused in B's acquisition and penetration calc.
    double3 triANode1, triANode2, triANode3;
    // Then allocate the optional quantities that will be needed in the force model (note: this one can't be in a
    // curly bracket, obviously...)
    _forceModelIngredientDefinition_;
    // Take care of 2 bodies in order, bodyA first, grab location and velocity to local cache
    // Decompose ContactType to get the types of A and B (known at compile time)
    constexpr deme::geoType_t AType = (CONTACT_TYPE >> 4);
    constexpr deme::geoType_t BType = (CONTACT_TYPE & 0xF);

        // ----------------------------------------------------------------
        // Based on A's type, equip info
        // ----------------------------------------------------------------
        if (AType == deme::GEO_T_SPHERE) {
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

            // Optional force model ingredients are loaded here...
            _forceModelIngredientAcqForA_;
            _forceModelGeoWildcardAcqForASph_;

            equipOwnerPosRot(simParams, granData, myOwner, myRelPos, AOwnerPos, bodyAPos, AOriQ);

            ARadius = myRadius;
            bodyAMatType = granData->sphereMaterialOffset[sphereID];
            extraMarginSize = granData->familyExtraMarginSize[AOwnerFamily];
        } else if (AType == deme::GEO_T_TRIANGLE) {
            // Geometry ID here is called sphereID, although it is not a sphere, it's more like triID. But naming it
            // sphereID makes the acquisition process cleaner.
            deme::bodyID_t sphereID = granData->idGeometryA[myContactID];
            deme::bodyID_t myOwner = granData->ownerMesh[sphereID];
            //// TODO: Is this OK?
            ARadius = DEME_HUGE_FLOAT;
            bodyAMatType = granData->triMaterialOffset[sphereID];

            // As the grace margin, the distance (negative overlap) just needs to be within the grace margin. So we pick
            // the larger of the 2 familyExtraMarginSize.
            extraMarginSize = granData->familyExtraMarginSize[AOwnerFamily];

            triANode1 = to_double3(granData->relPosNode1[sphereID]);
            triANode2 = to_double3(granData->relPosNode2[sphereID]);
            triANode3 = to_double3(granData->relPosNode3[sphereID]);

            // Get my mass info from either jitified arrays or global memory
            // Outputs myMass
            // Use an input named exactly `myOwner' which is the id of this owner
            {
                float myMass;
                _massAcqStrat_;
                AOwnerMass = myMass;
            }
            _forceModelIngredientAcqForA_;
            _forceModelGeoWildcardAcqForATri_;

            // bodyAPos is for a place holder for the outcome triANode1 position
            equipOwnerPosRot(simParams, granData, myOwner, triANode1, AOwnerPos, bodyAPos, AOriQ);
            triANode1 = bodyAPos;
            // Do this to node 2 and 3 as well
            applyOriQToVector3(triANode2.x, triANode2.y, triANode2.z, AOriQ.w, AOriQ.x, AOriQ.y, AOriQ.z);
            triANode2 += AOwnerPos;
            applyOriQToVector3(triANode3.x, triANode3.y, triANode3.z, AOriQ.w, AOriQ.x, AOriQ.y, AOriQ.z);
            triANode3 += AOwnerPos;
            // Assign the correct bodyAPos
            bodyAPos = triangleCentroid<double3>(triANode1, triANode2, triANode3);
        } else {
            // Currently, we only support sphere and mesh for body A
            ContactType = deme::NOT_A_CONTACT;
        }

        // ----------------------------------------------------------------
        // Then B, location and velocity, depending on type
        // ----------------------------------------------------------------
        if (BType == deme::GEO_T_SPHERE) {
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
            _forceModelIngredientAcqForB_;
            _forceModelGeoWildcardAcqForBSph_;

            equipOwnerPosRot(simParams, granData, myOwner, myRelPos, BOwnerPos, bodyBPos, BOriQ);

            BRadius = myRadius;
            bodyBMatType = granData->sphereMaterialOffset[sphereID];

            // As the grace margin, the distance (negative overlap) just needs to be within the grace margin. So we pick
            // the larger of the 2 familyExtraMarginSize.
            extraMarginSize = (extraMarginSize > granData->familyExtraMarginSize[BOwnerFamily])
                                  ? extraMarginSize
                                  : granData->familyExtraMarginSize[BOwnerFamily];

            // If B is a sphere, then A can only be a sphere
            checkSpheresOverlap<double, float>(bodyAPos.x, bodyAPos.y, bodyAPos.z, ARadius, bodyBPos.x, bodyBPos.y,
                                               bodyBPos.z, BRadius, contactPnt.x, contactPnt.y, contactPnt.z, B2A.x,
                                               B2A.y, B2A.z, overlapDepth);
            // If overlapDepth is negative then it might still be considered in contact, if the extra margins of A and B
            // combined is larger than abs(overlapDepth)
            if (overlapDepth < -extraMarginSize) {
                ContactType = deme::NOT_A_CONTACT;
            }

        } else if (BType == deme::GEO_T_TRIANGLE) {
            // Geometry ID here is called sphereID, although it is not a sphere, it's more like triID. But naming it
            // sphereID makes the acquisition process cleaner.
            deme::bodyID_t sphereID = granData->idGeometryB[myContactID];
            deme::bodyID_t myOwner = granData->ownerMesh[sphereID];
            //// TODO: Is this OK?
            BRadius = DEME_HUGE_FLOAT;
            bodyBMatType = granData->triMaterialOffset[sphereID];

            // As the grace margin, the distance (negative overlap) just needs to be within the grace margin. So we pick
            // the larger of the 2 familyExtraMarginSize.
            extraMarginSize = (extraMarginSize > granData->familyExtraMarginSize[BOwnerFamily])
                                  ? extraMarginSize
                                  : granData->familyExtraMarginSize[BOwnerFamily];
            // extraMarginSize here is purely family-based extra margin, so it can be used to determine if the user
            // potentially needs remote (non-contact) force calculation.
            bool needsNonContactPenetrationCalc = (extraMarginSize > 0.);

            double3 triBNode1 = to_double3(granData->relPosNode1[sphereID]);
            double3 triBNode2 = to_double3(granData->relPosNode2[sphereID]);
            double3 triBNode3 = to_double3(granData->relPosNode3[sphereID]);

            // Get my mass info from either jitified arrays or global memory
            // Outputs myMass
            // Use an input named exactly `myOwner' which is the id of this owner
            {
                float myMass;
                _massAcqStrat_;
                BOwnerMass = myMass;
            }
            _forceModelIngredientAcqForB_;
            _forceModelGeoWildcardAcqForBTri_;

            // bodyBPos is for a place holder for the outcome triBNode1 position
            equipOwnerPosRot(simParams, granData, myOwner, triBNode1, BOwnerPos, bodyBPos, BOriQ);
            triBNode1 = bodyBPos;
            // Do this to node 2 and 3 as well
            applyOriQToVector3(triBNode2.x, triBNode2.y, triBNode2.z, BOriQ.w, BOriQ.x, BOriQ.y, BOriQ.z);
            triBNode2 += BOwnerPos;
            applyOriQToVector3(triBNode3.x, triBNode3.y, triBNode3.z, BOriQ.w, BOriQ.x, BOriQ.y, BOriQ.z);
            triBNode3 += BOwnerPos;
            // Assign the correct bodyBPos
            bodyBPos = triangleCentroid<double3>(triBNode1, triBNode2, triBNode3);

            // If B is a triangle, then A can be a sphere or a triangle. But this branching is not too bad, as most
            // threads in this block will have the same ContactType.
            if (AType == deme::GEO_T_SPHERE) {
                double3 contact_normal;
                bool in_contact = checkTriSphereOverlap<double3, double>(
                    triBNode1, triBNode2, triBNode3, bodyAPos, ARadius, contact_normal, overlapDepth, contactPnt);
                B2A = to_float3(contact_normal);

                // Sphere--triangle is a bit tricky. Extra margin should only take effect when it comes from the
                // positive direction of the mesh facet. If not, sphere-setting-on-needle case will give huge
                // penetration since in that case, overlapDepth is very negative and this will be considered in-contact.
                // So the cases we exclude are: too far away while at the positive direction; not in contact while at
                // the negative side.
                // Also checkTriSphereOverlap gives positive number for overlapping cases
                if ((overlapDepth < -extraMarginSize) || (!in_contact && overlapDepth > 0.)) {
                    ContactType = deme::NOT_A_CONTACT;
                }
            } else if (AType == deme::GEO_T_TRIANGLE) {
                // Triangle--triangle contact, a bit more complex...
                double3 contact_normal;
                checkTriangleTriangleOverlap<double3, double>(triANode1, triANode2, triANode3, triBNode1, triBNode2,
                                                              triBNode3, contact_normal, overlapDepth, contactPnt,
                                                              needsNonContactPenetrationCalc);
                B2A = to_float3(contact_normal);
                // Fix ContactType if needed
                if (overlapDepth < -extraMarginSize) {
                    ContactType = deme::NOT_A_CONTACT;
                }
            }

        } else if (BType == deme::GEO_T_ANALYTICAL) {
            // Geometry ID here is called sphereID, although it is not a sphere, it's more like analyticalID. But naming
            // it sphereID makes the acquisition process cleaner.
            deme::objID_t sphereID = granData->idGeometryB[myContactID];
            deme::bodyID_t myOwner = objOwner[sphereID];
            // If B is analytical entity, its owner, relative location, material info is jitified.
            bodyBMatType = objMaterial[sphereID];
            BOwnerMass = objMass[sphereID];
            //// TODO: Is this OK?
            BRadius = DEME_HUGE_FLOAT;
            float3 myRelPos;
            float3 bodyBRot;
            myRelPos.x = objRelPosX[sphereID];
            myRelPos.y = objRelPosY[sphereID];
            myRelPos.z = objRelPosZ[sphereID];
            _forceModelIngredientAcqForB_;
            _forceModelGeoWildcardAcqForBAnal_;

            equipOwnerPosRot(simParams, granData, myOwner, myRelPos, BOwnerPos, bodyBPos, BOriQ);

            // As the grace margin, the distance (negative overlap) just needs to be within the grace margin. So we pick
            // the larger of the 2 familyExtraMarginSize.
            extraMarginSize = (extraMarginSize > granData->familyExtraMarginSize[BOwnerFamily])
                                  ? extraMarginSize
                                  : granData->familyExtraMarginSize[BOwnerFamily];

            // B's orientation (such as plane normal) is rotated with its owner too
            bodyBRot.x = objRotX[sphereID];
            bodyBRot.y = objRotY[sphereID];
            bodyBRot.z = objRotZ[sphereID];
            applyOriQToVector3<float, deme::oriQ_t>(bodyBRot.x, bodyBRot.y, bodyBRot.z, BOriQ.w, BOriQ.x, BOriQ.y,
                                                    BOriQ.z);

            // If B is an analytical entity, then A can be a sphere or a triangle. But this branching is not too bad, as
            // most threads in this block will have the same ContactType.
            if (AType == deme::GEO_T_SPHERE) {
                // Note for this test on dT side we don't enlarge entities
                checkSphereEntityOverlap<double3, float, double>(
                    bodyAPos, ARadius, objType[sphereID], bodyBPos, bodyBRot, objSize1[sphereID], objSize2[sphereID],
                    objSize3[sphereID], objNormal[sphereID], 0.0, contactPnt, B2A, overlapDepth);
                // Fix ContactType if needed
                if (overlapDepth < -extraMarginSize) {
                    ContactType = deme::NOT_A_CONTACT;
                }
            } else if (AType == deme::GEO_T_TRIANGLE) {
                calcTriEntityOverlap<double3, double>(
                    triANode1, triANode2, triANode3, objType[sphereID], bodyBPos, bodyBRot, objSize1[sphereID],
                    objSize2[sphereID], objSize3[sphereID], objNormal[sphereID], contactPnt, B2A, overlapDepth);
                // Fix ContactType if needed
                if (overlapDepth < -extraMarginSize) {
                    ContactType = deme::NOT_A_CONTACT;
                }
            }
        }

        _forceModelContactWildcardAcq_;
        if (ContactType != deme::NOT_A_CONTACT) {
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
            // NOTE!! "force" and all wildcards must be properly set by this piece of code
            { _DEMForceModel_; }

            // Write contact location values back to global memory
            _contactInfoWrite_;

            // If force model modifies owner wildcards, write them back here
            _forceModelOwnerWildcardWrite_;

            // Optionally, the forces can be reduced to acc right here (may be faster)
            _forceCollectInPlaceStrat_;
        } else {
            // The contact is no longer active, so we need to destroy its contact history recording
            _forceModelContactWildcardDestroy_;
        }

        // Updated contact wildcards need to be write back to global mem. It is here because contact wildcard may need
        // to be destroyed for non-contact, so it has to go last.
        _forceModelContactWildcardWrite_;
}

// 5 specialized kernels for different contact types
__global__ void calculateContactForces_SphSph(deme::DEMSimParams* simParams,
                                              deme::DEMDataDT* granData,
                                              size_t startOffset,
                                              size_t nContactPairs) {
    deme::contactPairs_t myContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myContactID < startOffset + nContactPairs) {
        calculateContactForcesImpl<deme::SPHERE_SPHERE_CONTACT>(simParams, granData, myContactID);
    }
}

__global__ void calculateContactForces_SphTri(deme::DEMSimParams* simParams,
                                              deme::DEMDataDT* granData,
                                              size_t startOffset,
                                              size_t nContactPairs) {
    deme::contactPairs_t myContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myContactID < startOffset + nContactPairs) {
        calculateContactForcesImpl<deme::SPHERE_TRIANGLE_CONTACT>(simParams, granData, myContactID);
    }
}

__global__ void calculateContactForces_SphAnal(deme::DEMSimParams* simParams,
                                               deme::DEMDataDT* granData,
                                               size_t startOffset,
                                               size_t nContactPairs) {
    deme::contactPairs_t myContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myContactID < startOffset + nContactPairs) {
        calculateContactForcesImpl<deme::SPHERE_ANALYTICAL_CONTACT>(simParams, granData, myContactID);
    }
}

__global__ void calculateContactForces_TriTri(deme::DEMSimParams* simParams,
                                              deme::DEMDataDT* granData,
                                              size_t startOffset,
                                              size_t nContactPairs) {
    deme::contactPairs_t myContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myContactID < startOffset + nContactPairs) {
        calculateContactForcesImpl<deme::TRIANGLE_TRIANGLE_CONTACT>(simParams, granData, myContactID);
    }
}

__global__ void calculateContactForces_TriAnal(deme::DEMSimParams* simParams,
                                               deme::DEMDataDT* granData,
                                               size_t startOffset,
                                               size_t nContactPairs) {
    deme::contactPairs_t myContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myContactID < startOffset + nContactPairs) {
        calculateContactForcesImpl<deme::TRIANGLE_ANALYTICAL_CONTACT>(simParams, granData, myContactID);
    }
}
