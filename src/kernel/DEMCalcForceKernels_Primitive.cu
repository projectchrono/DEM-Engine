// DEM force computation related custom kernels
#include <DEM/Defines.h>
#include <DEMCollisionKernels_SphSph.cuh>
#include <DEMCollisionKernels_SphTri_TriTri.cuh>
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

// Template device function for contact force calculation - will be called by 5 specialized kernels
template <deme::contact_t CONTACT_TYPE>
__device__ __forceinline__ void calculatePrimitiveContactForces_impl(deme::DEMSimParams* simParams,
                                                                     deme::DEMDataDT* granData,
                                                                     deme::contactPairs_t myPrimitiveContactID) {
    // Contact type is known at compile time
    deme::contact_t ContactType = CONTACT_TYPE;
    // The following quantities are always calculated, regardless of force model
    // Note it is strangely important to give init values to contactPnt here, otherwise on some systems it seems nvcc
    // optimizes it out before potentially using it to assign VRAM values, giving NaN results.
    double3 contactPnt = make_double3(0.0, 0.0, 0.0);  // contact point in global coords
    float3 B2A = make_float3(0.f, 0.f, 1.f);           // contact normal felt by A, pointing from B to A, unit vector
    // Penetration depth
    // Positive number for overlapping cases, negative for non-overlapping cases
    double overlapDepth = 0.0;
    // Area of the contact surface, or in the mesh--mesh case, area of the clipping polygon projection
    double overlapArea = 0.0;
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
    // One important thing: When writing results, the contact index is oganized by patch-based contact IDs.
    // For sph-sph and sph-anal contacts, this primitive sweep already generates the final results, so putting the
    // resulting into the correct place needs to be done here.
    deme::contactPairs_t myPatchContactID = granData->geomToPatchMap[myPrimitiveContactID];

    // ----------------------------------------------------------------
    // Based on A's type, equip info
    // ----------------------------------------------------------------
    if constexpr (AType == deme::GEO_T_SPHERE) {
        deme::bodyID_t sphereID = granData->idPrimitiveA[myPrimitiveContactID];
        deme::bodyID_t myOwner = granData->ownerClumpBody[sphereID];
        // Clump sphere's patch ID is just the sphereID itself
        deme::bodyID_t myPatchID = sphereID;

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
        // In priciple, material is patch-associated; but in clump, for now, we don't distinguish sphere and patch
        bodyAMatType = granData->sphereMaterialOffset[myPatchID];
        extraMarginSize = granData->familyExtraMarginSize[AOwnerFamily];
    } else if constexpr (AType == deme::GEO_T_TRIANGLE) {
        deme::bodyID_t triID = granData->idPrimitiveA[myPrimitiveContactID];
        deme::bodyID_t myOwner = granData->ownerTriMesh[triID];
        //// TODO: Is this OK?
        ARadius = DEME_HUGE_FLOAT;
        // If this is a triangle then it has a patch ID
        deme::bodyID_t myPatchID = granData->triPatchID[triID];
        bodyAMatType = granData->patchMaterialOffset[myPatchID];
        extraMarginSize = granData->familyExtraMarginSize[AOwnerFamily];

        triANode1 = to_double3(granData->relPosNode1[triID]);
        triANode2 = to_double3(granData->relPosNode2[triID]);
        triANode3 = to_double3(granData->relPosNode3[triID]);

        // Get my mass info from either jitified arrays or global memory
        // Outputs myMass
        // Use an input named exactly `myOwner' which is the id of this owner
        {
            float myMass;
            _massAcqStrat_;
            AOwnerMass = myMass;
        }
        _forceModelIngredientAcqForA_;
        _forceModelGeoWildcardAcqForAMeshPatch_;

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
    if constexpr (BType == deme::GEO_T_SPHERE) {
        deme::bodyID_t sphereID = granData->idPrimitiveB[myPrimitiveContactID];
        deme::bodyID_t myOwner = granData->ownerClumpBody[sphereID];
        // Clump sphere's patch ID is just the sphereID itself
        deme::bodyID_t myPatchID = sphereID;

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
        // In priciple, material is patch-associated; but in clump, for now, we don't distinguish sphere and patch
        bodyBMatType = granData->sphereMaterialOffset[myPatchID];

        // As the grace margin, the distance (negative overlap) just needs to be within the grace margin. So we pick
        // the larger of the 2 familyExtraMarginSize.
        extraMarginSize = (extraMarginSize > granData->familyExtraMarginSize[BOwnerFamily])
                              ? extraMarginSize
                              : granData->familyExtraMarginSize[BOwnerFamily];

        // If B is a sphere, then A can only be a sphere
        checkSpheresOverlap<double, float>(bodyAPos.x, bodyAPos.y, bodyAPos.z, ARadius, bodyBPos.x, bodyBPos.y,
                                           bodyBPos.z, BRadius, contactPnt.x, contactPnt.y, contactPnt.z, B2A.x, B2A.y,
                                           B2A.z, overlapDepth, overlapArea);
        // If overlapDepth is negative then it might still be considered in contact, if the extra margins of A and B
        // combined is larger than abs(overlapDepth)
        if (overlapDepth <= -extraMarginSize) {
            ContactType = deme::NOT_A_CONTACT;
        }

    } else if constexpr (BType == deme::GEO_T_TRIANGLE) {
        deme::bodyID_t triID = granData->idPrimitiveB[myPrimitiveContactID];
        deme::bodyID_t myOwner = granData->ownerTriMesh[triID];
        //// TODO: Is this OK?
        BRadius = DEME_HUGE_FLOAT;
        // If this is a triangle then it has a patch ID
        deme::bodyID_t myPatchID = granData->triPatchID[triID];
        bodyBMatType = granData->patchMaterialOffset[myPatchID];

        // As the grace margin, the distance (negative overlap) just needs to be within the grace margin. So we pick
        // the larger of the 2 familyExtraMarginSize.
        extraMarginSize = (extraMarginSize > granData->familyExtraMarginSize[BOwnerFamily])
                              ? extraMarginSize
                              : granData->familyExtraMarginSize[BOwnerFamily];

        double3 triBNode1 = to_double3(granData->relPosNode1[triID]);
        double3 triBNode2 = to_double3(granData->relPosNode2[triID]);
        double3 triBNode3 = to_double3(granData->relPosNode3[triID]);

        // Get my mass info from either jitified arrays or global memory
        // Outputs myMass
        // Use an input named exactly `myOwner' which is the id of this owner
        {
            float myMass;
            _massAcqStrat_;
            BOwnerMass = myMass;
        }
        _forceModelIngredientAcqForB_;
        _forceModelGeoWildcardAcqForBMeshPatch_;

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

        // If B is a triangle, then A can be a sphere or a triangle.
        if constexpr (AType == deme::GEO_T_SPHERE) {
            double3 contact_normal;
            // Note checkTriSphereOverlap gives positive number for overlapping cases.
            // Using checkTriSphereOverlap rather than the directional version is effectively saying if the overlap is
            // more than 2R, we don't want it; this is useful in preventing contact being detection from the other side
            // of a thin mesh.
            bool in_contact =
                checkTriSphereOverlap<double3, double>(triBNode1, triBNode2, triBNode3, bodyAPos, ARadius,
                                                       contact_normal, overlapDepth, overlapArea, contactPnt);
            B2A = to_float3(contact_normal);

            // If the solver says in contact, we do not question it
            if (!in_contact) {
                // Extra margin takes effect
                if (overlapDepth <= -extraMarginSize) {
                    ContactType = deme::NOT_A_CONTACT;
                }
            }

        } else if constexpr (AType == deme::GEO_T_TRIANGLE) {
            // Triangle--triangle contact, a bit more complex...
            double3 contact_normal;
            bool in_contact = checkTriangleTriangleOverlap<double3, double>(triANode1, triANode2, triANode3, triBNode1,
                                                                            triBNode2, triBNode3, contact_normal,
                                                                            overlapDepth, overlapArea, contactPnt);
            B2A = to_float3(contact_normal);

            // Record whether this tri-tri primitive contact satisfies SAT (is in physical contact)
            // Use the dedicated SAT check function to determine if triangles are truly in physical contact
            // Note: checkTriangleTriangleOverlap uses projection which can report contact even for non-physical
            // "submerged" cases, so we need the actual SAT test for accurate physical contact determination
            bool satisfiesSAT = checkTriangleTriangleSAT<double3, double>(triANode1, triANode2, triANode3, triBNode1,
                                                                          triBNode2, triBNode3);
            granData->contactSATSatisfied[myPrimitiveContactID] = satisfiesSAT ? 1 : 0;

            // Fix ContactType if needed
            // If the solver says in contact, we do not question it
            if (!in_contact) {
                // Then, if we have extra margin, we check that if the distance is within the extra margin
                if (overlapDepth <= -extraMarginSize) {
                    ContactType = deme::NOT_A_CONTACT;
                }
            }
        }

    } else if constexpr (BType == deme::GEO_T_ANALYTICAL) {
        deme::objID_t analyticalID = granData->idPrimitiveB[myPrimitiveContactID];
        deme::bodyID_t myOwner = objOwner[analyticalID];
        // For analytical entity, its patch ID is just its own component ID (but myPatchID is hardly used in this
        // analytical case)
        deme::bodyID_t myPatchID = analyticalID;
        // If B is analytical entity, its owner, relative location, material info is jitified.
        bodyBMatType = objMaterial[analyticalID];
        BOwnerMass = objMass[analyticalID];
        //// TODO: Is this OK?
        BRadius = DEME_HUGE_FLOAT;
        float3 myRelPos;
        float3 bodyBRot;
        myRelPos.x = objRelPosX[analyticalID];
        myRelPos.y = objRelPosY[analyticalID];
        myRelPos.z = objRelPosZ[analyticalID];
        _forceModelIngredientAcqForB_;
        _forceModelGeoWildcardAcqForBAnal_;

        equipOwnerPosRot(simParams, granData, myOwner, myRelPos, BOwnerPos, bodyBPos, BOriQ);

        // As the grace margin, the distance (negative overlap) just needs to be within the grace margin. So we pick
        // the larger of the 2 familyExtraMarginSize.
        extraMarginSize = (extraMarginSize > granData->familyExtraMarginSize[BOwnerFamily])
                              ? extraMarginSize
                              : granData->familyExtraMarginSize[BOwnerFamily];

        // B's orientation (such as plane normal) is rotated with its owner too
        bodyBRot.x = objRotX[analyticalID];
        bodyBRot.y = objRotY[analyticalID];
        bodyBRot.z = objRotZ[analyticalID];
        applyOriQToVector3<float, deme::oriQ_t>(bodyBRot.x, bodyBRot.y, bodyBRot.z, BOriQ.w, BOriQ.x, BOriQ.y, BOriQ.z);

        // If B is an analytical entity, then A can be a sphere or a triangle.
        if constexpr (AType == deme::GEO_T_SPHERE) {
            // Note for this test on dT side we don't enlarge entities
            checkSphereEntityOverlap<double3, float, double>(bodyAPos, ARadius, objType[analyticalID], bodyBPos,
                                                             bodyBRot, objSize1[analyticalID], objSize2[analyticalID],
                                                             objSize3[analyticalID], objNormal[analyticalID], 0.0,
                                                             contactPnt, B2A, overlapDepth, overlapArea);
            // Fix ContactType if needed
            if (overlapDepth <= -extraMarginSize) {
                ContactType = deme::NOT_A_CONTACT;
            }
        } else if constexpr (AType == deme::GEO_T_TRIANGLE) {
            calcTriEntityOverlap<double3, double>(triANode1, triANode2, triANode3, objType[analyticalID], bodyBPos,
                                                  bodyBRot, objSize1[analyticalID], objSize2[analyticalID],
                                                  objSize3[analyticalID], objNormal[analyticalID], contactPnt, B2A,
                                                  overlapDepth, overlapArea);
            // Fix ContactType if needed
            if (overlapDepth <= -extraMarginSize) {
                ContactType = deme::NOT_A_CONTACT;
            }
        }
    }

    if constexpr (CONTACT_TYPE == deme::SPHERE_SPHERE_CONTACT || CONTACT_TYPE == deme::SPHERE_ANALYTICAL_CONTACT) {
        _forceModelContactWildcardAcq_;

        // Essentials for storing and calculating contact info
        float3 force = make_float3(0, 0, 0);
        float3 torque_only_force = make_float3(0, 0, 0);
        // Local position of the contact point is always a piece of info we require... regardless of force model
        float3 locCPA = to_float3(contactPnt - AOwnerPos);
        float3 locCPB = to_float3(contactPnt - BOwnerPos);
        // Now map this contact point location to bodies' local ref
        applyOriQToVector3<float, deme::oriQ_t>(locCPA.x, locCPA.y, locCPA.z, AOriQ.w, -AOriQ.x, -AOriQ.y, -AOriQ.z);
        applyOriQToVector3<float, deme::oriQ_t>(locCPB.x, locCPB.y, locCPB.z, BOriQ.w, -BOriQ.x, -BOriQ.y, -BOriQ.z);

        if (ContactType != deme::NOT_A_CONTACT) {
            // The following part, the force model, is user-specifiable
            // NOTE!! "force" and all wildcards must be properly set by this piece of code
            { _DEMForceModel_; }

            // If force model modifies owner wildcards, write them back here
            _forceModelOwnerWildcardWrite_;
        } else {
            // The contact is no longer active, so we need to destroy its contact history recording
            _forceModelContactWildcardDestroy_;
        }

        // Note in DEME3, we do not clear force array anymore in each timestep, so always writing back force and contact
        // points, even for zero-force non-contacts, is needed (unless of course, the user instructed no force record).
        // This design has implications in our new two-step patch-based force calculation algorithm, as we re-use some
        // force-storing arrays for intermediate values.

        // Write contact location values back to global memory
        _contactInfoWrite_;

        // Optionally, the forces can be reduced to acc right here (may be faster)
        _forceCollectInPlaceStrat_;

        // Updated contact wildcards need to be write back to global mem. It is here because contact wildcard may need
        // to be destroyed for non-contact, so it has to go last.
        _forceModelContactWildcardWrite_;
    } else {  // If this is the kernel for a mesh-related contact, another follow-up kernel is needed to compute force
        // Use contactForces, contactPointGeometryAB to store the contact info for the next
        // kernel to compute forces. contactForces is used to store the contact normal. contactPointGeometryA is used to
        // store the (double) contact penetration. contactPointGeometryB is used to store the (double) contact area
        // contactTorque_convToForce is used to store the contact point position (cast from double3 to float3)

        // Store contact normal (B2A is already a float3)
        granData->contactForces[myPrimitiveContactID] = B2A;
        // Store contact penetration depth (double) in contactPointGeometryA (float3)
        granData->contactPointGeometryA[myPrimitiveContactID] = doubleToFloat3Storage(overlapDepth);
        // Store contact area (double) in contactPointGeometryB (float3)
        // If this is not a contact, we store 0.0 in the area, so it has no voting power in the next kernels. Note the
        // NOT_A_CONTACT control flow here and in the next few kernels is integrated in areas.
        granData->contactPointGeometryB[myPrimitiveContactID] =
            doubleToFloat3Storage((ContactType == deme::NOT_A_CONTACT || overlapArea <= 0.0) ? 0.0 : overlapArea);
        // Store contact point (cast from double3 to float3). Could make the following check, but hopefully it's not
        // necessary. if (!isfinite(contactPnt.x) || !isfinite(contactPnt.y) || !isfinite(contactPnt.z)) {
        //     DEME_ABORT_KERNEL(
        //         "Primitive contact No. %d of type %d has contact point with infinite component(s), something is "
        //         "wrong.\n",
        //         myPrimitiveContactID, ContactType);
        // }
        granData->contactTorque_convToForce[myPrimitiveContactID] = to_float3(contactPnt);
    }
}

// 5 specialized kernels for different contact types
__global__ void calculatePrimitiveContactForces_SphSph(deme::DEMSimParams* simParams,
                                                       deme::DEMDataDT* granData,
                                                       deme::contactPairs_t startOffset,
                                                       deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPrimitiveContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPrimitiveContactID < startOffset + nContactPairs) {
        calculatePrimitiveContactForces_impl<deme::SPHERE_SPHERE_CONTACT>(simParams, granData, myPrimitiveContactID);
    }
}

__global__ void calculatePrimitiveContactForces_SphTri(deme::DEMSimParams* simParams,
                                                       deme::DEMDataDT* granData,
                                                       deme::contactPairs_t startOffset,
                                                       deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPrimitiveContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPrimitiveContactID < startOffset + nContactPairs) {
        calculatePrimitiveContactForces_impl<deme::SPHERE_TRIANGLE_CONTACT>(simParams, granData, myPrimitiveContactID);
    }
}

__global__ void calculatePrimitiveContactForces_SphAnal(deme::DEMSimParams* simParams,
                                                        deme::DEMDataDT* granData,
                                                        deme::contactPairs_t startOffset,
                                                        deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPrimitiveContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPrimitiveContactID < startOffset + nContactPairs) {
        calculatePrimitiveContactForces_impl<deme::SPHERE_ANALYTICAL_CONTACT>(simParams, granData,
                                                                              myPrimitiveContactID);
    }
}

__global__ void calculatePrimitiveContactForces_TriTri(deme::DEMSimParams* simParams,
                                                       deme::DEMDataDT* granData,
                                                       deme::contactPairs_t startOffset,
                                                       deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPrimitiveContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPrimitiveContactID < startOffset + nContactPairs) {
        calculatePrimitiveContactForces_impl<deme::TRIANGLE_TRIANGLE_CONTACT>(simParams, granData,
                                                                              myPrimitiveContactID);
    }
}

__global__ void calculatePrimitiveContactForces_TriAnal(deme::DEMSimParams* simParams,
                                                        deme::DEMDataDT* granData,
                                                        deme::contactPairs_t startOffset,
                                                        deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPrimitiveContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPrimitiveContactID < startOffset + nContactPairs) {
        calculatePrimitiveContactForces_impl<deme::TRIANGLE_ANALYTICAL_CONTACT>(simParams, granData,
                                                                                myPrimitiveContactID);
    }
}
