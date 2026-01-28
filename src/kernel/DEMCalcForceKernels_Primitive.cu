// DEM force computation related custom kernels
#include <DEM/Defines.h>
#include <DEMCollisionKernels_SphSph.cuh>
#include <DEMCollisionKernels_SphTri_TriTri.cuh>
_kernelIncludes_;

inline __device__ float3 cylPeriodicRotatePosF(const float3& pos, const deme::DEMSimParams* simParams, float sin_span) {
    float3 pos_local = make_float3(pos.x - simParams->LBFX, pos.y - simParams->LBFY, pos.z - simParams->LBFZ);
    pos_local = cylPeriodicRotate(pos_local, simParams->cylPeriodicOrigin, simParams->cylPeriodicAxisVec,
                                  simParams->cylPeriodicU, simParams->cylPeriodicV, simParams->cylPeriodicCosSpan,
                                  sin_span);
    pos_local.x += simParams->LBFX;
    pos_local.y += simParams->LBFY;
    pos_local.z += simParams->LBFZ;
    return pos_local;
}

inline __device__ float3 cylPeriodicRotatePosF(const float3& pos,
                                               const deme::DEMSimParams* simParams,
                                               float cos_theta,
                                               float sin_theta) {
    float3 pos_local = make_float3(pos.x - simParams->LBFX, pos.y - simParams->LBFY, pos.z - simParams->LBFZ);
    pos_local = cylPeriodicRotate(pos_local, simParams->cylPeriodicOrigin, simParams->cylPeriodicAxisVec,
                                  simParams->cylPeriodicU, simParams->cylPeriodicV, cos_theta, sin_theta);
    pos_local.x += simParams->LBFX;
    pos_local.y += simParams->LBFY;
    pos_local.z += simParams->LBFZ;
    return pos_local;
}

inline __device__ float3 cylPeriodicRotatePosF(const float3& pos, const deme::DEMSimParams* simParams) {
    return cylPeriodicRotatePosF(pos, simParams, simParams->cylPeriodicSinSpan);
}

inline __device__ double3 cylPeriodicRotatePosD(const double3& pos, const deme::DEMSimParams* simParams, float sin_span) {
    float3 p = make_float3((float)pos.x, (float)pos.y, (float)pos.z);
    p = cylPeriodicRotatePosF(p, simParams, sin_span);
    return make_double3((double)p.x, (double)p.y, (double)p.z);
}

inline __device__ double3
cylPeriodicRotatePosD(const double3& pos, const deme::DEMSimParams* simParams, float cos_theta, float sin_theta) {
    float3 p = make_float3((float)pos.x, (float)pos.y, (float)pos.z);
    p = cylPeriodicRotatePosF(p, simParams, cos_theta, sin_theta);
    return make_double3((double)p.x, (double)p.y, (double)p.z);
}

inline __device__ double3 cylPeriodicRotatePosD(const double3& pos, const deme::DEMSimParams* simParams) {
    return cylPeriodicRotatePosD(pos, simParams, simParams->cylPeriodicSinSpan);
}

inline __device__ float3 cylPeriodicRotateVec(const float3& vec, const deme::DEMSimParams* simParams, float sin_span) {
    return cylPeriodicRotate(vec, make_float3(0.f, 0.f, 0.f), simParams->cylPeriodicAxisVec, simParams->cylPeriodicU,
                             simParams->cylPeriodicV, simParams->cylPeriodicCosSpan, sin_span);
}

inline __device__ float3
cylPeriodicRotateVec(const float3& vec, const deme::DEMSimParams* simParams, float cos_theta, float sin_theta) {
    return cylPeriodicRotate(vec, make_float3(0.f, 0.f, 0.f), simParams->cylPeriodicAxisVec, simParams->cylPeriodicU,
                             simParams->cylPeriodicV, cos_theta, sin_theta);
}

inline __device__ float3 cylPeriodicRotateVec(const float3& vec, const deme::DEMSimParams* simParams) {
    return cylPeriodicRotateVec(vec, simParams, simParams->cylPeriodicSinSpan);
}

inline __device__ float4 cylPeriodicRotateQuat(const float4& q, const deme::DEMSimParams* simParams, float sin_half_span) {
    float4 rotQ = make_float4(simParams->cylPeriodicAxisVec.x * sin_half_span,
                              simParams->cylPeriodicAxisVec.y * sin_half_span,
                              simParams->cylPeriodicAxisVec.z * sin_half_span,
                              simParams->cylPeriodicCosHalfSpan);
    float4 out = q;
    HamiltonProduct(out.w, out.x, out.y, out.z, rotQ.w, rotQ.x, rotQ.y, rotQ.z, out.w, out.x, out.y, out.z);
    out /= length(out);
    return out;
}

inline __device__ float4
cylPeriodicRotateQuat(const float4& q, const deme::DEMSimParams* simParams, float cos_half, float sin_half) {
    float4 rotQ = make_float4(simParams->cylPeriodicAxisVec.x * sin_half,
                              simParams->cylPeriodicAxisVec.y * sin_half,
                              simParams->cylPeriodicAxisVec.z * sin_half,
                              cos_half);
    float4 out = q;
    HamiltonProduct(out.w, out.x, out.y, out.z, rotQ.w, rotQ.x, rotQ.y, rotQ.z, out.w, out.x, out.y, out.z);
    out /= length(out);
    return out;
}

inline __device__ float4 cylPeriodicRotateQuat(const float4& q, const deme::DEMSimParams* simParams) {
    return cylPeriodicRotateQuat(q, simParams, simParams->cylPeriodicSinHalfSpan);
}

inline __device__ float cylPeriodicRelAngleGlobal(const float3& pos, const deme::DEMSimParams* simParams) {
    const float3 pos_local =
        make_float3(pos.x - simParams->LBFX, pos.y - simParams->LBFY, pos.z - simParams->LBFZ);
    return cylPeriodicRelAngle(pos_local, false, false, simParams);
}

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
    // `Body pos' in the primitive contact kernel means the position of the primitive itself, e.g., sphere center or
    // triangle nodes
    double3 AOwnerPos, bodyAPos, BOwnerPos, bodyBPos;
    // Radius always means radius of curvature; for triangle and analytical entity, it's set to a huge number
    float AOwnerMass, ARadius, BOwnerMass, BRadius;
    float4 AOriQ, BOriQ;
    double3 AOwnerPos_orig = make_double3(0.0, 0.0, 0.0);
    double3 BOwnerPos_orig = make_double3(0.0, 0.0, 0.0);
    float4 AOriQ_orig = make_float4(0.f, 0.f, 0.f, 1.f);
    float4 BOriQ_orig = make_float4(0.f, 0.f, 0.f, 1.f);
    deme::materialsOffset_t bodyAMatType, bodyBMatType;
    // The user-specified extra margin size (how much we should be lenient in determining `in-contact')
    float extraMarginSize = 0.;
    // Triangle A's three points are defined outside, as may be reused in B's acquisition and penetration calc.
    double3 triANode1, triANode2, triANode3;
    // Mesh's patch location may be needed for testing if this primitive contact respects the patch's general spatial
    // direction
    float3 triPatchPosA;
    // Then allocate the optional quantities that will be needed in the force model (note: this one can't be in a
    // curly bracket, obviously...)
    _forceModelIngredientDefinition_;
    bool wrapA = false;
    bool wrapB = false;
    bool wrapA_neg = false;
    bool wrapB_neg = false;
    bool isGhostA = false;
    bool isGhostB = false;
    bool discardGhostGhost = false;
    deme::bodyID_t ownerA = deme::NULL_BODYID;
    deme::bodyID_t ownerB = deme::NULL_BODYID;
    float wrapA_cos = 1.f;
    float wrapA_sin = 0.f;
    float wrapB_cos = 1.f;
    float wrapB_sin = 0.f;

    // Cylindrical periodic: desired minimum-image shift selection (decided on dT)
    int desiredShiftA = 0;
    int desiredShiftB = 0;
    int ghostShiftA = 0;
    int ghostShiftB = 0;
    int wrapShiftA = 0;
    int wrapShiftB = 0;
    float desiredBestDist2 = 0.f;
    float desiredRadAOwner = 0.f;
    float desiredRadBOwner = 0.f;
    bool cylPeriodicSkipPair = false;
    const deme::bodyID_t idA_raw = granData->idPrimitiveA[myPrimitiveContactID];
    const deme::bodyID_t idB_raw = granData->idPrimitiveB[myPrimitiveContactID];
    // Take care of 2 bodies in order, bodyA first, grab location and velocity to local cache
    // Decompose ContactType to get the types of A and B (known at compile time)
    constexpr deme::geoType_t AType = (CONTACT_TYPE >> 4);
    constexpr deme::geoType_t BType = (CONTACT_TYPE & 0xF);
    // One important thing: When writing results, the contact index is oganized by patch-based contact IDs.
    // For sph-sph and sph-anal contacts, this primitive sweep already generates the final results, so putting the
    // resulting into the correct place needs to be done here.
    deme::contactPairs_t myPatchContactID = granData->geomToPatchMap[myPrimitiveContactID];

    // Default: patch-direction check should not filter non-tri-tri contacts.
    // Tri-tri will overwrite this after computing patch direction.
    granData->contactPatchDirectionRespected[myPrimitiveContactID] = 1;

    // ----------------------------------------------------------------
    // Based on A's type, equip info
    // ----------------------------------------------------------------
    if constexpr (AType == deme::GEO_T_SPHERE) {
        deme::bodyID_t sphereID = idA_raw & deme::CYL_PERIODIC_SPHERE_ID_MASK;
        deme::bodyID_t myOwner = granData->ownerClumpBody[sphereID];
        ownerA = myOwner;
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
        deme::bodyID_t triID = idA_raw & deme::CYL_PERIODIC_SPHERE_ID_MASK;
        deme::bodyID_t myOwner = granData->ownerTriMesh[triID];
        ownerA = myOwner;
        //// TODO: Is this OK?
        ARadius = DEME_HUGE_FLOAT;
        // If this is a triangle then it has a patch ID
        deme::bodyID_t myPatchID = granData->triPatchID[triID];
        bodyAMatType = granData->patchMaterialOffset[myPatchID];
        extraMarginSize = granData->familyExtraMarginSize[AOwnerFamily];
        float3 relPosPatch = granData->relPosPatch[myPatchID];

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

        // Get triPatchPosA ready
        applyOriQToVector3(relPosPatch.x, relPosPatch.y, relPosPatch.z, AOriQ.w, AOriQ.x, AOriQ.y, AOriQ.z);
        triPatchPosA = relPosPatch + to_float3(AOwnerPos);
    } else {
        // Currently, we only support sphere and mesh for body A
        ContactType = deme::NOT_A_CONTACT;
    }

    // ----------------------------------------------------------------
    // Then B, location and velocity, depending on type
    // ----------------------------------------------------------------
    if constexpr (BType == deme::GEO_T_SPHERE) {
        deme::bodyID_t sphereID = idB_raw & deme::CYL_PERIODIC_SPHERE_ID_MASK;
        deme::bodyID_t myOwner = granData->ownerClumpBody[sphereID];
        ownerB = myOwner;
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

        AOwnerPos_orig = AOwnerPos;
        BOwnerPos_orig = BOwnerPos;
        AOriQ_orig = AOriQ;
        BOriQ_orig = BOriQ;
        if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f) {
            bool ghostA = false;
            bool ghostB = false;
            bool ghostA_neg = false;
            bool ghostB_neg = false;
            if constexpr (AType == deme::GEO_T_SPHERE || AType == deme::GEO_T_TRIANGLE) {
                cylPeriodicDecodeID(idA_raw, ghostA, ghostA_neg);
            }
            if constexpr (BType == deme::GEO_T_SPHERE || BType == deme::GEO_T_TRIANGLE) {
                cylPeriodicDecodeID(idB_raw, ghostB, ghostB_neg);
            }


            ghostShiftA = ghostA ? (ghostA_neg ? -1 : 1) : 0;
            ghostShiftB = ghostB ? (ghostB_neg ? -1 : 1) : 0;
            // Use only the kT ghost flag for geometry/force evaluation (base-wedge frame).
            wrapShiftA = ghostShiftA;
            wrapShiftB = ghostShiftB;
            if (granData->ownerCylWrapOffset) {
                if (ownerA != deme::NULL_BODYID && ownerA < simParams->nOwnerBodies) {
                    wrapShiftA += granData->ownerCylWrapOffset[ownerA];
                }
                if (ownerB != deme::NULL_BODYID && ownerB < simParams->nOwnerBodies) {
                    wrapShiftB += granData->ownerCylWrapOffset[ownerB];
                }
            }
            desiredShiftA = 0;
            desiredShiftB = 0;
            cylPeriodicSkipPair = false;
            if (ghostShiftA > 1 || ghostShiftA < -1 || ghostShiftB > 1 || ghostShiftB < -1) {
                ghostShiftA = 0;
                ghostShiftB = 0;
                cylPeriodicSkipPair = true;
                discardGhostGhost = true;
                ContactType = deme::NOT_A_CONTACT;
            }
            {
                float3 ownerA_sel = to_float3(bodyAPos);
                float3 ownerB_sel = to_float3(bodyBPos);
                bool availA_pos = true;
                bool availA_neg = true;
                bool availB_pos = true;
                bool availB_neg = true;
                desiredRadAOwner = 0.f;
                desiredRadBOwner = 0.f;
                if (granData->ownerBoundRadius) {
                    desiredRadAOwner = granData->ownerBoundRadius[ownerA];
                    desiredRadBOwner = granData->ownerBoundRadius[ownerB];
                    cylPeriodicGhostAvailability(ownerA_sel, desiredRadAOwner, simParams, availA_pos, availA_neg);
                    cylPeriodicGhostAvailability(ownerB_sel, desiredRadBOwner, simParams, availB_pos, availB_neg);
                }
                float bestDist = 0.f;
                cylPeriodicSelectGhostShiftByDistAvail(ownerA_sel, ownerB_sel, simParams, availA_pos, availA_neg,
                                                       availB_pos, availB_neg, desiredShiftA, desiredShiftB, bestDist);
                desiredBestDist2 = bestDist;
                // Deterministic owner-pair min-image selection for this dT step.
            }
            // Analytical objects are not periodic; discard ghosted contacts to avoid double counting.
            if (ownerA == deme::NULL_BODYID || ownerB == deme::NULL_BODYID) {
                if (ghostShiftA != 0 || ghostShiftB != 0) {
                    discardGhostGhost = true;
                }
                desiredShiftA = 0;
                desiredShiftB = 0;
            }
            // dT selects the active minimum-image periodic representation based on current configuration.
            // IMPORTANT: kT is allowed to emit multiple ghost-involved image pairs. dT recomputes which image is
            // the current minimum-image and keeps only the matching representation for this step.
            int baseShiftA = ghostShiftA;
            int baseShiftB = ghostShiftB;

            // Never allow ghost-ghost pairs in cylindrical wedge periodicity (non-minimal ±2 images).
            if (ghostShiftA != 0 && ghostShiftB != 0) {
                ContactType = deme::NOT_A_CONTACT;
                discardGhostGhost = true;
            } else {
                // Keep only the periodic image pair that matches the desired (minimum-image) shifts for this step.
                // If this candidate does not match, mark it as non-contact so it contributes neither force nor history.
                if (desiredShiftA != ghostShiftA || desiredShiftB != ghostShiftB) {
                    cylPeriodicSkipPair = true;
                }
            }
            // Effective ghosting for dT (after min-image selection / desired shifts).
            const bool activeGhostA = (baseShiftA != 0);
            const bool activeGhostB = (baseShiftB != 0);
            isGhostA = activeGhostA;
            isGhostB = activeGhostB;

            // Signal kT to keep/prepare ghosts for the desired periodic image(s) next step.
            // Use shared cylindrical periodic flags:
            // START/END for desired side, MISMATCH when kT-provided image differs from dT desired image.
            if (simParams->useCylPeriodicDiagCounters && granData->ownerCylGhostActive) {
                const bool mismatch = (desiredShiftA != ghostShiftA || desiredShiftB != ghostShiftB);
                const bool mismatch_force_relevant = mismatch && (ContactType != deme::NOT_A_CONTACT);
                if (mismatch_force_relevant) {
                    if (ownerA != deme::NULL_BODYID) {
                        atomicOr(granData->ownerCylGhostActive + ownerA, deme::CYL_GHOST_HINT_MISMATCH);
                    }
                    if (ownerB != deme::NULL_BODYID) {
                        atomicOr(granData->ownerCylGhostActive + ownerB, deme::CYL_GHOST_HINT_MISMATCH);
                    }
                }
                if (desiredShiftA > 0 && ownerA != deme::NULL_BODYID) {
                    atomicOr(granData->ownerCylGhostActive + ownerA, deme::CYL_GHOST_HINT_START);
                } else if (desiredShiftA < 0 && ownerA != deme::NULL_BODYID) {
                    atomicOr(granData->ownerCylGhostActive + ownerA, deme::CYL_GHOST_HINT_END);
                }
                if (desiredShiftB > 0 && ownerB != deme::NULL_BODYID) {
                    atomicOr(granData->ownerCylGhostActive + ownerB, deme::CYL_GHOST_HINT_START);
                } else if (desiredShiftB < 0 && ownerB != deme::NULL_BODYID) {
                    atomicOr(granData->ownerCylGhostActive + ownerB, deme::CYL_GHOST_HINT_END);
                }
            }

            // Keep IDs consistent with the selected periodic representation (important when this primitive contact
            // originated from an un-ghosted pair but dT selects a wrapped image).
            if (ContactType != deme::NOT_A_CONTACT) {
                if constexpr (AType == deme::GEO_T_SPHERE || AType == deme::GEO_T_TRIANGLE) {                }
                if constexpr (BType == deme::GEO_T_SPHERE || BType == deme::GEO_T_TRIANGLE) {                }
            }

            const bool kTGhostA = (ghostShiftA != 0);
            const bool kTGhostB = (ghostShiftB != 0);
            wrapA = (wrapShiftA != 0);
            wrapB = (wrapShiftB != 0);
            wrapA_neg = wrapShiftA < 0;
            wrapB_neg = wrapShiftB < 0;

            // Ghost-ghost pairs are never used in cylindrical wedge periodicity.
            if (kTGhostA && kTGhostB) {
                discardGhostGhost = true;
                ContactType = deme::NOT_A_CONTACT;
            }

            float cosA = 1.f, sinA = 0.f, cosHalfA = 1.f, sinHalfA = 0.f;
            float cosB = 1.f, sinB = 0.f, cosHalfB = 1.f, sinHalfB = 0.f;
            if (wrapA) {
                cylPeriodicShiftTrig(wrapShiftA, simParams, cosA, sinA, cosHalfA, sinHalfA);
                wrapA_cos = cosA;
                wrapA_sin = sinA;
            }
            if (wrapB) {
                cylPeriodicShiftTrig(wrapShiftB, simParams, cosB, sinB, cosHalfB, sinHalfB);
                wrapB_cos = cosB;
                wrapB_sin = sinB;
            }

            if (wrapA) {
                AOwnerPos = cylPeriodicRotatePosD(AOwnerPos, simParams, cosA, sinA);
                bodyAPos = cylPeriodicRotatePosD(bodyAPos, simParams, cosA, sinA);
                AOriQ = cylPeriodicRotateQuat(AOriQ, simParams, cosHalfA, sinHalfA);
#if _forceModelHasLinVel_
                ALinVel = cylPeriodicRotateVec(ALinVel, simParams, cosA, sinA);
#endif
#if _forceModelHasRotVel_
                ARotVel = cylPeriodicRotateVec(ARotVel, simParams, cosA, sinA);
#endif
                if constexpr (AType == deme::GEO_T_TRIANGLE) {
                    triPatchPosA = to_float3(cylPeriodicRotatePosD(to_double3(triPatchPosA), simParams, cosA, sinA));
                }
            }
            if (wrapB) {
                BOwnerPos = cylPeriodicRotatePosD(BOwnerPos, simParams, cosB, sinB);
                bodyBPos = cylPeriodicRotatePosD(bodyBPos, simParams, cosB, sinB);
                BOriQ = cylPeriodicRotateQuat(BOriQ, simParams, cosHalfB, sinHalfB);
#if _forceModelHasLinVel_
                BLinVel = cylPeriodicRotateVec(BLinVel, simParams, cosB, sinB);
#endif
#if _forceModelHasRotVel_
                BRotVel = cylPeriodicRotateVec(BRotVel, simParams, cosB, sinB);
#endif
            }
        }
        // If B is a sphere, then A can only be a sphere
        checkSpheresOverlap<double, float>(bodyAPos.x, bodyAPos.y, bodyAPos.z, ARadius, bodyBPos.x, bodyBPos.y,
                                           bodyBPos.z, BRadius, contactPnt.x, contactPnt.y, contactPnt.z, B2A.x, B2A.y,
                                           B2A.z, overlapDepth);
        // If overlapDepth is negative then it might still be considered in contact, if the extra margins of A and B
        // combined is larger than abs(overlapDepth)
        if (overlapDepth <= -extraMarginSize) {
            ContactType = deme::NOT_A_CONTACT;
        }

    } else if constexpr (BType == deme::GEO_T_TRIANGLE) {
        deme::bodyID_t triID = idB_raw & deme::CYL_PERIODIC_SPHERE_ID_MASK;
        deme::bodyID_t myOwner = granData->ownerTriMesh[triID];
        ownerB = myOwner;
        //// TODO: Is this OK?
        BRadius = DEME_HUGE_FLOAT;
        // If this is a triangle then it has a patch ID
        deme::bodyID_t myPatchID = granData->triPatchID[triID];
        bodyBMatType = granData->patchMaterialOffset[myPatchID];
        float3 relPosPatch = granData->relPosPatch[myPatchID];

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
        // Get triPatchPosB ready
        applyOriQToVector3(relPosPatch.x, relPosPatch.y, relPosPatch.z, BOriQ.w, BOriQ.x, BOriQ.y, BOriQ.z);
        float3 triPatchPosB = relPosPatch + to_float3(BOwnerPos);

        AOwnerPos_orig = AOwnerPos;
        BOwnerPos_orig = BOwnerPos;
        AOriQ_orig = AOriQ;
        BOriQ_orig = BOriQ;
        if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f) {
            bool ghostA = false;
            bool ghostB = false;
            bool ghostA_neg = false;
            bool ghostB_neg = false;
            if constexpr (AType == deme::GEO_T_SPHERE || AType == deme::GEO_T_TRIANGLE) {
                cylPeriodicDecodeID(idA_raw, ghostA, ghostA_neg);
            }
            if constexpr (BType == deme::GEO_T_SPHERE || BType == deme::GEO_T_TRIANGLE) {
                cylPeriodicDecodeID(idB_raw, ghostB, ghostB_neg);
            }


            ghostShiftA = ghostA ? (ghostA_neg ? -1 : 1) : 0;
            ghostShiftB = ghostB ? (ghostB_neg ? -1 : 1) : 0;
            wrapShiftA = ghostShiftA;
            wrapShiftB = ghostShiftB;
            if (granData->ownerCylWrapOffset) {
                if (ownerA != deme::NULL_BODYID && ownerA < simParams->nOwnerBodies) {
                    wrapShiftA += granData->ownerCylWrapOffset[ownerA];
                }
                if (ownerB != deme::NULL_BODYID && ownerB < simParams->nOwnerBodies) {
                    wrapShiftB += granData->ownerCylWrapOffset[ownerB];
                }
            }
            desiredShiftA = 0;
            desiredShiftB = 0;
            cylPeriodicSkipPair = false;
            if (ghostShiftA > 1 || ghostShiftA < -1 || ghostShiftB > 1 || ghostShiftB < -1) {
                ghostShiftA = 0;
                ghostShiftB = 0;
                cylPeriodicSkipPair = true;
                discardGhostGhost = true;
                ContactType = deme::NOT_A_CONTACT;
            }
            {
                float3 ownerA_sel = to_float3(AOwnerPos);
                float3 ownerB_sel = to_float3(BOwnerPos);
                bool availA_pos = true;
                bool availA_neg = true;
                bool availB_pos = true;
                bool availB_neg = true;
                desiredRadAOwner = 0.f;
                desiredRadBOwner = 0.f;
                if (granData->ownerBoundRadius) {
                    desiredRadAOwner = granData->ownerBoundRadius[ownerA];
                    desiredRadBOwner = granData->ownerBoundRadius[ownerB];
                    cylPeriodicGhostAvailability(ownerA_sel, desiredRadAOwner, simParams, availA_pos, availA_neg);
                    cylPeriodicGhostAvailability(ownerB_sel, desiredRadBOwner, simParams, availB_pos, availB_neg);
                }
                float bestDist = 0.f;
                cylPeriodicSelectGhostShiftByDistAvail(ownerA_sel, ownerB_sel, simParams, availA_pos, availA_neg,
                                                       availB_pos, availB_neg, desiredShiftA, desiredShiftB, bestDist);
                desiredBestDist2 = bestDist;
                // Deterministic owner-pair min-image selection for this dT step.
            }
            // Analytical objects are not periodic; only honor the shifts encoded in the IDs.
            int baseShiftA = ghostShiftA;
            int baseShiftB = ghostShiftB;
            if (ghostShiftA != 0 && ghostShiftB != 0) {
                ContactType = deme::NOT_A_CONTACT;
                discardGhostGhost = true;
            } else {
                if (desiredShiftA != ghostShiftA || desiredShiftB != ghostShiftB) {
                    cylPeriodicSkipPair = true;
                }
            }

// Effective ghosting for dT (after min-image selection / desired shifts).
            isGhostA = (baseShiftA != 0);
            isGhostB = (baseShiftB != 0);

            // Signal kT to keep/prepare ghosts for the desired periodic image(s) next step.
            // Use shared cylindrical periodic flags:
            // START/END for desired side, MISMATCH when kT-provided image differs from dT desired image.
            if (simParams->useCylPeriodicDiagCounters && granData->ownerCylGhostActive) {
                const bool mismatch = (desiredShiftA != ghostShiftA || desiredShiftB != ghostShiftB);
                const bool mismatch_force_relevant = mismatch && (ContactType != deme::NOT_A_CONTACT);
                if (mismatch_force_relevant) {
                    if (ownerA != deme::NULL_BODYID) {
                        atomicOr(granData->ownerCylGhostActive + ownerA, deme::CYL_GHOST_HINT_MISMATCH);
                    }
                    if (ownerB != deme::NULL_BODYID) {
                        atomicOr(granData->ownerCylGhostActive + ownerB, deme::CYL_GHOST_HINT_MISMATCH);
                    }
                }
                if (desiredShiftA > 0 && ownerA != deme::NULL_BODYID) {
                    atomicOr(granData->ownerCylGhostActive + ownerA, deme::CYL_GHOST_HINT_START);
                } else if (desiredShiftA < 0 && ownerA != deme::NULL_BODYID) {
                    atomicOr(granData->ownerCylGhostActive + ownerA, deme::CYL_GHOST_HINT_END);
                }
                if (desiredShiftB > 0 && ownerB != deme::NULL_BODYID) {
                    atomicOr(granData->ownerCylGhostActive + ownerB, deme::CYL_GHOST_HINT_START);
                } else if (desiredShiftB < 0 && ownerB != deme::NULL_BODYID) {
                    atomicOr(granData->ownerCylGhostActive + ownerB, deme::CYL_GHOST_HINT_END);
                }
            }

            // Keep IDs consistent with the selected periodic representation (important when this primitive contact
            // originated from an un-ghosted pair but dT selects a wrapped image).
            if (ContactType != deme::NOT_A_CONTACT) {
                if constexpr (AType == deme::GEO_T_SPHERE || AType == deme::GEO_T_TRIANGLE) {                }
                if constexpr (BType == deme::GEO_T_SPHERE || BType == deme::GEO_T_TRIANGLE) {                }
            }

            const bool kTGhostA = (ghostShiftA != 0);
            const bool kTGhostB = (ghostShiftB != 0);
            wrapA = (wrapShiftA != 0);
            wrapB = (wrapShiftB != 0);
            wrapA_neg = wrapShiftA < 0;
            wrapB_neg = wrapShiftB < 0;
            // Ghost-ghost pairs are never used in cylindrical wedge periodicity.
            if (kTGhostA && kTGhostB) {
                discardGhostGhost = true;
                ContactType = deme::NOT_A_CONTACT;
            }
            float cosA = 1.f, sinA = 0.f, cosHalfA = 1.f, sinHalfA = 0.f;
            float cosB = 1.f, sinB = 0.f, cosHalfB = 1.f, sinHalfB = 0.f;
            if (wrapA) {
                cylPeriodicShiftTrig(wrapShiftA, simParams, cosA, sinA, cosHalfA, sinHalfA);
                wrapA_cos = cosA;
                wrapA_sin = sinA;
            }
            if (wrapB) {
                cylPeriodicShiftTrig(wrapShiftB, simParams, cosB, sinB, cosHalfB, sinHalfB);
                wrapB_cos = cosB;
                wrapB_sin = sinB;
            }

            if (wrapA) {
                AOwnerPos = cylPeriodicRotatePosD(AOwnerPos, simParams, cosA, sinA);
                bodyAPos = cylPeriodicRotatePosD(bodyAPos, simParams, cosA, sinA);
                AOriQ = cylPeriodicRotateQuat(AOriQ, simParams, cosHalfA, sinHalfA);
#if _forceModelHasLinVel_
                ALinVel = cylPeriodicRotateVec(ALinVel, simParams, cosA, sinA);
#endif
#if _forceModelHasRotVel_
                ARotVel = cylPeriodicRotateVec(ARotVel, simParams, cosA, sinA);
#endif
                if constexpr (AType == deme::GEO_T_TRIANGLE) {
                    triANode1 = cylPeriodicRotatePosD(triANode1, simParams, cosA, sinA);
                    triANode2 = cylPeriodicRotatePosD(triANode2, simParams, cosA, sinA);
                    triANode3 = cylPeriodicRotatePosD(triANode3, simParams, cosA, sinA);
                    bodyAPos = triangleCentroid<double3>(triANode1, triANode2, triANode3);
                    triPatchPosA = to_float3(cylPeriodicRotatePosD(to_double3(triPatchPosA), simParams, cosA, sinA));
                }
            }
            if (wrapB) {
                BOwnerPos = cylPeriodicRotatePosD(BOwnerPos, simParams, cosB, sinB);
                bodyBPos = cylPeriodicRotatePosD(bodyBPos, simParams, cosB, sinB);
                BOriQ = cylPeriodicRotateQuat(BOriQ, simParams, cosHalfB, sinHalfB);
#if _forceModelHasLinVel_
                BLinVel = cylPeriodicRotateVec(BLinVel, simParams, cosB, sinB);
#endif
#if _forceModelHasRotVel_
                BRotVel = cylPeriodicRotateVec(BRotVel, simParams, cosB, sinB);
#endif
                triBNode1 = cylPeriodicRotatePosD(triBNode1, simParams, cosB, sinB);
                triBNode2 = cylPeriodicRotatePosD(triBNode2, simParams, cosB, sinB);
                triBNode3 = cylPeriodicRotatePosD(triBNode3, simParams, cosB, sinB);
                bodyBPos = triangleCentroid<double3>(triBNode1, triBNode2, triBNode3);
                triPatchPosB = to_float3(cylPeriodicRotatePosD(to_double3(triPatchPosB), simParams, cosB, sinB));
            }
        }
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

            // We require that in the tri--tri case, the contact also respects the patch--patch general direction. This
            // is because if the contact margin is very large, the algorithm can detect remote fake `submerge' cases
            // which involve the triangles of the wrong sides of the mesh particles. But in this case, the direction of
            // this contact is almost always opposite to the general direction of the 2 patches (in terms of B2A).
            float dotProd = dot(B2A, triPatchPosA - triPatchPosB);
            granData->contactPatchDirectionRespected[myPrimitiveContactID] = (dotProd > 0.f) ? 1 : 0;

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
        ownerB = myOwner;
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

        AOwnerPos_orig = AOwnerPos;
        BOwnerPos_orig = BOwnerPos;
        AOriQ_orig = AOriQ;
        BOriQ_orig = BOriQ;
        if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f) {
            bool ghostA = false;
            bool ghostB = false;
            bool ghostA_neg = false;
            bool ghostB_neg = false;
            if constexpr (AType == deme::GEO_T_SPHERE || AType == deme::GEO_T_TRIANGLE) {
                cylPeriodicDecodeID(idA_raw, ghostA, ghostA_neg);
            }
            if constexpr (BType == deme::GEO_T_SPHERE || BType == deme::GEO_T_TRIANGLE) {
                cylPeriodicDecodeID(idB_raw, ghostB, ghostB_neg);
            }


            ghostShiftA = ghostA ? (ghostA_neg ? -1 : 1) : 0;
            ghostShiftB = ghostB ? (ghostB_neg ? -1 : 1) : 0;
            wrapShiftA = ghostShiftA;
            wrapShiftB = ghostShiftB;
            if (granData->ownerCylWrapOffset) {
                if (ownerA != deme::NULL_BODYID && ownerA < simParams->nOwnerBodies) {
                    wrapShiftA += granData->ownerCylWrapOffset[ownerA];
                }
                if (ownerB != deme::NULL_BODYID && ownerB < simParams->nOwnerBodies) {
                    wrapShiftB += granData->ownerCylWrapOffset[ownerB];
                }
            }
            cylPeriodicSkipPair = false;
            desiredShiftA = 0;
            desiredShiftB = 0;
            if (ghostShiftA > 1 || ghostShiftA < -1 || ghostShiftB > 1 || ghostShiftB < -1) {
                ghostShiftA = 0;
                ghostShiftB = 0;
                cylPeriodicSkipPair = true;
                discardGhostGhost = true;
                ContactType = deme::NOT_A_CONTACT;
            }
            {
                // For periodic-vs-analytical contacts, pick one deterministic image branch for the periodic owner.
                // Use owner COM positions here so one owner keeps one branch across all its components.
                float3 ownerA_sel = to_float3(AOwnerPos);
                float3 ownerB_sel = to_float3(BOwnerPos);
                bool availA_pos = true;
                bool availA_neg = true;
                desiredRadAOwner = 0.f;
                desiredRadBOwner = 0.f;
                if (granData->ownerBoundRadius && ownerA != deme::NULL_BODYID && ownerA < simParams->nOwnerBodies) {
                    desiredRadAOwner = granData->ownerBoundRadius[ownerA];
                    cylPeriodicGhostAvailability(ownerA_sel, desiredRadAOwner, simParams, availA_pos, availA_neg);
                }
                float bestDist = 0.f;
                cylPeriodicSelectGhostShiftByDistAvail(ownerA_sel, ownerB_sel, simParams, availA_pos, availA_neg, false,
                                                       false, desiredShiftA, desiredShiftB, bestDist);
                desiredBestDist2 = bestDist;
            }
            // Effective ghosting for this candidate (encoded by kT).
            int baseShiftA = ghostShiftA;
            int baseShiftB = ghostShiftB;
            isGhostA = (baseShiftA != 0);
            isGhostB = (baseShiftB != 0);
            if (ghostShiftA != 0 && ghostShiftB != 0) {
                ContactType = deme::NOT_A_CONTACT;
                discardGhostGhost = true;
            } else {
                if (desiredShiftA != ghostShiftA || desiredShiftB != ghostShiftB) {
                    cylPeriodicSkipPair = true;
                }
            }

            // Signal kT to keep/prepare ghosts for the desired periodic image(s) next step.
            // Use shared cylindrical periodic flags:
            // START/END for desired side, MISMATCH when kT-provided image differs from dT desired image.
            if (simParams->useCylPeriodicDiagCounters && granData->ownerCylGhostActive) {
                const bool mismatch = (desiredShiftA != ghostShiftA || desiredShiftB != ghostShiftB);
                const bool mismatch_force_relevant = mismatch && (ContactType != deme::NOT_A_CONTACT);
                if (mismatch_force_relevant) {
                    if (ownerA != deme::NULL_BODYID) {
                        atomicOr(granData->ownerCylGhostActive + ownerA, deme::CYL_GHOST_HINT_MISMATCH);
                    }
                    if (ownerB != deme::NULL_BODYID) {
                        atomicOr(granData->ownerCylGhostActive + ownerB, deme::CYL_GHOST_HINT_MISMATCH);
                    }
                }
                if (desiredShiftA > 0 && ownerA != deme::NULL_BODYID) {
                    atomicOr(granData->ownerCylGhostActive + ownerA, deme::CYL_GHOST_HINT_START);
                } else if (desiredShiftA < 0 && ownerA != deme::NULL_BODYID) {
                    atomicOr(granData->ownerCylGhostActive + ownerA, deme::CYL_GHOST_HINT_END);
                }
                if (desiredShiftB > 0 && ownerB != deme::NULL_BODYID) {
                    atomicOr(granData->ownerCylGhostActive + ownerB, deme::CYL_GHOST_HINT_START);
                } else if (desiredShiftB < 0 && ownerB != deme::NULL_BODYID) {
                    atomicOr(granData->ownerCylGhostActive + ownerB, deme::CYL_GHOST_HINT_END);
                }
            }
            const bool kTGhostA = (ghostShiftA != 0);
            const bool kTGhostB = (ghostShiftB != 0);
            wrapA = (wrapShiftA != 0);
            wrapB = (wrapShiftB != 0);
            wrapA_neg = wrapShiftA < 0;
            wrapB_neg = wrapShiftB < 0;
            // Ghost-ghost pairs are never used in cylindrical wedge periodicity.
            if (kTGhostA && kTGhostB) {
                discardGhostGhost = true;
                ContactType = deme::NOT_A_CONTACT;
            }
            float cosA = 1.f, sinA = 0.f, cosHalfA = 1.f, sinHalfA = 0.f;
            float cosB = 1.f, sinB = 0.f, cosHalfB = 1.f, sinHalfB = 0.f;
            if (wrapA) {
                cylPeriodicShiftTrig(wrapShiftA, simParams, cosA, sinA, cosHalfA, sinHalfA);
                wrapA_cos = cosA;
                wrapA_sin = sinA;
            }
            if (wrapB) {
                cylPeriodicShiftTrig(wrapShiftB, simParams, cosB, sinB, cosHalfB, sinHalfB);
                wrapB_cos = cosB;
                wrapB_sin = sinB;
            }
            if (wrapA) {
                AOwnerPos = cylPeriodicRotatePosD(AOwnerPos, simParams, cosA, sinA);
                bodyAPos = cylPeriodicRotatePosD(bodyAPos, simParams, cosA, sinA);
                AOriQ = cylPeriodicRotateQuat(AOriQ, simParams, cosHalfA, sinHalfA);
#if _forceModelHasLinVel_
                ALinVel = cylPeriodicRotateVec(ALinVel, simParams, cosA, sinA);
#endif
#if _forceModelHasRotVel_
                ARotVel = cylPeriodicRotateVec(ARotVel, simParams, cosA, sinA);
#endif
                if constexpr (AType == deme::GEO_T_TRIANGLE) {
                    triANode1 = cylPeriodicRotatePosD(triANode1, simParams, cosA, sinA);
                    triANode2 = cylPeriodicRotatePosD(triANode2, simParams, cosA, sinA);
                    triANode3 = cylPeriodicRotatePosD(triANode3, simParams, cosA, sinA);
                    bodyAPos = triangleCentroid<double3>(triANode1, triANode2, triANode3);
                }
            }
            if (wrapB) {
                BOwnerPos = cylPeriodicRotatePosD(BOwnerPos, simParams, cosB, sinB);
                bodyBPos = cylPeriodicRotatePosD(bodyBPos, simParams, cosB, sinB);
                BOriQ = cylPeriodicRotateQuat(BOriQ, simParams, cosHalfB, sinHalfB);
#if _forceModelHasLinVel_
                BLinVel = cylPeriodicRotateVec(BLinVel, simParams, cosB, sinB);
#endif
#if _forceModelHasRotVel_
                BRotVel = cylPeriodicRotateVec(BRotVel, simParams, cosB, sinB);
#endif
            }
        }
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
                                                             contactPnt, B2A, overlapDepth);
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

        // Cylindrical periodic: store contact-history vectors (e.g., delta_tan) in the base-wedge frame.
        // For the active periodic image in this step, we rotate history into the active frame before the force model,
        // and rotate it back to base-wedge before writing back to global memory.
        int cylHistShift = 0;
        if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f) {
            cylHistShift = (wrapShiftA != 0) ? wrapShiftA : ((wrapShiftB != 0) ? wrapShiftB : 0);
        }


        // Essentials for storing and calculating contact info
        float3 force = make_float3(0, 0, 0);
        float3 torque_only_force = make_float3(0, 0, 0);
        // Local position of the contact point is always a piece of info we require... regardless of force model
        float3 locCPA = to_float3(contactPnt - AOwnerPos);
        float3 locCPB = to_float3(contactPnt - BOwnerPos);
        // Now map this contact point location to bodies' local ref
        applyOriQToVector3<float, deme::oriQ_t>(locCPA.x, locCPA.y, locCPA.z, AOriQ.w, -AOriQ.x, -AOriQ.y, -AOriQ.z);
        applyOriQToVector3<float, deme::oriQ_t>(locCPB.x, locCPB.y, locCPB.z, BOriQ.w, -BOriQ.x, -BOriQ.y, -BOriQ.z);

        // Decide whether this candidate contact is active for this dT step.
        // We may have multiple periodic image candidates for the same owner pair; if this one is not the current
        // minimum-image (cylPeriodicSkipPair), it must contribute neither force nor history update in this step.
        const deme::contact_t ContactType_candidate = ContactType;
        bool ownerBoundReject = false;
        if constexpr (BType != deme::GEO_T_ANALYTICAL) {
            if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f &&
            ContactType_candidate != deme::NOT_A_CONTACT && granData->ownerBoundRadius &&
            ownerA != deme::NULL_BODYID && ownerB != deme::NULL_BODYID && ownerA < simParams->nOwnerBodies &&
            ownerB < simParams->nOwnerBodies) {
                const float radA_owner = fmaxf(granData->ownerBoundRadius[ownerA], 0.f);
                const float radB_owner = fmaxf(granData->ownerBoundRadius[ownerB], 0.f);
                if (radA_owner > 0.f && radB_owner > 0.f) {
                    const float max_other_margin = simParams->dyn.beta + simParams->maxFamilyExtraMargin + extraMarginSize;
                    const float reach = radA_owner + radB_owner + fmaxf(max_other_margin, 0.f) + 1e-6f;
                    const float3 ownerA_pos = to_float3(AOwnerPos);
                    const float3 ownerB_pos = to_float3(BOwnerPos);
                    const float owner_d2 = cylPeriodicDist2(ownerA_pos, ownerB_pos);
                    ownerBoundReject = owner_d2 > reach * reach;
                }
            }
        }
        if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f && cylPeriodicSkipPair && !discardGhostGhost) {
            if (simParams->useCylPeriodicDiagCounters && granData->ownerCylSkipCount) {
                if (ownerA != deme::NULL_BODYID) {
                    atomicAdd(granData->ownerCylSkipCount + ownerA, 1u);
                }
                if (ownerB != deme::NULL_BODYID) {
                    atomicAdd(granData->ownerCylSkipCount + ownerB, 1u);
                }
            }
            if (simParams->useCylPeriodicDiagCounters && ContactType_candidate != deme::NOT_A_CONTACT &&
                granData->ownerCylSkipPotentialCount) {
                if (ownerA != deme::NULL_BODYID) {
                    atomicAdd(granData->ownerCylSkipPotentialCount + ownerA, 1u);
                }
                if (ownerB != deme::NULL_BODYID) {
                    atomicAdd(granData->ownerCylSkipPotentialCount + ownerB, 1u);
                }
            }
            if (ContactType_candidate != deme::NOT_A_CONTACT && granData->ownerCylSkipPotentialTotal) {
                atomicAdd(granData->ownerCylSkipPotentialTotal, 1u);
            }
        }
        if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f && ownerBoundReject && !discardGhostGhost) {
            if (simParams->useCylPeriodicDiagCounters && granData->ownerCylSkipCount) {
                if (ownerA != deme::NULL_BODYID) {
                    atomicAdd(granData->ownerCylSkipCount + ownerA, 1u);
                }
                if (ownerB != deme::NULL_BODYID) {
                    atomicAdd(granData->ownerCylSkipCount + ownerB, 1u);
                }
            }
            if (simParams->useCylPeriodicDiagCounters && ContactType_candidate != deme::NOT_A_CONTACT &&
                granData->ownerCylSkipPotentialCount) {
                if (ownerA != deme::NULL_BODYID) {
                    atomicAdd(granData->ownerCylSkipPotentialCount + ownerA, 1u);
                }
                if (ownerB != deme::NULL_BODYID) {
                    atomicAdd(granData->ownerCylSkipPotentialCount + ownerB, 1u);
                }
            }
            if (ContactType_candidate != deme::NOT_A_CONTACT && granData->ownerCylSkipPotentialTotal) {
                atomicAdd(granData->ownerCylSkipPotentialTotal, 1u);
            }
        }
        const bool activeForThisStep =
            (ContactType_candidate != deme::NOT_A_CONTACT) && (!discardGhostGhost) && (!cylPeriodicSkipPair) &&
            (!ownerBoundReject);

        // Rotate history (base-wedge -> active image) only when this contact is active.
        if (activeForThisStep && cylHistShift != 0) {
            const float sin_fwd = (cylHistShift < 0) ? -simParams->cylPeriodicSinSpan : simParams->cylPeriodicSinSpan;
            float3 dt = make_float3(delta_tan_x, delta_tan_y, delta_tan_z);
            dt = cylPeriodicRotateVec(dt, simParams, sin_fwd);
            delta_tan_x = dt.x;
            delta_tan_y = dt.y;
            delta_tan_z = dt.z;
        }

        // Force model execution (or skip for inactive periodic candidate)
        if (activeForThisStep) {
            // The following part, the force model, is user-specifiable
            // NOTE!! "force" and all wildcards must be properly set by this piece of code
            { _DEMForceModel_; }

            // If force model modifies owner wildcards, write them back here
            _forceModelOwnerWildcardWrite_;
        } else {
            // No contribution from this candidate in this step.
            force = make_float3(0.f, 0.f, 0.f);
            torque_only_force = make_float3(0.f, 0.f, 0.f);
            // For seam-branch mismatches, keep history (freeze) to avoid artificial de-sticking.
            // But if this candidate is geometrically invalid by owner bound rejection, destroy history.
            if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f &&
                ownerBoundReject && ContactType_candidate != deme::NOT_A_CONTACT && !discardGhostGhost) {
                _forceModelContactWildcardDestroy_;
            }
            // True non-contacts still destroy history.
            if (ContactType_candidate == deme::NOT_A_CONTACT || discardGhostGhost) {
                _forceModelContactWildcardDestroy_;
            }
        }

        // For output/contact-point bookkeeping, treat skipped periodic candidates as non-contacts.
        if (!activeForThisStep) {
            ContactType = deme::NOT_A_CONTACT;
        }

        // Note in DEME3, we do not clear force array anymore in each timestep, so always writing back force and contact
        // points, even for zero-force non-contacts, is needed (unless of course, the user instructed no force record).
        // This design has implications in our new two-step patch-based force calculation algorithm, as we re-use some
        // force-storing arrays for intermediate values.

        if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f && (wrapA || wrapB)) {
            double3 contactPntA = contactPnt;
            double3 contactPntB = contactPnt;
            if (wrapA) {
                contactPntA = cylPeriodicRotatePosD(contactPntA, simParams, wrapA_cos, -wrapA_sin);
            }
            if (wrapB) {
                contactPntB = cylPeriodicRotatePosD(contactPntB, simParams, wrapB_cos, -wrapB_sin);
            }
            AOwnerPos = AOwnerPos_orig;
            BOwnerPos = BOwnerPos_orig;
            AOriQ = AOriQ_orig;
            BOriQ = BOriQ_orig;
            locCPA = to_float3(contactPntA - AOwnerPos);
            locCPB = to_float3(contactPntB - BOwnerPos);
            applyOriQToVector3<float, deme::oriQ_t>(locCPA.x, locCPA.y, locCPA.z, AOriQ.w, -AOriQ.x, -AOriQ.y,
                                                    -AOriQ.z);
            applyOriQToVector3<float, deme::oriQ_t>(locCPB.x, locCPB.y, locCPB.z, BOriQ.w, -BOriQ.x, -BOriQ.y,
                                                    -BOriQ.z);
        }

        if (ContactType == deme::NOT_A_CONTACT) {
            locCPA = make_float3(0.f, 0.f, 0.f);
            locCPB = make_float3(0.f, 0.f, 0.f);
        }
        // Write contact location values back to global memory (after periodic wrap correction).
        _contactInfoWrite_;

        // Optionally, the forces can be reduced to acc right here (may be faster)
        _forceCollectInPlaceStrat_;

        // Updated contact wildcards need to be write back to global mem. It is here because contact wildcard may need
        // to be destroyed for non-contact, so it has to go last.
        // Rotate history back to base-wedge frame before writing to global memory.
        if (activeForThisStep && cylHistShift != 0) {
            const float sin_inv = (cylHistShift < 0) ? simParams->cylPeriodicSinSpan : -simParams->cylPeriodicSinSpan;
            float3 dt = make_float3(delta_tan_x, delta_tan_y, delta_tan_z);
            dt = cylPeriodicRotateVec(dt, simParams, sin_inv);
            delta_tan_x = dt.x;
            delta_tan_y = dt.y;
            delta_tan_z = dt.z;
        }

        _forceModelContactWildcardWrite_;
    } else {  // If this is the kernel for a mesh-related contact, another follow-up kernel is needed to compute force
        // Cylindrical periodic: if this primitive belongs to a non-minimum-image pair, it must not
        // contribute to patch aggregation (normals/penetration/area/contact point).
        if (cylPeriodicSkipPair) {
            if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f && !discardGhostGhost) {
                if (simParams->useCylPeriodicDiagCounters && granData->ownerCylSkipCount) {
                    if (ownerA != deme::NULL_BODYID) {
                        atomicAdd(granData->ownerCylSkipCount + ownerA, 1u);
                    }
                    if (ownerB != deme::NULL_BODYID) {
                        atomicAdd(granData->ownerCylSkipCount + ownerB, 1u);
                    }
                }
                if (simParams->useCylPeriodicDiagCounters && ContactType != deme::NOT_A_CONTACT &&
                    granData->ownerCylSkipPotentialCount) {
                    if (ownerA != deme::NULL_BODYID) {
                        atomicAdd(granData->ownerCylSkipPotentialCount + ownerA, 1u);
                    }
                    if (ownerB != deme::NULL_BODYID) {
                        atomicAdd(granData->ownerCylSkipPotentialCount + ownerB, 1u);
                    }
                }
                if (ContactType != deme::NOT_A_CONTACT && granData->ownerCylSkipPotentialTotal) {
                    atomicAdd(granData->ownerCylSkipPotentialTotal, 1u);
                }
            }
            ContactType = deme::NOT_A_CONTACT;
            overlapDepth = -1.0;
            overlapArea = 0.0;
            granData->contactPatchDirectionRespected[myPrimitiveContactID] = 0;
        }
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
DEME_KERNEL void calculatePrimitiveContactForces_SphSph(deme::DEMSimParams* simParams,
                                                       deme::DEMDataDT* granData,
                                                       deme::contactPairs_t startOffset,
                                                       deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPrimitiveContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPrimitiveContactID < startOffset + nContactPairs) {
        calculatePrimitiveContactForces_impl<deme::SPHERE_SPHERE_CONTACT>(simParams, granData, myPrimitiveContactID);
    }
}

DEME_KERNEL void calculatePrimitiveContactForces_SphTri(deme::DEMSimParams* simParams,
                                                       deme::DEMDataDT* granData,
                                                       deme::contactPairs_t startOffset,
                                                       deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPrimitiveContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPrimitiveContactID < startOffset + nContactPairs) {
        calculatePrimitiveContactForces_impl<deme::SPHERE_TRIANGLE_CONTACT>(simParams, granData, myPrimitiveContactID);
    }
}

DEME_KERNEL void calculatePrimitiveContactForces_SphAnal(deme::DEMSimParams* simParams,
                                                        deme::DEMDataDT* granData,
                                                        deme::contactPairs_t startOffset,
                                                        deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPrimitiveContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPrimitiveContactID < startOffset + nContactPairs) {
        calculatePrimitiveContactForces_impl<deme::SPHERE_ANALYTICAL_CONTACT>(simParams, granData,
                                                                              myPrimitiveContactID);
    }
}

DEME_KERNEL void calculatePrimitiveContactForces_TriTri(deme::DEMSimParams* simParams,
                                                       deme::DEMDataDT* granData,
                                                       deme::contactPairs_t startOffset,
                                                       deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPrimitiveContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPrimitiveContactID < startOffset + nContactPairs) {
        calculatePrimitiveContactForces_impl<deme::TRIANGLE_TRIANGLE_CONTACT>(simParams, granData,
                                                                              myPrimitiveContactID);
    }
}

DEME_KERNEL void calculatePrimitiveContactForces_TriAnal(deme::DEMSimParams* simParams,
                                                        deme::DEMDataDT* granData,
                                                        deme::contactPairs_t startOffset,
                                                        deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPrimitiveContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPrimitiveContactID < startOffset + nContactPairs) {
        calculatePrimitiveContactForces_impl<deme::TRIANGLE_ANALYTICAL_CONTACT>(simParams, granData,
                                                                                myPrimitiveContactID);
    }
}
