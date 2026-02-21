// DEM patch-based force computation related custom kernels
#include <DEM/Defines.h>
#include <DEMCollisionKernels_SphSph.cuh>
#include <DEMCollisionKernels_SphTri_TriTri.cuh>
_kernelIncludes_;

inline __device__ float3 cylPeriodicRotatePosF(const float3& pos, const deme::DEMSimParams* simParams, float sin_span) {
    float3 pos_local = make_float3(pos.x - simParams->LBFX, pos.y - simParams->LBFY, pos.z - simParams->LBFZ);
    pos_local =
        cylPeriodicRotate(pos_local, simParams->cylPeriodicOrigin, simParams->cylPeriodicAxisVec,
                          simParams->cylPeriodicU, simParams->cylPeriodicV, simParams->cylPeriodicCosSpan, sin_span);
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

inline __device__ double3 cylPeriodicRotatePosD(const double3& pos,
                                                const deme::DEMSimParams* simParams,
                                                float sin_span) {
    float3 p = make_float3((float)pos.x, (float)pos.y, (float)pos.z);
    p = cylPeriodicRotatePosF(p, simParams, sin_span);
    return make_double3((double)p.x, (double)p.y, (double)p.z);
}

inline __device__ double3 cylPeriodicRotatePosD(const double3& pos,
                                                const deme::DEMSimParams* simParams,
                                                float cos_theta,
                                                float sin_theta) {
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

inline __device__ float3 cylPeriodicRotateVec(const float3& vec,
                                              const deme::DEMSimParams* simParams,
                                              float cos_theta,
                                              float sin_theta) {
    return cylPeriodicRotate(vec, make_float3(0.f, 0.f, 0.f), simParams->cylPeriodicAxisVec, simParams->cylPeriodicU,
                             simParams->cylPeriodicV, cos_theta, sin_theta);
}

inline __device__ float3 cylPeriodicRotateVec(const float3& vec, const deme::DEMSimParams* simParams) {
    return cylPeriodicRotateVec(vec, simParams, simParams->cylPeriodicSinSpan);
}

inline __device__ float4 cylPeriodicRotateQuat(const float4& q,
                                               const deme::DEMSimParams* simParams,
                                               float sin_half_span) {
    float4 rotQ =
        make_float4(simParams->cylPeriodicAxisVec.x * sin_half_span, simParams->cylPeriodicAxisVec.y * sin_half_span,
                    simParams->cylPeriodicAxisVec.z * sin_half_span, simParams->cylPeriodicCosHalfSpan);
    float4 out = q;
    HamiltonProduct(out.w, out.x, out.y, out.z, rotQ.w, rotQ.x, rotQ.y, rotQ.z, out.w, out.x, out.y, out.z);
    out /= length(out);
    return out;
}

inline __device__ float4 cylPeriodicRotateQuat(const float4& q,
                                               const deme::DEMSimParams* simParams,
                                               float cos_half,
                                               float sin_half) {
    float4 rotQ = make_float4(simParams->cylPeriodicAxisVec.x * sin_half, simParams->cylPeriodicAxisVec.y * sin_half,
                              simParams->cylPeriodicAxisVec.z * sin_half, cos_half);
    float4 out = q;
    HamiltonProduct(out.w, out.x, out.y, out.z, rotQ.w, rotQ.x, rotQ.y, rotQ.z, out.w, out.x, out.y, out.z);
    out /= length(out);
    return out;
}

inline __device__ float4 cylPeriodicRotateQuat(const float4& q, const deme::DEMSimParams* simParams) {
    return cylPeriodicRotateQuat(q, simParams, simParams->cylPeriodicSinHalfSpan);
}

inline __device__ float cylPeriodicRelAngleGlobal(const float3& pos, const deme::DEMSimParams* simParams) {
    const float3 pos_local = make_float3(pos.x - simParams->LBFX, pos.y - simParams->LBFY, pos.z - simParams->LBFZ);
    return cylPeriodicRelAngle(pos_local, false, false, simParams);
}

inline __device__ float maxOwnerLocalLever(const deme::DEMSimParams* simParams,
                                           const deme::DEMDataDT* granData,
                                           deme::bodyID_t owner,
                                           float extraMarginSize) {
    // If we do not have a valid per-owner bound radius, do not clamp contact lever arm.
    float max_local_lever = DEME_HUGE_FLOAT;
    if (granData->ownerBoundRadius && owner != deme::NULL_BODYID && owner < simParams->nOwnerBodies) {
        const float bound_r = fmaxf(granData->ownerBoundRadius[owner], 0.f);
        if (isfinite(bound_r) && bound_r > DEME_TINY_FLOAT) {
            const float geom_tol =
                fmaxf(simParams->dyn.beta + simParams->maxFamilyExtraMargin + extraMarginSize, 0.f) + 1e-4f;
            max_local_lever = fmaxf(bound_r + geom_tol, 1e-3f);
        }
    }
    return max_local_lever;
}

inline __device__ void clampLocalContactPoint(float3& p, float max_norm) {
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) {
        p = make_float3(0.f, 0.f, 0.f);
        return;
    }
    if (!isfinite(max_norm) || max_norm >= 0.5f * DEME_HUGE_FLOAT) {
        return;
    }
    max_norm = fmaxf(max_norm, 1e-6f);
    const float m2 = p.x * p.x + p.y * p.y + p.z * p.z;
    const float max2 = max_norm * max_norm;
    if (m2 > max2) {
        const float inv_m = rsqrtf(m2);
        p *= (max_norm * inv_m);
    }
}

// Clamp penetration by the overlap of owner bounding spheres.
// This is a hard geometric invariant: if owner bounds do not overlap, geometry cannot overlap.
inline __device__ bool clampPatchPenetrationByOwnerBounds(const deme::DEMSimParams* simParams,
                                                          const deme::DEMDataDT* granData,
                                                          deme::bodyID_t ownerA,
                                                          deme::bodyID_t ownerB,
                                                          const double3& AOwnerPos,
                                                          const double3& BOwnerPos,
                                                          const double3& contactPnt,
                                                          const float3& contactNormal,
                                                          float extraMarginSize,
                                                          double& overlapDepth,
                                                          bool apply_shape_depth_cap) {
    if (!granData->ownerBoundRadius || ownerA == deme::NULL_BODYID || ownerB == deme::NULL_BODYID ||
        ownerA >= simParams->nOwnerBodies || ownerB >= simParams->nOwnerBodies) {
        return true;
    }

    const float radA = fmaxf(granData->ownerBoundRadius[ownerA], 0.f);
    const float radB = fmaxf(granData->ownerBoundRadius[ownerB], 0.f);
    if (!isfinite(radA) || !isfinite(radB) || radA <= DEME_TINY_FLOAT || radB <= DEME_TINY_FLOAT) {
        return true;
    }

    const double3 d = AOwnerPos - BOwnerPos;
    const double dist2 = dot(d, d);
    if (!(dist2 >= 0.0) || !isfinite(dist2)) {
        return false;
    }
    const double dist = sqrt(dist2);
    const double maxOverlap = static_cast<double>(radA) + static_cast<double>(radB) - dist;
    if (maxOverlap <= 0.0) {
        return false;
    }
    double cappedMaxOverlap = maxOverlap;
    if (apply_shape_depth_cap) {
        // Tri-tri penetration rate limiter:
        // cap overlap by what relative normal approach can create within one dT step.
        // This avoids force spikes from occasional geometric outliers without hardcoded size ratios.
        float3 n = contactNormal;
        const float n2 = n.x * n.x + n.y * n.y + n.z * n.z;
        if (isfinite(n2) && n2 > DEME_TINY_FLOAT) {
            const float inv_n = rsqrtf(n2);
            n *= inv_n;

            const float3 vA = make_float3(granData->vX[ownerA], granData->vY[ownerA], granData->vZ[ownerA]);
            const float3 vB = make_float3(granData->vX[ownerB], granData->vY[ownerB], granData->vZ[ownerB]);
            const float3 wA =
                make_float3(granData->omgBarX[ownerA], granData->omgBarY[ownerA], granData->omgBarZ[ownerA]);
            const float3 wB =
                make_float3(granData->omgBarX[ownerB], granData->omgBarY[ownerB], granData->omgBarZ[ownerB]);

            float3 rA = to_float3(contactPnt - AOwnerPos);
            float3 rB = to_float3(contactPnt - BOwnerPos);
            const float geom_tol =
                fmaxf(simParams->dyn.beta + simParams->maxFamilyExtraMargin + extraMarginSize, 0.f) + 1e-4f;
            clampLocalContactPoint(rA, fmaxf(radA + geom_tol, 1e-6f));
            clampLocalContactPoint(rB, fmaxf(radB + geom_tol, 1e-6f));
            const float3 vA_cp = vA + cross(wA, rA);
            const float3 vB_cp = vB + cross(wB, rB);

            // n points from B -> A, so positive closing speed is dot(vB - vA, n).
            const float closing_speed = fmaxf(dot(vB_cp - vA_cp, n), 0.f);
            unsigned int drift_steps = 1u;
            if (granData->pKTOwnedBuffer_maxDrift && *granData->pKTOwnedBuffer_maxDrift > 0u) {
                drift_steps = *granData->pKTOwnedBuffer_maxDrift;
            }
            const double drift_factor = sqrt(static_cast<double>(drift_steps));
            const double motion_cap =
                static_cast<double>(closing_speed) * static_cast<double>(simParams->dyn.h) * drift_factor +
                static_cast<double>(fmaxf(0.5f * simParams->dyn.beta + extraMarginSize, 0.f)) + 1e-7;
            if (motion_cap > 0.0) {
                cappedMaxOverlap = fmin(cappedMaxOverlap, motion_cap);
            }
        }
    }
    if (cappedMaxOverlap <= 0.0) {
        return false;
    }
    if (overlapDepth > cappedMaxOverlap) {
        overlapDepth = cappedMaxOverlap;
    }
    return true;
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

// Template device function for patch-based contact force calculation
template <deme::contact_t CONTACT_TYPE>
__device__ __forceinline__ void calculatePatchContactForces_impl(deme::DEMSimParams* simParams,
                                                                 deme::DEMDataDT* granData,
                                                                 const double* finalAreas,
                                                                 const float3* finalNormals,
                                                                 const double* finalPenetrations,
                                                                 const double3* finalContactPoints,
                                                                 deme::contactPairs_t myPatchContactID,
                                                                 deme::contactPairs_t startOffsetPatch) {
    // Contact type is known at compile time
    deme::contact_t ContactType = CONTACT_TYPE;

    // Calculate relative index for accessing the temp arrays (finalAreas, finalNormals, finalPenetrations,
    // finalContactPoints)
    deme::contactPairs_t relativeIndex = myPatchContactID - startOffsetPatch;

    // The following quantities are provided from the patch voting process
    float3 B2A = finalNormals[relativeIndex];                // contact normal felt by A, pointing from B to A
    double overlapDepth = finalPenetrations[relativeIndex];  // penetration depth
    double overlapArea = finalAreas[relativeIndex];          // total contact area for this patch pair

    // Contact point is computed via weighted average (weight = penetration * area)
    double3 contactPnt = finalContactPoints[relativeIndex];
    double3 AOwnerPos, bodyAPos, BOwnerPos, bodyBPos;
    float AOwnerMass, ARadius, BOwnerMass, BRadius;
    float4 AOriQ, BOriQ;
    double3 AOwnerPos_orig = make_double3(0.0, 0.0, 0.0);
    double3 BOwnerPos_orig = make_double3(0.0, 0.0, 0.0);
    float4 AOriQ_orig = make_float4(0.f, 0.f, 0.f, 1.f);
    float4 BOriQ_orig = make_float4(0.f, 0.f, 0.f, 1.f);
    deme::materialsOffset_t bodyAMatType, bodyBMatType;
    // The user-specified extra margin size (how much we should be lenient in determining `in-contact')
    float extraMarginSize = 0.;

    // Then allocate the optional quantities that will be needed in the force model
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
    const deme::bodyID_t idA_raw = granData->idPatchA[myPatchContactID];
    const deme::bodyID_t idB_raw = granData->idPatchB[myPatchContactID];
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

    // Decompose ContactType to get the types of A and B (known at compile time)
    constexpr deme::geoType_t AType = (CONTACT_TYPE >> 4);
    constexpr deme::geoType_t BType = (CONTACT_TYPE & 0xF);

    // ----------------------------------------------------------------
    // Based on A's type, equip info
    // ----------------------------------------------------------------
    if constexpr (AType == deme::GEO_T_SPHERE) {
        // For sphere-mesh contacts, patch A is a sphere
        // Note: For spheres, the patch ID is the same as the sphere ID
        deme::bodyID_t sphereID = idA_raw & deme::CYL_PERIODIC_SPHERE_ID_MASK;
        deme::bodyID_t myOwner = granData->ownerClumpBody[sphereID];
        ownerA = myOwner;
        deme::bodyID_t myPatchID = sphereID;  // For spheres, patch ID == sphere ID

        float3 myRelPos;
        float myRadius;
        // Get my component offset info from either jitified arrays or global memory
        { _componentAcqStrat_; }

        // Get my mass info
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
        bodyAMatType = granData->sphereMaterialOffset[myPatchID];
        extraMarginSize = granData->familyExtraMarginSize[AOwnerFamily];
    } else if constexpr (AType == deme::GEO_T_TRIANGLE) {
        // For mesh-mesh or mesh-analytical contacts, patch A is a mesh patch
        deme::bodyID_t myPatchID = idA_raw & deme::CYL_PERIODIC_SPHERE_ID_MASK;
        deme::bodyID_t myOwner = granData->ownerPatchMesh[myPatchID];
        ownerA = myOwner;
        ARadius = DEME_HUGE_FLOAT;
        bodyAMatType = granData->patchMaterialOffset[myPatchID];

        float3 myRelPos = granData->relPosPatch[myPatchID];

        // Get my mass info
        {
            float myMass;
            _massAcqStrat_;
            AOwnerMass = myMass;
        }
        _forceModelIngredientAcqForA_;
        _forceModelGeoWildcardAcqForAMeshPatch_;

        // In mesh case, bodyAPos is the patch position (not necessarily needed in the force model though)
        equipOwnerPosRot(simParams, granData, myOwner, myRelPos, AOwnerPos, bodyAPos, AOriQ);
        extraMarginSize = granData->familyExtraMarginSize[AOwnerFamily];
    } else {
        // Unsupported type
        ContactType = deme::NOT_A_CONTACT;
    }

    // ----------------------------------------------------------------
    // Then B, location and velocity, depending on type
    // ----------------------------------------------------------------
    if constexpr (BType == deme::GEO_T_TRIANGLE) {
        // For mesh-related contacts, patch B is a mesh patch
        deme::bodyID_t myPatchID = idB_raw & deme::CYL_PERIODIC_SPHERE_ID_MASK;
        deme::bodyID_t myOwner = granData->ownerPatchMesh[myPatchID];
        ownerB = myOwner;
        BRadius = DEME_HUGE_FLOAT;
        bodyBMatType = granData->patchMaterialOffset[myPatchID];

        float3 myRelPos = granData->relPosPatch[myPatchID];

        // Get my mass info
        {
            float myMass;
            _massAcqStrat_;
            BOwnerMass = myMass;
        }
        _forceModelIngredientAcqForB_;
        _forceModelGeoWildcardAcqForBMeshPatch_;

        // In mesh case, bodyBPos is the patch position (not necessarily needed in the force model though)
        equipOwnerPosRot(simParams, granData, myOwner, myRelPos, BOwnerPos, bodyBPos, BOriQ);

        // As the grace margin, the distance (negative overlap) just needs to be within the grace margin. So we pick
        // the larger of the 2 familyExtraMarginSize.
        extraMarginSize = (extraMarginSize > granData->familyExtraMarginSize[BOwnerFamily])
                              ? extraMarginSize
                              : granData->familyExtraMarginSize[BOwnerFamily];
        if (overlapDepth <= -extraMarginSize) {
            ContactType = deme::NOT_A_CONTACT;
        }

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
            // dT selects the active minimum-image periodic representation based on current configuration.
            // IMPORTANT: kT is allowed to emit multiple ghost-involved image pairs. dT recomputes which image is
            // the current minimum-image and keeps only the matching representation for this step.
            int baseShiftA = ghostShiftA;
            int baseShiftB = ghostShiftB;

            // Ghost-ghost pairs are never used in cylindrical wedge periodicity (avoid non-minimal ±2 images and
            // duplicates).
            if (ghostShiftA != 0 && ghostShiftB != 0) {
                ContactType = deme::NOT_A_CONTACT;
            } else {
                // Keep only the periodic image pair that matches the desired (minimum-image) shifts for this step.
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

            // Ensure IDs are consistent with the selected periodic representation so force accumulation and history
            // mapping stay coherent.
            if (ContactType != deme::NOT_A_CONTACT) {
                if constexpr (AType == deme::GEO_T_SPHERE || AType == deme::GEO_T_TRIANGLE) {
                }
                if constexpr (BType == deme::GEO_T_SPHERE || BType == deme::GEO_T_TRIANGLE) {
                }
            }

            // wrapShiftA/B use only the kT ghost flag for geometry/force evaluation.

            const bool kTGhostA = (ghostShiftA != 0);
            const bool kTGhostB = (ghostShiftB != 0);
            wrapA = (wrapShiftA != 0);
            wrapB = (wrapShiftB != 0);
            wrapA_neg = wrapShiftA < 0;
            wrapB_neg = wrapShiftB < 0;

            // Redundant safety: discard if not a contact (prevents using stale data paths).
            if (ContactType == deme::NOT_A_CONTACT) {
                discardGhostGhost = true;
            }
            // Ghost-ghost is always discarded (redundant with ContactType above).
            if (kTGhostA && kTGhostB) {
                discardGhostGhost = true;
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
    } else if constexpr (BType == deme::GEO_T_ANALYTICAL) {
        // For mesh-analytical contacts, patch B is an analytical entity
        deme::objID_t analyticalID = granData->idPatchB[myPatchContactID];
        deme::bodyID_t myOwner = objOwner[analyticalID];
        ownerB = myOwner;
        deme::bodyID_t myPatchID = analyticalID;

        bodyBMatType = objMaterial[analyticalID];
        BOwnerMass = objMass[analyticalID];
        BRadius = DEME_HUGE_FLOAT;
        float3 myRelPos;
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

            {
                // For periodic-vs-analytical contacts, choose one deterministic image branch of the periodic owner.
                // This keeps one canonical representation while preventing accidental contact dropout near the seam.
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

            const int baseShiftA = ghostShiftA;
            const int baseShiftB = ghostShiftB;

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

            // Keep IDs consistent (unghosted).
            if (ContactType != deme::NOT_A_CONTACT) {
                if constexpr (AType == deme::GEO_T_SPHERE || AType == deme::GEO_T_TRIANGLE) {
                }
            }

            wrapA = (wrapShiftA != 0);
            wrapB = (wrapShiftB != 0);
            wrapA_neg = wrapShiftA < 0;
            wrapB_neg = wrapShiftB < 0;

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
        if (overlapDepth <= -extraMarginSize) {
            ContactType = deme::NOT_A_CONTACT;
        }
    }

    // Patch-level geometric invariant guard for real owner-owner contacts.
    // Prevent rare depth outliers from injecting unphysical force/energy.
    if constexpr (BType != deme::GEO_T_ANALYTICAL) {
        constexpr bool apply_shape_depth_cap = (AType == deme::GEO_T_TRIANGLE) && (BType == deme::GEO_T_TRIANGLE);
        if (ContactType != deme::NOT_A_CONTACT &&
            !clampPatchPenetrationByOwnerBounds(simParams, granData, ownerA, ownerB, AOwnerPos, BOwnerPos, contactPnt,
                                                B2A, extraMarginSize, overlapDepth, apply_shape_depth_cap)) {
            ContactType = deme::NOT_A_CONTACT;
            overlapDepth = -DEME_HUGE_FLOAT;
            overlapArea = 0.0;
        }
    }

    // Now compute forces using the patch-based contact data
    _forceModelContactWildcardAcq_;

    // Cylindrical periodic: keep contact history (e.g., delta_tan) frame-invariant in the base wedge.
    // We rotate history into the active image frame for this step, and rotate it back before storing.
    int cylHistShift = 0;
    if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f) {
        cylHistShift = (wrapShiftA != 0) ? wrapShiftA : ((wrapShiftB != 0) ? wrapShiftB : 0);
    }

    // Essentials for storing and calculating contact info
    float3 force = make_float3(0, 0, 0);
    float3 torque_only_force = make_float3(0, 0, 0);
    // Local position of the contact point
    float3 locCPA = to_float3(contactPnt - AOwnerPos);
    float3 locCPB = to_float3(contactPnt - BOwnerPos);
    // Map contact point location to bodies' local reference frames
    applyOriQToVector3<float, deme::oriQ_t>(locCPA.x, locCPA.y, locCPA.z, AOriQ.w, -AOriQ.x, -AOriQ.y, -AOriQ.z);
    applyOriQToVector3<float, deme::oriQ_t>(locCPB.x, locCPB.y, locCPB.z, BOriQ.w, -BOriQ.x, -BOriQ.y, -BOriQ.z);
    {
        const float max_lever_A = maxOwnerLocalLever(simParams, granData, ownerA, extraMarginSize);
        const float max_lever_B = maxOwnerLocalLever(simParams, granData, ownerB, extraMarginSize);
        clampLocalContactPoint(locCPA, max_lever_A);
        clampLocalContactPoint(locCPB, max_lever_B);
    }

    const deme::contact_t ContactType_candidate = ContactType;
    bool ownerBoundReject = false;
    if constexpr (BType != deme::GEO_T_ANALYTICAL) {
        if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f &&
            ContactType_candidate != deme::NOT_A_CONTACT && granData->ownerBoundRadius && ownerA != deme::NULL_BODYID &&
            ownerB != deme::NULL_BODYID && ownerA < simParams->nOwnerBodies && ownerB < simParams->nOwnerBodies) {
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
    const bool activeForThisStep = (ContactType_candidate != deme::NOT_A_CONTACT) && !discardGhostGhost &&
                                   !cylPeriodicSkipPair && !ownerBoundReject;
    const deme::contact_t ContactType_forWrite = activeForThisStep ? ContactType_candidate : deme::NOT_A_CONTACT;
    ContactType = ContactType_forWrite;

    // Rotate history (base wedge -> active image) only if the contact is active this step.
    if (activeForThisStep && cylHistShift != 0) {
        const float sin_f = (cylHistShift < 0) ? -simParams->cylPeriodicSinSpan : simParams->cylPeriodicSinSpan;
        float3 dt = make_float3(delta_tan_x, delta_tan_y, delta_tan_z);
        dt = cylPeriodicRotateVec(dt, simParams, sin_f);
        delta_tan_x = dt.x;
        delta_tan_y = dt.y;
        delta_tan_z = dt.z;
    }

    if (activeForThisStep) {
        // The force model is user-specifiable
        // NOTE!! "force" and all wildcards must be properly set by this piece of code
        { _DEMForceModel_; }

        // If force model modifies owner wildcards, write them back here
        _forceModelOwnerWildcardWrite_;
    } else {
        // Not active this step: clear writeback buffers to avoid stale data reuse (DEME3 does not clear arrays per
        // step).
        force = make_float3(0.f, 0.f, 0.f);
        torque_only_force = make_float3(0.f, 0.f, 0.f);
        locCPA = make_float3(0.f, 0.f, 0.f);
        locCPB = make_float3(0.f, 0.f, 0.f);
        // For seam-branch mismatches, keep history (freeze) to avoid artificial de-sticking.
        // But if this candidate is geometrically invalid by owner bound rejection, destroy history.
        if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f && ownerBoundReject &&
            ContactType_candidate != deme::NOT_A_CONTACT && !discardGhostGhost) {
            _forceModelContactWildcardDestroy_;
        }
        // True non-contacts still destroy history.
        if (ContactType_candidate == deme::NOT_A_CONTACT || discardGhostGhost) {
            _forceModelContactWildcardDestroy_;
        }
    }

    // Rotate history back to base wedge (active image -> base) before storing.
    if (activeForThisStep && cylHistShift != 0) {
        const float sin_b = (cylHistShift < 0) ? simParams->cylPeriodicSinSpan : -simParams->cylPeriodicSinSpan;
        float3 dt = make_float3(delta_tan_x, delta_tan_y, delta_tan_z);
        dt = cylPeriodicRotateVec(dt, simParams, sin_b);
        delta_tan_x = dt.x;
        delta_tan_y = dt.y;
        delta_tan_z = dt.z;
    }

    // Note in DEME3, we do not clear force array anymore in each timestep, so always writing back force and contact
    // points, even for zero-force non-contacts, is needed (unless of course, the user instructed no force record). This
    // design has implications in our new two-step patch-based force calculation algorithm, as we re-use some
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
        applyOriQToVector3<float, deme::oriQ_t>(locCPA.x, locCPA.y, locCPA.z, AOriQ.w, -AOriQ.x, -AOriQ.y, -AOriQ.z);
        applyOriQToVector3<float, deme::oriQ_t>(locCPB.x, locCPB.y, locCPB.z, BOriQ.w, -BOriQ.x, -BOriQ.y, -BOriQ.z);
        {
            const float max_lever_A = maxOwnerLocalLever(simParams, granData, ownerA, extraMarginSize);
            const float max_lever_B = maxOwnerLocalLever(simParams, granData, ownerB, extraMarginSize);
            clampLocalContactPoint(locCPA, max_lever_A);
            clampLocalContactPoint(locCPB, max_lever_B);
        }
    }

    if (ContactType == deme::NOT_A_CONTACT) {
        locCPA = make_float3(0.f, 0.f, 0.f);
        locCPB = make_float3(0.f, 0.f, 0.f);
    }
    // Write contact location values back to global memory (after periodic wrap correction).
    _contactInfoWrite_;

    // Optionally, the forces can be reduced to acc right here (may be faster)
    _forceCollectInPlaceStrat_;

    // Updated contact wildcards need to be write back to global mem
    _forceModelContactWildcardWrite_;
}

// 3 specialized kernels for patch-based contact types
DEME_KERNEL void calculatePatchContactForces_SphTri(deme::DEMSimParams* simParams,
                                                    deme::DEMDataDT* granData,
                                                    const double* finalAreas,
                                                    const float3* finalNormals,
                                                    const double* finalPenetrations,
                                                    const double3* finalContactPoints,
                                                    deme::contactPairs_t startOffset,
                                                    deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPatchContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPatchContactID < startOffset + nContactPairs) {
        calculatePatchContactForces_impl<deme::SPHERE_TRIANGLE_CONTACT>(simParams, granData, finalAreas, finalNormals,
                                                                        finalPenetrations, finalContactPoints,
                                                                        myPatchContactID, startOffset);
    }
}

DEME_KERNEL void calculatePatchContactForces_TriTri(deme::DEMSimParams* simParams,
                                                    deme::DEMDataDT* granData,
                                                    const double* finalAreas,
                                                    const float3* finalNormals,
                                                    const double* finalPenetrations,
                                                    const double3* finalContactPoints,
                                                    deme::contactPairs_t startOffset,
                                                    deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPatchContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPatchContactID < startOffset + nContactPairs) {
        calculatePatchContactForces_impl<deme::TRIANGLE_TRIANGLE_CONTACT>(simParams, granData, finalAreas, finalNormals,
                                                                          finalPenetrations, finalContactPoints,
                                                                          myPatchContactID, startOffset);
    }
}

DEME_KERNEL void calculatePatchContactForces_TriAnal(deme::DEMSimParams* simParams,
                                                     deme::DEMDataDT* granData,
                                                     const double* finalAreas,
                                                     const float3* finalNormals,
                                                     const double* finalPenetrations,
                                                     const double3* finalContactPoints,
                                                     deme::contactPairs_t startOffset,
                                                     deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPatchContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPatchContactID < startOffset + nContactPairs) {
        calculatePatchContactForces_impl<deme::TRIANGLE_ANALYTICAL_CONTACT>(
            simParams, granData, finalAreas, finalNormals, finalPenetrations, finalContactPoints, myPatchContactID,
            startOffset);
    }
}
