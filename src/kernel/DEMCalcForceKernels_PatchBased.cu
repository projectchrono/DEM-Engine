// DEM patch-based force computation related custom kernels
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

    deme::materialsOffset_t bodyAMatType, bodyBMatType;
    // The user-specified extra margin size (how much we should be lenient in determining `in-contact')
    float extraMarginSize = 0.;

    // Then allocate the optional quantities that will be needed in the force model
    _forceModelIngredientDefinition_;

    deme::bodyID_t ownerA = deme::NULL_BODYID;
    deme::bodyID_t ownerB = deme::NULL_BODYID;

    const deme::bodyID_t idA_raw = granData->idPatchA[myPatchContactID];
    const deme::bodyID_t idB_raw = granData->idPatchB[myPatchContactID];
    // Cylindrical periodic: desired minimum-image shift selection (decided on dT)
    int desiredShiftA = 0;
    int desiredShiftB = 0;
    int ghostShiftA = 0;
    int ghostShiftB = 0;
    float desiredBestDist2 = 0.f;
    float desiredRadAOwner = 0.f;
    float desiredRadBOwner = 0.f;

    // Decompose ContactType to get the types of A and B (known at compile time)
    constexpr deme::geoType_t AType = (CONTACT_TYPE >> 4);
    constexpr deme::geoType_t BType = (CONTACT_TYPE & 0xF);

    // ----------------------------------------------------------------
    // Based on A's type, equip info
    // ----------------------------------------------------------------
    if constexpr (AType == deme::GEO_T_SPHERE) {
        // For sphere-mesh contacts, patch A is a sphere
        // Note: For spheres, the patch ID is the same as the sphere ID
        deme::bodyID_t sphereID = idA_raw;
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
        deme::bodyID_t myPatchID = idA_raw;
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
        deme::bodyID_t myPatchID = idB_raw;
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

        // As the grace margin, the distance (negative overlap) just needs to be within the grace margin. So we pick
        // the larger of the 2 familyExtraMarginSize.
        extraMarginSize = (extraMarginSize > granData->familyExtraMarginSize[BOwnerFamily])
                              ? extraMarginSize
                              : granData->familyExtraMarginSize[BOwnerFamily];
        if (overlapDepth <= -extraMarginSize) {
            ContactType = deme::NOT_A_CONTACT;
        }
    }

    //// NO CLAMPING. CURRENT FORCE IMPLEMENTATION IS PHYSICAL AND STABLE.
    // // Patch-level geometric invariant guard for real owner-owner contacts.
    // // Prevent rare depth outliers from injecting unphysical force/energy.
    // if constexpr (AType == deme::GEO_T_SPHERE && BType == deme::GEO_T_TRIANGLE) {
    //     if (ContactType != deme::NOT_A_CONTACT) {
    //         const float3 cpRelToSphere = to_float3(contactPnt - bodyAPos);
    //         const float cpRel2 = dot(cpRelToSphere, cpRelToSphere);
    //         const float shell_half_B = ownerShellHalfThickness(simParams, granData, ownerB);
    //         const float maxSphereReach =
    //             ARadius + shell_half_B +
    //             fmaxf(simParams->dyn.beta + simParams->maxFamilyExtraMargin + extraMarginSize, 0.f) + 1e-6f;
    //         if (!isfinite(cpRel2) || cpRel2 > maxSphereReach * maxSphereReach) {
    //             ContactType = deme::NOT_A_CONTACT;
    //             overlapDepth = -DEME_HUGE_FLOAT;
    //             overlapArea = 0.0;
    //         }
    //     }
    // }
    // if constexpr (BType != deme::GEO_T_ANALYTICAL) {
    //     constexpr bool apply_shape_depth_cap = (AType == deme::GEO_T_TRIANGLE) && (BType == deme::GEO_T_TRIANGLE);
    //     if (ContactType != deme::NOT_A_CONTACT &&
    //         !clampPatchPenetrationByOwnerBounds(simParams, granData, ownerA, ownerB, AOwnerPos, BOwnerPos,
    //                                             contactPnt, B2A, extraMarginSize, overlapDepth,
    //                                             apply_shape_depth_cap)) {
    //         ContactType = deme::NOT_A_CONTACT;
    //         overlapDepth = -DEME_HUGE_FLOAT;
    //         overlapArea = 0.0;
    //     }
    // }

    // Now compute forces using the patch-based contact data
    _forceModelContactWildcardAcq_;

    // Essentials for storing and calculating contact info
    float3 force = make_float3(0, 0, 0);
    float3 torque_only_force = make_float3(0, 0, 0);
    // Local position of the contact point
    float3 locCPA = to_float3(contactPnt - AOwnerPos);
    float3 locCPB = to_float3(contactPnt - BOwnerPos);
    // Map contact point location to bodies' local reference frames
    applyOriQToVector3(locCPA, make_float4(-AOriQ.x, -AOriQ.y, -AOriQ.z, AOriQ.w));
    applyOriQToVector3(locCPB, make_float4(-BOriQ.x, -BOriQ.y, -BOriQ.z, BOriQ.w));
    // {
    //     const float max_lever_A = maxOwnerLocalLever(simParams, granData, ownerA, extraMarginSize);
    //     const float max_lever_B = maxOwnerLocalLever(simParams, granData, ownerB, extraMarginSize);
    //     clampLocalContactPoint(locCPA, max_lever_A);
    //     clampLocalContactPoint(locCPB, max_lever_B);
    // }

    const deme::contact_t ContactType_candidate = ContactType;
    const bool activeForThisStep = (ContactType_candidate != deme::NOT_A_CONTACT);
    const deme::contact_t ContactType_forWrite = activeForThisStep ? ContactType_candidate : deme::NOT_A_CONTACT;
    ContactType = ContactType_forWrite;

    if (activeForThisStep) {
        // The force model is user-specifiable
        // NOTE!! "force" and all wildcards must be properly set by this piece of code
        { _DEMForceModel_; }

        // If force model modifies owner wildcards, write them back here
        _forceModelOwnerWildcardWrite_;
    } else {
        force = make_float3(0.f, 0.f, 0.f);
        torque_only_force = make_float3(0.f, 0.f, 0.f);
        locCPA = make_float3(0.f, 0.f, 0.f);
        locCPB = make_float3(0.f, 0.f, 0.f);
        if (ContactType_candidate == deme::NOT_A_CONTACT) {
            _forceModelContactWildcardDestroy_;
        }
    }

    // Note in DEME3, we do not clear force array anymore in each timestep, so always writing back force and contact
    // points, even for zero-force non-contacts, is needed (unless of course, the user instructed no force record). This
    // design has implications in our new two-step patch-based force calculation algorithm, as we re-use some
    // force-storing arrays for intermediate values.

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

/*
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

inline __device__ float ownerShellHalfThickness(const deme::DEMSimParams* simParams,
                                                const deme::DEMDataDT* granData,
                                                deme::bodyID_t ownerID) {
    if (!granData->ownerMeshShellHalfThickness || ownerID == deme::NULL_BODYID || ownerID >= simParams->nOwnerBodies) {
        return 0.f;
    }
    return fmaxf(granData->ownerMeshShellHalfThickness[ownerID], 0.f);
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
        // Rate-limit geometric depth outliers for tri-tri, but keep a scale-based lower bound so
        // existing sharp contacts are not over-clamped to near-zero overlap.
        float3 n = contactNormal;
        const float n2 = n.x * n.x + n.y * n.y + n.z * n.z;
        if (isfinite(n2) && n2 > DEME_TINY_FLOAT) {
            const float inv_n = rsqrtf(n2);
            n *= inv_n;

            const float3 vA = make_float3(granData->vX[ownerA], granData->vY[ownerA], granData->vZ[ownerA]);
            const float3 vB = make_float3(granData->vX[ownerB], granData->vY[ownerB], granData->vZ[ownerB]);
            const float3 wA = make_float3(granData->omgBarX[ownerA], granData->omgBarY[ownerA],
granData->omgBarZ[ownerA]); const float3 wB = make_float3(granData->omgBarX[ownerB], granData->omgBarY[ownerB],
granData->omgBarZ[ownerB]);

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
                4.0 * static_cast<double>(closing_speed) * static_cast<double>(simParams->dyn.h) * drift_factor +
                static_cast<double>(fmaxf(0.5f * simParams->dyn.beta + extraMarginSize, 0.f)) + 1e-7;

            const float pair_scale = sqrtf(fmaxf(radA * radB, 0.f));
            const float shape_scale = sqrtf(fmaxf(fminf(radA, radB) * pair_scale, 0.f));
            const double min_shape_cap = static_cast<double>(0.0041f * shape_scale);
            const double relaxed_cap = fmax(motion_cap, min_shape_cap);
            if (relaxed_cap > 0.0) {
                cappedMaxOverlap = fmin(cappedMaxOverlap, relaxed_cap);
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
*/
