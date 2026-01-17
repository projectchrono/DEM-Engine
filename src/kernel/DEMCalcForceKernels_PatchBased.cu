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
                                                                 const float3* finalRelVel,
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
    float3 patchRelVel = make_float3(0.f, 0.f, 0.f);
    if (simParams->usePatchRelVelOverride && finalRelVel != nullptr) {
        patchRelVel = finalRelVel[relativeIndex];
    }
    double3 AOwnerPos, bodyAPos, BOwnerPos, bodyBPos;
    float AOwnerMass, ARadius, BOwnerMass, BRadius;
    float4 AOriQ, BOriQ;
    deme::materialsOffset_t bodyAMatType, bodyBMatType;
    // The user-specified extra margin size (how much we should be lenient in determining `in-contact')
    float extraMarginSize = 0.;

    // Then allocate the optional quantities that will be needed in the force model
    _forceModelIngredientDefinition_;

    // Decompose ContactType to get the types of A and B (known at compile time)
    constexpr deme::geoType_t AType = (CONTACT_TYPE >> 4);
    constexpr deme::geoType_t BType = (CONTACT_TYPE & 0xF);

    // ----------------------------------------------------------------
    // Based on A's type, equip info
    // ----------------------------------------------------------------
    if constexpr (AType == deme::GEO_T_SPHERE) {
        // For sphere-mesh contacts, patch A is a sphere
        // Note: For spheres, the patch ID is the same as the sphere ID
        deme::bodyID_t sphereID = granData->idPatchA[myPatchContactID];
        deme::bodyID_t myOwner = granData->ownerClumpBody[sphereID];
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
        deme::bodyID_t myPatchID = granData->idPatchA[myPatchContactID];
        deme::bodyID_t myOwner = granData->ownerPatchMesh[myPatchID];
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
        deme::bodyID_t myPatchID = granData->idPatchB[myPatchContactID];
        deme::bodyID_t myOwner = granData->ownerPatchMesh[myPatchID];
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

    // Now compute forces using the patch-based contact data
    _forceModelContactWildcardAcq_;

    // Essentials for storing and calculating contact info
    float3 force = make_float3(0, 0, 0);
    float3 torque_only_force = make_float3(0, 0, 0);
    // Local position of the contact point
    float3 locCPA = to_float3(contactPnt - AOwnerPos);
    float3 locCPB = to_float3(contactPnt - BOwnerPos);
    // Map contact point location to bodies' local reference frames
    applyOriQToVector3<float, deme::oriQ_t>(locCPA.x, locCPA.y, locCPA.z, AOriQ.w, -AOriQ.x, -AOriQ.y, -AOriQ.z);
    applyOriQToVector3<float, deme::oriQ_t>(locCPB.x, locCPB.y, locCPB.z, BOriQ.w, -BOriQ.x, -BOriQ.y, -BOriQ.z);

    if (ContactType != deme::NOT_A_CONTACT) {
        // The force model is user-specifiable
        // NOTE!! "force" and all wildcards must be properly set by this piece of code
        { _DEMForceModel_; }

        // If force model modifies owner wildcards, write them back here
        _forceModelOwnerWildcardWrite_;
    } else {
        // The contact is no longer active, so we need to destroy its contact history recording
        _forceModelContactWildcardDestroy_;
    }

    // Note in DEME3, we do not clear force array anymore in each timestep, so always writing back force and contact
    // points, even for zero-force non-contacts, is needed (unless of course, the user instructed no force record). This
    // design has implications in our new two-step patch-based force calculation algorithm, as we re-use some
    // force-storing arrays for intermediate values.

    // Write contact location values back to global memory
    _contactInfoWrite_;

    // Optionally, the forces can be reduced to acc right here (may be faster)
    _forceCollectInPlaceStrat_;

    // Updated contact wildcards need to be write back to global mem
    _forceModelContactWildcardWrite_;
}

// 3 specialized kernels for patch-based contact types
__global__ void calculatePatchContactForces_SphTri(deme::DEMSimParams* simParams,
                                                   deme::DEMDataDT* granData,
                                                   const double* finalAreas,
                                                   const float3* finalNormals,
                                                   const double* finalPenetrations,
                                                   const double3* finalContactPoints,
                                                   const float3* finalRelVel,
                                                   deme::contactPairs_t startOffset,
                                                   deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPatchContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPatchContactID < startOffset + nContactPairs) {
        calculatePatchContactForces_impl<deme::SPHERE_TRIANGLE_CONTACT>(simParams, granData, finalAreas, finalNormals,
                                                                        finalPenetrations, finalContactPoints,
                                                                        finalRelVel,
                                                                        myPatchContactID, startOffset);
    }
}

__global__ void calculatePatchContactForces_TriTri(deme::DEMSimParams* simParams,
                                                   deme::DEMDataDT* granData,
                                                   const double* finalAreas,
                                                   const float3* finalNormals,
                                                   const double* finalPenetrations,
                                                   const double3* finalContactPoints,
                                                   const float3* finalRelVel,
                                                   deme::contactPairs_t startOffset,
                                                   deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPatchContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPatchContactID < startOffset + nContactPairs) {
        calculatePatchContactForces_impl<deme::TRIANGLE_TRIANGLE_CONTACT>(simParams, granData, finalAreas, finalNormals,
                                                                          finalPenetrations, finalContactPoints,
                                                                          finalRelVel,
                                                                          myPatchContactID, startOffset);
    }
}

__global__ void calculatePatchContactForces_TriAnal(deme::DEMSimParams* simParams,
                                                    deme::DEMDataDT* granData,
                                                    const double* finalAreas,
                                                    const float3* finalNormals,
                                                    const double* finalPenetrations,
                                                    const double3* finalContactPoints,
                                                    const float3* finalRelVel,
                                                    deme::contactPairs_t startOffset,
                                                    deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPatchContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPatchContactID < startOffset + nContactPairs) {
        calculatePatchContactForces_impl<deme::TRIANGLE_ANALYTICAL_CONTACT>(
            simParams, granData, finalAreas, finalNormals, finalPenetrations, finalContactPoints, finalRelVel,
            myPatchContactID, startOffset);
    }
}
