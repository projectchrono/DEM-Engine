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

// Template device function for patch-based contact force calculation
template <deme::contact_t CONTACT_TYPE>
__device__ __forceinline__ void calculatePatchContactForces_impl(deme::DEMSimParams* simParams,
                                                                 deme::DEMDataDT* granData,
                                                                 const double* totalAreas,
                                                                 const float3* finalNormals,
                                                                 const double* finalPenetrations,
                                                                 deme::contactPairs_t myPatchContactID,
                                                                 deme::contactPairs_t startOffsetPatch) {
    // Contact type is known at compile time
    deme::contact_t ContactType = CONTACT_TYPE;
    
    // Calculate relative index for accessing the temp arrays (totalAreas, finalNormals, finalPenetrations)
    deme::contactPairs_t relativeIndex = myPatchContactID - startOffsetPatch;
    
    // The following quantities are provided from the patch voting process
    float3 B2A = finalNormals[relativeIndex];  // contact normal felt by A, pointing from B to A
    double overlapDepth = finalPenetrations[relativeIndex];  // penetration depth
    double overlapArea = totalAreas[relativeIndex];  // total contact area for this patch pair
    
    // Contact point - we'll compute as centroid of the patch
    double3 contactPnt;
    double3 AOwnerPos, bodyAPos, BOwnerPos, bodyBPos;
    float AOwnerMass, ARadius, BOwnerMass, BRadius;
    float4 AOriQ, BOriQ;
    deme::materialsOffset_t bodyAMatType, bodyBMatType;
    
    // Then allocate the optional quantities that will be needed in the force model
    _forceModelIngredientDefinition_;
    
    // Decompose ContactType to get the types of A and B (known at compile time)
    constexpr deme::geoType_t AType = (CONTACT_TYPE >> 4);
    constexpr deme::geoType_t BType = (CONTACT_TYPE & 0xF);
    
    // Check if this is a valid contact (non-zero area)
    if (overlapArea <= 0.0) {
        ContactType = deme::NOT_A_CONTACT;
    }
    
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
    } else if constexpr (AType == deme::GEO_T_TRIANGLE) {
        // For mesh-mesh or mesh-analytical contacts, patch A is a mesh patch
        deme::bodyID_t myPatchID = granData->idPatchA[myPatchContactID];
        deme::bodyID_t myOwner = granData->patchOwnerMesh[myPatchID];
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

        equipOwnerPosRot(simParams, granData, myOwner, myRelPos, AOwnerPos, bodyAPos, AOriQ);
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
        deme::bodyID_t myOwner = granData->patchOwnerMesh[myPatchID];
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

        equipOwnerPosRot(simParams, granData, myOwner, myRelPos, BOwnerPos, bodyBPos, BOriQ);

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
    } else {
        // Unsupported type
        ContactType = deme::NOT_A_CONTACT;
    }

    // Calculate contact point as midpoint between patch centroids
    // This is a simplified approach for patch-based contacts. In the primitive phase, collision
    // detection algorithms compute exact contact points. Here, we use the centroid approximation
    // since we've already aggregated multiple primitive contacts into a single patch contact.
    // The approximation is acceptable because:
    // 1. The contact normal and penetration have been accurately computed via voting
    // 2. Forces are applied at the owner body level, so minor contact point variations have limited impact
    // 3. The torque calculations still use the correct local positions relative to each owner
    contactPnt = (bodyAPos + bodyBPos) * 0.5;

    // Now compute forces using the patch-based contact data
    _forceModelContactWildcardAcq_;
    if (ContactType != deme::NOT_A_CONTACT) {
        float3 force = make_float3(0, 0, 0);
        float3 torque_only_force = make_float3(0, 0, 0);
        
        // Local position of the contact point
        float3 locCPA = to_float3(contactPnt - AOwnerPos);
        float3 locCPB = to_float3(contactPnt - BOwnerPos);
        
        // Map contact point location to bodies' local reference frames
        applyOriQToVector3<float, deme::oriQ_t>(locCPA.x, locCPA.y, locCPA.z, AOriQ.w, -AOriQ.x, -AOriQ.y, -AOriQ.z);
        applyOriQToVector3<float, deme::oriQ_t>(locCPB.x, locCPB.y, locCPB.z, BOriQ.w, -BOriQ.x, -BOriQ.y, -BOriQ.z);
        
        // The force model is user-specifiable
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

    // Updated contact wildcards need to be write back to global mem
    _forceModelContactWildcardWrite_;
}

// 3 specialized kernels for patch-based contact types
__global__ void calculatePatchContactForces_SphTri(deme::DEMSimParams* simParams,
                                                   deme::DEMDataDT* granData,
                                                   const double* totalAreas,
                                                   const float3* finalNormals,
                                                   const double* finalPenetrations,
                                                   deme::contactPairs_t startOffset,
                                                   deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPatchContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPatchContactID < startOffset + nContactPairs) {
        calculatePatchContactForces_impl<deme::SPHERE_TRIANGLE_CONTACT>(simParams, granData, totalAreas, finalNormals,
                                                                        finalPenetrations, myPatchContactID, startOffset);
    }
}

__global__ void calculatePatchContactForces_TriTri(deme::DEMSimParams* simParams,
                                                   deme::DEMDataDT* granData,
                                                   const double* totalAreas,
                                                   const float3* finalNormals,
                                                   const double* finalPenetrations,
                                                   deme::contactPairs_t startOffset,
                                                   deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPatchContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPatchContactID < startOffset + nContactPairs) {
        calculatePatchContactForces_impl<deme::TRIANGLE_TRIANGLE_CONTACT>(simParams, granData, totalAreas, finalNormals,
                                                                          finalPenetrations, myPatchContactID, startOffset);
    }
}

__global__ void calculatePatchContactForces_TriAnal(deme::DEMSimParams* simParams,
                                                    deme::DEMDataDT* granData,
                                                    const double* totalAreas,
                                                    const float3* finalNormals,
                                                    const double* finalPenetrations,
                                                    deme::contactPairs_t startOffset,
                                                    deme::contactPairs_t nContactPairs) {
    deme::contactPairs_t myPatchContactID = startOffset + blockIdx.x * blockDim.x + threadIdx.x;
    if (myPatchContactID < startOffset + nContactPairs) {
        calculatePatchContactForces_impl<deme::TRIANGLE_ANALYTICAL_CONTACT>(simParams, granData, totalAreas,
                                                                            finalNormals, finalPenetrations,
                                                                            myPatchContactID, startOffset);
    }
}
