// DEM force computation related custom kernels
#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>

// Small helper for finite checks on device
inline __device__ bool deme_isfinite3(const float3& v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

inline __device__ float deme_len2(const float3& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

inline __device__ bool deme_sane_local_cp(const float3& p, float max_norm) {
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) {
        return false;
    }
    max_norm = fmaxf(max_norm, 1e-6f);
    return deme_len2(p) <= max_norm * max_norm;
}
_kernelIncludes_;

// Mass properties are below, if jitified mass properties are in use
_massDefs_;
_moiDefs_;

inline __device__ float3 cylPeriodicRotateVecSpan(const float3& vec,
                                                  const deme::DEMSimParams* simParams,
                                                  float sin_span) {
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

inline __device__ deme::bodyID_t getPatchOwnerSafe(const deme::DEMSimParams* simParams,
                                                   const deme::DEMDataDT* granData,
                                                   deme::bodyID_t patch_id,
                                                   deme::geoType_t type) {
    switch (type) {
        case deme::GEO_T_SPHERE:
            if (patch_id < simParams->nSpheresGM) {
                return granData->ownerClumpBody[patch_id];
            }
            break;
        case deme::GEO_T_TRIANGLE:
            if (patch_id < simParams->nMeshPatches) {
                return granData->ownerPatchMesh[patch_id];
            }
            break;
        case deme::GEO_T_ANALYTICAL:
            if (patch_id < simParams->nAnalGM) {
                return granData->ownerAnalBody[patch_id];
            }
            break;
        default:
            break;
    }
    return deme::NULL_BODYID;
}

// computes a ./ b
DEME_KERNEL void forceToAcc(deme::DEMSimParams* simParams, deme::DEMDataDT* granData, size_t n) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        deme::contact_t thisCntType = granData->contactTypePatch[myID];
        if (thisCntType == deme::NOT_A_CONTACT) {
            return;
        }
        if (!deme::isSupportedContactType(thisCntType)) {
            return;
        }
        const float3 F = granData->contactForces[myID];
        const float3 torque_only_force = granData->contactTorque_convToForce[myID];
        const deme::bodyID_t idPatchA_raw = granData->idPatchA[myID];
        const deme::bodyID_t idPatchB_raw = granData->idPatchB[myID];
        bool ghostA = false;
        bool ghostB = false;
        bool ghostA_neg = false;
        bool ghostB_neg = false;
        const deme::bodyID_t idPatchA = cylPeriodicDecodeID(idPatchA_raw, ghostA, ghostA_neg);
        const deme::bodyID_t idPatchB = cylPeriodicDecodeID(idPatchB_raw, ghostB, ghostB_neg);
        float3 forceA = F;
        float3 forceB = make_float3(-F.x, -F.y, -F.z);
        float3 torqueA = torque_only_force;
        float3 torqueB = make_float3(-torque_only_force.x, -torque_only_force.y, -torque_only_force.z);
        const deme::geoType_t typeA = deme::decodeTypeA(thisCntType);
        const deme::geoType_t typeB = deme::decodeTypeB(thisCntType);
        const deme::bodyID_t ownerA = getPatchOwnerSafe(simParams, granData, idPatchA, typeA);
        const deme::bodyID_t ownerB = getPatchOwnerSafe(simParams, granData, idPatchB, typeB);
        if (ownerA == deme::NULL_BODYID || ownerB == deme::NULL_BODYID || ownerA >= simParams->nOwnerBodies ||
            ownerB >= simParams->nOwnerBodies) {
            return;
        }

        if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f) {
            // Use only the kT ghost flag to rotate forces back to the base-wedge frame.
            int wrapShiftA = ghostA ? (ghostA_neg ? -1 : 1) : 0;
            int wrapShiftB = ghostB ? (ghostB_neg ? -1 : 1) : 0;
            if (granData->ownerCylWrapOffset) {
                wrapShiftA += granData->ownerCylWrapOffset[ownerA];
                wrapShiftB += granData->ownerCylWrapOffset[ownerB];
            }
            if (wrapShiftA != 0) {
                float cos_theta = 1.f, sin_theta = 0.f, cos_half = 1.f, sin_half = 0.f;
                cylPeriodicShiftTrig(-wrapShiftA, simParams, cos_theta, sin_theta, cos_half, sin_half);
                forceA = cylPeriodicRotateVec(forceA, simParams, cos_theta, sin_theta);
                torqueA = cylPeriodicRotateVec(torqueA, simParams, cos_theta, sin_theta);
            }
            if (wrapShiftB != 0) {
                float cos_theta = 1.f, sin_theta = 0.f, cos_half = 1.f, sin_half = 0.f;
                cylPeriodicShiftTrig(-wrapShiftB, simParams, cos_theta, sin_theta, cos_half, sin_half);
                forceB = cylPeriodicRotateVec(forceB, simParams, cos_theta, sin_theta);
                torqueB = cylPeriodicRotateVec(torqueB, simParams, cos_theta, sin_theta);
            }
        }

        // Take care of A
        {
            float myMass;
            float3 myMOI;
            const deme::bodyID_t idPatch = idPatchA;
            const float3 myCntPnt = granData->contactPointGeometryA[myID];
            const deme::bodyID_t myOwner = ownerA;
            // Get my mass info from either jitified arrays or global memory
            // Outputs myMass
            // Use an input named exactly `myOwner' which is the id of this owner
            {
                _massAcqStrat_;
                _moiAcqStrat_;
            }

            float max_local_lever = 2e-2f;
            if (granData->ownerBoundRadius && myOwner != deme::NULL_BODYID && myOwner < simParams->nOwnerBodies) {
                const float bound_r = fmaxf(granData->ownerBoundRadius[myOwner], 0.f);
                const float geom_tol = fmaxf(simParams->dyn.beta + simParams->maxFamilyExtraMargin, 0.f) + 1e-4f;
                max_local_lever = fmaxf(bound_r + geom_tol, 1e-3f);
            }
            const bool bad_vec = !deme_isfinite3(forceA) || !deme_isfinite3(torqueA);
            const bool bad_cp = !deme_sane_local_cp(myCntPnt, max_local_lever);

            atomicAdd(granData->aX + myOwner, forceA.x / myMass);
            atomicAdd(granData->aY + myOwner, forceA.y / myMass);
            atomicAdd(granData->aZ + myOwner, forceA.z / myMass);

            // Then ang acc
            if (!(bad_vec || bad_cp)) {
                const deme::oriQ_t myOriQw = granData->oriQw[myOwner];
                const deme::oriQ_t myOriQx = granData->oriQx[myOwner];
                const deme::oriQ_t myOriQy = granData->oriQy[myOwner];
                const deme::oriQ_t myOriQz = granData->oriQz[myOwner];

                // torque_inForceForm is usually the contribution of rolling resistance and it contributes to torque
                // only, not linear velocity
                float3 myF = (forceA + torqueA);
                // F is in global frame, but it needs to be in local to coordinate with moi and cntPnt
                applyOriQToVector3<float, deme::oriQ_t>(myF.x, myF.y, myF.z, myOriQw, -myOriQx, -myOriQy, -myOriQz);
                const float3 angAcc = cross(myCntPnt, myF) / myMOI;
                atomicAdd(granData->alphaX + myOwner, angAcc.x);
                atomicAdd(granData->alphaY + myOwner, angAcc.y);
                atomicAdd(granData->alphaZ + myOwner, angAcc.z);
            }
        }

        // Take care of B
        {
            float myMass;
            float3 myMOI;
            const deme::bodyID_t idPatch = idPatchB;
            const float3 myCntPnt = granData->contactPointGeometryB[myID];
            deme::bodyID_t myOwner = ownerB;

            // Get my mass info from either jitified arrays or global memory
            // Outputs myMass
            // Use an input named exactly `myOwner' which is the id of this owner
            {
                _massAcqStrat_;
                _moiAcqStrat_;
            }

            float max_local_lever = 2e-2f;
            if (granData->ownerBoundRadius && myOwner != deme::NULL_BODYID && myOwner < simParams->nOwnerBodies) {
                const float bound_r = fmaxf(granData->ownerBoundRadius[myOwner], 0.f);
                const float geom_tol = fmaxf(simParams->dyn.beta + simParams->maxFamilyExtraMargin, 0.f) + 1e-4f;
                max_local_lever = fmaxf(bound_r + geom_tol, 1e-3f);
            }
            const bool bad_vec = !deme_isfinite3(myCntPnt) || !deme_isfinite3(forceB) || !deme_isfinite3(torqueB);
            const bool bad_cp = !deme_sane_local_cp(myCntPnt, max_local_lever);

            atomicAdd(granData->aX + myOwner, forceB.x / myMass);
            atomicAdd(granData->aY + myOwner, forceB.y / myMass);
            atomicAdd(granData->aZ + myOwner, forceB.z / myMass);

            // Then ang acc
            if (!(bad_vec || bad_cp)) {
                const deme::oriQ_t myOriQw = granData->oriQw[myOwner];
                const deme::oriQ_t myOriQx = granData->oriQx[myOwner];
                const deme::oriQ_t myOriQy = granData->oriQy[myOwner];
                const deme::oriQ_t myOriQz = granData->oriQz[myOwner];

                // torque_inForceForm is usually the contribution of rolling resistance and it contributes to torque
                // only, not linear velocity
                float3 myF = (forceB + torqueB);
                // F is in global frame, but it needs to be in local to coordinate with moi and cntPnt
                applyOriQToVector3<float, deme::oriQ_t>(myF.x, myF.y, myF.z, myOriQw, -myOriQx, -myOriQy, -myOriQz);
                const float3 angAcc = cross(myCntPnt, myF) / myMOI;
                atomicAdd(granData->alphaX + myOwner, angAcc.x);
                atomicAdd(granData->alphaY + myOwner, angAcc.y);
                atomicAdd(granData->alphaZ + myOwner, angAcc.z);
            }
        }
    }
}
