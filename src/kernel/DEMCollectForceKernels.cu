// DEM force computation related custom kernels
#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>
_kernelIncludes_;

// Mass properties are below, if jitified mass properties are in use
_massDefs_;
_moiDefs_;

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
        const deme::bodyID_t idPatchA = granData->idPatchA[myID];
        const deme::bodyID_t idPatchB = granData->idPatchB[myID];
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

        // Take care of A
        {
            float myMass;
            const deme::bodyID_t idPatch = idPatchA;
            const deme::bodyID_t myOwner = ownerA;
            // Get my mass info from either jitified arrays or global memory
            // Outputs myMass
            // Use an input named exactly `myOwner' which is the id of this owner
            { _massAcqStrat_; }

            if (!isfinite3(forceA) || !isfinite3(torqueA)) {
                DEME_ABORT_KERNEL("forceToAcc found invalid force/torque for A: contact %llu, owner %llu, type %u.\n",
                                  static_cast<unsigned long long>(myID), static_cast<unsigned long long>(myOwner),
                                  static_cast<unsigned int>(thisCntType));
            }

            atomicAdd(granData->aX + myOwner, forceA.x / myMass);
            atomicAdd(granData->aY + myOwner, forceA.y / myMass);
            atomicAdd(granData->aZ + myOwner, forceA.z / myMass);

            // torque_inForceForm is usually the contribution of rolling resistance and it contributes to torque
            // only, not linear velocity. A zero resultant cannot produce angular acceleration, so do not read a
            // contact-point slot that a margin-only pair may never have written. Besides avoiding cross(NaN, 0), this
            // skips the orientation and angular atomics for the common zero-force case without dropping linear force.
            // The in-force-kernel collection policy does not need this memory-safety guard because it consumes the
            // locally computed contact point instead of reading a persistent contact-point array.
            float3 myF = forceA + torqueA;
            if (myF.x != 0.0f || myF.y != 0.0f || myF.z != 0.0f) {
                float3 myMOI;
                { _moiAcqStrat_; }
                const float3 myCntPnt = granData->contactPointGeometryA[myID];
                if (!isfinite3(myCntPnt)) {
                    DEME_ABORT_KERNEL(
                        "forceToAcc found invalid contact point for A: contact %llu, owner %llu, type %u.\n",
                        static_cast<unsigned long long>(myID), static_cast<unsigned long long>(myOwner),
                        static_cast<unsigned int>(thisCntType));
                }

                const deme::oriQ_t myOriQw = granData->oriQw[myOwner];
                const deme::oriQ_t myOriQx = granData->oriQx[myOwner];
                const deme::oriQ_t myOriQy = granData->oriQy[myOwner];
                const deme::oriQ_t myOriQz = granData->oriQz[myOwner];
                // F is in global frame, but it needs to be in local to coordinate with moi and cntPnt
                applyOriQToVector3(myF, make_float4(-myOriQx, -myOriQy, -myOriQz, myOriQw));
                const float3 angAcc = cross(myCntPnt, myF) / myMOI;
                atomicAdd(granData->alphaX + myOwner, angAcc.x);
                atomicAdd(granData->alphaY + myOwner, angAcc.y);
                atomicAdd(granData->alphaZ + myOwner, angAcc.z);
            }
        }

        // Take care of B
        {
            float myMass;
            const deme::bodyID_t idPatch = idPatchB;
            deme::bodyID_t myOwner = ownerB;

            // Get my mass info from either jitified arrays or global memory
            // Outputs myMass
            // Use an input named exactly `myOwner' which is the id of this owner
            { _massAcqStrat_; }

            if (!isfinite3(forceB) || !isfinite3(torqueB)) {
                DEME_ABORT_KERNEL("forceToAcc found invalid force/torque for B: contact %llu, owner %llu, type %u.\n",
                                  static_cast<unsigned long long>(myID), static_cast<unsigned long long>(myOwner),
                                  static_cast<unsigned int>(thisCntType));
            }

            atomicAdd(granData->aX + myOwner, forceB.x / myMass);
            atomicAdd(granData->aY + myOwner, forceB.y / myMass);
            atomicAdd(granData->aZ + myOwner, forceB.z / myMass);

            // torque_inForceForm is usually the contribution of rolling resistance and it contributes to torque
            // only, not linear velocity. Keep B's guard independent because its contact point has separate storage.
            float3 myF = forceB + torqueB;
            if (myF.x != 0.0f || myF.y != 0.0f || myF.z != 0.0f) {
                float3 myMOI;
                { _moiAcqStrat_; }
                const float3 myCntPnt = granData->contactPointGeometryB[myID];
                if (!isfinite3(myCntPnt)) {
                    DEME_ABORT_KERNEL(
                        "forceToAcc found invalid contact point for B: contact %llu, owner %llu, type %u.\n",
                        static_cast<unsigned long long>(myID), static_cast<unsigned long long>(myOwner),
                        static_cast<unsigned int>(thisCntType));
                }

                const deme::oriQ_t myOriQw = granData->oriQw[myOwner];
                const deme::oriQ_t myOriQx = granData->oriQx[myOwner];
                const deme::oriQ_t myOriQy = granData->oriQy[myOwner];
                const deme::oriQ_t myOriQz = granData->oriQz[myOwner];
                // F is in global frame, but it needs to be in local to coordinate with moi and cntPnt
                applyOriQToVector3(myF, make_float4(-myOriQx, -myOriQy, -myOriQz, myOriQw));
                const float3 angAcc = cross(myCntPnt, myF) / myMOI;
                atomicAdd(granData->alphaX + myOwner, angAcc.x);
                atomicAdd(granData->alphaY + myOwner, angAcc.y);
                atomicAdd(granData->alphaZ + myOwner, angAcc.z);
            }
        }
    }
}

DEME_KERNEL void aggregateCombinedOwnersAcc(deme::DEMSimParams* simParams,
                                            deme::DEMDataDT* granData,
                                            deme::bodyID_t nOwners) {
    // Device-side aggregation pass:
    // each non-master member contributes its already-accumulated linear/angular accelerations to its master.
    // Contributions are mass/MOI weighted so that master-side integration follows the equivalent-group properties.
    deme::bodyID_t owner = blockIdx.x * blockDim.x + threadIdx.x;
    if (owner >= nOwners || simParams->nCombinedOwners == 0 || granData->ownerCombinedMaster == nullptr) {
        return;
    }

    const deme::bodyID_t master = granData->ownerCombinedMaster[owner];
    // Master owners are intentionally skipped in this pass: only non-master members contribute to master.
    if (master == deme::NULL_BODYID || master == owner || master >= simParams->nOwnerBodies) {
        return;
    }

    float myMass;
    float3 myMOI;
    deme::bodyID_t myOwner = owner;
    {
        _massAcqStrat_;
        _moiAcqStrat_;
    }
    const float memberMass = myMass;
    const float3 memberMOI = myMOI;

    float masterMass = 0.f;
    float3 masterMOI = make_float3(0);
    if (granData->ownerCombinedMasterMass != nullptr) {
        masterMass = granData->ownerCombinedMasterMass[master];
    }
    if (granData->ownerCombinedMasterMOI != nullptr) {
        masterMOI = granData->ownerCombinedMasterMOI[master];
    }
    if (masterMass <= DEME_TINY_FLOAT || !isfinite(masterMass)) {
        // Defensive fallback: if equivalent master properties are unavailable, fall back to master's own properties.
        myOwner = master;
        {
            _massAcqStrat_;
            _moiAcqStrat_;
        }
        masterMass = myMass;
        masterMOI = myMOI;
    }

    if (memberMass > DEME_TINY_FLOAT && masterMass > DEME_TINY_FLOAT) {
        // Convert member acceleration contribution to master acceleration space.
        const float ratio_m = memberMass / masterMass;
        atomicAdd(granData->aX + master, granData->aX[owner] * ratio_m);
        atomicAdd(granData->aY + master, granData->aY[owner] * ratio_m);
        atomicAdd(granData->aZ + master, granData->aZ[owner] * ratio_m);
    }

    if (masterMOI.x > DEME_TINY_FLOAT && masterMOI.y > DEME_TINY_FLOAT && masterMOI.z > DEME_TINY_FLOAT) {
        float4 qMaster = make_float4(granData->oriQx[master], granData->oriQy[master], granData->oriQz[master],
                                     granData->oriQw[master]);
        float3 memberForceMaster =
            memberMass * make_float3(granData->aX[owner], granData->aY[owner], granData->aZ[owner]);
        applyOriQToVector3(memberForceMaster, make_float4(-qMaster.x, -qMaster.y, -qMaster.z, qMaster.w));

        float3 torqueMaster = make_float3(0);
        if (granData->ownerCombinedRelPos != nullptr) {
            torqueMaster += cross(granData->ownerCombinedRelPos[owner], memberForceMaster);
        }

        float3 memberTorqueMaster =
            memberMOI * make_float3(granData->alphaX[owner], granData->alphaY[owner], granData->alphaZ[owner]);
        if (granData->ownerCombinedRelOriQ != nullptr) {
            applyOriQToVector3(memberTorqueMaster, granData->ownerCombinedRelOriQ[owner]);
        }
        torqueMaster += memberTorqueMaster;

        atomicAdd(granData->alphaX + master, torqueMaster.x / masterMOI.x);
        atomicAdd(granData->alphaY + master, torqueMaster.y / masterMOI.y);
        atomicAdd(granData->alphaZ + master, torqueMaster.z / masterMOI.z);
    }

    granData->aX[owner] = 0.f;
    granData->aY[owner] = 0.f;
    granData->aZ[owner] = 0.f;
    granData->alphaX[owner] = 0.f;
    granData->alphaY[owner] = 0.f;
    granData->alphaZ[owner] = 0.f;
}

DEME_KERNEL void reimposeCombinedOwners(deme::DEMSimParams* simParams,
                                        deme::DEMDataDT* granData,
                                        deme::bodyID_t nOwners) {
    // Device-side rigid re-imposition pass:
    // after integrating masters, overwrite each member's state from
    //   master state + fixed member transform in master frame.
    // This enforces rigid relative motion without host/device synchronization.
    deme::bodyID_t owner = blockIdx.x * blockDim.x + threadIdx.x;
    if (owner >= nOwners || simParams->nCombinedOwners == 0 || granData->ownerCombinedMaster == nullptr ||
        granData->ownerCombinedRelPos == nullptr || granData->ownerCombinedRelOriQ == nullptr) {
        return;
    }

    const deme::bodyID_t master = granData->ownerCombinedMaster[owner];
    // Master owners are intentionally skipped in this pass: only non-master members are re-imposed.
    if (master == deme::NULL_BODYID || master == owner || master >= simParams->nOwnerBodies) {
        return;
    }

    double mX, mY, mZ;
    voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
        mX, mY, mZ, granData->voxelID[master], granData->locX[master], granData->locY[master], granData->locZ[master],
        _nvXp2_, _nvYp2_, _voxelSize_, _l_);
    mX += (double)simParams->LBFX;
    mY += (double)simParams->LBFY;
    mZ += (double)simParams->LBFZ;

    float4 qMaster =
        make_float4(granData->oriQx[master], granData->oriQy[master], granData->oriQz[master], granData->oriQw[master]);
    const float3 relLocal = granData->ownerCombinedRelPos[owner];
    float3 relWorld = relLocal;
    applyOriQToVector3(relWorld, qMaster);

    double X = mX + (double)relWorld.x;
    double Y = mY + (double)relWorld.y;
    double Z = mZ + (double)relWorld.z;
    X -= (double)simParams->LBFX;
    Y -= (double)simParams->LBFY;
    Z -= (double)simParams->LBFZ;
    positionToVoxelID<deme::voxelID_t, deme::subVoxelPos_t, double>(granData->voxelID[owner], granData->locX[owner],
                                                                    granData->locY[owner], granData->locZ[owner], X, Y,
                                                                    Z, _nvXp2_, _nvYp2_, _voxelSize_, _l_);

    float4 qRel = granData->ownerCombinedRelOriQ[owner];
    float qW, qX, qY, qZ;
    HamiltonProduct(qW, qX, qY, qZ, qMaster.w, qMaster.x, qMaster.y, qMaster.z, qRel.w, qRel.x, qRel.y, qRel.z);
    const float qn = sqrtf(qW * qW + qX * qX + qY * qY + qZ * qZ);
    if (qn > DEME_TINY_FLOAT && isfinite(qn)) {
        const float inv_qn = 1.f / qn;
        qW *= inv_qn;
        qX *= inv_qn;
        qY *= inv_qn;
        qZ *= inv_qn;
    } else {
        qW = 1.f;
        qX = qY = qZ = 0.f;
    }
    granData->oriQw[owner] = qW;
    granData->oriQx[owner] = qX;
    granData->oriQy[owner] = qY;
    granData->oriQz[owner] = qZ;

    float3 wMasterLocal = make_float3(granData->omgBarX[master], granData->omgBarY[master], granData->omgBarZ[master]);
    float3 wMasterWorld = wMasterLocal;
    applyOriQToVector3(wMasterWorld, qMaster);

    float3 vMaster = make_float3(granData->vX[master], granData->vY[master], granData->vZ[master]);
    // Rigid-body kinematics for member linear velocity: v = v_master + omega_master x r.
    float3 vMember = vMaster + cross(wMasterWorld, relWorld);
    granData->vX[owner] = vMember.x;
    granData->vY[owner] = vMember.y;
    granData->vZ[owner] = vMember.z;

    float4 qMemberConj = make_float4(-qX, -qY, -qZ, qW);
    float3 wMemberLocal = wMasterWorld;
    applyOriQToVector3(wMemberLocal, qMemberConj);
    granData->omgBarX[owner] = wMemberLocal.x;
    granData->omgBarY[owner] = wMemberLocal.y;
    granData->omgBarZ[owner] = wMemberLocal.z;
}
