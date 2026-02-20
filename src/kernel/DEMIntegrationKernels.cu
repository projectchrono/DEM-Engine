// DEM integration related custom kernels
#include <DEMHelperKernels.cuh>
#include <DEM/Defines.h>
_kernelIncludes_;

// Apply presecibed velocity and report whether the `true' physics should be skipped, rather than added on top of
// that
template <typename T1, typename T2, typename T3, typename T4>
inline __device__ void applyPrescribedVel(bool& LinVelXPrescribed,
                                          bool& LinVelYPrescribed,
                                          bool& LinVelZPrescribed,
                                          bool& RotVelXPrescribed,
                                          bool& RotVelYPrescribed,
                                          bool& RotVelZPrescribed,
                                          T1& vX,
                                          T1& vY,
                                          T1& vZ,
                                          T2& omgBarX,
                                          T2& omgBarY,
                                          T2& omgBarZ,
                                          T3 X,
                                          T3 Y,
                                          T3 Z,
                                          T4 oriQw,
                                          T4 oriQx,
                                          T4 oriQy,
                                          T4 oriQz,
                                          deme::bodyID_t ownerID,
                                          const deme::family_t& family,
                                          const float& t) {
    switch (family) {
        _velPrescriptionStrategy_;
        default:
            // Default can just do nothing
            return;
    }
}

// Apply presecibed location and report whether the `true' physics should be skipped, rather than added on top of that
template <typename T1, typename T2, typename T3, typename T4>
inline __device__ void applyPrescribedPos(bool& LinXPrescribed,
                                          bool& LinYPrescribed,
                                          bool& LinZPrescribed,
                                          bool& RotPrescribed,
                                          T1& X,
                                          T1& Y,
                                          T1& Z,
                                          T2& oriQw,
                                          T2& oriQx,
                                          T2& oriQy,
                                          T2& oriQz,
                                          T3 vX,
                                          T3 vY,
                                          T3 vZ,
                                          T4 omgBarX,
                                          T4 omgBarY,
                                          T4 omgBarZ,
                                          deme::bodyID_t ownerID,
                                          const deme::family_t& family,
                                          const float& t) {
    switch (family) {
        _posPrescriptionStrategy_;
        default:
            // Default can just do nothing
            return;
    }
}

// Apply extra accelerations for family numbers
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
inline __device__ void applyAddedAcceleration(T1& accX,
                                              T1& accY,
                                              T1& accZ,
                                              T2& angAccX,
                                              T2& angAccY,
                                              T2& angAccZ,
                                              T3 X,
                                              T3 Y,
                                              T3 Z,
                                              T4 oriQw,
                                              T4 oriQx,
                                              T4 oriQy,
                                              T4 oriQz,
                                              T5 vX,
                                              T5 vY,
                                              T5 vZ,
                                              T6 omgBarX,
                                              T6 omgBarY,
                                              T6 omgBarZ,
                                              deme::bodyID_t ownerID,
                                              const deme::family_t& family,
                                              const float& t) {
    switch (family) {
        _accPrescriptionStrategy_;
        default:
            return;
    }
}

inline __device__ void integrateVelPos(deme::bodyID_t ownerID,
                                       deme::DEMSimParams* simParams,
                                       deme::DEMDataDT* granData,
                                       float3& v,
                                       float3& omgBar,
                                       float h,
                                       float t) {
    // Acquisition phase...
    deme::family_t family_code = granData->familyID[ownerID];
    bool LinVelXPrescribed = false, LinVelYPrescribed = false, LinVelZPrescribed = false, RotVelXPrescribed = false,
         RotVelYPrescribed = false, RotVelZPrescribed = false;
    bool LinXPrescribed = false, LinYPrescribed = false, LinZPrescribed = false, RotPrescribed = false;
    double X, Y, Z;
    // Keep tab of the old... we'll need that
    float3 old_v = make_float3(granData->vX[ownerID], granData->vY[ownerID], granData->vZ[ownerID]);
    float3 old_omgBar = make_float3(granData->omgBarX[ownerID], granData->omgBarY[ownerID], granData->omgBarZ[ownerID]);

    {
        // Now XYZ gets the old position. We can write them directly back, then it is equivalent to being LinPrescribed.
        voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
            X, Y, Z, granData->voxelID[ownerID], granData->locX[ownerID], granData->locY[ownerID],
            granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        // Do this and we get the `true' pos... Needed for prescription
        X += (double)simParams->LBFX;
        Y += (double)simParams->LBFY;
        Z += (double)simParams->LBFZ;

        // The user may directly change v and omgBar info in global memory in applyPrescribedVel (XYZ and oriQ in this
        // call are read-only)
        applyPrescribedVel(LinVelXPrescribed, LinVelYPrescribed, LinVelZPrescribed, RotVelXPrescribed,
                           RotVelYPrescribed, RotVelZPrescribed, granData->vX[ownerID], granData->vY[ownerID],
                           granData->vZ[ownerID], granData->omgBarX[ownerID], granData->omgBarY[ownerID],
                           granData->omgBarZ[ownerID], X, Y, Z, granData->oriQw[ownerID], granData->oriQx[ownerID],
                           granData->oriQy[ownerID], granData->oriQz[ownerID], ownerID, family_code, (float)t);
        // The user may directly change oriQ info (vX and omgBar in this call are read-only)
        applyPrescribedPos(LinXPrescribed, LinYPrescribed, LinZPrescribed, RotPrescribed, X, Y, Z,
                           granData->oriQw[ownerID], granData->oriQx[ownerID], granData->oriQy[ownerID],
                           granData->oriQz[ownerID], granData->vX[ownerID], granData->vY[ownerID],
                           granData->vZ[ownerID], granData->omgBarX[ownerID], granData->omgBarY[ownerID],
                           granData->omgBarZ[ownerID], ownerID, family_code, (float)t);
    }

    // Operation phase...
    {
        // User's addition of accelerations won't affect acc arrays in global memory; that is, if the user query the
        // contact acceleration, still they don't get the part they applied in this acc prescription
        float3 v_update = make_float3(0, 0, 0), omgBar_update = make_float3(0, 0, 0);
        float3 extra_acc = make_float3(0, 0, 0), extra_angAcc = make_float3(0, 0, 0);
        applyAddedAcceleration(extra_acc.x, extra_acc.y, extra_acc.z, extra_angAcc.x, extra_angAcc.y, extra_angAcc.z, X,
                               Y, Z, granData->oriQw[ownerID], granData->oriQx[ownerID], granData->oriQy[ownerID],
                               granData->oriQz[ownerID], granData->vX[ownerID], granData->vY[ownerID],
                               granData->vZ[ownerID], granData->omgBarX[ownerID], granData->omgBarY[ownerID],
                               granData->omgBarZ[ownerID], ownerID, family_code, (float)t);

        if (!LinVelXPrescribed) {
            v_update.x = (granData->aX[ownerID] + extra_acc.x + simParams->Gx) * h;
            granData->vX[ownerID] += v_update.x;
        } else {
            old_v.x = granData->vX[ownerID];
        }
        if (!LinVelYPrescribed) {
            v_update.y = (granData->aY[ownerID] + extra_acc.y + simParams->Gy) * h;
            granData->vY[ownerID] += v_update.y;
        } else {
            old_v.y = granData->vY[ownerID];
        }
        if (!LinVelZPrescribed) {
            v_update.z = (granData->aZ[ownerID] + extra_acc.z + simParams->Gz) * h;
            granData->vZ[ownerID] += v_update.z;
        } else {
            old_v.z = granData->vZ[ownerID];
        }

        if (!RotVelXPrescribed) {
            omgBar_update.x = (granData->alphaX[ownerID] + extra_angAcc.x) * h;
            granData->omgBarX[ownerID] += omgBar_update.x;
        } else {
            old_omgBar.x = granData->omgBarX[ownerID];
        }
        if (!RotVelYPrescribed) {
            omgBar_update.y = (granData->alphaY[ownerID] + extra_angAcc.y) * h;
            granData->omgBarY[ownerID] += omgBar_update.y;
        } else {
            old_omgBar.y = granData->omgBarY[ownerID];
        }
        if (!RotVelZPrescribed) {
            omgBar_update.z = (granData->alphaZ[ownerID] + extra_angAcc.z) * h;
            granData->omgBarZ[ownerID] += omgBar_update.z;
        } else {
            old_omgBar.z = granData->omgBarZ[ownerID];
        }

        // We need to set v and omgBar, and they will be used in position/quaternion update
        _integrationVelocityPassOnStrategy_;
    }

    // With v and omgBar. update pos now...
    {
        bool wrapped = false;
        // Cylindrical periodicity: reset wrap count for this owner for the current integration step.
        granData->ownerCylWrapK[ownerID] = 0;
        float cos_delta = 1.f;
        float sin_delta = 0.f;
        float cos_half = 1.f;
        float sin_half = 0.f;

        if (!LinXPrescribed) {
            // Impllicitly, pos integration strategy is here
            X += (double)v.x * h;
        }
        if (!LinYPrescribed) {
            Y += (double)v.y * h;
        }
        if (!LinZPrescribed) {
            Z += (double)v.z * h;
        }

        if (simParams->useCylPeriodic &&
            (granData->ownerTypes[ownerID] == deme::OWNER_T_CLUMP ||
             granData->ownerTypes[ownerID] == deme::OWNER_T_MESH)) {
            const float span = simParams->cylPeriodicSpan;
            if (span > 0.f) {
                const float3 origin = simParams->cylPeriodicOrigin;
                const float3 lbf = make_float3(simParams->LBFX, simParams->LBFY, simParams->LBFZ);
                float3 pos_local = make_float3((float)X - lbf.x, (float)Y - lbf.y, (float)Z - lbf.z);
                const float3 pos_global = pos_local - origin;
                const float dist_start = dot(pos_global, simParams->cylPeriodicStartNormal);
                const float dist_end = dot(pos_global, simParams->cylPeriodicEndNormal);
                int k = 0;
                float kf = 0.f;
                // Keep a canonical COM representation in the base wedge.
                // Delaying wrap until the entire owner crosses a seam leaves COMs outside the wedge and creates
                // large branch mismatch windows between kT/dT near the seam.
                const float wrap_eps = 1e-7f;
                if (dist_start < -wrap_eps && dist_end > wrap_eps) {
                    // Extremely rare: outside both planes (large jump). Pick the larger overshoot.
                    if (-dist_start >= dist_end) {
                        k = -1;
                        kf = -1.f;
                    } else {
                        k = 1;
                        kf = 1.f;
                    }
                } else if (dist_start < -wrap_eps) {
                    k = -1;
                    kf = -1.f;
                } else if (dist_end > wrap_eps) {
                    k = 1;
                    kf = 1.f;
                }
                if (k != 0) {
                    granData->ownerCylWrapK[ownerID] = k;
                    if (granData->ownerCylWrapOffset) {
                        granData->ownerCylWrapOffset[ownerID] += k;
                    }
                    const float delta = -kf * span;
                    if (k == 1) {
                        cos_delta = simParams->cylPeriodicCosSpan;
                        sin_delta = -simParams->cylPeriodicSinSpan;
                        cos_half = simParams->cylPeriodicCosHalfSpan;
                        sin_half = -simParams->cylPeriodicSinHalfSpan;
                    } else if (k == -1) {
                        cos_delta = simParams->cylPeriodicCosSpan;
                        sin_delta = simParams->cylPeriodicSinSpan;
                        cos_half = simParams->cylPeriodicCosHalfSpan;
                        sin_half = simParams->cylPeriodicSinHalfSpan;
                    } else {
                        sincosf(delta, &sin_delta, &cos_delta);
                        sincosf(0.5f * delta, &sin_half, &cos_half);
                    }
                    pos_local = cylPeriodicRotate(pos_local, origin, simParams->cylPeriodicAxisVec,
                                                  simParams->cylPeriodicU, simParams->cylPeriodicV, cos_delta,
                                                  sin_delta);
                    X = (double)(pos_local.x + lbf.x);
                    Y = (double)(pos_local.y + lbf.y);
                    Z = (double)(pos_local.z + lbf.z);
                    wrapped = true;
                }

                const float min_r = simParams->cylPeriodicMinRadius;
                if (min_r > 0.f) {
                    pos_local = make_float3((float)X - lbf.x, (float)Y - lbf.y, (float)Z - lbf.z);
                    const float3 pos_rel = pos_local - origin;
                    const float pu = dot(pos_rel, simParams->cylPeriodicU);
                    const float pv = dot(pos_rel, simParams->cylPeriodicV);
                    const float r2 = pu * pu + pv * pv;
                    const float min_r2 = min_r * min_r;
                    if (r2 < min_r2) {
                        const float axial = dot(pos_rel, simParams->cylPeriodicAxisVec);
                        float pu_new = pu;
                        float pv_new = pv;
                        if (r2 > 0.f) {
                            const float scale = min_r / sqrtf(r2);
                            pu_new *= scale;
                            pv_new *= scale;
                        } else {
                            pu_new = min_r;
                            pv_new = 0.f;
                        }
                        const float3 pos_new = origin + simParams->cylPeriodicAxisVec * axial +
                                               simParams->cylPeriodicU * pu_new + simParams->cylPeriodicV * pv_new;
                        X = (double)(pos_new.x + lbf.x);
                        Y = (double)(pos_new.y + lbf.y);
                        Z = (double)(pos_new.z + lbf.z);
                    }
                }
            }
        }

        // Undo the influence of LBF...
        X -= (double)simParams->LBFX;
        Y -= (double)simParams->LBFY;
        Z -= (double)simParams->LBFZ;
        positionToVoxelID<deme::voxelID_t, deme::subVoxelPos_t, double>(
            granData->voxelID[ownerID], granData->locX[ownerID], granData->locY[ownerID], granData->locZ[ownerID], X, Y,
            Z, _nvXp2_, _nvYp2_, _voxelSize_, _l_);

        if (!RotPrescribed) {
            // Then integrate the quaternion
            // 1st Taylor series multiplier. First use it to record delta rotation...
            // Refer to
            // https://stackoverflow.com/questions/24197182/efficient-quaternion-angular-velocity/24201879#24201879
            const float3 ha = 0.5 * h * omgBar;
            float4 oriQ = make_float4(ha.x, ha.y, ha.z, 1.0);  // xyzw
            // Note: Yes it is Quat * deltaRot, not the other way around. Then store result in oriQ.
            HamiltonProduct(oriQ.w, oriQ.x, oriQ.y, oriQ.z, granData->oriQw[ownerID], granData->oriQx[ownerID],
                            granData->oriQy[ownerID], granData->oriQz[ownerID], oriQ.w, oriQ.x, oriQ.y, oriQ.z);
            // Normalizing it is essential. Note even if you use an exp map to update quaternion, you still need to
            // normalize.
            oriQ /= length(oriQ);
            granData->oriQw[ownerID] = oriQ.w;
            granData->oriQx[ownerID] = oriQ.x;
            granData->oriQy[ownerID] = oriQ.y;
            granData->oriQz[ownerID] = oriQ.z;
        }

        if (wrapped) {
            float3 vel = make_float3(granData->vX[ownerID], granData->vY[ownerID], granData->vZ[ownerID]);
            vel = cylPeriodicRotate(vel, make_float3(0.f, 0.f, 0.f), simParams->cylPeriodicAxisVec,
                                    simParams->cylPeriodicU, simParams->cylPeriodicV, cos_delta, sin_delta);
            granData->vX[ownerID] = vel.x;
            granData->vY[ownerID] = vel.y;
            granData->vZ[ownerID] = vel.z;

            float3 omg = make_float3(granData->omgBarX[ownerID], granData->omgBarY[ownerID], granData->omgBarZ[ownerID]);
            omg = cylPeriodicRotate(omg, make_float3(0.f, 0.f, 0.f), simParams->cylPeriodicAxisVec,
                                    simParams->cylPeriodicU, simParams->cylPeriodicV, cos_delta, sin_delta);
            granData->omgBarX[ownerID] = omg.x;
            granData->omgBarY[ownerID] = omg.y;
            granData->omgBarZ[ownerID] = omg.z;

            float4 rotQ = make_float4(simParams->cylPeriodicAxisVec.x * sin_half,
                                      simParams->cylPeriodicAxisVec.y * sin_half,
                                      simParams->cylPeriodicAxisVec.z * sin_half, cos_half);
            float4 oriQ = make_float4(granData->oriQx[ownerID], granData->oriQy[ownerID], granData->oriQz[ownerID],
                                      granData->oriQw[ownerID]);
            HamiltonProduct(oriQ.w, oriQ.x, oriQ.y, oriQ.z, rotQ.w, rotQ.x, rotQ.y, rotQ.z, oriQ.w, oriQ.x, oriQ.y,
                            oriQ.z);
            oriQ /= length(oriQ);
            granData->oriQw[ownerID] = oriQ.w;
            granData->oriQx[ownerID] = oriQ.x;
            granData->oriQy[ownerID] = oriQ.y;
            granData->oriQz[ownerID] = oriQ.z;
        }
    }
}

// inline __device__ void locateNewVoxel(deme::voxelID_t& voxel, int64_t& locX_tmp, int64_t& locY_tmp, int64_t&
// locZ_tmp) {
//     deme::voxelID_t voxelX;
//     deme::voxelID_t voxelY;
//     deme::voxelID_t voxelZ;
//     IDChopper<deme::voxelID_t, deme::voxelID_t>(voxelX, voxelY, voxelZ, voxel, _nvXp2_, _nvYp2_);

//     // MAX_SUBVOXEL is int64 and large enough to handle VOXEL_RES_POWER2 == 16 or 32
//     voxelX += div_floor<int64_t, int64_t>(locX_tmp, deme::MAX_SUBVOXEL);
//     voxelY += div_floor<int64_t, int64_t>(locY_tmp, deme::MAX_SUBVOXEL);
//     voxelZ += div_floor<int64_t, int64_t>(locZ_tmp, deme::MAX_SUBVOXEL);
//     locX_tmp = mod_floor<int64_t, int64_t>(locX_tmp, deme::MAX_SUBVOXEL);
//     locY_tmp = mod_floor<int64_t, int64_t>(locY_tmp, deme::MAX_SUBVOXEL);
//     locZ_tmp = mod_floor<int64_t, int64_t>(locZ_tmp, deme::MAX_SUBVOXEL);

//     IDPacker<deme::voxelID_t, deme::voxelID_t>(voxel, voxelX, voxelY, voxelZ, _nvXp2_, _nvYp2_);
// }

DEME_KERNEL void integrateOwners(deme::DEMSimParams* simParams, deme::DEMDataDT* granData, double timeElapsed) {
    deme::bodyID_t ownerID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ownerID < simParams->nOwnerBodies) {
        // These 2 quantities mean the velocity and ang vel used for updating position/quaternion for this step.
        // Depending on the integration scheme in use, they can be different.
        float3 v, omgBar;
        integrateVelPos(ownerID, simParams, granData, v, omgBar, simParams->dyn.h, (float)timeElapsed);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        simParams->dyn.timeElapsed = timeElapsed + (double)simParams->dyn.h;
    }
}



// Rotate selected per-contact wildcard vector triplets when an owner wraps across the cylindrical periodic seam.
// This preserves contact-history continuity (e.g., tangential displacement vectors) under the periodic mapping.
DEME_KERNEL void cylPeriodicRotateContactWildcards(deme::DEMSimParams* simParams,
                                                  deme::DEMDataDT* granData,
                                                  size_t nContacts) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nContacts) {
        return;
    }
    if (!simParams->useCylPeriodic || simParams->nCylPeriodicWCTriplets == 0) {
        return;
    }
    // Contact wildcards are indexed in patch-contact space (myPatchContactID).
    // Therefore this kernel must operate strictly on patch-contact slots.
    deme::bodyID_t ownerA = deme::NULL_BODYID;
    deme::bodyID_t ownerB = deme::NULL_BODYID;

    const deme::contact_t cntTypePatch =
        granData->contactTypePatch ? granData->contactTypePatch[tid] : deme::NOT_A_CONTACT;
    if (cntTypePatch == deme::NOT_A_CONTACT) {
        return;
    }
    const deme::bodyID_t patchA = granData->idPatchA[tid];
    const deme::bodyID_t patchB = granData->idPatchB[tid];
    ownerA = DEME_GET_PATCH_OWNER_ID(patchA, deme::decodeTypeA(cntTypePatch));
    ownerB = DEME_GET_PATCH_OWNER_ID(patchB, deme::decodeTypeB(cntTypePatch));

    int kA = 0;
    int kB = 0;
    const bool ownerA_valid = (ownerA != deme::NULL_BODYID) && (ownerA < simParams->nOwnerBodies);
    const bool ownerB_valid = (ownerB != deme::NULL_BODYID) && (ownerB < simParams->nOwnerBodies);
    if (ownerA_valid) {
        kA = granData->ownerCylWrapK[ownerA];
    }
    if (ownerB_valid) {
        kB = granData->ownerCylWrapK[ownerB];
    }

    if (kA == 0 && kB == 0) {
        return;
    }
    // If both owners wrapped in this step and their wrap counts differ, rotating the contact history is
    // ambiguous (the pair did not undergo a single rigid transform). In that case, reset wildcard triplets.
    if (kA != 0 && kB != 0 && kA != kB) {
        for (unsigned int t = 0; t < simParams->nCylPeriodicWCTriplets; t++) {
            const int3 ijk = simParams->cylPeriodicWCTriplets[t];
            granData->contactWildcards[ijk.x][tid] = 0.f;
            granData->contactWildcards[ijk.y][tid] = 0.f;
            granData->contactWildcards[ijk.z][tid] = 0.f;
        }
        return;
    }
    // Otherwise, the pair experienced a single rigid wrap transform; rotate the history accordingly.
    const int k = (kA != 0) ? kA : kB;

    const float span = simParams->cylPeriodicSpan;
    const float delta = -((float)k) * span;

    float cos_delta, sin_delta;
    // Fast-path for single-span wraps.
    if (k == 1) {
        cos_delta = simParams->cylPeriodicCosSpan;
        sin_delta = -simParams->cylPeriodicSinSpan;
    } else if (k == -1) {
        cos_delta = simParams->cylPeriodicCosSpan;
        sin_delta = simParams->cylPeriodicSinSpan;
    } else {
        sincosf(delta, &sin_delta, &cos_delta);
    }

    // Rotate all registered wildcard vector triplets in-place.
    const float3 origin = make_float3(0.f);
    for (unsigned int t = 0; t < simParams->nCylPeriodicWCTriplets; t++) {
        const int3 ijk = simParams->cylPeriodicWCTriplets[t];
        float3 v = make_float3(granData->contactWildcards[ijk.x][tid],
                               granData->contactWildcards[ijk.y][tid],
                               granData->contactWildcards[ijk.z][tid]);
        v = cylPeriodicRotate(v, origin, simParams->cylPeriodicAxisVec, simParams->cylPeriodicU, simParams->cylPeriodicV,
                              cos_delta, sin_delta);
        granData->contactWildcards[ijk.x][tid] = v.x;
        granData->contactWildcards[ijk.y][tid] = v.y;
        granData->contactWildcards[ijk.z][tid] = v.z;
    }
}
