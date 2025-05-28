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

__global__ void integrateOwners(deme::DEMSimParams* simParams, deme::DEMDataDT* granData) {
    deme::bodyID_t ownerID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ownerID < simParams->nOwnerBodies) {
        // These 2 quantities mean the velocity and ang vel used for updating position/quaternion for this step.
        // Depending on the integration scheme in use, they can be different.
        float3 v, omgBar;
        integrateVelPos(ownerID, simParams, granData, v, omgBar, (float)simParams->h, (float)simParams->timeElapsed);
    }
}
