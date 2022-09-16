// DEM integration related custom kernels
#include <kernel/DEMHelperKernels.cu>
#include <DEM/Defines.h>

// Apply presecibed velocity and report whether the `true' physics should be skipped, rather than added on top of that
template <typename T1, typename T2>
inline __device__ void applyPrescribedVel(bool& LinPrescribed,
                                          bool& RotPrescribed,
                                          T1& vX,
                                          T1& vY,
                                          T1& vZ,
                                          T2& omgBarX,
                                          T2& omgBarY,
                                          T2& omgBarZ,
                                          const deme::family_t& family,
                                          const float& t) {
    switch (family) {
        _velPrescriptionStrategy_;
        default:
            LinPrescribed = false;
            RotPrescribed = false;
    }
}

// Apply presecibed location and report whether the `true' physics should be skipped, rather than added on top of that
template <typename T1, typename T2>
inline __device__ void applyPrescribedPos(bool& LinPrescribed,
                                          bool& RotPrescribed,
                                          T1& X,
                                          T1& Y,
                                          T1& Z,
                                          T2& oriQw,
                                          T2& oriQx,
                                          T2& oriQy,
                                          T2& oriQz,
                                          const deme::family_t& family,
                                          const float& t) {
    switch (family) {
        _posPrescriptionStrategy_;
        default:
            LinPrescribed = false;
            RotPrescribed = false;
    }
}

inline __device__ void integrateVel(deme::bodyID_t thisClump,
                                    deme::DEMSimParams* simParams,
                                    deme::DEMDataDT* granData,
                                    float3& v,
                                    float3& omgBar,
                                    float h,
                                    float t) {
    deme::family_t family_code = granData->familyID[thisClump];
    bool LinPrescribed = false, RotPrescribed = false;

    // Keep tab of the old... we'll need that
    float3 old_v = make_float3(granData->vX[thisClump], granData->vY[thisClump], granData->vZ[thisClump]);
    float3 old_omgBar =
        make_float3(granData->omgBarX[thisClump], granData->omgBarY[thisClump], granData->omgBarZ[thisClump]);

    // The user may directly change and omgBar info in global memory in applyPrescribedVel
    applyPrescribedVel<float, float>(LinPrescribed, RotPrescribed, granData->vX[thisClump], granData->vY[thisClump],
                                     granData->vZ[thisClump], granData->omgBarX[thisClump],
                                     granData->omgBarY[thisClump], granData->omgBarZ[thisClump], family_code, (float)t);

    float3 v_update = make_float3(0, 0, 0), omgBar_update = make_float3(0, 0, 0);

    if (!LinPrescribed) {
        v_update.x = (granData->aX[thisClump] + simParams->Gx) * h;
        v_update.y = (granData->aY[thisClump] + simParams->Gy) * h;
        v_update.z = (granData->aZ[thisClump] + simParams->Gz) * h;
        granData->vX[thisClump] += v_update.x;
        granData->vY[thisClump] += v_update.y;
        granData->vZ[thisClump] += v_update.z;
    }

    if (!RotPrescribed) {
        omgBar_update.x = granData->alphaX[thisClump] * h;
        omgBar_update.y = granData->alphaY[thisClump] * h;
        omgBar_update.z = granData->alphaZ[thisClump] * h;
        granData->omgBarX[thisClump] += omgBar_update.x;
        granData->omgBarY[thisClump] += omgBar_update.y;
        granData->omgBarZ[thisClump] += omgBar_update.z;
    }

    // We need to set v and omgBar, and they will be used in position/quaternion update
    _integrationVelocityPassOnStrategy_;
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

inline __device__ void integratePos(deme::bodyID_t thisClump,
                                    deme::DEMDataDT* granData,
                                    float3 v,
                                    float3 omgBar,
                                    float h,
                                    float t) {
    // This block is not needed, with our current way of integration...
    // int64_t locX_tmp = (int64_t)granData->locX[thisClump];
    // int64_t locY_tmp = (int64_t)granData->locY[thisClump];
    // int64_t locZ_tmp = (int64_t)granData->locZ[thisClump];
    // locateNewVoxel(newVoxel, locX_tmp, locY_tmp, locZ_tmp);
    // locX_tmp += (int64_t)((double)v.x / _l_ * h);
    // locY_tmp += (int64_t)((double)v.y / _l_ * h);
    // locZ_tmp += (int64_t)((double)v.z / _l_ * h);

    double X, Y, Z;
    // Now XYZ gets the old position. We can write them directly back, then it is equivalent to being LinPrescribed.
    voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
        X, Y, Z, granData->voxelID[thisClump], granData->locX[thisClump], granData->locY[thisClump],
        granData->locZ[thisClump], _nvXp2_, _nvYp2_, _voxelSize_, _l_);

    deme::family_t family_code = granData->familyID[thisClump];
    bool LinPrescribed = false, RotPrescribed = false;
    applyPrescribedPos<double, deme::oriQ_t>(LinPrescribed, RotPrescribed, X, Y, Z, granData->oriQw[thisClump],
                                             granData->oriQx[thisClump], granData->oriQy[thisClump],
                                             granData->oriQz[thisClump], family_code, (float)t);

    if (!LinPrescribed) {
        // Pos integration strategy here
        X += (double)v.x * h;
        Y += (double)v.y * h;
        Z += (double)v.z * h;
    }
    positionToVoxelID<deme::voxelID_t, deme::subVoxelPos_t, double>(
        granData->voxelID[thisClump], granData->locX[thisClump], granData->locY[thisClump], granData->locZ[thisClump],
        X, Y, Z, _nvXp2_, _nvYp2_, _voxelSize_, _l_);

    if (!RotPrescribed) {
        // Then integrate the quaternion
        // Exp map-based rotation angle calculation
        double deltaQ0 = 1;
        double deltaQ1 = omgBar.x;
        double deltaQ2 = omgBar.y;
        double deltaQ3 = omgBar.z;
        double len = sqrt(deltaQ1 * deltaQ1 + deltaQ2 * deltaQ2 + deltaQ3 * deltaQ3);
        double theta = 0.5 * h * len;  // 0.5*dt*len, delta rotation
        if (len > 0) {
            deltaQ0 = cos(theta);
            double s = sin(theta) / len;
            deltaQ1 *= s;
            deltaQ2 *= s;
            deltaQ3 *= s;
        }
        // Note: Yes it is Quat * deltaRot, not the other way around. Also, Hamilton product should automatically
        // maintain the unit-ness of quaternions.
        HamiltonProduct<float>(granData->oriQw[thisClump], granData->oriQx[thisClump], granData->oriQy[thisClump],
                               granData->oriQz[thisClump], granData->oriQw[thisClump], granData->oriQx[thisClump],
                               granData->oriQy[thisClump], granData->oriQz[thisClump], deltaQ0, deltaQ1, deltaQ2,
                               deltaQ3);
    }
}

__global__ void integrateOwners(deme::DEMSimParams* simParams, deme::DEMDataDT* granData) {
    deme::bodyID_t thisClump = blockIdx.x * blockDim.x + threadIdx.x;
    if (thisClump < simParams->nOwnerBodies) {
        // These 2 quantities mean the velocity and ang vel used for updating position/quaternion for this step.
        // Depending on the integration scheme in use, they can be different.
        float3 v, omgBar;
        integrateVel(thisClump, simParams, granData, v, omgBar, simParams->h, simParams->timeElapsed);
        integratePos(thisClump, granData, v, omgBar, simParams->h, simParams->timeElapsed);
    }
}
