// DEM integration related custom kernels
#include <kernel/DEMHelperKernels.cu>
#include <DEM/DEMDefines.h>

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
                                          const sgps::family_t& family,
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
                                          T2& ori0,
                                          T2& ori1,
                                          T2& ori2,
                                          T2& ori3,
                                          const sgps::family_t& family,
                                          const float& t) {
    switch (family) {
        _posPrescriptionStrategy_;
        default:
            LinPrescribed = false;
            RotPrescribed = false;
    }
}

// For now, write a custom kernel (instead of cub-based), and change it later
inline __device__ void integrateVel(sgps::bodyID_t thisClump, sgps::DEMDataDT* granData, float h, float t) {
    // Even prescribed motion should leverage custom integrators, so we put the prescription condition at a ``inner''
    // location.
    sgps::family_t family_code = granData->familyID[thisClump];
    bool LinPrescribed = false, RotPrescribed = false;
    applyPrescribedVel<float, float>(LinPrescribed, RotPrescribed, granData->vX[thisClump], granData->vY[thisClump],
                                     granData->vZ[thisClump], granData->omgBarX[thisClump],
                                     granData->omgBarY[thisClump], granData->omgBarZ[thisClump], family_code, (float)t);

    if (!LinPrescribed) {
        granData->vX[thisClump] += granData->aX[thisClump] * h;
        granData->vY[thisClump] += granData->aY[thisClump] * h;
        granData->vZ[thisClump] += granData->aZ[thisClump] * h;
    }

    if (!RotPrescribed) {
        granData->omgBarX[thisClump] += granData->alphaX[thisClump] * h;
        granData->omgBarY[thisClump] += granData->alphaY[thisClump] * h;
        granData->omgBarZ[thisClump] += granData->alphaZ[thisClump] * h;
    }
}

inline __device__ void locateNewVoxel(sgps::voxelID_t& voxel, int64_t& locX_tmp, int64_t& locY_tmp, int64_t& locZ_tmp) {
    sgps::voxelID_t voxelX;
    sgps::voxelID_t voxelY;
    sgps::voxelID_t voxelZ;
    IDChopper<sgps::voxelID_t, sgps::voxelID_t>(voxelX, voxelY, voxelZ, voxel, _nvXp2_, _nvYp2_);

    // DEM_MAX_SUBVOXEL is int64 and large enough to handle DEM_VOXEL_RES_POWER2 == 16 or 32
    voxelX += div_floor<int64_t, int64_t>(locX_tmp, sgps::DEM_MAX_SUBVOXEL);
    voxelY += div_floor<int64_t, int64_t>(locY_tmp, sgps::DEM_MAX_SUBVOXEL);
    voxelZ += div_floor<int64_t, int64_t>(locZ_tmp, sgps::DEM_MAX_SUBVOXEL);
    locX_tmp = mod_floor<int64_t, int64_t>(locX_tmp, sgps::DEM_MAX_SUBVOXEL);
    locY_tmp = mod_floor<int64_t, int64_t>(locY_tmp, sgps::DEM_MAX_SUBVOXEL);
    locZ_tmp = mod_floor<int64_t, int64_t>(locZ_tmp, sgps::DEM_MAX_SUBVOXEL);

    // TODO: Should add a check here where, if negative voxel component spotted, stop the simulation

    IDPacker<sgps::voxelID_t, sgps::voxelID_t>(voxel, voxelX, voxelY, voxelZ, _nvXp2_, _nvYp2_);
}

inline __device__ void integratePos(sgps::bodyID_t thisClump, sgps::DEMDataDT* granData, float h, float t) {
    // Location accuracy is up to integer level anyway
    int64_t locX_tmp = (int64_t)granData->locX[thisClump];
    int64_t locY_tmp = (int64_t)granData->locY[thisClump];
    int64_t locZ_tmp = (int64_t)granData->locZ[thisClump];

    sgps::family_t family_code = granData->familyID[thisClump];
    bool LinPrescribed = false, RotPrescribed = false;
    applyPrescribedPos<int64_t, sgps::oriQ_t>(
        LinPrescribed, RotPrescribed, locX_tmp, locY_tmp, locZ_tmp, granData->oriQ0[thisClump],
        granData->oriQ1[thisClump], granData->oriQ2[thisClump], granData->oriQ3[thisClump], family_code, (float)t);

    if (!LinPrescribed) {
        locX_tmp += (int64_t)((double)granData->vX[thisClump] / _l_ * h);
        locY_tmp += (int64_t)((double)granData->vY[thisClump] / _l_ * h);
        locZ_tmp += (int64_t)((double)granData->vZ[thisClump] / _l_ * h);
        sgps::voxelID_t newVoxel = granData->voxelID[thisClump];
        locateNewVoxel(newVoxel, locX_tmp, locY_tmp, locZ_tmp);
        granData->voxelID[thisClump] = newVoxel;
        granData->locX[thisClump] = locX_tmp;
        granData->locY[thisClump] = locY_tmp;
        granData->locZ[thisClump] = locZ_tmp;
    }

    if (!RotPrescribed) {
        // Then integrate the quaternion
        // Exp map-based rotation angle calculation
        double deltaQ0;
        double deltaQ1 = granData->omgBarX[thisClump];
        double deltaQ2 = granData->omgBarY[thisClump];
        double deltaQ3 = granData->omgBarZ[thisClump];
        double len = sqrt(deltaQ1 * deltaQ1 + deltaQ2 * deltaQ2 + deltaQ3 * deltaQ3);
        // TODO: Why not just store omgBar, not hOmgBar (it's even better for integer representation purposes)
        // TODO: Talk with Dan about this (is he insisting this because it has implications in rA?)
        double theta = 0.5 * h * len;  // 0.5*dt*len, delta rotation
        if (len > SGPS_DEM_TINY_FLOAT) {
            deltaQ0 = cos(theta);
            double s = sin(theta) / len;
            deltaQ1 *= s;
            deltaQ2 *= s;
            deltaQ3 *= s;
        } else {
            deltaQ0 = 1;
            deltaQ1 = 0;
            deltaQ2 = 0;
            deltaQ3 = 0;
        }
        // Hamilton product should maintain the unit-ness of quaternions
        HamiltonProduct<float>(granData->oriQ0[thisClump], granData->oriQ1[thisClump], granData->oriQ2[thisClump],
                               granData->oriQ3[thisClump], deltaQ0, deltaQ1, deltaQ2, deltaQ3);
    }
}

__global__ void integrateClumps(sgps::DEMDataDT* granData, float h, float t) {
    sgps::bodyID_t thisClump = blockIdx.x * blockDim.x + threadIdx.x;
    if (thisClump < _nOwnerBodies_) {
        integrateVel(thisClump, granData, h, t);
        integratePos(thisClump, granData, h, t);
    }
}
