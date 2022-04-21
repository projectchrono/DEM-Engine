// DEM integration related custom kernels
#include <granular/DataStructs.h>
#include <kernel/DEMHelperKernels.cu>
#include <granular/GranularDefines.h>
#include <kernel/DEMPrescribedIntegrationKernels.cu>

// For now, write a custom kernel (instead of cub-based), and change it later
inline __device__ void integrateVel(sgps::bodyID_t thisClump, sgps::DEMDataDT* granData, double h) {
    // Even prescribed motion should leverage custom integrators, so we put the prescription condition at a ``inner''
    // location.
    sgps::family_t family_code = granData->familyID[thisClump];
    bool LinPrescribed, RotPrescribed;
    applyPrescribedVel<float, float>(LinPrescribed, RotPrescribed, granData->vX[thisClump], granData->vY[thisClump],
                                     granData->vZ[thisClump], granData->omgBarX[thisClump],
                                     granData->omgBarY[thisClump], granData->omgBarZ[thisClump], family_code);

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

inline __device__ void integratePos(sgps::bodyID_t thisClump, sgps::DEMDataDT* granData, double h) {
    // Location accuracy is up to integer level anyway
    int64_t locX_tmp = (int64_t)granData->locX[thisClump];
    int64_t locY_tmp = (int64_t)granData->locY[thisClump];
    int64_t locZ_tmp = (int64_t)granData->locZ[thisClump];

    sgps::family_t family_code = granData->familyID[thisClump];
    bool LinPrescribed, RotPrescribed;
    applyPrescribedPos<int64_t, sgps::oriQ_t>(LinPrescribed, RotPrescribed, locX_tmp, locY_tmp, locZ_tmp,
                                              granData->oriQ0[thisClump], granData->oriQ1[thisClump],
                                              granData->oriQ2[thisClump], granData->oriQ3[thisClump], family_code);

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
        if (len > 1.0e-12) {
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

__global__ void integrateClumps(sgps::DEMDataDT* granData, double h) {
    sgps::bodyID_t thisClump = blockIdx.x * blockDim.x + threadIdx.x;
    if (thisClump < _nOwnerBodies_) {
        integrateVel(thisClump, granData, h);
        integratePos(thisClump, granData, h);
    }
}
