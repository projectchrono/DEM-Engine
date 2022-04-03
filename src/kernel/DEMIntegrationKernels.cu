// DEM integration related custom kernels
#include <granular/DataStructs.h>
#include <kernel/DEMHelperKernels.cu>
#include <granular/GranularDefines.h>
#include <kernel/DEMPrescribedIntegrationKernels.cu>

// For now, write a custom kernel (instead of cub-based), and change it later
inline __device__ void integrateVel(sgps::bodyID_t thisClump, sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData) {
    // Even prescribed motion should leverage custom integrators, so we put the prescription condition at a ``inner'' location.
    sgps::family_t family_code = granData->familyID[thisClump];
    bool LinPrescribed, RotPrescribed;
    applyPrescribedVel<float,float>(LinPrescribed, RotPrescribed,granData->hvX[thisClump],granData->hvY[thisClump],granData->hvZ[thisClump],granData->hOmgBarX[thisClump],granData->hOmgBarY[thisClump],granData->hOmgBarZ[thisClump],family_code);

    if (!LinPrescribed) {
    granData->hvX[thisClump] += granData->h2aX[thisClump];
    granData->hvY[thisClump] += granData->h2aY[thisClump];
    granData->hvZ[thisClump] += granData->h2aZ[thisClump];
    }

    if (!RotPrescribed) {
    granData->hOmgBarX[thisClump] += granData->h2AlphaX[thisClump];
    granData->hOmgBarY[thisClump] += granData->h2AlphaY[thisClump];
    granData->hOmgBarZ[thisClump] += granData->h2AlphaZ[thisClump];
    }
}

inline __device__ void locateNewVoxel(sgps::voxelID_t& voxel,
                                      int64_t& locX_tmp,
                                      int64_t& locY_tmp,
                                      int64_t& locZ_tmp,
                                      sgps::DEMSimParams* simParams) {
    // can handle VOXEL_RES_POWER2 == 16 or 32
    int64_t max_loc = ((int64_t)1 << sgps::VOXEL_RES_POWER2);
    sgps::voxelID_t voxelX;
    sgps::voxelID_t voxelY;
    sgps::voxelID_t voxelZ;
    IDChopper<sgps::voxelID_t, sgps::voxelID_t>(voxelX, voxelY, voxelZ, voxel, simParams->nvXp2, simParams->nvYp2);

    voxelX += div_floor<int64_t, int64_t>(locX_tmp, max_loc);
    voxelY += div_floor<int64_t, int64_t>(locY_tmp, max_loc);
    voxelZ += div_floor<int64_t, int64_t>(locZ_tmp, max_loc);
    locX_tmp = mod_floor<int64_t, int64_t>(locX_tmp, max_loc);
    locY_tmp = mod_floor<int64_t, int64_t>(locY_tmp, max_loc);
    locZ_tmp = mod_floor<int64_t, int64_t>(locZ_tmp, max_loc);

    // TODO: Should add a check here where, if negative voxel component spotted, stop the simulation

    IDPacker<sgps::voxelID_t, sgps::voxelID_t>(voxel, voxelX, voxelY, voxelZ, simParams->nvXp2, simParams->nvYp2);
}

inline __device__ void integratePos(sgps::bodyID_t thisClump, sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData) {
    // Location accuracy is up to integer level anyway
    int64_t locX_tmp = (int64_t)granData->locX[thisClump];
    int64_t locY_tmp = (int64_t)granData->locY[thisClump];
    int64_t locZ_tmp = (int64_t)granData->locZ[thisClump];
    
    sgps::family_t family_code = granData->familyID[thisClump];
    bool LinPrescribed, RotPrescribed;
    applyPrescribedPos<int64_t,sgps::oriQ_t>(LinPrescribed, RotPrescribed,locX_tmp,locY_tmp,locZ_tmp,granData->oriQ0[thisClump],granData->oriQ1[thisClump],granData->oriQ2[thisClump],granData->oriQ3[thisClump],family_code);

    if (!LinPrescribed) {
    locX_tmp += granData->hvX[thisClump];
    locY_tmp += granData->hvY[thisClump];
    locZ_tmp += granData->hvZ[thisClump];
    sgps::voxelID_t newVoxel = granData->voxelID[thisClump];
    locateNewVoxel(newVoxel, locX_tmp, locY_tmp, locZ_tmp, simParams);
    granData->voxelID[thisClump] = newVoxel;
    granData->locX[thisClump] = locX_tmp;
    granData->locY[thisClump] = locY_tmp;
    granData->locZ[thisClump] = locZ_tmp;
    }

    if (!RotPrescribed) {
    // Then integrate the quaternion
    // Exp map-based rotation angle calculation
    double deltaQ0;
    double deltaQ1 = granData->hOmgBarX[thisClump];
    double deltaQ2 = granData->hOmgBarY[thisClump];
    double deltaQ3 = granData->hOmgBarZ[thisClump];
    double len = sqrt(deltaQ1 * deltaQ1 + deltaQ2 * deltaQ2 + deltaQ3 * deltaQ3);
    // TODO: Why not just store omgBar, not hOmgBar (it's even better for integer representation purposes)
    // TODO: Talk with Dan about this (is he insisting this because it has implications in rA?)
    double theta = len * 0.5;  // 0.5*dt*len, but dt is already included in hOmgBar
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

__global__ void integrateClumps(sgps::DEMSimParams* simParams,
                                sgps::DEMDataDT* granData,
                                sgps::DEMTemplate* granTemplates) {
    sgps::bodyID_t thisClump = blockIdx.x * blockDim.x + threadIdx.x;
    if (thisClump < simParams->nClumpBodies) {
        integrateVel(thisClump, simParams, granData);
        integratePos(thisClump, simParams, granData);
    }
}
