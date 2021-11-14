// DEM integration related custom kernels
#include <granular/DataStructs.h>
#include <kernel/DEMHelperKernels.cu>
#include <granular/GranularDefines.h>

// For now, write a custom kernel (instead of cub-based), and change it later
inline __device__ void integrateLinVel(unsigned int thisClump,
                                       sgps::DEMSimParams* simParams,
                                       sgps::DEMDataDT* granData) {
    // Explicit update
    granData->hvX[thisClump] += granData->h2aX[thisClump];
    granData->hvY[thisClump] += granData->h2aY[thisClump];
    granData->hvZ[thisClump] += granData->h2aZ[thisClump];
}

inline __device__ void locateNewVoxel(sgps::voxelID_t& voxel,
                                      int64_t& locX_tmp,
                                      int64_t& locY_tmp,
                                      int64_t& locZ_tmp,
                                      sgps::DEMSimParams* simParams) {
    int max_loc = (1 << VOXEL_RES_POWER2);
    sgps::voxelID_t voxelX;
    sgps::voxelID_t voxelY;
    sgps::voxelID_t voxelZ;
    IDChopper<sgps::voxelID_t, sgps::voxelID_t>(voxelX, voxelY, voxelZ, voxel, simParams->nvXp2, simParams->nvYp2);

    voxelX += div_floor<int64_t, int>(locX_tmp, max_loc);
    voxelY += div_floor<int64_t, int>(locY_tmp, max_loc);
    voxelZ += div_floor<int64_t, int>(locZ_tmp, max_loc);
    locX_tmp = mod_floor<int64_t, int>(locX_tmp, max_loc);
    locY_tmp = mod_floor<int64_t, int>(locY_tmp, max_loc);
    locZ_tmp = mod_floor<int64_t, int>(locZ_tmp, max_loc);

    // TODO: Should add a check here where, if negative voxel component spotted, stop the simulation

    IDPacker<sgps::voxelID_t, sgps::voxelID_t>(voxel, voxelX, voxelY, voxelZ, simParams->nvXp2, simParams->nvYp2);
}

inline __device__ void integrateLinPos(unsigned int thisClump,
                                       sgps::DEMSimParams* simParams,
                                       sgps::DEMDataDT* granData) {
    // Location accuracy is up to integer level anyway
    int64_t locX_tmp = (int64_t)granData->locX[thisClump] + granData->hvX[thisClump];
    int64_t locY_tmp = (int64_t)granData->locY[thisClump] + granData->hvY[thisClump];
    int64_t locZ_tmp = (int64_t)granData->locZ[thisClump] + granData->hvZ[thisClump];
    sgps::voxelID_t newVoxel = granData->voxelID[thisClump];
    locateNewVoxel(newVoxel, locX_tmp, locY_tmp, locZ_tmp, simParams);
    granData->voxelID[thisClump] = newVoxel;
    granData->locX[thisClump] = locX_tmp;
    granData->locY[thisClump] = locY_tmp;
    granData->locZ[thisClump] = locZ_tmp;
}

__global__ void integrateClumps(sgps::DEMSimParams* simParams, sgps::DEMDataDT* granData) {
    unsigned int thisClump = blockIdx.x * blockDim.x + threadIdx.x;
    if (thisClump < simParams->nClumpBodies) {
        integrateLinVel(thisClump, simParams, granData);
        integrateLinPos(thisClump, simParams, granData);
    }
}
