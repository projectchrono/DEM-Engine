// DEM device-side helper kernel collection
#include <helper_math.cuh>
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>

// inline __device__ double3 voxelID2LRFPosition
// inline __device__ voxelID_default_t position2VoxelID

// Sign function
template <typename T>
inline __device__ int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

// Integer division that rounds towards -infty
template <typename T1, typename T2>
inline __device__ T1 div_floor(T1 a, T2 b) {
    T1 res = a / b;
    T1 rem = a % b;
    // Correct division result downwards if up-rounding happened,
    // (for non-zero remainder of sign different than the divisor).
    T1 corr = (rem != 0 && ((rem < 0) != (b < 0)));
    return res - corr;
}

// Modulus that rounds towards -infty
template <typename T1, typename T2>
inline __device__ T1 mod_floor(T1 a, T2 b) {
    if (b < 0)  // you can check for b == 0 separately and do what you want
        return -mod_floor(-a, -b);
    T1 ret = a % b;
    if (ret < 0)
        ret += b;
    return ret;
}

// Chops voxelID into XYZ components
inline __device__ void voxelIDChopper(sgps::voxelID_default_t& X,
                                      sgps::voxelID_default_t& Y,
                                      sgps::voxelID_default_t& Z,
                                      sgps::voxelID_default_t& voxelID,
                                      sgps::DEMSimParams* simParams) {
    X = voxelID & (((sgps::voxelID_default_t)1 << simParams->nvXp2) - 1);  // & operation here equals modulo
    Y = (voxelID >> simParams->nvXp2) & (((sgps::voxelID_default_t)1 << simParams->nvYp2) - 1);
    Z = (voxelID) >> (simParams->nvXp2 + simParams->nvYp2);
}

// Packs XYZ components back to voxelID
inline __device__ void voxelIDPacker(sgps::voxelID_default_t& voxelID,
                                     sgps::voxelID_default_t& X,
                                     sgps::voxelID_default_t& Y,
                                     sgps::voxelID_default_t& Z,
                                     sgps::DEMSimParams* simParams) {
    voxelID = 0;
    voxelID += X;
    voxelID += Y << simParams->nvXp2;
    voxelID += Z << (simParams->nvXp2 + simParams->nvYp2);
}
