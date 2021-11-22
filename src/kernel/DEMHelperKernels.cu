// DEM device-side helper kernel collection
#include <helper_math.cuh>
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>

// inline __device__ double3 voxelID2LRFPosition
// inline __device__ voxelID_t position2VoxelID

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

// Chops a long ID (typically voxelID) into XYZ components
template <typename T1, typename T2>
inline __device__ void IDChopper(T1& X, T1& Y, T1& Z, T2& ID, unsigned char& nvXp2, unsigned char& nvYp2) {
    X = ID & (((T1)1 << nvXp2) - 1);  // & operation here equals modulo
    Y = (ID >> nvXp2) & (((T1)1 << nvYp2) - 1);
    Z = (ID) >> (nvXp2 + nvYp2);
}

// Packs XYZ components back to a long ID (typically voxelID)
template <typename T1, typename T2>
inline __device__ void IDPacker(T1& ID, T2& X, T2& Y, T2& Z, unsigned char& nvXp2, unsigned char& nvYp2) {
    ID = 0;
    ID += X;
    ID += Y << nvXp2;
    ID += Z << (nvXp2 + nvYp2);
}

template <typename T1, typename T2>
inline __device__ void applyOriQToVector3(T1& X, T1& Y, T1& Z) {
    // Now does nothing
}

template <typename T1>
inline __device__ T1 distSquared(T1 x1, T1 y1, T1 z1, T1 x2, T1 y2, T1 z2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
}
