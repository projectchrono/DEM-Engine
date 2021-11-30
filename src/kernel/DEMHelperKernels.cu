// DEM device-side helper kernel collection
#include <helper_math.cuh>
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>

// inline __device__ voxelID_t position2VoxelID

// Sign function
template <typename T1>
inline __device__ int sgn(const T1& val) {
    return (T1(0) < val) - (val < T1(0));
}

// Integer division that rounds towards -infty
template <typename T1, typename T2>
inline __device__ T1 div_floor(const T1& a, const T2& b) {
    T1 res = a / b;
    T1 rem = a % b;
    // Correct division result downwards if up-rounding happened,
    // (for non-zero remainder of sign different than the divisor).
    T1 corr = (rem != 0 && ((rem < 0) != (b < 0)));
    return res - corr;
}

// Modulus that rounds towards -infty
template <typename T1, typename T2>
inline __device__ T1 mod_floor(const T1& a, const T2& b) {
    if (b < 0)  // you can check for b == 0 separately and do what you want
        return -mod_floor(-a, -b);
    T1 ret = a % b;
    if (ret < 0)
        ret += b;
    return ret;
}

// Chops a long ID (typically voxelID) into XYZ components
template <typename T1, typename T2>
inline __device__ void IDChopper(T1& X,
                                 T1& Y,
                                 T1& Z,
                                 const T2& ID,
                                 const unsigned char& nvXp2,
                                 const unsigned char& nvYp2) {
    X = ID & (((T1)1 << nvXp2) - 1);  // & operation here equals modulo
    Y = (ID >> nvXp2) & (((T1)1 << nvYp2) - 1);
    Z = (ID) >> (nvXp2 + nvYp2);
}

// Packs XYZ components back to a long ID (typically voxelID)
template <typename T1, typename T2>
inline __device__ void IDPacker(T1& ID,
                                const T2& X,
                                const T2& Y,
                                const T2& Z,
                                const unsigned char& nvXp2,
                                const unsigned char& nvYp2) {
    ID = 0;
    ID += X;
    ID += Y << nvXp2;
    ID += Z << (nvXp2 + nvYp2);
}

// From a voxelID to (usually double-precision) xyz coordinate
template <typename T1, typename T2, typename T3>
inline __device__ void voxelID2Position(T1& X,
                                        T1& Y,
                                        T1& Z,
                                        const T2& ID,
                                        const T3& subPosX,
                                        const T3& subPosY,
                                        const T3& subPosZ,
                                        const unsigned char& nvXp2,
                                        const unsigned char& nvYp2,
                                        const T1& voxelSize,
                                        const T1& l) {
    T2 voxelIDX, voxelIDY, voxelIDZ;
    IDChopper<T2, T2>(voxelIDX, voxelIDY, voxelIDZ, ID, nvXp2, nvYp2);
    X = (T1)voxelIDX * voxelSize + (T1)subPosX * l;
    Y = (T1)voxelIDY * voxelSize + (T1)subPosY * l;
    Z = (T1)voxelIDZ * voxelSize + (T1)subPosZ * l;
}

template <typename T1, typename T2>
inline __device__ void applyOriQToVector3(T1& X, T1& Y, T1& Z) {
    // Now does nothing
}

template <typename T1>
inline __device__ T1 distSquared(const T1& x1, const T1& y1, const T1& z1, const T1& x2, const T1& y2, const T1& z2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
}

// Normalize a 3-component vector
template <typename T1>
inline __device__ void normalizeVector3(T1& x, T1& y, T1& z) {
    T1 magnitude = sqrt(x * x + y * y + z * z);
    // TODO: Think about whether this is safe
    x /= magnitude;
    y /= magnitude;
    z /= magnitude;
}

// Return whether 2 spheres intersect and the intersection point coordinates
template <typename T1>
inline __device__ void checkSpheresOverlap(const T1& XA,
                                           const T1& YA,
                                           const T1& ZA,
                                           const T1& radA,
                                           const T1& XB,
                                           const T1& YB,
                                           const T1& ZB,
                                           const T1& radB,
                                           T1& CPX,
                                           T1& CPY,
                                           T1& CPZ,
                                           bool& overlap) {
    T1 centerDist2 = distSquared<T1>(XA, YA, ZA, XB, YB, ZB);
    if (centerDist2 > (radA + radB) * (radA + radB)) {
        overlap = false;
        return;
    }
    // If getting this far, then 2 spheres have an intersection, let's calculate the intersection point
    overlap = true;
    T1 A2BVecX = XB - XA;
    T1 A2BVecY = YB - YA;
    T1 A2BVecZ = ZB - ZA;
    normalizeVector3<double>(A2BVecX, A2BVecY, A2BVecZ);
    T1 halfOverlapDepth = (radA + radB - sqrt(centerDist2)) / (T1)2;
    // From center of A, towards center of B, move a distance of radA, then backtrack a bit, for half the overlap depth
    CPX = XA + (radA - halfOverlapDepth) * A2BVecX;
    CPY = YA + (radA - halfOverlapDepth) * A2BVecY;
    CPZ = ZA + (radA - halfOverlapDepth) * A2BVecZ;
}

// Another version of checkSpheresOverlap which also gives the penetration length and bodyA's outward contact normal
template <typename T1>
inline __device__ void checkSpheresOverlap(const T1& XA,
                                           const T1& YA,
                                           const T1& ZA,
                                           const T1& radA,
                                           const T1& XB,
                                           const T1& YB,
                                           const T1& ZB,
                                           const T1& radB,
                                           T1& CPX,
                                           T1& CPY,
                                           T1& CPZ,
                                           T1& normalX,
                                           T1& normalY,
                                           T1& normalZ,
                                           T1& overlapDepth,
                                           bool& overlap) {
    T1 centerDist2 = distSquared<T1>(XA, YA, ZA, XB, YB, ZB);
    if (centerDist2 > (radA + radB) * (radA + radB)) {
        overlap = false;
        return;
    }
    // If getting this far, then 2 spheres have an intersection, let's calculate the intersection point
    overlap = true;
    normalX = XB - XA;
    normalY = YB - YA;
    normalZ = ZB - ZA;
    normalizeVector3<double>(normalX, normalY, normalZ);
    overlapDepth = radA + radB - sqrt(centerDist2);
    // From center of A, towards center of B, move a distance of radA, then backtrack a bit, for half the overlap depth
    CPX = XA + (radA - overlapDepth / (T1)2) * normalX;
    CPY = YA + (radA - overlapDepth / (T1)2) * normalY;
    CPZ = ZA + (radA - overlapDepth / (T1)2) * normalZ;
}

template <typename T1>
inline __device__ T1
getPointBinID(const double& X, const double& Y, const double& Z, const double& binSize, const T1& nbX, const T1& nbY) {
    T1 binIDX = X / binSize;
    T1 binIDY = Y / binSize;
    T1 binIDZ = Z / binSize;
    return binIDX + binIDY * nbX + binIDZ * nbX * nbY;
}
