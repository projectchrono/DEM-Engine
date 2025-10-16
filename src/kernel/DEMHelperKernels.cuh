// DEM device-side helper kernel collection

#ifndef DEME_HELPER_KERNELS_CUH
#define DEME_HELPER_KERNELS_CUH

#include "CUDAMathHelpers.cuh"
// If being statically compiled using CMake, then include DEM/Defines.h using relative path.
// Otherwise, use the angle bracket form for jitify to pick it up from the build dir.
#if __has_include("../DEM/Defines.h")
    #include "../DEM/Defines.h"
#else
    #include <DEM/Defines.h>
#endif

// inline __device__ voxelID_t position2VoxelID

////////////////////////////////////////////////////////////////////////////////
// A few helper functions specific to DEM module
////////////////////////////////////////////////////////////////////////////////

// Sign function
template <typename T1>
inline __host__ __device__ int sgn(const T1& val) {
    return (T1(0) < val) - (val < T1(0));
}

template <typename T1>
inline __host__ __device__ T1 dot3(const T1& x1, const T1& x2, const T1& x3, const T1& y1, const T1& y2, const T1& y3) {
    return x1 * y1 + x2 * y2 + x3 * y3;
}

// Integer division that rounds towards -infty
template <typename T1, typename T2>
inline __host__ __device__ T1 div_floor(const T1& a, const T2& b) {
    T1 res = a / b;
    T1 rem = a % b;
    // Correct division result downwards if up-rounding happened,
    // (for non-zero remainder of sign different than the divisor).
    T1 corr = (rem != 0 && ((rem < 0) != (b < 0)));
    return res - corr;
}

// Modulus that rounds towards -infty
template <typename T1, typename T2>
inline __host__ __device__ T1 mod_floor(const T1& a, const T2& b) {
    if (b < 0)  // you can check for b == 0 separately and do what you want
        return -mod_floor(-a, -b);
    T1 ret = a % b;
    if (ret < 0)
        ret += b;
    return ret;
}

// In an upper-triangular (including the diagonal part) matrix, given i and j, this function returns the index of the
// corresponding flatten-ed non-zero entries (col-major like in matlab). This function does not assume i <= j. It is
// used in locating masks that maps the contact between families.
template <typename T1>
inline __host__ __device__ T1 locateMaskPair(const T1& i, const T1& j) {
    if (i > j)
        return locateMaskPair(j, i);
    return (1 + j) * j / 2 + i;
}

// Magic function that converts an index of a flatten-ed upper-triangular matrix (EXCLUDING the diagonal) to its
// corresponding i and j. It is ROW-major. It is used to map contact pair numbers in a bin.
template <typename T1>
inline __host__ __device__ void recoverCntPair(T1& i, T1& j, const T1& ind, const T1& n) {
    i = n - 2 - (T1)(sqrt((float)(4 * n * (n - 1) - 7 - 8 * ind)) / 2.0 - 0.5);
    j = ind + i + 1 + (n - i) * ((n - i) - 1) / 2 - n * (n - 1) / 2;
}

// Make sure a T1 type triplet falls in a range, then output as T2 type
template <typename T1, typename T2>
inline __host__ __device__ T2 clampBetween3Comp(const T1& data, const T2& low, const T2& high) {
    T2 res;
    res.x = DEME_MIN(DEME_MAX(data.x, low.x), high.x);
    res.y = DEME_MIN(DEME_MAX(data.y, low.y), high.y);
    res.z = DEME_MIN(DEME_MAX(data.z, low.z), high.z);
    return res;
}

// Make sure a T1 type triplet falls in a range, then output as T2 type
template <typename T1, typename T2>
inline __host__ __device__ T2 clampBetween(const T1& data, const T2& low, const T2& high) {
    T2 res;
    res = DEME_MIN(DEME_MAX(data, low), high);
    return res;
}

// Chops a long ID (typically voxelID) into XYZ components
template <typename T1, typename T2>
inline __host__ __device__ void IDChopper(T1& X,
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
inline __host__ __device__ void IDPacker(T1& ID,
                                         const T2& X,
                                         const T2& Y,
                                         const T2& Z,
                                         const unsigned char& nvXp2,
                                         const unsigned char& nvYp2) {
    ID = X;
    ID += Y << nvXp2;
    ID += Z << (nvXp2 + nvYp2);
}

// From a voxelID to (usually double-precision) xyz coordinate
template <typename T1, typename T2, typename T3>
inline __host__ __device__ void voxelIDToPosition(T1& X,
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

// From xyz coordinate (usually double-precision) to voxelID
template <typename T1, typename T2, typename T3>
inline __host__ __device__ void positionToVoxelID(T1& ID,
                                                  T2& subPosX,
                                                  T2& subPosY,
                                                  T2& subPosZ,
                                                  const T3& X,
                                                  const T3& Y,
                                                  const T3& Z,
                                                  const unsigned char& nvXp2,
                                                  const unsigned char& nvYp2,
                                                  const T3& voxelSize,
                                                  const T3& l) {
    deme::voxelID_t voxelNumX = X / voxelSize;
    deme::voxelID_t voxelNumY = Y / voxelSize;
    deme::voxelID_t voxelNumZ = Z / voxelSize;
    subPosX = (X - (T3)voxelNumX * voxelSize) / l;
    subPosY = (Y - (T3)voxelNumY * voxelSize) / l;
    subPosZ = (Z - (T3)voxelNumZ * voxelSize) / l;

    ID = voxelNumX;
    ID += voxelNumY << nvXp2;
    ID += voxelNumZ << (nvXp2 + nvYp2);
}

template <typename T1, typename T2>
inline __host__ __device__ void
applyOriQToVector3(T1& X, T1& Y, T1& Z, const T2& Qw, const T2& Qx, const T2& Qy, const T2& Qz) {
    T1 oldX = X;
    T1 oldY = Y;
    T1 oldZ = Z;
    X = ((T2)2.0 * (Qw * Qw + Qx * Qx) - (T2)1.0) * oldX + ((T2)2.0 * (Qx * Qy - Qw * Qz)) * oldY +
        ((T2)2.0 * (Qx * Qz + Qw * Qy)) * oldZ;
    Y = ((T2)2.0 * (Qx * Qy + Qw * Qz)) * oldX + ((T2)2.0 * (Qw * Qw + Qy * Qy) - (T2)1.0) * oldY +
        ((T2)2.0 * (Qy * Qz - Qw * Qx)) * oldZ;
    Z = ((T2)2.0 * (Qx * Qz - Qw * Qy)) * oldX + ((T2)2.0 * (Qy * Qz + Qw * Qx)) * oldY +
        ((T2)2.0 * (Qw * Qw + Qz * Qz) - (T2)1.0) * oldZ;
}

template <typename T1, typename T2, typename T3>
inline __host__ __device__ void applyFrameTransformLocalToGlobal(T1& pos, const T2& vec, const T3& rot_Q) {
    applyOriQToVector3(pos.x, pos.y, pos.z, rot_Q.w, rot_Q.x, rot_Q.y, rot_Q.z);
    pos.x += vec.x;
    pos.y += vec.y;
    pos.z += vec.z;
}

template <typename T1>
inline __host__ __device__ T1
distSquared(const T1& x1, const T1& y1, const T1& z1, const T1& x2, const T1& y2, const T1& z2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
}

// Get the maginitude of a 3-component vector
template <typename T1>
inline __host__ __device__ T1 magVector3(T1& x, T1& y, T1& z) {
    return sqrt(x * x + y * y + z * z);
}

// Normalize a 3-component vector
template <typename T1>
inline __host__ __device__ void normalizeVector3(T1& x, T1& y, T1& z) {
    T1 magnitude = sqrt(x * x + y * y + z * z);
    // TODO: Think about whether this is safe
    // if (magnitude < DEME_TINY_FLOAT) {
    //     printf("Caution!\n");
    // }
    x /= magnitude;
    y /= magnitude;
    z /= magnitude;
}

// Calculate the centroid of a triangle
template <typename T1>
inline __host__ __device__ T1 triangleCentroid(const T1& p1, const T1& p2, const T1& p3) {
    return (p1 + p2 + p3) / 3.;
}

// Calculate the incenter of a triangle
template <typename T1>
inline __host__ __device__ T1 triangleIncenter(const T1& p1, const T1& p2, const T1& p3) {
    float a = length(p2 - p3);
    float b = length(p1 - p3);
    float c = length(p1 - p2);
    T1 res;
    res.x = (a * p1.x + b * p2.x + c * p3.x) / (a + b + c);
    res.y = (a * p1.y + b * p2.y + c * p3.y) / (a + b + c);
    res.z = (a * p1.z + b * p2.z + c * p3.z) / (a + b + c);
    return res;
}

// Hamilton product of 2 quaternions
template <typename T1, typename T2, typename T3>
inline __host__ __device__ void HamiltonProduct(T1& A,
                                                T1& B,
                                                T1& C,
                                                T1& D,
                                                const T2 a1,
                                                const T2 b1,
                                                const T2 c1,
                                                const T2 d1,
                                                const T3 a2,
                                                const T3 b2,
                                                const T3 c2,
                                                const T3 d2) {
    A = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2;
    B = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2;
    C = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2;
    D = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2;
}

/*
// This version of checkSpheresOverlap is not used anymore for code clarity, and the fact that every use case now
// requires overlapDepth.
template <typename T1>
inline __device__ deme::contact_t checkSpheresOverlap(const T1& XA,
                                                      const T1& YA,
                                                      const T1& ZA,
                                                      const T1& radA,
                                                      const T1& XB,
                                                      const T1& YB,
                                                      const T1& ZB,
                                                      const T1& radB,
                                                      T1& CPX,
                                                      T1& CPY,
                                                      T1& CPZ) {
    T1 centerDist2 = distSquared<T1>(XA, YA, ZA, XB, YB, ZB);
    deme::contact_t contactType;
    if (centerDist2 > (radA + radB) * (radA + radB)) {
        contactType = deme::NOT_A_CONTACT;
    } else {
        contactType = deme::SPHERE_SPHERE_CONTACT;
    }
    // If getting this far, then 2 spheres have an intersection, let's calculate the intersection point
    float B2AVecX = XA - XB;
    float B2AVecY = YA - YB;
    float B2AVecZ = ZA - ZB;
    normalizeVector3<float>(B2AVecX, B2AVecY, B2AVecZ);
    T1 halfOverlapDepth = (radA + radB - sqrt(centerDist2)) / (T1)2;
    // From center of B, towards center of A, move a distance of radB, then backtrack a bit, for half the overlap depth
    CPX = XB + (radB - halfOverlapDepth) * B2AVecX;
    CPY = YB + (radB - halfOverlapDepth) * B2AVecY;
    CPZ = ZB + (radB - halfOverlapDepth) * B2AVecZ;
    return contactType;
}
*/

/**
 * Template arguments:
 *   - T1: the floating point accuracy level for contact point location/penetration depth
 *   - T2: the floating point accuracy level for the relative position of 2 bodies involved
 *
 * Basic idea: determines whether 2 spheres intersect and the intersection point coordinates which also gives the
 * penetration length and bodyB's outward contact normal.
 *
 */
template <typename T1, typename T2>
inline __host__ __device__ deme::contact_t checkSpheresOverlap(const T1& XA,
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
                                                               T2& normalX,
                                                               T2& normalY,
                                                               T2& normalZ,
                                                               T1& overlapDepth) {
    T1 centerDist2 = distSquared<T1>(XA, YA, ZA, XB, YB, ZB);
    deme::contact_t contactType;
    if (centerDist2 > (radA + radB) * (radA + radB)) {
        contactType = deme::NOT_A_CONTACT;
    } else {
        contactType = deme::SPHERE_SPHERE_CONTACT;
    }
    // If getting this far, then 2 spheres have an intersection, let's calculate the intersection point
    normalX = XA - XB;
    normalY = YA - YB;
    normalZ = ZA - ZB;
    normalizeVector3<T2>(normalX, normalY, normalZ);
    overlapDepth = radA + radB - sqrt(centerDist2);
    // From center of B, towards center of A, move a distance of radB, then backtrack a bit, for half the overlap depth
    CPX = XB + (radB - overlapDepth / (T1)2) * normalX;
    CPY = YB + (radB - overlapDepth / (T1)2) * normalY;
    CPZ = ZB + (radB - overlapDepth / (T1)2) * normalZ;
    return contactType;
}

// Compute the binID for a point in space
template <typename T1>
inline __host__ __device__ T1
getPointBinID(const double& X, const double& Y, const double& Z, const double& binSize, const T1& nbX, const T1& nbY) {
    T1 binIDX = X / binSize;
    T1 binIDY = Y / binSize;
    T1 binIDZ = Z / binSize;
    return binIDX + binIDY * nbX + binIDZ * nbX * nbY;
}

// Compute the binID using its indices in X, Y and Z directions
template <typename T1>
inline __host__ __device__ T1
binIDFrom3Indices(const T1& X, const T1& Y, const T1& Z, const T1& nbX, const T1& nbY, const T1& nbZ) {
    if ((X < nbX) && (Y < nbY) && (Z < nbZ)) {
        return X + Y * nbX + Z * nbX * nbY;
    } else {
        return deme::NULL_BINID;
    }
}

// This utility function returns the normal to the triangular face defined by
// the vertices A, B, and C. The face is assumed to be non-degenerate.
// Note that order of vertices is important!
template <typename T1>
inline __host__ __device__ T1 face_normal(const T1& A, const T1& B, const T1& C) {
    return normalize(cross(B - A, C - A));
}

// Binary search on GPU, which is probably quite divergent... use if absolutely needs to
// T2 must be a singed type
template <typename T1, typename T2>
inline __host__ __device__ bool cuda_binary_search(T1* A, const T1& val, T2 imin, T2 imax, T2& res) {
    while (imax >= imin) {
        T2 imid = imin + (imax - imin) / 2;
        if (val == A[imid]) {
            res = imid;
            return true;
        } else if (val > A[imid]) {
            imin = imid + 1;
        } else {
            imax = imid - 1;
        }
    }

    return false;
}
template <typename T1, typename T2>
inline __host__ __device__ bool cuda_binary_search(T1* A, const T1& val, T2 imin, T2 imax) {
    while (imax >= imin) {
        T2 imid = imin + (imax - imin) / 2;
        if (val == A[imid]) {
            return true;
        } else if (val > A[imid]) {
            imin = imid + 1;
        } else {
            imax = imid - 1;
        }
    }

    return false;
}

/**
 * Template arguments:
 *   - T1: the floating point accuracy level for the point coordinates
 *
 * Basic idea: calculate a vector that goes from B to A, and pack it as a float3
 *
 */
template <typename T1>
inline __host__ __device__ float3
vectorAB(const T1& AX, const T1& AY, const T1& AZ, const T1& BX, const T1& BY, const T1& BZ) {
    return make_float3(AX - BX, AY - BY, AZ - BZ);
}

/**
 * Template arguments:
 *   - T1: the floating point accuracy level for the point coordinate and the frame O coordinate
 *
 * Basic idea: calculate a point's (typically contact point) local coordinate in a specific frame, then return as a
 * float3
 *
 */
template <typename T1>
inline __host__ __device__ float3 findLocalCoord(const T1& X,
                                                 const T1& Y,
                                                 const T1& Z,
                                                 const T1& Ox,
                                                 const T1& Oy,
                                                 const T1& Oz,
                                                 const deme::oriQ_t& oriQw,
                                                 const deme::oriQ_t& oriQx,
                                                 const deme::oriQ_t& oriQy,
                                                 const deme::oriQ_t& oriQz) {
    float locX, locY, locZ;
    locX = X - Ox;
    locY = Y - Oy;
    locZ = Z - Oz;
    // To find the contact point in the local (body) frame, just apply inverse quaternion to OP vector in global frame
    applyOriQToVector3<float, deme::oriQ_t>(locX, locY, locZ, oriQw, -oriQx, -oriQy, -oriQz);
    return make_float3(locX, locY, locZ);
}

/// Calculate the contact params based on the 2 contact material types given
template <typename T1>
inline __host__ __device__ void matProxy2ContactParam(T1& E_eff,
                                                      T1& G_eff,
                                                      const T1& Y1,
                                                      const T1& nu1,
                                                      const T1& Y2,
                                                      const T1& nu2) {
    T1 invE = ((T1)1 - nu1 * nu1) / Y1 + ((T1)1 - nu2 * nu2) / Y2;
    E_eff = (T1)1 / invE;
    T1 invG = (T1)2 * ((T1)2 - nu1) * ((T1)1 + nu1) / Y1 + (T1)2 * ((T1)2 - nu2) * ((T1)1 + nu2) / Y2;
    G_eff = (T1)1 / invG;
}

/// Calculate the contact params based on the 2 contact material types given (no-tangent version)
template <typename T1>
inline __host__ __device__ void matProxy2ContactParam(T1& E_eff,
                                                      const T1& Y1,
                                                      const T1& nu1,
                                                      const T1& Y2,
                                                      const T1& nu2) {
    T1 invE = ((T1)1 - nu1 * nu1) / Y1 + ((T1)1 - nu2 * nu2) / Y2;
    E_eff = (T1)1 / invE;
}

/*
// This version of checkSphereEntityOverlap is not used anymore for code clarity, and the fact that every use case now
// requires overlapDepth.
template <typename T1, typename T2>
inline __device__ deme::contact_t checkSphereEntityOverlap(const T1& A,
                                                           const T2& radA,
                                                           const deme::objType_t& typeB,
                                                           const T1& B,
                                                           const float3& dirB,
                                                           const float& size1B,
                                                           const float& size2B,
                                                           const float& size3B,
                                                           const float& normal_sign,
                                                           const float& beta4Entity) {
    deme::contact_t contactType;
    switch (typeB) {
        case (deme::ANAL_OBJ_TYPE_PLANE): {
            const T1 plane2sph = A - B;
            // Plane is directional, and the direction is given by plane rotation
            const double dist = dot(plane2sph, dirB);
            const double overlapDepth = (radA + beta4Entity - dist);
            if (overlapDepth < 0.0) {
                contactType = deme::NOT_A_CONTACT;
            } else {
                contactType = deme::SPHERE_PLANE_CONTACT;
            }
            return contactType;
        }
        case (deme::ANAL_OBJ_TYPE_PLATE): {
            return deme::NOT_A_CONTACT;
        }
        case (deme::ANAL_OBJ_TYPE_CYL_INF): {
            T1 sph2cyl = B - A;
            // Projection along cyl axis direction
            const double proj_dist = dot(sph2cyl, dirB);
            // Radial vector from cylinder center to sphere center, along inward direction
            sph2cyl -= proj_dist * dirB;
            const double dist_delta_r = length(sph2cyl);
            const double overlapDepth = radA - abs(size1B - dist_delta_r - beta4Entity);
            if (overlapDepth <= DEME_TINY_FLOAT) {
                contactType = deme::NOT_A_CONTACT;
            } else {
                contactType = deme::SPHERE_CYL_CONTACT;
            }
            return contactType;
        }
        default:
            return deme::NOT_A_CONTACT;
    }
}
*/

// Check whether a sphere and an analytical boundary are in contact, and gives overlap depth, contact point and contact
// normal.
template <typename T1, typename T2, typename T3>
inline __host__ __device__ deme::contact_t checkSphereEntityOverlap(const T1& A,
                                                                    const T2& radA,
                                                                    const deme::objType_t& typeB,
                                                                    const T1& B,
                                                                    const float3& dirB,
                                                                    const float& size1B,
                                                                    const float& size2B,
                                                                    const float& size3B,
                                                                    const float& normal_sign,
                                                                    const float& beta4Entity,
                                                                    T1& CP,
                                                                    float3& cntNormal,
                                                                    T3& overlapDepth) {
    deme::contact_t contactType;
    switch (typeB) {
        case (deme::ANAL_OBJ_TYPE_PLANE): {
            const T1 plane2sph = A - B;
            // Plane is directional, and the direction is given by plane rotation
            const T3 dist = dot(plane2sph, dirB);
            overlapDepth = (radA + beta4Entity - dist);
            if (overlapDepth < 0.0) {
                contactType = deme::NOT_A_CONTACT;
            } else {
                contactType = deme::SPHERE_PLANE_CONTACT;
            }
            // From sphere center, go along negative plane normal for (dist + overlapDepth / 2)
            CP = A - to_real3<float3, T1>(dirB * (dist + overlapDepth / 2.0));
            // Contact normal (B to A) is the same as plane normal
            cntNormal = dirB;
            return contactType;
        }
        case (deme::ANAL_OBJ_TYPE_PLATE): {
            return deme::NOT_A_CONTACT;
        }
        case (deme::ANAL_OBJ_TYPE_CYL_INF): {
            T1 sph2cyl = B - A;
            // Projection along cyl axis direction
            const T3 proj_dist = dot(sph2cyl, dirB);
            // Radial vector from cylinder center to sphere center, along inward direction
            sph2cyl -= proj_dist * dirB;
            const T3 dist_delta_r = length(sph2cyl);
            overlapDepth = radA - abs(size1B - dist_delta_r - beta4Entity);
            if (overlapDepth <= DEME_TINY_FLOAT) {
                contactType = deme::NOT_A_CONTACT;
            } else {
                contactType = deme::SPHERE_CYL_CONTACT;
            }
            // dist_delta_r is 0 only when cylinder is thinner than sphere rad...
            cntNormal = to_real3<T1, float3>(normal_sign / dist_delta_r * sph2cyl);
            CP = A - to_real3<float3, T1>(cntNormal * (radA - overlapDepth / 2.0));
            return contactType;
        }
        default:
            return deme::NOT_A_CONTACT;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Triangle-specific helper kernels
////////////////////////////////////////////////////////////////////////////////

/// Takes in a triangle ID and figures out an SD AABB for broadphase use
__inline__ __device__ void boundingBoxIntersectBin(deme::binID_t* L,
                                                   deme::binID_t* U,
                                                   const float3& vA,
                                                   const float3& vB,
                                                   const float3& vC,
                                                   deme::DEMSimParams* simParams) {
    float3 min_pt;
    min_pt.x = DEME_MIN(vA.x, DEME_MIN(vB.x, vC.x));
    min_pt.y = DEME_MIN(vA.y, DEME_MIN(vB.y, vC.y));
    min_pt.z = DEME_MIN(vA.z, DEME_MIN(vB.z, vC.z));

    // Enlarge bounding box, so that no triangle lies right between 2 layers of bins
    min_pt -= DEME_BIN_ENLARGE_RATIO_FOR_FACETS * simParams->binSize;
    // A point on a mesh can be out of the simulation world. In this case, becasue we only need to detect their contact
    // with spheres, and spheres are all in the simulation world, so we just clamp out the bins that are outside the
    // simulation world.
    int3 min_bin =
        clampBetween3Comp<float3, int3>(min_pt / simParams->binSize, make_int3(0, 0, 0),
                                        make_int3(simParams->nbX - 1, simParams->nbY - 1, simParams->nbZ - 1));

    float3 max_pt;
    max_pt.x = DEME_MAX(vA.x, DEME_MAX(vB.x, vC.x));
    max_pt.y = DEME_MAX(vA.y, DEME_MAX(vB.y, vC.y));
    max_pt.z = DEME_MAX(vA.z, DEME_MAX(vB.z, vC.z));

    max_pt += DEME_BIN_ENLARGE_RATIO_FOR_FACETS * simParams->binSize;
    int3 max_bin =
        clampBetween3Comp<float3, int3>(max_pt / simParams->binSize, make_int3(0, 0, 0),
                                        make_int3(simParams->nbX - 1, simParams->nbY - 1, simParams->nbZ - 1));

    L[0] = min_bin.x;
    L[1] = min_bin.y;
    L[2] = min_bin.z;

    U[0] = max_bin.x;
    U[1] = max_bin.y;
    U[2] = max_bin.z;
}

#endif
