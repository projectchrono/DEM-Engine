// DEM collision-related kernel collection

#ifndef DEME_COLLI_KERNELS_SS_CUH
#define DEME_COLLI_KERNELS_SS_CUH

#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>

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
                                                               T1& overlapDepth,
                                                               T1& overlapArea) {
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
    // overlap area in the sph-sph case will involve only half the overlap depth
    // overlapArea = deme::PI * (radA * radA - (radA - overlapDepth / (T1)2) * (radA - overlapDepth / (T1)2));
    // Simplify it...
    overlapArea = deme::PI * (radA * overlapDepth - overlapDepth * overlapDepth / (T1)4);
    // From center of B, towards center of A, move a distance of radB, then backtrack a bit, for half the overlap depth
    CPX = XB + (radB - overlapDepth / (T1)2) * normalX;
    CPY = YB + (radB - overlapDepth / (T1)2) * normalY;
    CPZ = ZB + (radB - overlapDepth / (T1)2) * normalZ;
    return contactType;
}

// Check whether a sphere and an analytical boundary are in contact, and gives overlap depth, contact point and contact
// normal. Returned contact type is only useful for kT to sort contact types, as for dT's force calculation, the flavor
// used is determined by type B's actual objType.
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
                                                                    T3& overlapDepth,
                                                                    T3& overlapArea) {
    deme::contact_t contactType;
    switch (typeB) {
        case (deme::ANAL_OBJ_TYPE_PLANE): {
            // Plane is directional, and the direction is given by plane rotation
            const T3 dist = planeSignedDistance<T3>(A, B, dirB);
            overlapDepth = (radA + beta4Entity - dist);
            if (overlapDepth < 0.0) {
                contactType = deme::NOT_A_CONTACT;
                overlapArea = 0.0;
            } else {
                contactType = deme::SPHERE_ANALYTICAL_CONTACT;
                // Approximate overlap area as circle area
                overlapArea = deme::PI * (radA * radA - (dist - beta4Entity) * (dist - beta4Entity));
            }
            // From sphere center, go along reverse plane normal for (dist + overlapDepth / 2)
            CP = A - to_real3<float3, T1>(dirB * (dist + overlapDepth / 2.0));
            // Contact normal (B to A) is the same as plane normal
            cntNormal = dirB;
            return contactType;
        }
        case (deme::ANAL_OBJ_TYPE_PLATE): {
            return deme::NOT_A_CONTACT;
        }
        case (deme::ANAL_OBJ_TYPE_CYL_INF): {
            T1 cyl2sph = cylRadialDistanceVec<T1>(A, B, dirB);
            const T3 dist_delta_r = length(cyl2sph);
            overlapDepth = radA - abs(size1B - dist_delta_r - beta4Entity);
            if (overlapDepth <= DEME_TINY_FLOAT) {
                contactType = deme::NOT_A_CONTACT;
                overlapArea = 0.0;
            } else {
                contactType = deme::SPHERE_ANALYTICAL_CONTACT;
                // Approximate overlap area as circle area
                // overlapArea = deme::PI * (radA * radA - (radA - overlapDepth) * (radA - overlapDepth));
                // Simplify it...
                overlapArea = deme::PI * (2.0 * radA * overlapDepth - overlapDepth * overlapDepth);
            }
            // dist_delta_r is 0 only when cylinder is thinner than sphere rad...
            // Also, inward normal is 1, outward is -1, so flip normal_sign for B2A vector
            cntNormal = to_real3<T1, float3>(-normal_sign / dist_delta_r * cyl2sph);
            CP = A - to_real3<float3, T1>(cntNormal * (radA - overlapDepth / 2.0));
            return contactType;
        }
        default:
            return deme::NOT_A_CONTACT;
    }
}

#endif
