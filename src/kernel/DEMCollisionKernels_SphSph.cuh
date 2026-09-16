// DEM collision-related kernel collection

#ifndef DEME_COLLI_KERNELS_SS_CUH
#define DEME_COLLI_KERNELS_SS_CUH

#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>

template <typename T1>
inline __device__ void equipOwnerPosRot(deme::DEMSimParams* simParams,
                                        deme::DEMDataDT* granData,
                                        const deme::bodyID_t& myOwner,
                                        T1& relPos,
                                        double3& ownerPos,
                                        double3& bodyPos,
                                        float4& oriQ) {
    voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
        ownerPos.x, ownerPos.y, ownerPos.z, granData->voxelID[myOwner], granData->locX[myOwner],
        granData->locY[myOwner], granData->locZ[myOwner], simParams->nvXp2, simParams->nvYp2, simParams->voxelSize,
        simParams->l);
    // Do this and we get the `true' pos...
    ownerPos.x += simParams->LBFX;
    ownerPos.y += simParams->LBFY;
    ownerPos.z += simParams->LBFZ;
    oriQ.w = granData->oriQw[myOwner];
    oriQ.x = granData->oriQx[myOwner];
    oriQ.y = granData->oriQy[myOwner];
    oriQ.z = granData->oriQz[myOwner];
    applyOriQToVector3(relPos, oriQ);
    bodyPos.x = ownerPos.x + (double)relPos.x;
    bodyPos.y = ownerPos.y + (double)relPos.y;
    bodyPos.z = ownerPos.z + (double)relPos.z;
}

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
__host__ __device__ deme::contact_t checkSpheresOverlap(const T1& XA,
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
    deme::contact_t contactTypePrimitive;
    if (centerDist2 > (radA + radB) * (radA + radB)) {
        contactTypePrimitive = deme::NOT_A_CONTACT;
    } else {
        contactTypePrimitive = deme::SPHERE_SPHERE_CONTACT;
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
    return contactTypePrimitive;
}

// Return a deterministic radial direction for points on a cone axis, where the closest direction is non-unique.
inline __host__ __device__ float3 anyPerpendicularUnitVector(const float3& axis) {
    float3 reference = (fabsf(axis.x) < 0.5f) ? make_float3(1, 0, 0) : make_float3(0, 1, 0);
    float3 perpendicular = cross(axis, reference);
    if (length(perpendicular) < DEME_TINY_FLOAT) {
        reference = make_float3(0, 0, 1);
        perpendicular = cross(axis, reference);
    }
    return normalize(perpendicular);
}

// Check whether a sphere and an analytical boundary are in contact, and gives overlap depth, contact point and contact
// normal. Returned contact type is only useful for kT to sort contact types, as for dT's force calculation, the flavor
// used is determined by type B's actual objType.
template <typename T1, typename T2, typename T3>
__host__ __device__ deme::contact_t checkSphereEntityOverlap(const T1& A,
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
    deme::contact_t contactTypePrimitive;
    switch (typeB) {
        case (deme::ANAL_OBJ_TYPE_PLANE): {
            // Plane is directional, and the direction is given by plane rotation
            const T3 dist = planeSignedDistance<T3>(A, B, dirB);
            overlapDepth = (radA + beta4Entity - dist);
            if (overlapDepth < 0.0) {
                contactTypePrimitive = deme::NOT_A_CONTACT;
                overlapArea = 0.0;
            } else {
                contactTypePrimitive = deme::SPHERE_ANALYTICAL_CONTACT;
                // Approximate overlap area as circle area
                overlapArea = deme::PI * (radA * radA - (dist - beta4Entity) * (dist - beta4Entity));
            }
            // From sphere center, go along reverse plane normal for (dist + overlapDepth / 2)
            CP = A - to_real3<float3, T1>(dirB * (dist + overlapDepth / 2.0));
            // Contact normal (B to A) is the same as plane normal
            cntNormal = dirB;
            return contactTypePrimitive;
        }
        case (deme::ANAL_OBJ_TYPE_CYL_INF): {
            T1 cyl2sph = cylRadialDistanceVec<T1>(A, B, dirB);
            const T3 dist_delta_r = length(cyl2sph);
            // The margin expands an exterior cylinder and shrinks an interior one in the direction of its normal.
            const float cyl_rad = size1B - normal_sign * beta4Entity;
            overlapDepth = radA - normal_sign * (cyl_rad - dist_delta_r);
            if (overlapDepth <= DEME_TINY_FLOAT) {
                contactTypePrimitive = deme::NOT_A_CONTACT;
                overlapArea = 0.0;
            } else {
                contactTypePrimitive = deme::SPHERE_ANALYTICAL_CONTACT;
                // Approximate overlap area as circle area
                // overlapArea = deme::PI * (radA * radA - (radA - overlapDepth) * (radA - overlapDepth));
                // Simplify it...
                overlapArea = deme::PI * (2.0 * radA * overlapDepth - overlapDepth * overlapDepth);
            }
            // Inward normal is 1, outward is -1, so flip normal_sign for B2A vector
            // A sphere exactly on the cylinder axis has no radial direction. Use the cylinder axis as a deterministic
            // finite fallback instead of dividing by zero.
            if (dist_delta_r >= (T3)DEME_TINY_FLOAT) {
                cntNormal = to_real3<T1, float3>(-normal_sign / dist_delta_r * cyl2sph);
                CP = A - to_real3<float3, T1>(cntNormal * (radA - overlapDepth / 2.0));
            } else {
                cntNormal = dirB;
                CP = A;
            }
            return contactTypePrimitive;
        }
        case (deme::ANAL_OBJ_TYPE_CONE_INF):
        case (deme::ANAL_OBJ_TYPE_CONE): {
            const T3 cone_slope = (T3)size1B;
            const T3 min_h = (T3)size2B;
            const T3 max_h = (T3)size3B;
            const T1 tip2sph = A - B;
            const T1 cone_axis = to_real3<float3, T1>(dirB);
            const T3 axial_dist = dot(tip2sph, cone_axis);
            const T1 radial_vec = tip2sph - cone_axis * axial_dist;
            const T3 radial_dist = length(radial_vec);
            const float3 radial_dir = (radial_dist >= (T3)DEME_TINY_FLOAT)
                                          ? to_real3<T1, float3>(radial_vec / radial_dist)
                                          : anyPerpendicularUnitVector(dirB);

            // Project the sphere center onto the cone generator in (axial distance, radius) space. Clamping this
            // projection supplies exact apex/rim contact for bounded cone segments without adding mesh facets.
            const T3 inv_slope_metric = (T3)1.0 / ((T3)1.0 + cone_slope * cone_slope);
            T3 closest_h = (axial_dist + cone_slope * radial_dist) * inv_slope_metric;
            bool closest_is_edge = false;
            if (closest_h < min_h) {
                closest_h = min_h;
                closest_is_edge = true;
            } else if (closest_h > max_h) {
                closest_h = max_h;
                closest_is_edge = true;
            }

            if (closest_is_edge) {
                const T1 closest =
                    B + cone_axis * closest_h + to_real3<float3, T1>(radial_dir) * (cone_slope * closest_h);
                const T1 feature2sph = A - closest;
                const T3 dist_to_feature = length(feature2sph);
                overlapDepth = (T3)radA + (T3)beta4Entity - dist_to_feature;
                contactTypePrimitive = (overlapDepth < (T3)0.0) ? deme::NOT_A_CONTACT : deme::SPHERE_ANALYTICAL_CONTACT;
                overlapArea = (contactTypePrimitive == deme::NOT_A_CONTACT)
                                  ? (T3)0.0
                                  : deme::PI * ((T3)2.0 * (T3)radA * overlapDepth - overlapDepth * overlapDepth);

                if (dist_to_feature >= (T3)DEME_TINY_FLOAT) {
                    cntNormal = to_real3<T1, float3>(feature2sph / dist_to_feature);
                } else {
                    const T3 side_normal_len = sqrt((T3)1.0 + cone_slope * cone_slope);
                    cntNormal = normal_sign * ((float)cone_slope * dirB - radial_dir) / side_normal_len;
                }
                CP = A - to_real3<float3, T1>(cntNormal * ((T3)radA - overlapDepth / (T3)2.0));
                return contactTypePrimitive;
            }

            // Directional side contact follows the analytical-cylinder convention: inward normals retain particles
            // inside the cone, while outward normals model the exterior of a solid cone.
            const T3 side_normal_len = sqrt((T3)1.0 + cone_slope * cone_slope);
            const T3 signed_gap = (T3)normal_sign * (cone_slope * axial_dist - radial_dist) / side_normal_len;
            overlapDepth = (T3)radA + (T3)beta4Entity - signed_gap;
            contactTypePrimitive = (overlapDepth < (T3)0.0) ? deme::NOT_A_CONTACT : deme::SPHERE_ANALYTICAL_CONTACT;
            overlapArea = (contactTypePrimitive == deme::NOT_A_CONTACT)
                              ? (T3)0.0
                              : deme::PI * ((T3)2.0 * (T3)radA * overlapDepth - overlapDepth * overlapDepth);
            cntNormal = normal_sign * ((float)cone_slope * dirB - radial_dir) / side_normal_len;
            CP = A - to_real3<float3, T1>(cntNormal * ((T3)radA - overlapDepth / (T3)2.0));
            return contactTypePrimitive;
        }
        default:
            return deme::NOT_A_CONTACT;
    }
}

#endif
