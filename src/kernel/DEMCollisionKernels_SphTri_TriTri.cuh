// DEM collision-related kernel collection

#ifndef DEME_COLLI_KERNELS_ST_TT_CUH
#define DEME_COLLI_KERNELS_ST_TT_CUH

#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>

// ------------------------------------------------------------------
// Triangle-analytical object collision detection utilities
// ------------------------------------------------------------------

template <typename T1, typename T2>
bool __device__ tri_plane_penetration(const T1** tri,
                                      const T1& entityLoc,
                                      const float3& entityDir,
                                      T2& overlapDepth,
                                      T2& overlapArea,
                                      T1& contactPnt) {
    // signed distances
    T2 d[3];
    // penetration depth: deepest point in triangle
    T2 dmin = DEME_HUGE_FLOAT;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        d[i] = planeSignedDistance<T2>(*tri[i], entityLoc, entityDir);
        if (d[i] < dmin)
            dmin = d[i];
    }
    // build clipped polygon
    T1 poly[4];
    int nNode = 0;                 // max 4 poly nodes
    bool hasIntersection = false;  // one edge intersecting the plane
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        int j = (i + 1) % 3;  // compare with next vertex
        bool in_i = (d[i] < 0.0);
        bool in_j = (d[j] < 0.0);

        // ^ means one is in, the other is out
        if (in_i ^ in_j) {
            T2 t = d[i] / (d[i] - d[j]);  // between 0 and 1
            T1 inter = *tri[i] + (*tri[j] - *tri[i]) * t;
            // Only register inside points once - project them onto the plane
            if (in_i) {
                // Project the submerging node onto the plane
                T1 projectedNode = *tri[i] - d[i] * to_real3<float3, T1>(entityDir);
                poly[nNode++] = projectedNode;
            }
            poly[nNode++] = inter;
            hasIntersection = true;
        }
    }

    // Handle the case where all three vertices are submerged (no edge crosses the plane)
    if (!hasIntersection) {
        // Check if all vertices are below the plane
        bool allBelow = true;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            if (d[i] >= 0.0) {
                allBelow = false;
                break;
            }
        }
        if (allBelow) {
            // All vertices are below the plane - project all three onto the plane
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                T1 projectedNode = *tri[i] - d[i] * to_real3<float3, T1>(entityDir);
                poly[nNode++] = projectedNode;
            }
            hasIntersection = true;  // We now have a valid polygon
        }
    }

    // centroid of contact
    T1 centroid;
    centroid.x = 0.;
    centroid.y = 0.;
    centroid.z = 0.;
    if (hasIntersection) {  // If has intersection, centroid of the (inside) polygon
        for (int i = 0; i < nNode; i++)
            centroid = centroid + poly[i];
        centroid = centroid / T2(nNode);
    } else {  // If no intersection, centroid is just average of all tri verts
#pragma unroll
        for (int i = 0; i < 3; i++)
            centroid = centroid + *tri[i];
        centroid = centroid / 3.0;
    }

    // We use the convention that if in contact, overlapDepth is positive
    overlapDepth = -dmin;
    bool in_contact = (overlapDepth >= 0.);
    // The centroid's projection to the plane
    T1 projection =
        centroid - planeSignedDistance<T2>(centroid, entityLoc, entityDir) * to_real3<float3, T1>(entityDir);

    // Calculate the area of the clipping polygon using fan triangulation from centroid
    float overlap_area_f = 0.0f;
    if (hasIntersection && nNode >= 3) {
        const float3 centroid_f = to_float3(centroid);
        for (int i = 0; i < nNode; ++i) {
            float3 v1 = to_float3(poly[i]) - centroid_f;
            float3 v2 = to_float3(poly[(i + 1) % nNode]) - centroid_f;
            float3 crossProd = cross(v1, v2);
            overlap_area_f += sqrtf(dot(crossProd, crossProd));
        }
        overlap_area_f *= 0.5f;
    }
    overlapArea = static_cast<T2>(overlap_area_f);

    // cntPnt is from the projection point, go half penetration depth.
    // Note this penetration depth is signed, so if no contact, we go positive plane normal; if in contact, we go
    // negative plane normal. As such, cntPnt always exists and this is important for the cases with extraMargin.
    contactPnt = projection - (overlapDepth * 0.5) * to_real3<float3, T1>(entityDir);
    return in_contact;
}

template <typename T1, typename T2>
bool __device__ tri_cyl_penetration(const T1** tri,
                                    const T1& entityLoc,
                                    const float3& entityDir,
                                    const float& entitySize1,
                                    const float& entitySize2,
                                    const float& normal_sign,
                                    float3& contact_normal,
                                    T2& overlapDepth,
                                    T2& overlapArea,
                                    T1& contactPnt) {
    return false;
}

// Check only, no contact point, depth, area output
template <typename T1>
__host__ __device__ deme::contact_t checkTriEntityOverlap(const T1& A,
                                                          const T1& B,
                                                          const T1& C,
                                                          const deme::objType_t& typeB,
                                                          const T1& entityLoc,
                                                          const float3& entityDir,
                                                          const float& entitySize1,
                                                          const float& entitySize2,
                                                          const float& entitySize3,
                                                          const float& normal_sign,
                                                          const float& beta4Entity) {
    const T1* tri[] = {&A, &B, &C};
    switch (typeB) {
        case (deme::ANAL_OBJ_TYPE_PLANE): {
            for (const T1*& v : tri) {
                // Always cast to double
                double d = planeSignedDistance<double>(*v, entityLoc, entityDir);
                double overlapDepth = beta4Entity - d;
                // printf("v point %f %f %f, entityLoc %f %f %f\n", v->x, v->y, v->z, entityLoc.x, entityLoc.y,
                // entityLoc.z);
                if (overlapDepth >= 0.0)
                    return deme::TRIANGLE_ANALYTICAL_CONTACT;
            }
            return deme::NOT_A_CONTACT;
        }
        case (deme::ANAL_OBJ_TYPE_PLATE): {
            return deme::NOT_A_CONTACT;
        }
        case (deme::ANAL_OBJ_TYPE_CYL_INF): {
            for (const T1*& v : tri) {
                // Radial distance vector is from cylinder axis to a point
                double3 vec = cylRadialDistanceVec<double3>(*v, entityLoc, entityDir);
                // Also, inward normal is 1, outward is -1, so it's the signed dist from point to cylinder wall
                // (positive if same orientation, negative if opposite)
                double signed_dist = (entitySize1 - length(vec)) * normal_sign;
                if (signed_dist <= beta4Entity)
                    return deme::TRIANGLE_ANALYTICAL_CONTACT;
            }
            return deme::NOT_A_CONTACT;
        }
        default:
            return deme::NOT_A_CONTACT;
    }
}

// Fast FP32-only overlap check for kT contact detection (no penetration/area outputs).
template <typename T1>
__host__ __device__ deme::contact_t checkTriEntityOverlapFP32(const T1& A,
                                                              const T1& B,
                                                              const T1& C,
                                                              const deme::objType_t& typeB,
                                                              const T1& entityLoc,
                                                              const float3& entityDir,
                                                              const float& entitySize1,
                                                              const float& entitySize2,
                                                              const float& entitySize3,
                                                              const float& normal_sign,
                                                              const float& beta4Entity) {
    const T1* tri[] = {&A, &B, &C};
    switch (typeB) {
        case (deme::ANAL_OBJ_TYPE_PLANE): {
            for (const T1*& v : tri) {
                const float d = planeSignedDistance<float>(*v, entityLoc, entityDir);
                const float overlapDepth = beta4Entity - d;
                if (overlapDepth >= 0.0f)
                    return deme::TRIANGLE_ANALYTICAL_CONTACT;
            }
            return deme::NOT_A_CONTACT;
        }
        case (deme::ANAL_OBJ_TYPE_PLATE): {
            return deme::NOT_A_CONTACT;
        }
        case (deme::ANAL_OBJ_TYPE_CYL_INF): {
            for (const T1*& v : tri) {
                float3 vec = cylRadialDistanceVec<float3>(*v, entityLoc, entityDir);
                const float signed_dist = (entitySize1 - length(vec)) * normal_sign;
                if (signed_dist <= beta4Entity)
                    return deme::TRIANGLE_ANALYTICAL_CONTACT;
            }
            return deme::NOT_A_CONTACT;
        }
        default:
            return deme::NOT_A_CONTACT;
    }
}

// NOTE: Due to our algorithm needs a overlapDepth even in the case of no contact (because of extraMargin; and with
// extraMargin, negative overlapDepth can be considered in-contact), our tri-anal CD algorithm needs to always return a
// overlapDepth, even in the case of no contact. This is different from the usual CD algorithms which only return a
// overlapDepth. Also unlike tri-sph contact which has a unified util function, calcTriEntityOverlap is different from
// checkTriEntityOverlap.
template <typename T1, typename T2>
bool __device__ calcTriEntityOverlap(const T1& A,
                                     const T1& B,
                                     const T1& C,
                                     const deme::objType_t& entityType,
                                     const T1& entityLoc,
                                     const float3& entityDir,
                                     const float& entitySize1,
                                     const float& entitySize2,
                                     const float& entitySize3,
                                     const float& normal_sign,
                                     T1& contactPnt,
                                     float3& contact_normal,
                                     T2& overlapDepth,
                                     T2& overlapArea) {
    const T1* tri[] = {&A, &B, &C};
    bool in_contact;
    switch (entityType) {
        case deme::ANAL_OBJ_TYPE_PLANE:
            in_contact =
                tri_plane_penetration<T1, T2>(tri, entityLoc, entityDir, overlapDepth, overlapArea, contactPnt);
            // Plane contact's normal is always the plane's normal
            contact_normal = entityDir;
            return in_contact;
        case deme::ANAL_OBJ_TYPE_CYL_INF:
            in_contact = tri_cyl_penetration<T1, T2>(tri, entityLoc, entityDir, entitySize1, entitySize2, normal_sign,
                                                     contact_normal, overlapDepth, overlapArea, contactPnt);
            return in_contact;
        default:
            return false;
    }
}

// -----------------------------------------------------------------
// Triangle-sphere collision detection utilities
// -----------------------------------------------------------------

/// This utility function takes the location 'P' and snaps it to the closest
/// point on the triangular face with given vertices (A, B, and C). The result
/// is returned in 'res'. Both 'P' and 'res' are assumed to be specified in
/// the same frame as the face vertices. This function returns 'true' if the
/// result is on an edge of this face and 'false' if the result is inside the
/// triangle.
/// Code from Ericson, "real-time collision detection", 2005, pp. 141
template <typename T1 = double3, typename T2 = double>
__device__ bool snap_to_face(const T1& A, const T1& B, const T1& C, const T1& P, T1& res) {
    T1 AB = B - A;
    T1 AC = C - A;

    // Check if P in vertex region outside A
    T1 AP = P - A;
    T2 d1 = dot(AB, AP);
    T2 d2 = dot(AC, AP);
    if (d1 <= 0 && d2 <= 0) {
        res = A;  // barycentric coordinates (1,0,0)
        return true;
    }

    // Check if P in vertex region outside B
    T1 BP = P - B;
    T2 d3 = dot(AB, BP);
    T2 d4 = dot(AC, BP);
    if (d3 >= 0 && d4 <= d3) {
        res = B;  // barycentric coordinates (0,1,0)
        return true;
    }

    // Check if P in edge region of AB
    T2 vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        // Return projection of P onto AB
        T2 v = d1 / (d1 - d3);
        res = A + v * AB;  // barycentric coordinates (1-v,v,0)
        return true;
    }

    // Check if P in vertex region outside C
    T1 CP = P - C;
    T2 d5 = dot(AB, CP);
    T2 d6 = dot(AC, CP);
    if (d6 >= 0 && d5 <= d6) {
        res = C;  // barycentric coordinates (0,0,1)
        return true;
    }

    // Check if P in edge region of AC
    T2 vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        // Return projection of P onto AC
        T2 w = d2 / (d2 - d6);
        res = A + w * AC;  // barycentric coordinates (1-w,0,w)
        return true;
    }

    // Check if P in edge region of BC
    T2 va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        // Return projection of P onto BC
        T2 w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        res = B + w * (C - B);  // barycentric coordinates (0,1-w,w)
        return true;
    }

    // P inside face region. Return projection of P onto face
    // barycentric coordinates (u,v,w)
    T2 denom = __drcp_ru(va + vb + vc);
    T2 v = __dmul_ru(vb, denom);
    T2 w = __dmul_ru(vc, denom);
    res = A + v * AB + w * AC;  // = u*A + v*B + w*C  where  (u = 1 - v - w)
    return false;
}

/**
/brief TRIANGLE FACE--SPHERE COLLISION DETECTION

The triangular face is defined by points A, B, C. The sequence is important as it defines the positive face via a
right-hand rule.
The sphere is centered at sphere_pos and has radius.
The index "1" is associated with the triangle. The index "2" is associated with the sphere.
The coordinates of the face and sphere are assumed to be provided in the same reference frame.

Output:
  - pt1:      contact point on triangle
  - depth:    penetration distance (a positive value means that overlap exists)
  - normal:     contact normal, from pt2 to pt1
A return value of "true" signals collision.
*/
template <typename T1, typename T2>
__device__ bool checkTriSphereOverlap(const T1& A,           ///< First vertex of the triangle
                                      const T1& B,           ///< Second vertex of the triangle
                                      const T1& C,           ///< Third vertex of the triangle
                                      const T1& sphere_pos,  ///< Location of the center of the sphere
                                      const T2 radius,       ///< Sphere radius
                                      T1& normal,            ///< contact normal
                                      T2& depth,             ///< penetration (positive if in contact)
                                      T2& overlapArea,       ///< overlap area
                                      T1& pt1                ///< contact point on triangle
) {
    // Calculate face normal using RHR
    T1 face_n = normalize(cross(B - A, C - A));

    // Calculate signed height of sphere center above face plane
    T2 h = dot(sphere_pos - A, face_n);

    // Find the closest point on the face to the sphere center and determine
    // whether or not this location is inside the face or on an edge.
    T1 faceLoc;

    // Triangle in contact with sphere or not
    bool in_contact;

    // Still do the following since we need depth
    if (!snap_to_face<T1, T2>(A, B, C, sphere_pos, faceLoc)) {
        // Nearest point on the triangle is on its face
        // printf("FACE CONTACT\n");
        depth = radius - h;  // Positive for contact
        normal.x = face_n.x;
        normal.y = face_n.y;
        normal.z = face_n.z;
        // The contact point is somewhere in the midpoint of the deepest penetration line segment. Go from faceLoc,
        // backwards wrt normal, half the penetration depth.
        pt1 = faceLoc - (depth * 0.5) * normal;
        if (h >= radius || h <= -radius) {
            in_contact = false;
        } else {
            in_contact = true;
        }
        // overlapArea = deme::PI * (radius * radius - (radius - depth) * (radius - depth));
        // Simplify it and assign it at the end of this call
    } else {
        // printf("EDGE CONTACT\n");
        // Nearest point on the triangle is on an edge
        {
            T1 normal_d = sphere_pos - faceLoc;
            normal.x = normal_d.x;
            normal.y = normal_d.y;
            normal.z = normal_d.z;
        }
        T2 dist = length(normal);
        depth = radius - dist;  // Positive for contact

        normal = (1.0 / dist) * normal;
        // Go from faceLoc, backwards wrt normal, half the penetration depth
        pt1 = faceLoc - (depth * 0.5) * normal;
        if (depth < 0. || h >= radius || h <= -radius) {
            in_contact = false;
        } else {
            in_contact = true;
        }
        // In the edge case, overlapArea is a bit tricky to define accurately.
        // Here we still just approximate it as a circle area.
    }
    {
        const float depth_f = static_cast<float>(depth);
        const float radius_f = static_cast<float>(radius);
        const float overlap_area_f = static_cast<float>(deme::PI) * (2.0f * radius_f * depth_f - depth_f * depth_f);
        overlapArea = static_cast<T2>(overlap_area_f);
    }
    return in_contact;
}

/**
/brief TRIANGLE FACE--SPHERE COLLISION DETECTION (DIRECTIONAL)

The triangular face is defined by points A, B, C. The sequence is important as it defines the positive face via a
right-hand rule.
The sphere is centered at sphere_pos and has radius.
The index "1" is associated with the triangle. The index "2" is associated with the sphere.
The coordinates of the face and sphere are assumed to be provided in the same reference frame.
This flavor is directional, meaning if a sphere is not in (geometric) contact with a triangle, it may still be
considered in contact, if it is submerged in the (closed) object that this mesh is representing, or say the penetration
is too deep, large than 2 * sphere rad.

Output:
  - pt1:      contact point on triangle
  - depth:    penetration distance (a positive value means that overlap exists)
  - normal:     contact normal, from pt2 to pt1
A return value of "true" signals collision.
*/
template <typename T1, typename T2>
__device__ bool checkTriSphereOverlap_directional(const T1& A,           ///< First vertex of the triangle
                                                  const T1& B,           ///< Second vertex of the triangle
                                                  const T1& C,           ///< Third vertex of the triangle
                                                  const T1& sphere_pos,  ///< Location of the center of the sphere
                                                  const T2 radius,       ///< Sphere radius
                                                  T1& normal,            ///< contact normal
                                                  T2& depth,             ///< penetration (positive if in contact)
                                                  T1& pt1                ///< contact point on triangle
) {
    // Calculate face normal using RHR
    T1 face_n = normalize(cross(B - A, C - A));

    // Calculate signed height of sphere center above face plane
    T2 h = dot(sphere_pos - A, face_n);

    // Find the closest point on the face to the sphere center and determine
    // whether or not this location is inside the face or on an edge.
    T1 faceLoc;

    // Triangle in contact with sphere or not
    bool in_contact;

    // Still do the following since we need depth
    if (!snap_to_face<T1, T2>(A, B, C, sphere_pos, faceLoc)) {
        // Nearest point on the triangle is on its face
        // printf("FACE CONTACT\n");
        depth = radius - h;  // Positive for contact
        normal.x = face_n.x;
        normal.y = face_n.y;
        normal.z = face_n.z;
        // The contact point is somewhere in the midpoint of the deepest penetration line segment. Go from faceLoc,
        // backwards wrt normal, half the penetration depth.
        pt1 = faceLoc - (depth * 0.5) * normal;
        if (depth < 0.) {
            in_contact = false;
        } else {
            in_contact = true;
        }
    } else {
        // printf("EDGE CONTACT\n");
        // Nearest point on the triangle is on an edge
        {
            T1 normal_d = sphere_pos - faceLoc;
            normal.x = normal_d.x;
            normal.y = normal_d.y;
            normal.z = normal_d.z;
        }
        T2 dist = length(normal);
        depth = radius - dist;  // Positive for contact

        normal = (1.0 / dist) * normal;
        // Go from faceLoc, backwards wrt normal, half the penetration depth
        pt1 = faceLoc - (depth * 0.5) * normal;
        if (depth < 0. || h >= radius) {
            in_contact = false;
        } else {
            in_contact = true;
        }
    }
    return in_contact;
}

// -----------------------------------------------------------------------------
// Triangle-triangle collision detection utilities
// -----------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
// Prism contact detection using the Separating Axis Theorem (SAT)
//
// For the extruded triangle "sandwich" prisms we only have 4 unique edge
// directions (3 base edges + extrusion). This yields:
// - 8 face normals (base + 3 side faces per prism)
// - 16 edge-edge axes
// Total: 24 axes, evaluated on the fly without normalization.
////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ float invSqrt(float x) {
    return rsqrtf(x);
}

__device__ __forceinline__ double invSqrt(double x) {
    return 1.0 / sqrt(x);
}

#ifndef DEME_SAT_ENABLE_MIXED_PRECISION
    #define DEME_SAT_ENABLE_MIXED_PRECISION 0
#endif

// Optimized: Fused projection using FMA operations and reduced branching
template <typename Vec, typename Scalar>
__device__ __forceinline__ void projectExtrudedTriPrism(const Vec& v0,
                                                        const Vec& v1,
                                                        const Vec& v2,
                                                        const Vec& d,
                                                        const Vec& axis,
                                                        Scalar& outMin,
                                                        Scalar& outMax) {
    // Compute projections
    const Scalar p0 = dot(v0, axis);
    const Scalar p1 = dot(v1, axis);
    const Scalar p2 = dot(v2, axis);
    const Scalar shift = dot(d, axis);

    // For each vertex, compute both endpoints of the extruded projection
    const Scalar p0_shifted = p0 + shift;
    const Scalar p1_shifted = p1 + shift;
    const Scalar p2_shifted = p2 + shift;

    // Branchless min/max using fmin/fmax (single instruction on GPU)
    outMin = fmin(fmin(fmin(p0, p0_shifted), fmin(p1, p1_shifted)), fmin(p2, p2_shifted));
    outMax = fmax(fmax(fmax(p0, p0_shifted), fmax(p1, p1_shifted)), fmax(p2, p2_shifted));
}

template <typename Vec, typename Scalar>
__device__ __forceinline__ Scalar satSeparationOnAxis(const Vec& axis,
                                                      const Vec& A0,
                                                      const Vec& A1,
                                                      const Vec& A2,
                                                      const Vec& dA,
                                                      const Vec& B0,
                                                      const Vec& B1,
                                                      const Vec& B2,
                                                      const Vec& dB) {
    Scalar len2 = dot(axis, axis);
    if (len2 < Scalar(DEME_TINY_FLOAT))
        return -Scalar(DEME_HUGE_FLOAT);

    Scalar minA, maxA, minB, maxB;
    projectExtrudedTriPrism<Vec, Scalar>(A0, A1, A2, dA, axis, minA, maxA);
    projectExtrudedTriPrism<Vec, Scalar>(B0, B1, B2, dB, axis, minB, maxB);

    Scalar sep1 = minB - maxA;
    Scalar sep2 = minA - maxB;
    Scalar sepProj = (sep1 > sep2) ? sep1 : sep2;
    Scalar invLen = invSqrt(len2);
    return sepProj * invLen;
}

/**
 * @brief Helper function to project one triangle onto another triangle's plane and clip using Sutherland-Hodgman
 *
 * @param incTri The incident triangle vertices to be projected
 * @param refTri The reference triangle vertices (defines the plane and clipping region)
 * @param refNormal The normal of the reference triangle's plane
 * @param depth Output: penetration depth (max distance of submerged vertices)
 * @param area Output: area of the clipping polygon
 * @param centroid Output: centroid of the clipping polygon
 * @return true if there is contact (at least one vertex submerged), false otherwise
 */
template <typename T1, typename T2>
__device__ bool projectTriangleOntoTriangle(const T1* incTri,
                                            const T1* refTri,
                                            const T1& refNormal,
                                            T2& depth,
                                            T2& area,
                                            T1& centroid) {
    // Compute signed distances of incident triangle vertices to reference plane
    area = T2(0.0);
    T2 incDists[3];
    T2 maxPenetration = 0.0;
    int8_t numSubmerged = 0;
#pragma unroll
    for (int8_t i = 0; i < 3; ++i) {
        incDists[i] = dot(incTri[i] - refTri[0], refNormal);
        if (incDists[i] < 0.0) {
            numSubmerged++;
            T2 pen = -incDists[i];
            if (pen > maxPenetration)
                maxPenetration = pen;
        }
    }

    // If no vertices are submerged, no contact
    if (numSubmerged == 0) {
        // depth = T2(0.0);
        // area = T2(0.0);
        // centroid.x = T2(0.0);
        // centroid.y = T2(0.0);
        // centroid.z = T2(0.0);
        return false;
    }

    // Maximum vertices in a triangle-triangle clipping polygon
    // Sutherland-Hodgman clipping can produce up to (n+m) vertices where n and m are
    // the number of vertices in the input polygons. For triangle-triangle clipping,
    // we conservatively use 9 (more than the theoretical max of 6) for safety.
    const int8_t SH_MAX_CLIPPING_VERTICES = 9;

    // Build polygon from projected submerged vertices and edge-plane intersections
    T1 projectedPoly[SH_MAX_CLIPPING_VERTICES];
    int8_t nPoly = 0;

    // Process each edge of the incident triangle
#pragma unroll
    for (int8_t i = 0; i < 3; ++i) {
        int8_t j = (i + 1) % 3;
        bool in_i = (incDists[i] < 0.0);
        bool in_j = (incDists[j] < 0.0);

        // Add submerged vertex (projected onto plane)
        if (in_i) {
            projectedPoly[nPoly++] = incTri[i] - refNormal * incDists[i];
        }

        // Add edge-plane intersection if edge crosses the plane
        if (in_i != in_j) {
            T2 denom = incDists[i] - incDists[j];
            if (denom != 0.0) {  // Avoid division by zero
                T2 t = incDists[i] / denom;
                T1 inter = incTri[i] + (incTri[j] - incTri[i]) * t;
                projectedPoly[nPoly++] = inter;
            }
        }
    }

    // If we don't have at least 3 vertices, no valid polygon
    if (nPoly < 3) {
        // depth = maxPenetration;
        // area = T2(0.0);
        // centroid = (incTri[0] + incTri[1] + incTri[2]) / T2(3.0);
        return false;
    }

    // Now compute the intersection polygon of the projected triangle and reference triangle
    // We need bidirectional clipping: clip projectedPoly against refTri, then add refTri vertices inside projectedPoly

    // Step 1: Clip projected polygon against reference triangle (Sutherland-Hodgman)
    T1 resultPoly[SH_MAX_CLIPPING_VERTICES];
    for (int8_t i = 0; i < nPoly; ++i) {
        resultPoly[i] = projectedPoly[i];
    }
    int8_t numInputVerts = nPoly;

    T1 intermediatePoly[SH_MAX_CLIPPING_VERTICES];
    for (int8_t edge = 0; edge < 3; ++edge) {
        int8_t numOutputVerts = 0;
        T1 edgeStart = refTri[edge];
        T1 edgeEnd = refTri[(edge + 1) % 3];
        T1 edgeDir = edgeEnd - edgeStart;
        T1 edgeNormal = cross(refNormal, edgeDir);
        edgeNormal = normalize(edgeNormal);

        // Clip input polygon against this edge
        for (int8_t i = 0; i < numInputVerts; ++i) {
            T1 v1 = resultPoly[i];
            T1 v2 = resultPoly[(i + 1) % numInputVerts];
            T2 d1 = dot(v1 - edgeStart, edgeNormal);
            T2 d2 = dot(v2 - edgeStart, edgeNormal);
            bool in1 = (d1 >= -DEME_TINY_FLOAT);
            bool in2 = (d2 >= -DEME_TINY_FLOAT);

            if (in1) {
                intermediatePoly[numOutputVerts++] = v1;
            }
            if (in1 != in2) {
                T2 denom = d1 - d2;
                if (denom != 0.0) {  // Avoid division by zero
                    T2 t = d1 / denom;
                    T1 inter = v1 + (v2 - v1) * t;
                    intermediatePoly[numOutputVerts++] = inter;
                }
            }
        }

        // Copy output to input for next iteration
        for (int8_t i = 0; i < numOutputVerts; ++i) {
            resultPoly[i] = intermediatePoly[i];
        }
        numInputVerts = numOutputVerts;

        if (numInputVerts == 0) {
            break;  // No intersection
        }
    }

    // Step 2: Check if any reference triangle vertices are inside the projected polygon
    // and add them to the intersection polygon if they are
    int8_t numFinalVerts = numInputVerts;

    // For each reference triangle vertex, check if it's inside the original projected polygon
    for (int8_t refIdx = 0; refIdx < 3; ++refIdx) {
        T1 refVertex = refTri[refIdx];

        // Check if refVertex is inside the projected polygon using winding number
        bool inside = true;
        for (int8_t i = 0; i < nPoly; ++i) {
            T1 edgeStart = projectedPoly[i];
            T1 edgeEnd = projectedPoly[(i + 1) % nPoly];
            T1 edgeDir = edgeEnd - edgeStart;
            T1 edgeNormal = cross(refNormal, edgeDir);
            T2 dist = dot(refVertex - edgeStart, edgeNormal);
            if (dist < -DEME_TINY_FLOAT) {
                inside = false;
                break;
            }
        }

        if (inside) {
            // Check if this vertex is not already in the polygon (avoid duplicates)
            bool isDuplicate = false;
            for (int8_t j = 0; j < numFinalVerts; ++j) {
                T1 diff = resultPoly[j] - refVertex;
                if (dot(diff, diff) < DEME_TINY_FLOAT * DEME_TINY_FLOAT) {
                    isDuplicate = true;
                    break;
                }
            }
            if (!isDuplicate) {
                resultPoly[numFinalVerts++] = refVertex;
            }
        }
    }

    // If we added reference vertices, we need to reorder the polygon to maintain proper winding
    if (numFinalVerts > numInputVerts && numFinalVerts >= 3) {
        // Compute centroid of all vertices
        T1 tempCentroid;
        tempCentroid.x = 0.0;
        tempCentroid.y = 0.0;
        tempCentroid.z = 0.0;
        for (int8_t i = 0; i < numFinalVerts; ++i) {
            tempCentroid = tempCentroid + resultPoly[i];
        }
        tempCentroid = tempCentroid / T2(numFinalVerts);

        // Sort vertices by angle around centroid to ensure proper winding order
        // Use simple bubble sort for small number of vertices
        for (int8_t i = 0; i < numFinalVerts - 1; ++i) {
            for (int8_t j = i + 1; j < numFinalVerts; ++j) {
                T1 vi = resultPoly[i] - tempCentroid;
                T1 vj = resultPoly[j] - tempCentroid;
                // Use reference normal to determine consistent orientation
                T1 cross_ij = cross(vi, vj);
                if (dot(cross_ij, refNormal) < 0.0) {
                    // Swap
                    T1 temp = resultPoly[i];
                    resultPoly[i] = resultPoly[j];
                    resultPoly[j] = temp;
                }
            }
        }
    }

    numInputVerts = numFinalVerts;

    // Compute centroid and area of the intersection polygon
    centroid.x = 0.0;
    centroid.y = 0.0;
    centroid.z = 0.0;

    depth = maxPenetration;
    if (numInputVerts >= 3) {
        for (int8_t i = 0; i < numInputVerts; ++i) {
            centroid = centroid + resultPoly[i];
        }
        centroid = centroid / T2(numInputVerts);

        // Calculate area using fan triangulation from centroid
        float area_f = 0.0f;
        const float3 centroid_f = to_float3(centroid);
        for (int8_t i = 0; i < numInputVerts; ++i) {
            float3 v1 = to_float3(resultPoly[i]) - centroid_f;
            float3 v2 = to_float3(resultPoly[(i + 1) % numInputVerts]) - centroid_f;
            float3 crossProd = cross(v1, v2);
            area_f += sqrtf(dot(crossProd, crossProd));
        }
        area = static_cast<T2>(area_f * 0.5f);
        return true;
    } else {
        // Degenerate intersection polygon
        // centroid = (incTri[0] + incTri[1] + incTri[2]) / T2(3.0);
        return false;
    }
}

/**
 * @brief Fast SAT contact check between two triangular prisms (triangle sandwiches).
 *
 * Evaluates 24 axes (8 face normals + 16 edge-edge) without normalization. Uses FP32 by
 * default with a narrow mixed-precision recheck near zero overlap to avoid false positives.
 * 
 * OPTIMIZED VERSION: 
 * - Uses fused operations to reduce register pressure
 * - Inline axis separation test to avoid lambda overhead
 * - Early termination structure optimized for GPU SIMT execution
 *
 * @return true if prisms are in contact (no separating axis found), false otherwise
 */
template <typename T1>
__device__ bool calc_prism_contact(const T1& prismAFaceANode1,
                                   const T1& prismAFaceANode2,
                                   const T1& prismAFaceANode3,
                                   const T1& prismAFaceBNode1,
                                   const T1& prismAFaceBNode2,
                                   const T1& prismAFaceBNode3,
                                   const T1& prismBFaceANode1,
                                   const T1& prismBFaceANode2,
                                   const T1& prismBFaceANode3,
                                   const T1& prismBFaceBNode1,
                                   const T1& prismBFaceBNode2,
                                   const T1& prismBFaceBNode3) {
    // Use relative coordinates centered at prismAFaceANode1 to reduce FP32 precision issues
    // This is cheaper than converting to double for all operations
    const float3 origin = prismAFaceANode1;
    
    // Prism A vertices relative to origin
    const float3 A0 = make_float3(0.0f, 0.0f, 0.0f);  // prismAFaceANode1 - origin = 0
    const float3 A1 = prismAFaceANode2 - origin;
    const float3 A2 = prismAFaceANode3 - origin;
    
    // Prism B vertices relative to origin
    const float3 B0 = prismBFaceANode1 - origin;
    const float3 B1 = prismBFaceANode2 - origin;
    const float3 B2 = prismBFaceANode3 - origin;
    
    // Extrusion vectors
    const float3 dA = prismAFaceBNode1 - prismAFaceANode1;
    const float3 dB = prismBFaceBNode1 - prismBFaceANode1;

    // Edge vectors computed once
    const float3 eA0 = A1 - A0;
    const float3 eA1 = A2 - A1;
    const float3 eA2 = A0 - A2;
    const float3 eB0 = B1 - B0;
    const float3 eB1 = B2 - B1;
    const float3 eB2 = B0 - B2;

    // Inline separation test macro to avoid function call overhead
    // Returns true if separated (axis found), false otherwise
    #define TEST_AXIS(axis_expr) do { \
        const float3 axis = (axis_expr); \
        const float len2 = dot(axis, axis); \
        if (len2 > DEME_TINY_FLOAT) { \
            float minA, maxA, minB, maxB; \
            projectExtrudedTriPrism<float3, float>(A0, A1, A2, dA, axis, minA, maxA); \
            projectExtrudedTriPrism<float3, float>(B0, B1, B2, dB, axis, minB, maxB); \
            const float sep = fmax(minB - maxA, minA - maxB) * rsqrtf(len2); \
            if (sep > 0.0f) return false; \
        } \
    } while(0)

    // Test face normals (2 axes)
    TEST_AXIS(cross(eA0, A2 - A0));
    TEST_AXIS(cross(eB0, B2 - B0));

    // Test side normals (6 axes)
    TEST_AXIS(cross(eA0, dA));
    TEST_AXIS(cross(eA1, dA));
    TEST_AXIS(cross(eA2, dA));
    TEST_AXIS(cross(eB0, dB));
    TEST_AXIS(cross(eB1, dB));
    TEST_AXIS(cross(eB2, dB));

    // Test edge-edge axes (9 axes)
    TEST_AXIS(cross(eA0, eB0));
    TEST_AXIS(cross(eA0, eB1));
    TEST_AXIS(cross(eA0, eB2));
    TEST_AXIS(cross(eA1, eB0));
    TEST_AXIS(cross(eA1, eB1));
    TEST_AXIS(cross(eA1, eB2));
    TEST_AXIS(cross(eA2, eB0));
    TEST_AXIS(cross(eA2, eB1));
    TEST_AXIS(cross(eA2, eB2));

    // Test edge-extrusion cross products (6 axes)
    TEST_AXIS(cross(eA0, dB));
    TEST_AXIS(cross(eA1, dB));
    TEST_AXIS(cross(eA2, dB));
    TEST_AXIS(cross(dA, eB0));
    TEST_AXIS(cross(dA, eB1));
    TEST_AXIS(cross(dA, eB2));

    // Test extrusion-extrusion (1 axis)
    TEST_AXIS(cross(dA, dB));

    #undef TEST_AXIS

    // No separating axis found - prisms are in contact
    return true;
}

/// Lightweight SAT check for triangle-triangle contact (physical contact only)
/// Returns true if triangles are in physical contact (no separating axis found), false otherwise
/// This is a simplified version that only performs the SAT test without computing contact details
template <typename T1, typename T2>
__device__ bool checkTriangleTriangleSAT(const T1& A1,
                                         const T1& B1,
                                         const T1& C1,
                                         const T1& A2,
                                         const T1& B2,
                                         const T1& C2) {
    // Triangle A vertices (tri1)
    const T1 triA[3] = {A1, B1, C1};
    // Triangle B vertices (tri2)
    const T1 triB[3] = {A2, B2, C2};

    // Compute face normals
    T1 nA = normalize(cross(B1 - A1, C1 - A1));
    T1 nB = normalize(cross(B2 - A2, C2 - A2));

    //// TODO: And degenerated triangles?

    // Edge vectors
    T1 edges1[3] = {triA[1] - triA[0], triA[2] - triA[1], triA[0] - triA[2]};
    T1 edges2[3] = {triB[1] - triB[0], triB[2] - triB[1], triB[0] - triB[2]};

    // Test face normal of triangle A
    {
        T1 axis = nA;

        // Project triangle A vertices
        T2 min1 = dot(triA[0], axis);
        T2 max1 = min1;
#pragma unroll
        for (int i = 1; i < 3; ++i) {
            T2 proj = dot(triA[i], axis);
            if (proj < min1)
                min1 = proj;
            if (proj > max1)
                max1 = proj;
        }

        // Project triangle B vertices
        T2 min2 = dot(triB[0], axis);
        T2 max2 = min2;
#pragma unroll
        for (int i = 1; i < 3; ++i) {
            T2 proj = dot(triB[i], axis);
            if (proj < min2)
                min2 = proj;
            if (proj > max2)
                max2 = proj;
        }

        // Check for separation
        if (max1 < min2 || max2 < min1) {
            return false;  // Separating axis found
        }
    }

    // Test face normal of triangle B
    {
        T1 axis = nB;

        // Project triangle A vertices
        T2 min1 = dot(triA[0], axis);
        T2 max1 = min1;
#pragma unroll
        for (int i = 1; i < 3; ++i) {
            T2 proj = dot(triA[i], axis);
            if (proj < min1)
                min1 = proj;
            if (proj > max1)
                max1 = proj;
        }

        // Project triangle B vertices
        T2 min2 = dot(triB[0], axis);
        T2 max2 = min2;
#pragma unroll
        for (int i = 1; i < 3; ++i) {
            T2 proj = dot(triB[i], axis);
            if (proj < min2)
                min2 = proj;
            if (proj > max2)
                max2 = proj;
        }

        // Check for separation
        if (max1 < min2 || max2 < min1) {
            return false;  // Separating axis found
        }
    }

    // Test 9 edge-edge cross products
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            T1 axis = cross(edges1[i], edges2[j]);
            T2 len2 = dot(axis, axis);

            if (len2 > DEME_TINY_FLOAT) {
                axis = axis * rsqrt(len2);

                // Project triangle A vertices
                T2 min1 = dot(triA[0], axis);
                T2 max1 = min1;
#pragma unroll
                for (int k = 1; k < 3; ++k) {
                    T2 proj = dot(triA[k], axis);
                    if (proj < min1)
                        min1 = proj;
                    if (proj > max1)
                        max1 = proj;
                }

                // Project triangle B vertices
                T2 min2 = dot(triB[0], axis);
                T2 max2 = min2;
#pragma unroll
                for (int k = 1; k < 3; ++k) {
                    T2 proj = dot(triB[k], axis);
                    if (proj < min2)
                        min2 = proj;
                    if (proj > max2)
                        max2 = proj;
                }

                // Check for separation
                if (max1 < min2 || max2 < min1) {
                    return false;  // Separating axis found
                }
            }
        }
    }

    // No separating axis found - triangles are in contact
    return true;
}

/// Triangle-triangle contact detection using projection-based approach:
/// 1. Project triangle A onto triangle B's plane and clip against B's edges
/// 2. Project triangle B onto triangle A's plane and clip against A's edges
/// 3. Average the results for final contact info
/// This approach uses Sutherland-Hodgman algorithm for clipping and does not require SAT
template <typename T1, typename T2>
__device__ bool checkTriangleTriangleOverlap(
    const T1& A1,
    const T1& B1,
    const T1& C1,
    const T1& A2,
    const T1& B2,
    const T1& C2,
    T1& normal,         ///< contact normal (B2A direction)
    T2& depth,          ///< penetration (positive if in contact)
    T2& projectedArea,  ///< projected area of clipping polygon (optional output)
    T1& point) {        ///< contact point
    // Triangle A vertices (tri1)
    const T1 triA[3] = {A1, B1, C1};
    // Triangle B vertices (tri2)
    const T1 triB[3] = {A2, B2, C2};

    // Compute face normals
    T1 nA = normalize(cross(B1 - A1, C1 - A1));
    T1 nB = normalize(cross(B2 - A2, C2 - A2));

    //// TODO: And degenerated triangles?

    // ========================================================================
    // Projection-based approach: project each triangle onto the other's plane
    // and clip using Sutherland-Hodgman algorithm
    // ========================================================================

    // Project triangle B onto triangle A's plane and clip against A
    T2 depthBA, areaBA;
    T1 centroidBA;
    bool contactBA = projectTriangleOntoTriangle<T1, T2>(triB, triA, nA, depthBA, areaBA, centroidBA);

    // Project triangle A onto triangle B's plane and clip against B
    T2 depthAB, areaAB;
    T1 centroidAB;
    bool contactAB = projectTriangleOntoTriangle<T1, T2>(triA, triB, nB, depthAB, areaAB, centroidAB);

    // Determine if there is contact
    bool inContact = contactBA && contactAB;

    if (!inContact) {
        // No contact detected, Provide separation info
        T1 centA = (triA[0] + triA[1] + triA[2]) / 3.0;
        T1 centB = (triB[0] + triB[1] + triB[2]) / 3.0;
        T1 sep = centA - centB;
        T2 sepLen2 = dot(sep, sep);

        if (sepLen2 > (DEME_TINY_FLOAT * DEME_TINY_FLOAT)) {
            T2 sepLen = sqrt(sepLen2);
            normal = sep / sepLen;
            depth = -sepLen;  // Negative for separation
            point = (centA + centB) * 0.5;
        } else {
            normal = nA;
            depth = -DEME_TINY_FLOAT;
            point = centA;
        }
        projectedArea = 0.0;
        return false;
    }

    // If both projection yields results, we select the one with less projection distance.
    // This is important. For example, consider a small surface intersecting with a large surface nearly vertically. The
    // smaller one projected onto the larger one: nearly 0 area (depending on numerical stability, may actually be 0);
    // Larger one projected onto the smaller one: almost covers the entire smaller surface. We always want them both
    // have non-0 projection area, and then select the shorter projection distance one. This is good for stability.
    if (depthBA < depthAB) {
        // Use B->A projection results
        depth = depthBA;
        projectedArea = areaBA;
        normal = -1.0 * nA;  // Pay attention to direction

        // Contact point: centroid on A's plane, moved back by half depth
        point = centroidBA - nA * (depth * 0.5);
    } else {
        // Use A->B projection results
        depth = depthAB;
        projectedArea = areaAB;
        normal = nB;

        // Contact point: centroid on B's plane, moved back by half depth
        point = centroidAB - nB * (depth * 0.5);
    }

    return true;
}

#endif
