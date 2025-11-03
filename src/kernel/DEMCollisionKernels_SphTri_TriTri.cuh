// DEM collision-related kernel collection

#ifndef DEME_COLLI_KERNELS_ST_TT_CUH
#define DEME_COLLI_KERNELS_ST_TT_CUH

#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>

// ------------------------------------------------------------------
// Triangle-analytical object collision detection utilities
// ------------------------------------------------------------------

template <typename T1, typename T2>
inline bool __device__ tri_plane_penetration(const T1** tri,
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
        centroid = centroid / T2(3);
    }

    // We use the convention that if in contact, overlapDepth is positive
    overlapDepth = -dmin;
    bool in_contact = (overlapDepth >= 0.);
    // The centroid's projection to the plane
    T1 projection =
        centroid - planeSignedDistance<T2>(centroid, entityLoc, entityDir) * to_real3<float3, T1>(entityDir);

    // Calculate the area of the clipping polygon using fan triangulation from centroid
    overlapArea = 0.0;
    if (hasIntersection && nNode >= 3) {
        for (int i = 0; i < nNode; ++i) {
            T1 v1 = poly[i] - centroid;
            T1 v2 = poly[(i + 1) % nNode] - centroid;
            T1 crossProd = cross(v1, v2);
            overlapArea += sqrt(dot(crossProd, crossProd));
        }
        overlapArea *= 0.5;
    }

    // cntPnt is from the projection point, go half penetration depth.
    // Note this penetration depth is signed, so if no contact, we go positive plane normal; if in contact, we go
    // negative plane normal. As such, cntPnt always exists and this is important for the cases with extraMargin.
    contactPnt = projection - (overlapDepth * 0.5) * to_real3<float3, T1>(entityDir);
    return in_contact;
}

template <typename T1, typename T2>
inline bool __device__ tri_cyl_penetration(const T1** tri,
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
inline __host__ __device__ deme::contact_t checkTriEntityOverlap(const T1& A,
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

// NOTE: Due to our algorithm needs a overlapDepth even in the case of no contact (because of extraMargin; and with
// extraMargin, negative overlapDepth can be considered in-contact), our tri-anal CD algorithm needs to always return a
// overlapDepth, even in the case of no contact. This is different from the usual CD algorithms which only return a
// overlapDepth. Also unlike tri-sph contact which has a unified util function, calcTriEntityOverlap is different from
// checkTriEntityOverlap.
template <typename T1, typename T2>
inline bool __device__ calcTriEntityOverlap(const T1& A,
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
    overlapArea = deme::PI * (2.0 * radius * depth - depth * depth);
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
// A prism is formed by two parallel triangular faces (bases) connected by three
// rectangular side faces. For proper contact detection between two prisms, SAT
// requires testing multiple potential separating axes:
//
// 1. Face normals of both triangular bases (2 axes)
// 2. Face normals of all rectangular side faces (6 axes, 3 per prism)
// 3. Cross products of edges from different prisms to detect edge-edge contacts
//    - Base edges × Base edges (9 axes)
//    - Height edges × Base edges (18 axes)
//
// This comprehensive approach ensures detection of:
// - Face-face contacts (parallel prisms)
// - Edge-face contacts (side intersecting base/side)
// - Edge-edge contacts
// - Complete containment (one prism inside another)
////////////////////////////////////////////////////////////////////////////////

template <typename T1, typename T2>
inline __device__ void select_projection(const T1& pts, const T1& axis, T2& min_p, T2& max_p) {
    T2 p = dot(pts, axis);
    if (p < min_p)
        min_p = p;
    if (p > max_p)
        max_p = p;
}

template <typename T1, typename T2>
inline __device__ void project_points_on_axis(const T1* prism, const T1& axis, T2& out_min, T2& out_max) {
    T2 min_p = dot(prism[0], axis);
    T2 max_p = min_p;
    for (int i = 1; i < 6; ++i) {
        select_projection(prism[i], axis, min_p, max_p);
    }
    out_min = min_p;
    out_max = max_p;
}

template <typename T>
inline __device__ bool projections_overlap(T minA, T maxA, T minB, T maxB) {
    return !(maxA < minB || maxB < minA);
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
inline __device__ bool projectTriangleOntoTriangle(const T1* incTri,
                                                   const T1* refTri,
                                                   const T1& refNormal,
                                                   T2& depth,
                                                   T2& area,
                                                   T1& centroid) {
    // Compute signed distances of incident triangle vertices to reference plane
    T2 incDists[3];
    T2 maxPenetration = T2(0.0);
    int numSubmerged = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
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
    const unsigned short int SH_MAX_CLIPPING_VERTICES = 9;

    // Build polygon from projected submerged vertices and edge-plane intersections
    T1 projectedPoly[SH_MAX_CLIPPING_VERTICES];
    int nPoly = 0;

    // Process each edge of the incident triangle
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        int j = (i + 1) % 3;
        bool in_i = (incDists[i] < 0.0);
        bool in_j = (incDists[j] < 0.0);

        // Add submerged vertex (projected onto plane)
        if (in_i && nPoly < SH_MAX_CLIPPING_VERTICES) {
            projectedPoly[nPoly++] = incTri[i] - refNormal * incDists[i];
        }

        // Add edge-plane intersection if edge crosses the plane
        if (in_i != in_j && nPoly < SH_MAX_CLIPPING_VERTICES) {
            T2 denom = incDists[i] - incDists[j];
            if (denom != T2(0.0)) {  // Avoid division by zero
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

    // Now clip the projected polygon against the reference triangle using Sutherland-Hodgman
    T1 inputPoly[SH_MAX_CLIPPING_VERTICES];
    for (int i = 0; i < nPoly; ++i) {
        inputPoly[i] = projectedPoly[i];
    }
    int numInputVerts = nPoly;

    T1 outputPoly[SH_MAX_CLIPPING_VERTICES];
    for (int edge = 0; edge < 3; ++edge) {
        int numOutputVerts = 0;
        T1 edgeStart = refTri[edge];
        T1 edgeEnd = refTri[(edge + 1) % 3];
        T1 edgeDir = edgeEnd - edgeStart;
        T1 edgeNormal = cross(refNormal, edgeDir);
        edgeNormal = normalize(edgeNormal);

        // Clip input polygon against this edge
        for (int i = 0; i < numInputVerts; ++i) {
            T1 v1 = inputPoly[i];
            T1 v2 = inputPoly[(i + 1) % numInputVerts];
            T2 d1 = dot(v1 - edgeStart, edgeNormal);
            T2 d2 = dot(v2 - edgeStart, edgeNormal);
            bool in1 = (d1 >= -DEME_TINY_FLOAT);
            bool in2 = (d2 >= -DEME_TINY_FLOAT);

            if (in1 && numOutputVerts < SH_MAX_CLIPPING_VERTICES) {
                outputPoly[numOutputVerts++] = v1;
            }
            if (in1 != in2 && numOutputVerts < SH_MAX_CLIPPING_VERTICES) {
                T2 denom = d1 - d2;
                if (denom != T2(0.0)) {  // Avoid division by zero
                    T2 t = d1 / denom;
                    T1 inter = v1 + (v2 - v1) * t;
                    outputPoly[numOutputVerts++] = inter;
                }
            }
        }

        // Copy output to input for next iteration
        for (int i = 0; i < numOutputVerts; ++i) {
            inputPoly[i] = outputPoly[i];
        }
        numInputVerts = numOutputVerts;

        if (numInputVerts == 0) {
            break;  // No intersection
        }
    }

    // Compute centroid and area of the clipping polygon
    centroid.x = T2(0.0);
    centroid.y = T2(0.0);
    centroid.z = T2(0.0);

    area = T2(0.0);
    depth = maxPenetration;
    if (numInputVerts >= 3) {
        for (int i = 0; i < numInputVerts; ++i) {
            centroid = centroid + inputPoly[i];
        }
        centroid = centroid / T2(numInputVerts);

        // Calculate area using fan triangulation from centroid
        for (int i = 0; i < numInputVerts; ++i) {
            T1 v1 = inputPoly[i] - centroid;
            T1 v2 = inputPoly[(i + 1) % numInputVerts] - centroid;
            T1 crossProd = cross(v1, v2);
            area += sqrt(dot(crossProd, crossProd));
        }
        area *= T2(0.5);
        return true;
    } else {
        // Degenerate clipping polygon
        // centroid = (incTri[0] + incTri[1] + incTri[2]) / T2(3.0);
        return false;
    }
}

/**
 * @brief Detect contact between two triangular prisms using comprehensive SAT.
 *
 * Each prism is defined by two triangular faces (Face A and Face B) with 3 vertices each.
 * Vertices are ordered: Face A nodes 1-3, then Face B nodes 1-3 (corresponding vertices).
 * The function tests up to 35 potential separating axes to ensure complete coverage
 * of all contact scenarios including parallel prisms, side-side intersections, and
 * containment cases.
 *
 * @return true if prisms are in contact (no separating axis found), false otherwise
 */
template <typename T1>
inline __device__ bool calc_prism_contact(const T1& prismAFaceANode1,
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
    // Increased axis count to accommodate additional side face normals and height edge tests
    // Max axes: 2 base normals + 6 side face normals + 9 base edge-edge + 18 height edge cross products = 35
    float3 axes[35];
    unsigned short int axisCount = 0;

    // Pack as stack arrays for easier looping
    T1 prismA[6] = {prismAFaceANode1, prismAFaceANode2, prismAFaceANode3,
                    prismAFaceBNode1, prismAFaceBNode2, prismAFaceBNode3};
    T1 prismB[6] = {prismBFaceANode1, prismBFaceANode2, prismBFaceANode3,
                    prismBFaceBNode1, prismBFaceBNode2, prismBFaceBNode3};

    // Base triangle normals (both top and bottom faces)
    T1 A_faceNormal = cross(prismA[1] - prismA[0], prismA[2] - prismA[0]);
    T1 B_faceNormal = cross(prismB[1] - prismB[0], prismB[2] - prismB[0]);

    axes[axisCount++] = normalize(A_faceNormal);
    axes[axisCount++] = normalize(B_faceNormal);

    // Edges of each prism base and height edges (connecting corresponding vertices of the two bases)
    T1 A_baseEdges[3] = {prismA[1] - prismA[0], prismA[2] - prismA[1], prismA[0] - prismA[2]};
    T1 B_baseEdges[3] = {prismB[1] - prismB[0], prismB[2] - prismB[1], prismB[0] - prismB[2]};

    // Height edges connecting corresponding vertices of the two triangular bases
    // Note: Due to winding order to maintain opposite normals, the vertex correspondence is:
    // FaceA[0,1,2] corresponds to FaceB[0,2,1] (i.e., Node1->Node1, Node2->Node3, Node3->Node2)
    T1 A_heightEdges[3] = {prismA[3] - prismA[0], prismA[5] - prismA[1], prismA[4] - prismA[2]};
    T1 B_heightEdges[3] = {prismB[3] - prismB[0], prismB[5] - prismB[1], prismB[4] - prismB[2]};

    // Side face normals for prism A (3 rectangular side faces)
    // Each side face is formed by an edge of the base and the corresponding height edges
    for (unsigned short int i = 0; i < 3; ++i) {
        // For each base edge, compute the normal of the rectangular side face
        // The side face is formed by base edge i and the two height edges at its endpoints
        T1 sideNormal = cross(A_baseEdges[i], A_heightEdges[i]);
        float len = length(sideNormal);
        if (len > DEME_TINY_FLOAT)
            axes[axisCount++] = sideNormal / len;
    }

    // Side face normals for prism B (3 rectangular side faces)
    for (unsigned short int i = 0; i < 3; ++i) {
        T1 sideNormal = cross(B_baseEdges[i], B_heightEdges[i]);
        float len = length(sideNormal);
        if (len > DEME_TINY_FLOAT)
            axes[axisCount++] = sideNormal / len;
    }

    // Edge-edge cross products: base edges of A with base edges of B
    for (unsigned short int i = 0; i < 3; ++i) {
        for (unsigned short int j = 0; j < 3; ++j) {
            T1 cp = cross(A_baseEdges[i], B_baseEdges[j]);
            float len = length(cp);
            if (len > DEME_TINY_FLOAT)
                axes[axisCount++] = cp / len;
        }
    }

    // Edge-edge cross products: height edges of A with base edges of B
    for (unsigned short int i = 0; i < 3; ++i) {
        for (unsigned short int j = 0; j < 3; ++j) {
            T1 cp = cross(A_heightEdges[i], B_baseEdges[j]);
            float len = length(cp);
            if (len > DEME_TINY_FLOAT)
                axes[axisCount++] = cp / len;
        }
    }

    // Edge-edge cross products: base edges of A with height edges of B
    for (unsigned short int i = 0; i < 3; ++i) {
        for (unsigned short int j = 0; j < 3; ++j) {
            T1 cp = cross(A_baseEdges[i], B_heightEdges[j]);
            float len = length(cp);
            if (len > DEME_TINY_FLOAT)
                axes[axisCount++] = cp / len;
        }
    }

    // SAT test: check all computed axes
    // Note: This correctly handles the containment case (one prism completely inside another).
    // When prism A is inside prism B, for any axis, the projection range of A [minA, maxA]
    // will be contained within the projection range of B [minB, maxB], causing all overlap
    // checks to pass. With no separating axis found, the function returns true (in contact).
    for (unsigned short int i = 0; i < axisCount; ++i) {
        float minA, maxA, minB, maxB;
        project_points_on_axis(prismA, axes[i], minA, maxA);
        project_points_on_axis(prismB, axes[i], minB, maxB);
        if (!projections_overlap(minA, maxA, minB, maxB))
            return false;  // separating axis found -> no contact
    }

    /*
    // Contact confirmed... find lex smallest point from both prisms, used for the contact point
    // The contact point does not need to be accurate, but consistent in terms of two metrics:
    // 1. It should be the same for the same pair of prisms, regardless of their order.
    // 2. It should be in the bin that one of the triangles lives
    // And we use the computed midpoint of closest vertex pair
    T1 closestA = prismA[0];
    T1 closestB = prismB[0];
    float minDist2 = DEME_HUGE_FLOAT;

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            T1 diff = prismA[i] - prismB[j];
            float d2 = dot(diff, diff);
            if (d2 < minDist2 || (d2 == minDist2 && (i < j))) {
                minDist2 = d2;
                closestA = prismA[i];
                closestB = prismB[j];
            }
        }
    }
    contactPointOut = (closestA + closestB) * 0.5;
    */

    return true;
}

/// Triangle-triangle contact detection using projection-based approach:
/// 1. Project triangle A onto triangle B's plane and clip against B's edges
/// 2. Project triangle B onto triangle A's plane and clip against A's edges
/// 3. Average the results for final contact info
/// This approach uses Sutherland-Hodgman algorithm for clipping and does not require SAT
template <typename T1, typename T2>
inline __device__ bool checkTriangleTriangleOverlap(
    const T1& A1,
    const T1& B1,
    const T1& C1,
    const T1& A2,
    const T1& B2,
    const T1& C2,
    T1& normal,                      ///< contact normal (B2A direction)
    T2& depth,                       ///< penetration (positive if in contact)
    T2& projectedArea,               ///< projected area of clipping polygon (optional output)
    T1& point,                       ///< contact point
    bool outputNoContact = false) {  ///< output info even when no contact
    // Triangle A vertices (tri1)
    const T1 triA[3] = {A1, B1, C1};
    // Triangle B vertices (tri2)
    const T1 triB[3] = {A2, B2, C2};

    // Compute face normals
    T1 nA_unnorm = cross(B1 - A1, C1 - A1);
    T1 nB_unnorm = cross(B2 - A2, C2 - A2);

    T2 lenA2 = dot(nA_unnorm, nA_unnorm);
    T2 lenB2 = dot(nB_unnorm, nB_unnorm);

    // Check for degenerate triangles
    if (lenA2 <= DEME_TINY_FLOAT * DEME_TINY_FLOAT || lenB2 <= DEME_TINY_FLOAT * DEME_TINY_FLOAT) {
        return false;
    }

    T1 nA = nA_unnorm * rsqrt(lenA2);
    T1 nB = nB_unnorm * rsqrt(lenB2);

    // ========================================================================
    // Projection-based approach: project each triangle onto the other's plane
    // and clip using Sutherland-Hodgman algorithm
    // ========================================================================

    // Calculate reference triangle areas for axis hygiene check
    T2 areaRefA = T2(0.5) * sqrt(lenA2);
    T2 areaRefB = T2(0.5) * sqrt(lenB2);
    
    // Project triangle B onto triangle A's plane and clip against A
    T2 depthBA, areaBA;
    T1 centroidBA;
    bool contactBA = projectTriangleOntoTriangle<T1, T2>(triB, triA, nA, depthBA, areaBA, centroidBA);

    // Project triangle A onto triangle B's plane and clip against B
    T2 depthAB, areaAB;
    T1 centroidAB;
    bool contactAB = projectTriangleOntoTriangle<T1, T2>(triA, triB, nB, depthAB, areaAB, centroidAB);
    
    // Axis hygiene check: if projected area is suspiciously small, verify stability with rotated trials
    const T2 SUSPICIOUS_AREA_THRESHOLD = T2(0.05);  // 5% of reference area
    const T2 ROTATION_ANGLE = T2(0.1);  // Small rotation angle in radians (~5.7 degrees)
    
    if (contactBA && areaBA < SUSPICIOUS_AREA_THRESHOLD * areaRefA) {
        // Suspicious overlap - perform stability check with 4 trial rotations
        T2 minArea = areaBA;
        
        // Find two orthogonal directions in the plane of triangle A
        T1 edge0 = triA[1] - triA[0];
        T2 edge0Len2 = dot(edge0, edge0);
        if (edge0Len2 > DEME_TINY_FLOAT) {
            T1 tangent1 = edge0 * rsqrt(edge0Len2);
            T1 tangent2 = cross(nA, tangent1);
            
            // Try 4 rotations: +/- rotation around tangent1 and tangent2
            for (int dir = 0; dir < 2; ++dir) {
                T1 axis = (dir == 0) ? tangent1 : tangent2;
                for (int sign = -1; sign <= 1; sign += 2) {
                    T2 angle = T2(sign) * ROTATION_ANGLE;
                    // Rodrigues rotation formula: v' = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
                    T2 cosAngle = cos(angle);
                    T2 sinAngle = sin(angle);
                    T1 rotatedNormal = nA * cosAngle + cross(axis, nA) * sinAngle;
                    
                    T2 trialDepth, trialArea;
                    T1 trialCentroid;
                    bool trialContact = projectTriangleOntoTriangle<T1, T2>(triB, triA, rotatedNormal, trialDepth, trialArea, trialCentroid);
                    
                    if (trialContact && trialArea < minArea) {
                        minArea = trialArea;
                    } else if (!trialContact) {
                        minArea = T2(0.0);
                        break;
                    }
                }
            }
            
            areaBA = minArea;
            if (minArea <= DEME_TINY_FLOAT) {
                contactBA = false;
            }
        }
    }
    
    if (contactAB && areaAB < SUSPICIOUS_AREA_THRESHOLD * areaRefB) {
        // Suspicious overlap - perform stability check with 4 trial rotations
        T2 minArea = areaAB;
        
        // Find two orthogonal directions in the plane of triangle B
        T1 edge0 = triB[1] - triB[0];
        T2 edge0Len2 = dot(edge0, edge0);
        if (edge0Len2 > DEME_TINY_FLOAT) {
            T1 tangent1 = edge0 * rsqrt(edge0Len2);
            T1 tangent2 = cross(nB, tangent1);
            
            // Try 4 rotations: +/- rotation around tangent1 and tangent2
            for (int dir = 0; dir < 2; ++dir) {
                T1 axis = (dir == 0) ? tangent1 : tangent2;
                for (int sign = -1; sign <= 1; sign += 2) {
                    T2 angle = T2(sign) * ROTATION_ANGLE;
                    // Rodrigues rotation formula: v' = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
                    T2 cosAngle = cos(angle);
                    T2 sinAngle = sin(angle);
                    T1 rotatedNormal = nB * cosAngle + cross(axis, nB) * sinAngle;
                    
                    T2 trialDepth, trialArea;
                    T1 trialCentroid;
                    bool trialContact = projectTriangleOntoTriangle<T1, T2>(triA, triB, rotatedNormal, trialDepth, trialArea, trialCentroid);
                    
                    if (trialContact && trialArea < minArea) {
                        minArea = trialArea;
                    } else if (!trialContact) {
                        minArea = T2(0.0);
                        break;
                    }
                }
            }
            
            areaAB = minArea;
            if (minArea <= DEME_TINY_FLOAT) {
                contactAB = false;
            }
        }
    }

    // Determine if there is contact
    bool inContact = contactBA || contactAB;

    if (!inContact) {
        // No contact detected
        if (outputNoContact) {
            // Provide separation info
            T1 centA = (triA[0] + triA[1] + triA[2]) / T2(3.0);
            T1 centB = (triB[0] + triB[1] + triB[2]) / T2(3.0);
            T1 sep = centA - centB;
            T2 sepLen2 = dot(sep, sep);

            if (sepLen2 > (DEME_TINY_FLOAT * DEME_TINY_FLOAT)) {
                T2 sepLen = sqrt(sepLen2);
                normal = sep / sepLen;
                depth = -sepLen;  // Negative for separation
                point = (centA + centB) * T2(0.5);
            } else {
                normal = nA;
                depth = -DEME_HUGE_FLOAT;
                point = centA;
            }
            projectedArea = T2(0.0);
        }
        return false;
    }

    // Average the results from both projections
    if (contactBA && contactAB) {
        // Both directions have contact - average the results
        depth = (depthBA + depthAB) / T2(2.0);
        projectedArea = (areaBA + areaAB) / T2(2.0);

        // Contact normal: average of the two normals (pointing from B to A)
        // Note: nA points outward from A, so when B is submerged below A, B2A is the inverse of nA
        // Similarly, nB is the contact normal when A is submerged below B
        T1 normalBA = -1.0 * nA;
        T1 normalAB = nB;
        T1 normalSum = normalBA + normalAB;
        T2 normalSumLen2 = dot(normalSum, normalSum);
        if (normalSumLen2 > DEME_TINY_FLOAT * DEME_TINY_FLOAT) {
            normal = normalSum * rsqrt(normalSumLen2);
        } else {
            // Normals are nearly opposite - use one of them
            normal = nB;
        }

        // Contact point: midpoint between the two centroids
        T1 midCentroid = (centroidBA + centroidAB) / T2(2.0);
        point = midCentroid;
    } else if (contactBA) {
        // Only B->A projection has contact
        depth = depthBA;
        projectedArea = areaBA;
        normal = -1.0 * nA;  // Pay attention to direction

        // Contact point: centroid on A's plane, moved back by half depth
        point = centroidBA - nA * (depth * T2(0.5));
    } else {
        // Only A->B projection has contact
        depth = depthAB;
        projectedArea = areaAB;
        normal = nB;

        // Contact point: centroid on B's plane, moved back by half depth
        point = centroidAB - nB * (depth * T2(0.5));
    }

    return true;
}

#endif
