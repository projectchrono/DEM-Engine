// DEM collision-related kernel collection

#ifndef DEME_COLLI_KERNELS_CUH
#define DEME_COLLI_KERNELS_CUH

#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>

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

/// Triangle-triangle contact detection with 2-stage approach:
/// Case 2: Complete submersion - all vertices of one triangle below the other's plane
///         Uses distance filtering to avoid false positives from far-away pairs
/// Case 1: Partial overlap - use SAT to find MTV and contact via clipping polygon
///         Handles all other cases including complete separation
template <typename T1, typename T2>
inline __device__ bool checkTriangleTriangleOverlap(
    const T1& A1,
    const T1& B1,
    const T1& C1,
    const T1& A2,
    const T1& B2,
    const T1& C2,
    T1& normal,                      ///< contact normal
    T2& depth,                       ///< penetration (positive if in contact)
    T2& projectedArea,               ///< projected area of clipping polygon (optional output)
    T1& point,                       ///< contact point
    bool& shouldDropContact,         ///< true if solver thinks this true contact is redundant
    bool outputNoContact = false) {  ///< output info even when no contact
    // Default: don't drop contact
    shouldDropContact = false;

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
    if (lenA2 <= DEME_TINY_FLOAT || lenB2 <= DEME_TINY_FLOAT) {
        return false;
    }

    T1 nA = nA_unnorm * rsqrt(lenA2);
    T1 nB = nB_unnorm * rsqrt(lenB2);

    // Compute signed distances
    T2 dB[3];  // B vertices to A's plane
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        dB[i] = dot(triB[i] - triA[0], nA);
    }

    T2 dA[3];  // A vertices to B's plane
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        dA[i] = dot(triA[i] - triB[0], nB);
    }

    // Count vertices below planes
    int numB_below_A = 0;
    int numA_below_B = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        if (dB[i] < 0.0)
            numB_below_A++;
        if (dA[i] < 0.0)
            numA_below_B++;
    }

    // ========================================================================
    // CASE 2: Complete submersion - all vertices of one triangle below the other's plane
    // This is checked FIRST to handle triangles completely inside a mesh.
    // To avoid false positives from triangles far away on opposite sides of a mesh,
    // we add a distance check: only activate if triangles are reasonably close.
    // ========================================================================
    if (numB_below_A == 3 || numA_below_B == 3) {
        // Compute triangle centroids and distance
        T1 centA = (triA[0] + triA[1] + triA[2]) / T2(3.0);
        T1 centB = (triB[0] + triB[1] + triB[2]) / T2(3.0);
        T1 centDiff = centA - centB;
        T2 centDist2 = dot(centDiff, centDiff);

        // Estimate characteristic size of triangles
        // Use the maximum edge length of both triangles as a size metric
        T1 edges1[3] = {triA[1] - triA[0], triA[2] - triA[1], triA[0] - triA[2]};
        T1 edges2[3] = {triB[1] - triB[0], triB[2] - triB[1], triB[0] - triB[2]};

        T2 maxEdgeLen2 = T2(0.0);
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            T2 len2_A = dot(edges1[i], edges1[i]);
            T2 len2_B = dot(edges2[i], edges2[i]);
            if (len2_A > maxEdgeLen2)
                maxEdgeLen2 = len2_A;
            if (len2_B > maxEdgeLen2)
                maxEdgeLen2 = len2_B;
        }

        // Distance threshold: triangles must be within a reasonable range
        // Use 3x the maximum edge length as the threshold
        // This allows for true submersion while rejecting far-away pairs
        T2 distanceThreshold2 = T2(9.0) * maxEdgeLen2;  // (3 * maxEdgeLen)^2

        // Only proceed with submersion case if triangles are close enough
        if (centDist2 <= distanceThreshold2) {
            // Determine which triangle is submerged
            bool B_submerged = (numB_below_A == 3);
            const T1* subTri = B_submerged ? triB : triA;
            const T1* refTri = B_submerged ? triA : triB;
            T1 refNormal = B_submerged ? nA : nB;
            const T2* subDists = B_submerged ? dB : dA;

            // Find deepest penetration (largest magnitude negative distance)
            T2 maxPenetration = T2(0.0);
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                T2 pen = -subDists[i];  // positive value
                if (pen > maxPenetration)
                    maxPenetration = pen;
            }

            // Build clipping polygon by projecting submerged vertices onto reference triangle
            // First, project each vertex of the submerged triangle onto the reference plane
            T1 projectedOntoPlane[3];
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                // subDists[i] is negative for vertices below the plane
                // To project onto plane: move by -subDists[i] along refNormal direction
                projectedOntoPlane[i] = subTri[i] - refNormal * subDists[i];
            }

            // Now find the intersection of the projected triangle with the reference triangle
            // This forms the clipping polygon
            // We'll use Sutherland-Hodgman algorithm to clip the projected triangle against the reference triangle
            const int MAX_CLIP_VERTS = 9;  // Max vertices for triangle-triangle intersection
            T1 clippingPoly[MAX_CLIP_VERTS];
            int numClipVerts = 0;

            // Initialize with the projected triangle
            T1 inputPoly[MAX_CLIP_VERTS];
            for (int i = 0; i < 3; ++i) {
                inputPoly[i] = projectedOntoPlane[i];
            }
            int numInputVerts = 3;

            // Clip against each edge of the reference triangle
            T1 outputPoly[MAX_CLIP_VERTS];
            for (int edge = 0; edge < 3; ++edge) {
                int numOutputVerts = 0;
                T1 edgeStart = refTri[edge];
                T1 edgeEnd = refTri[(edge + 1) % 3];
                T1 edgeDir = edgeEnd - edgeStart;
                T1 edgeNormal = cross(refNormal, edgeDir);
                T2 edgeNormalLen2 = dot(edgeNormal, edgeNormal);
                if (edgeNormalLen2 > DEME_TINY_FLOAT) {
                    edgeNormal = edgeNormal * rsqrt(edgeNormalLen2);
                }

                // Clip input polygon against this edge
                for (int i = 0; i < numInputVerts; ++i) {
                    T1 v1 = inputPoly[i];
                    T1 v2 = inputPoly[(i + 1) % numInputVerts];
                    T2 d1 = dot(v1 - edgeStart, edgeNormal);
                    T2 d2 = dot(v2 - edgeStart, edgeNormal);
                    bool in1 = (d1 >= -DEME_TINY_FLOAT);
                    bool in2 = (d2 >= -DEME_TINY_FLOAT);

                    // Inside point forms in the output polygon
                    if (in1 && numOutputVerts < MAX_CLIP_VERTS) {
                        outputPoly[numOutputVerts++] = v1;
                    }
                    if (in1 != in2 && numOutputVerts < MAX_CLIP_VERTS) {
                        // Edge crosses the clipping edge
                        T2 denom = d1 - d2;
                        if (denom != T2(0.0)) {  // Safety check for division by zero
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

            numClipVerts = numInputVerts;
            for (int i = 0; i < numClipVerts; ++i) {
                clippingPoly[i] = inputPoly[i];
            }

            // Now we have the clipping polygon
            // Check if we have a valid contact (at least 3 vertices)
            if (numClipVerts >= 3) {
                // Compute centroid of clipping polygon
                T1 centroid;
                centroid.x = T2(0.0);
                centroid.y = T2(0.0);
                centroid.z = T2(0.0);

                for (int i = 0; i < numClipVerts; ++i) {
                    centroid = centroid + clippingPoly[i];
                }
                centroid = centroid / T2(numClipVerts);

                // Calculate the area of the clipping polygon
                // Use the fan triangulation method from the centroid
                T2 area = T2(0.0);
                for (int i = 0; i < numClipVerts; ++i) {
                    T1 v1 = clippingPoly[i] - centroid;
                    T1 v2 = clippingPoly[(i + 1) % numClipVerts] - centroid;
                    T1 crossProd = cross(v1, v2);
                    area += sqrt(dot(crossProd, crossProd));
                }
                area *= T2(0.5);
                projectedArea = area;

                depth = maxPenetration;
                normal = B_submerged ? nA : (nB * T2(-1.0));  // From B to A
                T2 centroidDist = dot(centroid - refTri[0], refNormal);
                point = centroid - refNormal * (centroidDist + depth * T2(0.5));

                // Successfully handled submersion case
                return true;
            } else {
                // Not enough vertices in clipping polygon, drop this contact
                // This is fine as other triangles will have head-on contact
                shouldDropContact = true;
                projectedArea = T2(0.0);
                if (outputNoContact) {
                    // Still need to provide some output
                    depth = maxPenetration;
                    T1 sep = centA - centB;
                    T2 sepLen2 = dot(sep, sep);
                    if (sepLen2 > DEME_TINY_FLOAT) {
                        normal = sep * rsqrt(sepLen2);
                    } else {
                        normal = nA;
                    }
                    point = (centA + centB) * T2(0.5);
                }
                // It is always contact in this case; yet we may instruct caller to drop it
                return true;
            }
        }
        // If triangles are too far apart, fall through to Case 1 (SAT)
    }

    // ========================================================================
    // CASE 1: Partial overlap - use SAT to find MTV
    // ========================================================================

    T1 edges1[3] = {triA[1] - triA[0], triA[2] - triA[1], triA[0] - triA[2]};
    T1 edges2[3] = {triB[1] - triB[0], triB[2] - triB[1], triB[0] - triB[2]};

    // Track the MTV (minimum translation vector)
    T2 minOverlap = DEME_HUGE_FLOAT;
    T1 mtv_axis;
    int axisType = -1;  // 0: face1, 1: face2, 2+: edge-edge
    int edgeIndexA = -1, edgeIndexB = -1;
    bool foundSeparation = false;
    T2 maxSeparation = -DEME_HUGE_FLOAT;

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

        // Check overlap or separation
        if (max1 < min2 || max2 < min1) {
            // Separating axis found
            T2 separation = (max1 < min2) ? (min2 - max1) : (min1 - max2);
            // Recording maxSeparation is for when outputNoContact is true
            if (separation > maxSeparation) {
                maxSeparation = separation;
                mtv_axis = (max1 < min2) ? axis : (axis * T2(-1.0));
                axisType = 0;
                foundSeparation = true;
            }
            if (!outputNoContact)
                return false;
        } else {
            // Calculate overlap
            T2 overlap = DEME_MIN(max1 - min2, max2 - min1);
            if (overlap < minOverlap) {
                minOverlap = overlap;
                mtv_axis = axis;
                axisType = 0;
                // Ensure MTV points from tri2 to tri1
                if (max1 - min2 < max2 - min1) {
                    mtv_axis = axis * T2(-1.0);
                }
            }
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

        // Check overlap or separation
        if (max1 < min2 || max2 < min1) {
            // Separating axis found
            T2 separation = (max1 < min2) ? (min2 - max1) : (min1 - max2);
            if (separation > maxSeparation) {
                maxSeparation = separation;
                mtv_axis = (max1 < min2) ? axis : (axis * T2(-1.0));
                axisType = 1;
                foundSeparation = true;
            }
            if (!outputNoContact)
                return false;
        } else {
            // Calculate overlap
            T2 overlap = DEME_MIN(max1 - min2, max2 - min1);
            if (overlap < minOverlap) {
                minOverlap = overlap;
                mtv_axis = axis;
                axisType = 1;
                // Ensure MTV points from tri2 to tri1
                if (max1 - min2 < max2 - min1) {
                    mtv_axis = axis * T2(-1.0);
                }
            }
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

                // Check overlap or separation
                if (max1 < min2 || max2 < min1) {
                    // Separating axis found
                    T2 separation = (max1 < min2) ? (min2 - max1) : (min1 - max2);
                    if (separation > maxSeparation) {
                        maxSeparation = separation;
                        mtv_axis = (max1 < min2) ? axis : (axis * T2(-1.0));
                        axisType = 2;
                        edgeIndexA = i;
                        edgeIndexB = j;
                        foundSeparation = true;
                    }
                    if (!outputNoContact)
                        return false;
                } else {
                    // Calculate overlap
                    T2 overlap = DEME_MIN(max1 - min2, max2 - min1);
                    if (overlap < minOverlap) {
                        minOverlap = overlap;
                        mtv_axis = axis;
                        axisType = 2;
                        edgeIndexA = i;
                        edgeIndexB = j;
                        // Ensure MTV points from tri2 to tri1
                        if (max1 - min2 < max2 - min1) {
                            mtv_axis = axis * T2(-1.0);
                        }
                    }
                }
            }
        }
    }

    // Safety check: if no valid axis was found (degenerate triangles), return false
    if (axisType == -1 && !outputNoContact) {
        return false;
    }
    // printf("Best overlap: %f on axis type %d\n", float(minOverlap), axisType);

    // Determine contact status
    bool inContact = !foundSeparation;

    if (inContact) {
        // Contact found - compute clipping polygon for contact point
        depth = minOverlap;
        normal = mtv_axis;

        // Use clipping polygon approach similar to Case 1
        if (axisType == 0 || axisType == 1) {
            // Face-based contact
            const T1* refTri = (axisType == 0) ? triA : triB;
            const T1* incTri = (axisType == 0) ? triB : triA;
            T1 refNormal = (axisType == 0) ? nA : nB;

            // Compute signed distances of incident triangle vertices to reference plane
            T2 incDists[3];
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                incDists[i] = dot(incTri[i] - refTri[0], refNormal);
            }

            // Project all incident triangle vertices onto the reference plane
            T1 projectedOntoPlane[3];
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                projectedOntoPlane[i] = incTri[i] - refNormal * incDists[i];
            }

            // Clip the projected incident triangle against the reference triangle
            // using Sutherland-Hodgman algorithm
            T1 clippingPoly[9];
            int numClipVerts = 0;

            // Initialize with the projected triangle
            T1 inputPoly[9];
            for (int i = 0; i < 3; ++i) {
                inputPoly[i] = projectedOntoPlane[i];
            }
            int numInputVerts = 3;

            // Clip against each edge of the reference triangle
            T1 outputPoly[9];
            for (int edge = 0; edge < 3; ++edge) {
                int numOutputVerts = 0;
                T1 edgeStart = refTri[edge];
                T1 edgeEnd = refTri[(edge + 1) % 3];
                T1 edgeDir = edgeEnd - edgeStart;
                T1 edgeNormal = cross(refNormal, edgeDir);
                T2 edgeNormalLen2 = dot(edgeNormal, edgeNormal);
                if (edgeNormalLen2 > DEME_TINY_FLOAT) {
                    edgeNormal = edgeNormal * rsqrt(edgeNormalLen2);
                }

                // Clip input polygon against this edge
                for (int i = 0; i < numInputVerts; ++i) {
                    T1 v1 = inputPoly[i];
                    T1 v2 = inputPoly[(i + 1) % numInputVerts];
                    T2 d1 = dot(v1 - edgeStart, edgeNormal);
                    T2 d2 = dot(v2 - edgeStart, edgeNormal);
                    bool in1 = (d1 >= -DEME_TINY_FLOAT);
                    bool in2 = (d2 >= -DEME_TINY_FLOAT);

                    // Inside point forms in the output polygon
                    if (in1) {
                        outputPoly[numOutputVerts++] = v1;
                    }
                    if (in1 != in2) {
                        // Edge crosses the clipping edge
                        T2 t = d1 / (d1 - d2);
                        T1 inter = v1 + (v2 - v1) * t;
                        outputPoly[numOutputVerts++] = inter;
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

            numClipVerts = numInputVerts;
            for (int i = 0; i < numClipVerts; ++i) {
                clippingPoly[i] = inputPoly[i];
            }

            // Compute centroid and area of the clipping polygon
            T1 centroid;
            centroid.x = T2(0.0);
            centroid.y = T2(0.0);
            centroid.z = T2(0.0);

            projectedArea = T2(0.0);
            if (numClipVerts >= 3) {
                for (int i = 0; i < numClipVerts; ++i) {
                    centroid = centroid + clippingPoly[i];
                }
                centroid = centroid / T2(numClipVerts);

                // Calculate the area using fan triangulation
                T2 area = T2(0.0);
                for (int i = 0; i < numClipVerts; ++i) {
                    T1 v1 = clippingPoly[i] - centroid;
                    T1 v2 = clippingPoly[(i + 1) % numClipVerts] - centroid;
                    T1 crossProd = cross(v1, v2);
                    area += sqrt(dot(crossProd, crossProd));
                }
                area *= T2(0.5);
                projectedArea = area;
            } else {
                centroid = (incTri[0] + incTri[1] + incTri[2]) / T2(3.0);
            }

            T2 centroidDist = dot(centroid - refTri[0], refNormal);
            point = centroid - refNormal * (centroidDist + depth * T2(0.5));
        } else {
            // Edge-edge contact (axisType == 2)
            int nextA = (edgeIndexA + 1) % 3;
            T1 edgeA_start = triA[edgeIndexA];
            T1 edgeA = edges1[edgeIndexA];

            int nextB = (edgeIndexB + 1) % 3;
            T1 edgeB_start = triB[edgeIndexB];
            T1 edgeB = edges2[edgeIndexB];

            // Compute closest points on edges
            T1 r = edgeB_start - edgeA_start;
            T2 a = dot(edgeA, edgeA);
            T2 e = dot(edgeB, edgeB);
            T2 f = dot(edgeB, r);

            T2 s, t;
            if (a <= DEME_TINY_FLOAT && e <= DEME_TINY_FLOAT) {
                s = t = T2(0.0);
            } else if (a <= DEME_TINY_FLOAT) {
                s = T2(0.0);
                t = clampBetween(f / e, T2(0.0), T2(1.0));
            } else {
                T2 c = dot(edgeA, r);
                if (e <= DEME_TINY_FLOAT) {
                    t = T2(0.0);
                    s = clampBetween(-c / a, T2(0.0), T2(1.0));
                } else {
                    T2 b = dot(edgeA, edgeB);
                    T2 denom = a * e - b * b;

                    if (denom != T2(0.0)) {
                        s = clampBetween((b * f - c * e) / denom, T2(0.0), T2(1.0));
                    } else {
                        s = T2(0.0);
                    }

                    t = (b * s + f) / e;

                    if (t < T2(0.0)) {
                        t = T2(0.0);
                        s = clampBetween(-c / a, T2(0.0), T2(1.0));
                    } else if (t > T2(1.0)) {
                        t = T2(1.0);
                        s = clampBetween((b - c) / a, T2(0.0), T2(1.0));
                    }
                }
            }

            T1 closestA = edgeA_start + edgeA * s;
            T1 closestB = edgeB_start + edgeB * t;
            point = (closestA + closestB) * T2(0.5);

            // For edge-edge contact, use projection approach similar to face-based contact
            // Choose one triangle as reference (use triA) and project triB onto its plane
            const T1* refTri = triA;
            const T1* incTri = triB;
            T1 refNormal = nA;

            // Compute signed distances of incident triangle vertices to reference plane
            T2 incDists[3];
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                incDists[i] = dot(incTri[i] - refTri[0], refNormal);
            }

            // Project all incident triangle vertices onto the reference plane
            T1 projectedOntoPlane[3];
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                projectedOntoPlane[i] = incTri[i] - refNormal * incDists[i];
            }

            // Clip the projected incident triangle against the reference triangle
            T1 clippingPoly[9];
            int numClipVerts = 0;

            T1 inputPoly[9];
            for (int i = 0; i < 3; ++i) {
                inputPoly[i] = projectedOntoPlane[i];
            }
            int numInputVerts = 3;

            // Clip against each edge of the reference triangle
            T1 outputPoly[9];
            for (int edge = 0; edge < 3; ++edge) {
                int numOutputVerts = 0;
                T1 edgeStart = refTri[edge];
                T1 edgeEnd = refTri[(edge + 1) % 3];
                T1 edgeDir = edgeEnd - edgeStart;
                T1 edgeNormal = cross(refNormal, edgeDir);
                T2 edgeNormalLen2 = dot(edgeNormal, edgeNormal);
                if (edgeNormalLen2 > DEME_TINY_FLOAT) {
                    edgeNormal = edgeNormal * rsqrt(edgeNormalLen2);
                }

                for (int i = 0; i < numInputVerts; ++i) {
                    T1 v1 = inputPoly[i];
                    T1 v2 = inputPoly[(i + 1) % numInputVerts];
                    T2 d1 = dot(v1 - edgeStart, edgeNormal);
                    T2 d2 = dot(v2 - edgeStart, edgeNormal);
                    bool in1 = (d1 >= -DEME_TINY_FLOAT);
                    bool in2 = (d2 >= -DEME_TINY_FLOAT);

                    if (in1) {
                        outputPoly[numOutputVerts++] = v1;
                    }
                    if (in1 != in2) {
                        T2 t_clip = d1 / (d1 - d2);
                        T1 inter = v1 + (v2 - v1) * t_clip;
                        outputPoly[numOutputVerts++] = inter;
                    }
                }

                for (int i = 0; i < numOutputVerts; ++i) {
                    inputPoly[i] = outputPoly[i];
                }
                numInputVerts = numOutputVerts;

                if (numInputVerts == 0) {
                    break;
                }
            }

            numClipVerts = numInputVerts;
            for (int i = 0; i < numClipVerts; ++i) {
                clippingPoly[i] = inputPoly[i];
            }

            // Calculate projected area if we have a non-degenerate polygon
            projectedArea = T2(0.0);
            if (numClipVerts >= 3) {
                T1 centroid;
                centroid.x = T2(0.0);
                centroid.y = T2(0.0);
                centroid.z = T2(0.0);

                for (int i = 0; i < numClipVerts; ++i) {
                    centroid = centroid + clippingPoly[i];
                }
                centroid = centroid / T2(numClipVerts);

                T2 area = T2(0.0);
                for (int i = 0; i < numClipVerts; ++i) {
                    T1 v1 = clippingPoly[i] - centroid;
                    T1 v2 = clippingPoly[(i + 1) % numClipVerts] - centroid;
                    T1 crossProd = cross(v1, v2);
                    area += sqrt(dot(crossProd, crossProd));
                }
                area *= T2(0.5);
                projectedArea = area;
            }
            // If numClipVerts < 3, the polygon is degenerate and projectedArea remains 0
        }

        return true;
    } else {
        // No contact - separation found
        projectedArea = 0.0;
        if (outputNoContact) {
            // Provide separation info
            depth = -maxSeparation;
            normal = mtv_axis;

            // Estimate contact point as midpoint trajectory
            T1 centA = (triA[0] + triA[1] + triA[2]) / T2(3.0);
            T1 centB = (triB[0] + triB[1] + triB[2]) / T2(3.0);
            point = (centA + centB) * T2(0.5);
        }
        return false;
    }
}

#endif
