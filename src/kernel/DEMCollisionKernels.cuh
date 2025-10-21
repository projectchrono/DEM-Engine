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
        // Note checkTriSphereOverlap is used in force calc, so we follow the convention that the contact point is
        // somewhere in the midpoint of the deepest penetration line segment. Go from faceLoc, backwards wrt normal,
        // half the penetration depth.
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
        // Note checkTriSphereOverlap_directional is never used in force calc, so pt1 being an approximation is ok, just
        // leave it as a point on facet
        pt1 = faceLoc;
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
        pt1 = faceLoc;
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
inline bool __device__
tri_plane_penetration(const T1** tri, const T1& entityLoc, const float3& entityDir, T2& overlapDepth, T1& contactPnt) {
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
            // Only register inside points once
            if (in_i)
                poly[nNode++] = *tri[i];
            poly[nNode++] = inter;
            hasIntersection = true;
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
                                            T2& overlapDepth) {
    const T1* tri[] = {&A, &B, &C};
    bool in_contact;
    switch (entityType) {
        case deme::ANAL_OBJ_TYPE_PLANE:
            in_contact = tri_plane_penetration<T1, T2>(tri, entityLoc, entityDir, overlapDepth, contactPnt);
            // Plane contact's normal is always the plane's normal
            contact_normal = entityDir;
            return in_contact;
        case deme::ANAL_OBJ_TYPE_CYL_INF:
            in_contact = tri_cyl_penetration<T1, T2>(tri, entityLoc, entityDir, entitySize1, entitySize2, normal_sign,
                                                     contact_normal, overlapDepth, contactPnt);
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
    float3 axes[11];
    unsigned short int axisCount = 0;
    // Pack as stack arrays for easier looping
    T1 prismA[6] = {prismAFaceANode1, prismAFaceANode2, prismAFaceANode3,
                    prismAFaceBNode1, prismAFaceBNode2, prismAFaceBNode3};
    T1 prismB[6] = {prismBFaceANode1, prismBFaceANode2, prismBFaceANode3,
                    prismBFaceBNode1, prismBFaceBNode2, prismBFaceBNode3};

    // Base triangle normals
    T1 A_faceNormal = cross(prismA[1] - prismA[0], prismA[2] - prismA[0]);
    T1 B_faceNormal = cross(prismB[1] - prismB[0], prismB[2] - prismB[0]);

    axes[axisCount++] = normalize(A_faceNormal);
    axes[axisCount++] = normalize(B_faceNormal);

    // Edges of each prism base
    {
        T1 A_edges[3] = {prismA[1] - prismA[0], prismA[2] - prismA[1], prismA[0] - prismA[2]};
        T1 B_edges[3] = {prismB[1] - prismB[0], prismB[2] - prismB[1], prismB[0] - prismB[2]};

        // Edge–edge cross products
        for (unsigned short int i = 0; i < 3; ++i) {
            for (unsigned short int j = 0; j < 3; ++j) {
                T1 cp = cross(A_edges[i], B_edges[j]);
                float len2 = dot(cp, cp);
                if (len2 > DEME_TINY_FLOAT)
                    axes[axisCount++] = normalize(cp);
            }
        }
    }

    // SAT test
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

/// SAT narrow-phase returning MTV info, across the two face normals (nA, nB), and the 9 edge--edge cross axes (edgeAi ×
/// edgeBj). It then derives the contact point depending on which axis “won”: If FACE_A: project the penetrating
/// B-vertices onto A’s plane and average. If FACE_B: symmetric. If EDGE_EDGE: compute closest points between the
/// winning edge pair and average them.
template <typename T1, typename T2>
inline __device__ bool checkTriangleTriangleOverlap(const T1& A1,
                                                    const T1& B1,
                                                    const T1& C1,
                                                    const T1& A2,
                                                    const T1& B2,
                                                    const T1& C2,
                                                    T1& normal,   ///< contact normal
                                                    T2& depth,    ///< penetration (positive if in contact)
                                                    T1& point) {  ///< contact point
    // Triangle 1 vertices
    const T1 tri1[3] = {A1, B1, C1};
    // Triangle 2 vertices
    const T1 tri2[3] = {A2, B2, C2};
    
    // Compute face normals
    T1 edge1_0 = B1 - A1;
    T1 edge1_1 = C1 - B1;
    T1 edge1_2 = A1 - C1;
    T1 normal1 = cross(edge1_0, C1 - A1);
    
    T1 edge2_0 = B2 - A2;
    T1 edge2_1 = C2 - B2;
    T1 edge2_2 = A2 - C2;
    T1 normal2 = cross(edge2_0, C2 - A2);
    
    // Track the MTV (minimum translation vector)
    T2 minOverlap = DEME_HUGE_FLOAT;
    T1 mtv_axis;
    int axisType = -1;  // 0: face1, 1: face2, 2+: edge-edge
    int edgeIndexA = -1, edgeIndexB = -1;
    
    // Store edges for later use
    T1 edges1[3] = {edge1_0, edge1_1, edge1_2};
    T1 edges2[3] = {edge2_0, edge2_1, edge2_2};
    
    // Test face normal of triangle 1
    {
        T2 len2 = dot(normal1, normal1);
        if (len2 > DEME_TINY_FLOAT) {
            T1 axis = normal1 * (T2(1.0) / sqrt(len2));
            
            // Project triangle 1 vertices
            T2 min1 = dot(tri1[0], axis);
            T2 max1 = min1;
            for (int i = 1; i < 3; ++i) {
                T2 proj = dot(tri1[i], axis);
                if (proj < min1) min1 = proj;
                if (proj > max1) max1 = proj;
            }
            
            // Project triangle 2 vertices
            T2 min2 = dot(tri2[0], axis);
            T2 max2 = min2;
            for (int i = 1; i < 3; ++i) {
                T2 proj = dot(tri2[i], axis);
                if (proj < min2) min2 = proj;
                if (proj > max2) max2 = proj;
            }
            
            // Check overlap
            if (max1 < min2 || max2 < min1) {
                return false;  // Separating axis found
            }
            
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
    
    // Test face normal of triangle 2
    {
        T2 len2 = dot(normal2, normal2);
        if (len2 > DEME_TINY_FLOAT) {
            T1 axis = normal2 * (T2(1.0) / sqrt(len2));
            
            // Project triangle 1 vertices
            T2 min1 = dot(tri1[0], axis);
            T2 max1 = min1;
            for (int i = 1; i < 3; ++i) {
                T2 proj = dot(tri1[i], axis);
                if (proj < min1) min1 = proj;
                if (proj > max1) max1 = proj;
            }
            
            // Project triangle 2 vertices
            T2 min2 = dot(tri2[0], axis);
            T2 max2 = min2;
            for (int i = 1; i < 3; ++i) {
                T2 proj = dot(tri2[i], axis);
                if (proj < min2) min2 = proj;
                if (proj > max2) max2 = proj;
            }
            
            // Check overlap
            if (max1 < min2 || max2 < min1) {
                return false;  // Separating axis found
            }
            
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
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            T1 axis = cross(edges1[i], edges2[j]);
            T2 len2 = dot(axis, axis);
            
            if (len2 > DEME_TINY_FLOAT) {
                axis = axis * (T2(1.0) / sqrt(len2));
                
                // Project triangle 1 vertices
                T2 min1 = dot(tri1[0], axis);
                T2 max1 = min1;
                for (int k = 1; k < 3; ++k) {
                    T2 proj = dot(tri1[k], axis);
                    if (proj < min1) min1 = proj;
                    if (proj > max1) max1 = proj;
                }
                
                // Project triangle 2 vertices
                T2 min2 = dot(tri2[0], axis);
                T2 max2 = min2;
                for (int k = 1; k < 3; ++k) {
                    T2 proj = dot(tri2[k], axis);
                    if (proj < min2) min2 = proj;
                    if (proj > max2) max2 = proj;
                }
                
                // Check overlap
                if (max1 < min2 || max2 < min1) {
                    return false;  // Separating axis found
                }
                
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
    
    // If we reach here, triangles are overlapping
    // Safety check: if no valid axis was found (degenerate triangles), return false
    if (axisType == -1) {
        return false;
    }
    
    depth = minOverlap;
    normal = mtv_axis;
    
    // Calculate contact point based on which axis "won"
    if (axisType == 0) {
        // Face A: project penetrating B-vertices onto A's plane and average
        T1 sum;
        sum.x = 0.0;
        sum.y = 0.0;
        sum.z = 0.0;
        int count = 0;
        T1 planeNormal = normalize(normal1);
        for (int i = 0; i < 3; ++i) {
            T2 dist = dot(tri2[i] - tri1[0], planeNormal);
            if (dist < 0.0) {  // Vertex is on penetrating side
                T1 proj = tri2[i] - planeNormal * dist;
                sum = sum + proj;
                count++;
            }
        }
        if (count > 0) {
            point = sum / T2(count);
        } else {
            // Fallback: use centroid of triangle 2
            point = (tri2[0] + tri2[1] + tri2[2]) / T2(3.0);
        }
    } else if (axisType == 1) {
        // Face B: project penetrating A-vertices onto B's plane and average
        T1 sum;
        sum.x = 0.0;
        sum.y = 0.0;
        sum.z = 0.0;
        int count = 0;
        T1 planeNormal = normalize(normal2);
        for (int i = 0; i < 3; ++i) {
            T2 dist = dot(tri1[i] - tri2[0], planeNormal);
            if (dist < 0.0) {  // Vertex is on penetrating side
                T1 proj = tri1[i] - planeNormal * dist;
                sum = sum + proj;
                count++;
            }
        }
        if (count > 0) {
            point = sum / T2(count);
        } else {
            // Fallback: use centroid of triangle 1
            point = (tri1[0] + tri1[1] + tri1[2]) / T2(3.0);
        }
    } else {
        // Edge-edge: compute closest points between the winning edge pair
        // Edge from triangle 1
        int nextA = (edgeIndexA + 1) % 3;
        T1 edgeA_start = tri1[edgeIndexA];
        T1 edgeA_end = tri1[nextA];
        T1 edgeA = edges1[edgeIndexA];
        
        // Edge from triangle 2
        int nextB = (edgeIndexB + 1) % 3;
        T1 edgeB_start = tri2[edgeIndexB];
        T1 edgeB_end = tri2[nextB];
        T1 edgeB = edges2[edgeIndexB];
        
        // Compute closest points on two line segments
        T1 r = edgeB_start - edgeA_start;
        T2 a = dot(edgeA, edgeA);
        T2 e = dot(edgeB, edgeB);
        T2 f = dot(edgeB, r);
        
        T2 s, t;
        if (a <= DEME_TINY_FLOAT && e <= DEME_TINY_FLOAT) {
            // Both edges degenerate to points
            s = t = 0.0;
        } else if (a <= DEME_TINY_FLOAT) {
            // First edge degenerates to a point
            s = 0.0;
            t = clamp(f / e, T2(0.0), T2(1.0));
        } else {
            T2 c = dot(edgeA, r);
            if (e <= DEME_TINY_FLOAT) {
                // Second edge degenerates to a point
                t = 0.0;
                s = clamp(-c / a, T2(0.0), T2(1.0));
            } else {
                // General case
                T2 b = dot(edgeA, edgeB);
                T2 denom = a * e - b * b;
                
                if (denom != T2(0.0)) {
                    s = clamp((b * f - c * e) / denom, T2(0.0), T2(1.0));
                } else {
                    s = 0.0;
                }
                
                t = (b * s + f) / e;
                
                if (t < T2(0.0)) {
                    t = 0.0;
                    s = clamp(-c / a, T2(0.0), T2(1.0));
                } else if (t > T2(1.0)) {
                    t = 1.0;
                    s = clamp((b - c) / a, T2(0.0), T2(1.0));
                }
            }
        }
        
        T1 closestA = edgeA_start + edgeA * s;
        T1 closestB = edgeB_start + edgeB * t;
        point = (closestA + closestB) * T2(0.5);
    }
    
    return true;
}

#endif
