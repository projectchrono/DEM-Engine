// DEM collision-related kernel collection

#ifndef DEME_COLLI_KERNELS_CU
#define DEME_COLLI_KERNELS_CU

#include <DEM/Defines.h>

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
  - depth:    penetration distance (a negative value means that overlap exists)
  - normal:     contact normal, from pt2 to pt1
A return value of "true" signals collision.
*/
template <typename T1, typename T2>
__device__ bool triangle_sphere_CD(const T1& A,           ///< First vertex of the triangle
                                   const T1& B,           ///< Second vertex of the triangle
                                   const T1& C,           ///< Third vertex of the triangle
                                   const T1& sphere_pos,  ///< Location of the center of the sphere
                                   const T2 radius,       ///< Sphere radius
                                   T1& normal,            ///< contact normal
                                   T2& depth,             ///< penetration (negative if in contact)
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
        depth = h - radius;
        normal.x = face_n.x;
        normal.y = face_n.y;
        normal.z = face_n.z;
        pt1 = faceLoc;
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
        depth = dist - radius;

        normal = (1.0 / dist) * normal;
        pt1 = faceLoc;
        if (depth >= 0. || h >= radius || h <= -radius) {
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
  - depth:    penetration distance (a negative value means that overlap exists)
  - normal:     contact normal, from pt2 to pt1
A return value of "true" signals collision.
*/
template <typename T1, typename T2>
__device__ bool triangle_sphere_CD_directional(const T1& A,           ///< First vertex of the triangle
                                               const T1& B,           ///< Second vertex of the triangle
                                               const T1& C,           ///< Third vertex of the triangle
                                               const T1& sphere_pos,  ///< Location of the center of the sphere
                                               const T2 radius,       ///< Sphere radius
                                               T1& normal,            ///< contact normal
                                               T2& depth,             ///< penetration (negative if in contact)
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
        depth = h - radius;
        normal.x = face_n.x;
        normal.y = face_n.y;
        normal.z = face_n.z;
        pt1 = faceLoc;
        if (depth >= 0.) {
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
        depth = dist - radius;

        normal = (1.0 / dist) * normal;
        pt1 = faceLoc;
        if (depth >= 0. || h >= radius) {
            in_contact = false;
        } else {
            in_contact = true;
        }
    }
    return in_contact;
}

#endif
