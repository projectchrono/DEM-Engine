//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#ifndef SGPS_DEM_HOST_HELPERS
#define SGPS_DEM_HOST_HELPERS

#include <iostream>
#include <sstream>
#include <list>
#include <cmath>
#include <vector>
#include <algorithm>
#include <regex>
#include <nvmath/helper_math.cuh>

#include <DEM/DEMDefines.h>
#include <DEM/DEMStructs.h>

namespace sgps {

// In an upper-triangular matrix, given i and j, this function returns the index of the corresponding flatten-ed
// non-zero entries. This function does not assume i <= j.
template <typename T1>
inline T1 locateMaskPair(const T1& i, const T1& j) {
    if (i > j)
        return locateMaskPair(j, i);
    return (1 + j) * j / 2 + i;
}

template <typename T1>
inline void displayArray(T1* arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        std::cout << +(arr[i]) << " ";
    }
    std::cout << std::endl;
}

inline void displayFloat3(float3* arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        std::cout << "(" << +(arr[i].x) << ", " << +(arr[i].y) << ", " << +(arr[i].z) << "), ";
    }
    std::cout << std::endl;
}

template <typename T1>
inline void hostPrefixScan(T1* arr, size_t n) {
    T1 buffer_previous = arr[0];
    arr[0] = 0;
    for (size_t i = 1; i < n; i++) {
        T1 item_to_add = buffer_previous;
        buffer_previous = arr[i];
        arr[i] = arr[i - 1] + item_to_add;
    }
}

template <typename T1>
inline void elemSwap(T1* x, T1* y) {
    T1 tmp = *x;
    *x = *y;
    *y = tmp;
}

// Test if 2 types of DEM materials are the same
inline bool is_DEM_material_same(const std::shared_ptr<DEMMaterial>& a, const std::shared_ptr<DEMMaterial>& b) {
    if (std::abs(a->rho - b->rho) > SGPS_DEM_TINY_FLOAT) {
        return false;
    }
    if (std::abs(a->E - b->E) > SGPS_DEM_TINY_FLOAT) {
        return false;
    }
    if (std::abs(a->nu - b->nu) > SGPS_DEM_TINY_FLOAT) {
        return false;
    }
    if (std::abs(a->CoR - b->CoR) > SGPS_DEM_TINY_FLOAT) {
        return false;
    }
    if (std::abs(a->mu - b->mu) > SGPS_DEM_TINY_FLOAT) {
        return false;
    }
    if (std::abs(a->Crr - b->Crr) > SGPS_DEM_TINY_FLOAT) {
        return false;
    }
    return true;
}

/// Check if this_material is in loaded_materials: if yes, return the correspnding index in loaded_materials; if not,
/// load it and return the correspnding index in loaded_materials (the last element)
inline unsigned int stash_material_in_templates(std::vector<std::shared_ptr<DEMMaterial>>& loaded_materials,
                                                const std::shared_ptr<DEMMaterial>& this_material) {
    auto is_same = [&](const std::shared_ptr<DEMMaterial>& ptr) { return is_DEM_material_same(ptr, this_material); };
    // Is this material already loaded? (most likely yes)
    auto it_mat = std::find_if(loaded_materials.begin(), loaded_materials.end(), is_same);
    if (it_mat != loaded_materials.end()) {
        // Already in, then just get where it's located in the m_loaded_materials array
        return std::distance(loaded_materials.begin(), it_mat);
    } else {
        // Not already in, come on. Load it, and then get it into this_clump_sp_mat_ids. This is unlikely, unless the
        // users made a shared_ptr themselves.
        loaded_materials.push_back(this_material);
        return loaded_materials.size() - 1;
    }
}

/// Replace all instances of a certain pattern from a string then return it
inline std::string replace_pattern(const std::string& in, const std::string& from, const std::string& to) {
    return std::regex_replace(in, std::regex(from), to);
}

/// Sachin Gupta's work on removing comments from a piece of code, from
/// https://www.geeksforgeeks.org/remove-comments-given-cc-program/
inline std::string remove_comments(const std::string& prgm) {
    size_t n = prgm.length();
    std::string res;

    // Flags to indicate that single line and multiple line comments
    // have started or not.
    bool s_cmt = false;
    bool m_cmt = false;

    // Traverse the given program
    for (size_t i = 0; i < n; i++) {
        // If single line comment flag is on, then check for end of it
        if (s_cmt == true && prgm[i] == '\n')
            s_cmt = false;

        // If multiple line comment is on, then check for end of it
        else if (m_cmt == true && prgm[i] == '*' && prgm[i + 1] == '/')
            m_cmt = false, i++;

        // If this character is in a comment, ignore it
        else if (s_cmt || m_cmt)
            continue;

        // Check for beginning of comments and set the appropriate flags
        else if (prgm[i] == '/' && prgm[i + 1] == '/')
            s_cmt = true, i++;
        else if (prgm[i] == '/' && prgm[i + 1] == '*')
            m_cmt = true, i++;

        // If current character is a non-comment character, append it to res
        else
            res += prgm[i];
    }
    return res;
}

/// Remove comments and newlines from a piece of code, making it suitable for jitification in the framework of SGPS
inline std::string compact_code(const std::string& prgm) {
    std::string res;
    res = remove_comments(prgm);
    res = replace_pattern(res, "\n", "");
    return res;
}

/// Rotate a vector about an unit axis by an angle
inline float3 Rodrigues(const float3 vec, const float3 axis, const float theta) {
    float3 res;
    res = vec * cos(theta) + cross(axis, vec) * sin(theta) + axis * dot(axis, vec) * (1. - cos(theta));
    return res;
}

template <typename T1, typename T2>
inline void hostSortByKey(T1* keys, T2* vals, size_t n) {
    // Just bubble sort it
    bool swapped;
    for (size_t i = 0; i < n - 1; i++) {
        swapped = false;
        for (size_t j = 0; j < n - i - 1; j++) {
            if (keys[j] > keys[j + 1]) {
                elemSwap<T1>(&keys[j], &keys[j + 1]);
                elemSwap<T2>(&vals[j], &vals[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) {
            break;
        }
    }
}

template <typename T1>
inline void hostScanForJumpsNum(T1* arr, size_t n, unsigned int minSegLen, size_t& total_found) {
    size_t i = 0;
    size_t found = 0;
    while (i < n - 1) {
        size_t thisIndx = i;
        T1 thisItem = arr[i];
        do {
            i++;
        } while (arr[i] == thisItem && i < n - 1);
        if (i - thisIndx >= minSegLen || (i == n - 1 && i - thisIndx + 1 >= minSegLen && arr[i] == thisItem)) {
            found++;
        }
    }
    total_found = found;
}

// Tell each active bin where to find its touching spheres
template <typename T1, typename T2, typename T3>
inline void hostScanForJumps(T1* arr, T1* arr_elem, T2* jump_loc, T3* jump_len, size_t n, unsigned int minSegLen) {
    size_t total_found = 0;
    T2 i = 0;
    unsigned int thisSegLen;
    while (i < n - 1) {
        thisSegLen = 0;
        T2 thisIndx = i;
        T1 thisItem = arr[i];
        do {
            i++;
            thisSegLen++;
        } while (arr[i] == thisItem && i < n - 1);
        if (i - thisIndx >= minSegLen || (i == n - 1 && i - thisIndx + 1 >= minSegLen && arr[i] == thisItem)) {
            jump_loc[total_found] = thisIndx;
            if (i == n - 1 && i - thisIndx + 1 >= minSegLen && arr[i] == thisItem)
                thisSegLen++;
            jump_len[total_found] = thisSegLen;
            arr_elem[total_found] = arr[thisIndx];
            total_found++;
        }
    }
}

// Chops a long ID (typically voxelID) into XYZ components
template <typename T1, typename T2>
inline void hostIDChopper(T1& X, T1& Y, T1& Z, const T2& ID, const unsigned char& nvXp2, const unsigned char& nvYp2) {
    X = ID & (((T1)1 << nvXp2) - 1);  // & operation here equals modulo
    Y = (ID >> nvXp2) & (((T1)1 << nvYp2) - 1);
    Z = (ID) >> (nvXp2 + nvYp2);
}

// Packs XYZ components back to a long ID (typically voxelID)
template <typename T1, typename T2>
inline void hostIDPacker(T1& ID,
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
inline void hostVoxelID2Position(T1& X,
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
    hostIDChopper<T2, T2>(voxelIDX, voxelIDY, voxelIDZ, ID, nvXp2, nvYp2);
    X = (T1)voxelIDX * voxelSize + (T1)subPosX * l;
    Y = (T1)voxelIDY * voxelSize + (T1)subPosY * l;
    Z = (T1)voxelIDZ * voxelSize + (T1)subPosZ * l;
}

template <typename T1>
std::vector<T1> hostUniqueVector(const std::vector<T1>& vec) {
    std::vector<T1> unique_vec(vec);
    // Need sort first!
    std::sort(unique_vec.begin(), unique_vec.end());
    auto tmp_it = std::unique(unique_vec.begin(), unique_vec.end());
    unique_vec.resize(std::distance(unique_vec.begin(), tmp_it));
    return unique_vec;
}

// We collect h2a in fact, not force
inline void hostCollectForces(inertiaOffset_t* inertiaPropOffsets,
                              bodyID_t* idA,
                              bodyID_t* idB,
                              float3* contactForces,
                              float* clump_aX,
                              float* clump_aY,
                              float* clump_aZ,
                              bodyID_t* ownerClumpBody,
                              float* massOwnerBody,
                              float h,
                              size_t n,
                              double l) {
    for (size_t i = 0; i < n; i++) {
        bodyID_t bodyA = idA[i];
        bodyID_t bodyB = idB[i];
        const float3 F = contactForces[i];
        bodyID_t AOwner = ownerClumpBody[bodyA];
        float AMass = massOwnerBody[inertiaPropOffsets[AOwner]];
        clump_aX[AOwner] += (double)F.x / AMass / l * h * h;
        clump_aY[AOwner] += (double)F.y / AMass / l * h * h;
        clump_aZ[AOwner] += (double)F.z / AMass / l * h * h;

        bodyID_t BOwner = ownerClumpBody[bodyB];
        float BMass = massOwnerBody[inertiaPropOffsets[BOwner]];
        clump_aX[BOwner] += -(double)F.x / BMass / l * h * h;
        clump_aY[BOwner] += -(double)F.y / BMass / l * h * h;
        clump_aZ[BOwner] += -(double)F.z / BMass / l * h * h;
    }
}

// Note we collect h2Alpha, not torque
inline void hostCollectTorques(inertiaOffset_t* inertiaPropOffsets,
                               bodyID_t* idA,
                               bodyID_t* idB,
                               float3* contactForces,
                               float3* contactLocA,
                               float3* contactLocB,
                               float* clump_alphaX,
                               float* clump_alphaY,
                               float* clump_alphaZ,
                               bodyID_t* ownerClumpBody,
                               float* mmiXX,
                               float* mmiYY,
                               float* mmiZZ,
                               float h,
                               size_t n,
                               double l) {
    for (size_t i = 0; i < n; i++) {
        // First, recover the force
        float3 F = contactForces[i];
        // Then, compute alpha as torque/moi
        float3 CPA = contactLocA[i];
        float3 CPB = contactLocB[i];

        bodyID_t bodyA = idA[i];
        bodyID_t bodyB = idB[i];
        bodyID_t AOwner = ownerClumpBody[bodyA];
        bodyID_t BOwner = ownerClumpBody[bodyB];
        inertiaOffset_t AMOIOffset = inertiaPropOffsets[AOwner];
        inertiaOffset_t BMOIOffset = inertiaPropOffsets[BOwner];
        float3 AMOI, BMOI;
        AMOI.x = mmiXX[AMOIOffset];
        AMOI.y = mmiYY[AMOIOffset];
        AMOI.z = mmiZZ[AMOIOffset];
        BMOI.x = mmiXX[BMOIOffset];
        BMOI.y = mmiYY[BMOIOffset];
        BMOI.z = mmiZZ[BMOIOffset];
        float3 alphaA = cross(CPA, F) / AMOI;
        float3 alphaB = cross(CPB, -F) / BMOI;

        clump_alphaX[AOwner] += (double)alphaA.x * h * h;
        clump_alphaY[AOwner] += (double)alphaA.y * h * h;
        clump_alphaZ[AOwner] += (double)alphaA.z * h * h;

        clump_alphaX[BOwner] += (double)alphaB.x * h * h;
        clump_alphaY[BOwner] += (double)alphaB.y * h * h;
        clump_alphaZ[BOwner] += (double)alphaB.z * h * h;
    }
}

/// A light-weight grid/box sampler that can be used to generate the initial stage of the DEM system
inline std::vector<float3> DEMBoxGridSampler(float3 BoxCenter,
                                             float3 HalfDims,
                                             float GridSizeX,
                                             float GridSizeY = -1.0,
                                             float GridSizeZ = -1.0) {
    if (GridSizeY < 0)
        GridSizeY = GridSizeX;
    if (GridSizeZ < 0)
        GridSizeZ = GridSizeX;
    std::vector<float3> points;
    for (float z = BoxCenter.z - HalfDims.z; z <= BoxCenter.z + HalfDims.z; z += GridSizeZ) {
        for (float y = BoxCenter.y - HalfDims.y; y <= BoxCenter.y + HalfDims.y; y += GridSizeY) {
            for (float x = BoxCenter.x - HalfDims.x; x <= BoxCenter.x + HalfDims.x; x += GridSizeX) {
                float3 xyz;
                xyz.x = x;
                xyz.y = y;
                xyz.z = z;
                points.push_back(xyz);
            }
        }
    }
    return points;
}

/// A light-weight sampler that generates a shell made of particles that resembles a cylindrical surface
inline std::vector<float3> DEMCylSurfSampler(float3 CylCenter,
                                             float3 CylAxis,
                                             float CylRad,
                                             float CylHeight,
                                             float ParticleRad,
                                             float spacing = 1.2f) {
    std::vector<float3> points;
    float perimeter = 2.0 * SGPS_PI * CylRad;
    unsigned int NumRows = perimeter / (spacing * ParticleRad);
    float RadIncr = 2.0 * SGPS_PI / (float)(NumRows);
    float SideIncr = spacing * ParticleRad;
    float3 UnitCylAxis = normalize(CylAxis);
    float3 RadDir;
    {
        float3 PerpVec;
        PerpVec.x = -UnitCylAxis.z;
        PerpVec.y = 0.f;
        PerpVec.z = UnitCylAxis.x;
        RadDir = normalize(PerpVec);
    }
    for (unsigned int i = 0; i < NumRows; i++) {
        std::vector<float3> thisRow;
        float3 thisRowSt = CylCenter + UnitCylAxis * (CylHeight / 2.) + RadDir * CylRad;
        for (float d = 0.; d <= CylHeight; d += SideIncr) {
            float3 point;
            point = thisRowSt + UnitCylAxis * (-d);
            thisRow.push_back(point);
        }
        points.insert(points.end(), thisRow.begin(), thisRow.end());
        RadDir = Rodrigues(RadDir, UnitCylAxis, RadIncr);
    }
    return points;
}

/// Host version of applying a quaternion to a vector
template <typename T1, typename T2>
inline void hostapplyOriQToVector3(T1& X, T1& Y, T1& Z, const T2& Q0, const T2& Q1, const T2& Q2, const T2& Q3) {
    T1 oldX = X;
    T1 oldY = Y;
    T1 oldZ = Z;
    X = ((T2)2.0 * (Q0 * Q0 + Q1 * Q1) - (T2)1.0) * oldX + ((T2)2.0 * (Q1 * Q2 - Q0 * Q3)) * oldY +
        ((T2)2.0 * (Q1 * Q3 + Q0 * Q2)) * oldZ;
    Y = ((T2)2.0 * (Q1 * Q2 + Q0 * Q3)) * oldX + ((T2)2.0 * (Q0 * Q0 + Q2 * Q2) - (T2)1.0) * oldY +
        ((T2)2.0 * (Q2 * Q3 - Q0 * Q1)) * oldZ;
    Z = ((T2)2.0 * (Q1 * Q3 - Q0 * Q2)) * oldX + ((T2)2.0 * (Q2 * Q3 + Q0 * Q1)) * oldY +
        ((T2)2.0 * (Q0 * Q0 + Q3 * Q3) - (T2)1.0) * oldZ;
}

// Default accuracy is 17. This accuracy is especially needed for MOIs and length-unit (l).
inline std::string to_string_with_precision(const double a_value, const unsigned int n = 17) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

// Release the memory allocated for an array
template <typename T1>
inline void deallocate_array(std::vector<T1>& arr) {
    std::vector<T1>().swap(arr);
}

}  // namespace sgps

#endif
