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
#include <numeric>
#include <algorithm>
#include <regex>
#include <fstream>
#include <filesystem>
#include <nvmath/helper_math.cuh>

#include <DEM/DEMDefines.h>

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
inline T1 vector_sum(const std::vector<T1>& vect) {
    T1 sum_of_elems = std::accumulate(vect.begin(), vect.end(), T1(0));
    return sum_of_elems;
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

//// TODO: Why is there a namespace (?) issue that makes us able to use make_float3 properly in demo scripts, but not in
/// any of these DEM system h or cpp files?
inline float3 host_make_float3(float a, float b, float c) {
    float3 f;
    f.x = a;
    f.y = b;
    f.z = c;
    return f;
}
inline float4 host_make_float4(float a, float b, float c, float d) {
    float4 f;
    f.x = a;
    f.y = b;
    f.z = c;
    f.w = d;
    return f;
}

/// Replace all instances of a certain pattern from a string then return it
inline std::string replace_pattern(const std::string& in, const std::string& from, const std::string& to) {
    return std::regex_replace(in, std::regex(from), to);
}

/// Replace all instances of certain patterns from a string, based on a mapping passed as an argument
inline std::string replace_patterns(const std::string& in,
                                    const std::unordered_map<std::string, std::string>& mapping) {
    std::string str = in;
    for (const auto& rep : mapping) {
        str = std::regex_replace(str, std::regex(rep.first), rep.second);
    }
    return str;
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

/// Remove comments and newlines from a piece of code, so that string replacement happens `in-line'. When the code is
/// compacted, it gives more understandable compiler error msg, but make it explode if user's code needs to have '\n'
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
inline void hostVoxelIDToPosition(T1& X,
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

// From xyz coordinate (usually double-precision) to voxelID
template <typename T1, typename T2, typename T3>
inline void hostPositionToVoxelID(T1& ID,
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
    voxelID_t voxelNumX = X / voxelSize;
    voxelID_t voxelNumY = Y / voxelSize;
    voxelID_t voxelNumZ = Z / voxelSize;
    subPosX = (X - (T3)voxelNumX * voxelSize) / l;
    subPosY = (Y - (T3)voxelNumY * voxelSize) / l;
    subPosZ = (Z - (T3)voxelNumZ * voxelSize) / l;

    ID = voxelNumX;
    ID += voxelNumY << nvXp2;
    ID += voxelNumZ << (nvXp2 + nvYp2);
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

template <typename T1>
inline bool check_exist(const std::vector<T1>& vec, const T1& key) {
    return std::find(vec.begin(), vec.end(), key) != vec.end();
}

template <typename T1, typename T2>
inline bool check_exist(const std::unordered_map<T1, T2>& map, const T1& key) {
    return map.find(key) != map.end();
}

/// Host version of applying a quaternion to a vector
template <typename T1, typename T2>
inline void hostApplyOriQToVector3(T1& X, T1& Y, T1& Z, const T2& Q0, const T2& Q1, const T2& Q2, const T2& Q3) {
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
template <typename T1, typename T2>
inline void deallocate_array(std::unordered_map<T1, T2>& mapping) {
    mapping.clear();
}

// Load content from a file to a string
inline std::string read_file_to_string(const std::filesystem::path& sourcefile) {
    std::ifstream t(sourcefile);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

}  // namespace sgps

#endif
