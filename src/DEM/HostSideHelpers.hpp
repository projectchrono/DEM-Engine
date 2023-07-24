//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_HOST_HELPERS
#define DEME_HOST_HELPERS

#include <iostream>
#include <sstream>
#include <list>
#include <cmath>
#include <set>
#include <vector>
#include <numeric>
#include <algorithm>
#include <regex>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <nvmath/helper_math.cuh>
#include <DEM/VariableTypes.h>
// #include <DEM/Defines.h>

namespace deme {

// Sign function
template <typename T1>
inline int sign_func(const T1& val) {
    return (T1(0) < val) - (val < T1(0));
}

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

inline bool isBetween(const float3& coord, const float3& L, const float3& U) {
    if (coord.x < L.x || coord.y < L.y || coord.z < L.z) {
        return false;
    }
    if (coord.x > U.x || coord.y > U.y || coord.z > U.z) {
        return false;
    }
    return true;
}

// Make sure a T1 type triplet falls in a range, then output as T2 type
template <typename T1, typename T2>
inline T2 hostClampBetween(const T1& data, const T2& low, const T2& high) {
    T2 res;
    res = DEME_MIN(DEME_MAX(data, low), high);
    return res;
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

/// Return a perpendicular unit vector of the input vector. Will not work well if the input vector is too short;
/// consider normalizing it if it is the case.
template <typename T1>
inline T1 findPerpendicular(const T1& input_vec) {
    T1 PerpVec;
    PerpVec.x = -input_vec.z;
    PerpVec.y = 0.;
    PerpVec.z = input_vec.x;
    if (length(PerpVec) < 1e-6) {
        PerpVec.x = 0.;
        PerpVec.y = -input_vec.z;
        PerpVec.z = input_vec.y;
    }
    return normalize(PerpVec);
}

/// Generate a mapping between arr1 and arr2, where this mapping has the same length as arr1, and each element i in the
/// mapping is arr1[i]'s corresponding (same) element number in arr2. Both arr1 and arr2 need to be sorted.
template <typename T1>
inline void hostMergeSearchMapGen(T1* arr1, T1* arr2, T1* map, size_t size1, size_t size2, T1 NULL_ID) {
    size_t ind2 = 0;
    for (size_t ind1 = 0; ind1 < size1; ind1++) {
        map[ind1] = NULL_ID;
        while (ind2 < size2) {
            if (arr1[ind1] < arr2[ind2]) {
                // arr1 should be one step ahead, but if not, no match is in arr2
                break;
            } else if (arr1[ind1] == arr2[ind2]) {
                map[ind1] = ind2;
                break;
            } else {
                ind2++;
            }
        }
    }
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
inline float4 host_make_float4(float x, float y, float z, float w) {
    float4 f;
    f.x = x;
    f.y = y;
    f.z = z;
    f.w = w;
    return f;
}

inline size_t hostCalcBinNum(binID_t& nbX,
                             binID_t& nbY,
                             binID_t& nbZ,
                             double m_voxelSize,
                             double m_binSize,
                             unsigned char nvXp2,
                             unsigned char nvYp2,
                             unsigned char nvZp2) {
    nbX = (binID_t)(m_voxelSize * (double)((size_t)1 << nvXp2) / m_binSize) + 1;
    nbY = (binID_t)(m_voxelSize * (double)((size_t)1 << nvYp2) / m_binSize) + 1;
    nbZ = (binID_t)(m_voxelSize * (double)((size_t)1 << nvZp2) / m_binSize) + 1;
    return (size_t)nbX * (size_t)nbY * (size_t)nbZ;
}

/// @brief  Check if the string has only spaces.
inline bool is_all_spaces(const std::string& str) {
    return str.find_first_not_of(' ') == str.npos;
}

// Check if a word exists in a string (but it does not need to be a whole word)
inline bool match_pattern(const std::string& sentence, const std::string& word) {
    std::smatch m;
    std::regex r(word);
    return std::regex_search(sentence, m, r);
}

/// Check if a word exists in a string as a whole word, from bames53 on StackOverflow
/// https://stackoverflow.com/questions/22516463/how-do-i-find-a-complete-word-not-part-of-it-in-a-string-in-c
inline bool match_whole_word(const std::string& sentence, const std::string& word) {
    std::smatch m;
    std::regex r("\\b" + word + "\\b");
    return std::regex_search(sentence, m, r);
}

inline bool any_whole_word_match(const std::string& sentence, const std::set<std::string>& words) {
    for (const auto& word : words) {
        if (match_whole_word(sentence, word))
            return true;
    }
    return false;
}

inline bool all_whole_word_match(const std::string& sentence,
                                 const std::set<std::string>& words,
                                 std::string& non_match) {
    for (const auto& word : words) {
        if (!match_whole_word(sentence, word)) {
            non_match = word;
            return false;
        }
    }
    return true;
}

/// Change all characters in a string to upper case.
inline std::string str_to_upper(const std::string& input) {
    std::string output = input;
    std::transform(input.begin(), input.end(), output.begin(), ::toupper);
    return output;
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
/// compacted, it gives more understandable compiler error msg, but makes it explode if user's code needs to have '\n'
inline std::string compact_code(const std::string& prgm) {
    std::string res;
    res = remove_comments(prgm);
    res = replace_pattern(res, "\n", "");
    return res;
}

/// Get a quaternion from axis and angle
inline float4 QuatFromAxisAngle(const float3& axis, const float& theta) {
    float4 Q;
    Q.x = axis.x * sin(theta / 2);
    Q.y = axis.y * sin(theta / 2);
    Q.z = axis.z * sin(theta / 2);
    Q.w = cos(theta / 2);
    return Q;
}

/// Host version of Quaternion product
inline float4 hostHamiltonProduct(const float4& Q1, const float4& Q2) {
    float4 Q;
    Q.w = Q1.w * Q2.w - Q1.x * Q2.x - Q1.y * Q2.y - Q1.z * Q2.z;
    Q.x = Q1.w * Q2.x + Q1.x * Q2.w + Q1.y * Q2.z - Q1.z * Q2.y;
    Q.y = Q1.w * Q2.y - Q1.x * Q2.z + Q1.y * Q2.w + Q1.z * Q2.x;
    Q.z = Q1.w * Q2.z + Q1.x * Q2.y - Q1.y * Q2.x + Q1.z * Q2.w;
    return Q;
}

/// Rotate a quaternion about an unit axis by an angle
inline float4 RotateQuat(const float4& quat, const float3& axis, const float& theta) {
    // Rotation to quaternion first
    float4 rot = QuatFromAxisAngle(axis, theta);
    // Apply
    return hostHamiltonProduct(rot, quat);
}

/// Rotate a vector about an unit axis by an angle
inline float3 Rodrigues(const float3& vec, const float3& axis, const float& theta) {
    float3 res;
    res = vec * cos(theta) + cross(axis, vec) * sin(theta) + axis * dot(axis, vec) * (1. - cos(theta));
    return res;
}

// Remove elements of a vector based on bool array
template <typename T1>
inline std::vector<T1> hostRemoveElem(const std::vector<T1>& vec, const std::vector<bool>& flags) {
    auto v = vec;
    v.erase(std::remove_if(v.begin(), v.end(), [&flags, &v](const T1& i) { return flags.at(&i - v.data()); }), v.end());
    return v;
}

// Contribution from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
template <typename T1>
inline std::vector<size_t> hostSortIndices(const std::vector<T1>& v) {
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return idx;
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

template <typename T1, typename T2>
inline bool inBoxRegion(const T1& X,
                        const T1& Y,
                        const T1& Z,
                        const std::pair<T2, T2>& Xrange,
                        const std::pair<T2, T2>& Yrange,
                        const std::pair<T2, T2>& Zrange) {
    if ((X >= Xrange.first) && (X <= Xrange.second) && (Y >= Yrange.first) && (Y <= Yrange.second) &&
        (Z >= Zrange.first) && (Z <= Zrange.second)) {
        return true;
    } else {
        return false;
    }
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
inline bool check_exist(const std::set<T1>& the_set, const T1& key) {
    return the_set.find(key) != the_set.end();
}

template <typename T1>
inline bool check_exist(const std::vector<T1>& vec, const T1& key) {
    return std::find(vec.begin(), vec.end(), key) != vec.end();
}

template <typename T1, typename T2>
inline bool check_exist(const std::unordered_map<T1, T2>& map, const T1& key) {
    return map.find(key) != map.end();
}

template <typename T1, typename T2>
inline size_t find_offset_in_list(const T1& list, const T2& key) {
    auto it = std::find(list.begin(), list.end(), key);
    return std::distance(list.begin(), it);
}

inline std::vector<std::string> parse_string_line(const std::string& in_str, const char separator = ',') {
    std::vector<std::string> result;
    std::stringstream s_stream(in_str);
    while (s_stream.good()) {
        std::string substr;
        getline(s_stream, substr, separator);
        result.push_back(substr);
    }
    return result;
}

/// Host version of applying a quaternion to a vector
template <typename T1, typename T2>
inline void hostApplyOriQToVector3(T1& X, T1& Y, T1& Z, const T2& Qw, const T2& Qx, const T2& Qy, const T2& Qz) {
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

/// Host version of applying a local rotation then a translation.
template <typename T1, typename T2, typename T3>
inline void applyFrameTransformLocalToGlobal(T1& pos, const T2& vec, const T3& rot_Q) {
    hostApplyOriQToVector3(pos.x, pos.y, pos.z, rot_Q.w, rot_Q.x, rot_Q.y, rot_Q.z);
    pos.x += vec.x;
    pos.y += vec.y;
    pos.z += vec.z;
}
/// Apply a local rotation then a translation, then return the result.
inline std::vector<double> FrameTransformLocalToGlobal(const std::vector<double>& pos,
                                                       const std::vector<double>& vec,
                                                       const std::vector<double>& rot_Q) {
    double3 deme_pos, deme_vec;
    double4 deme_Q;
    deme_pos.x = pos[0];
    deme_pos.y = pos[1];
    deme_pos.z = pos[2];
    deme_vec.x = vec[0];
    deme_vec.y = vec[1];
    deme_vec.z = vec[2];
    deme_Q.x = rot_Q[0];
    deme_Q.y = rot_Q[1];
    deme_Q.z = rot_Q[2];
    deme_Q.w = rot_Q[3];
    applyFrameTransformLocalToGlobal<double3, double3, double4>(deme_pos, deme_vec, deme_Q);
    return {deme_pos.x, deme_pos.y, deme_pos.z};
}

/// Host version of translating the inverse of the provided vec then applying a local inverse rotation of the provided
/// rot_Q.
template <typename T1, typename T2, typename T3>
inline void applyFrameTransformGlobalToLocal(T1& pos, const T2& vec, const T3& rot_Q) {
    pos.x -= vec.x;
    pos.y -= vec.y;
    pos.z -= vec.z;
    hostApplyOriQToVector3(pos.x, pos.y, pos.z, rot_Q.w, -rot_Q.x, -rot_Q.y, -rot_Q.z);
}
/// Translating the inverse of the provided vec then applying a local inverse rotation of the provided rot_Q, then
/// return the result.
inline std::vector<double> FrameTransformGlobalToLocal(const std::vector<double>& pos,
                                                       const std::vector<double>& vec,
                                                       const std::vector<double>& rot_Q) {
    double3 deme_pos, deme_vec;
    double4 deme_Q;
    deme_pos.x = pos[0];
    deme_pos.y = pos[1];
    deme_pos.z = pos[2];
    deme_vec.x = vec[0];
    deme_vec.y = vec[1];
    deme_vec.z = vec[2];
    deme_Q.x = rot_Q[0];
    deme_Q.y = rot_Q[1];
    deme_Q.z = rot_Q[2];
    deme_Q.w = rot_Q[3];
    applyFrameTransformGlobalToLocal<double3, double3, double4>(deme_pos, deme_vec, deme_Q);
    return {deme_pos.x, deme_pos.y, deme_pos.z};
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

/// Find the offset of an element in an array
template <typename T1>
inline size_t find_array_offset(T1* arr, T1 elem, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (arr[i] == elem)
            return i;
    }
    return n;
}

// A smaller hasher that helps determine the indentifier type. Contribution from Nick and hare1039 on Stackoverflow,
// https://stackoverflow.com/questions/650162/why-cant-the-switch-statement-be-applied-on-strings.
constexpr unsigned int hash_charr(const char* s, int off = 0) {
    return !s[off] ? 7001 : (hash_charr(s, off + 1) * 33) ^ s[off];
}
constexpr inline unsigned int operator"" _(const char* s, size_t) {
    return hash_charr(s);
}

// Load content from a file to a string
inline std::string read_file_to_string(const std::filesystem::path& sourcefile) {
    std::ifstream t(sourcefile);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

// Asserters (for the convenience of Python wrapper)
template <typename T>
inline void assertThreeElements(const std::vector<T>& vec, const std::string& func_name, const std::string& var_name) {
    if (vec.size() != 3) {
        std::stringstream out;
        out << func_name << "'s " << var_name << " argument needs to be a length 3 list/vector. The provided size is "
            << vec.size() << ".\n";
        throw std::runtime_error(out.str());
    }
}
template <typename T>
inline void assertFourElements(const std::vector<T>& vec, const std::string& func_name, const std::string& var_name) {
    if (vec.size() != 4) {
        std::stringstream out;
        out << func_name << "'s " << var_name << " argument needs to be a length 4 list/vector. The provided size is "
            << vec.size() << ".\n";
        throw std::runtime_error(out.str());
    }
}
template <typename T>
inline void assertThreeElementsVector(const std::vector<std::vector<T>>& vec,
                                      const std::string& func_name,
                                      const std::string& var_name) {
    if (vec.at(0).size() != 3) {
        std::stringstream out;
        out << func_name << "'s " << var_name
            << " argument needs to be a list/vector of length 3 vectors (in other words, n by 3 matrix).\n The "
               "provided size is "
            << vec.size() << " by " << vec.at(0).size() << ".\n";
        throw std::runtime_error(out.str());
    }
}
template <typename T>
inline void assertFourElementsVector(const std::vector<std::vector<T>>& vec,
                                     const std::string& func_name,
                                     const std::string& var_name) {
    if (vec.at(0).size() != 4) {
        std::stringstream out;
        out << func_name << "'s " << var_name
            << " argument needs to be a list/vector of length 4 vectors (in other words, n by 4 matrix).\n The "
               "provided size is "
            << vec.size() << " by " << vec.at(0).size() << ".\n";
        throw std::runtime_error(out.str());
    }
}

}  // namespace deme

#endif
