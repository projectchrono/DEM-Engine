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
#include <random>
#include <thread>
#include <future>
#include <utility>
#include <tuple>
#include <type_traits>

#include "../kernel/DEMHelperKernels.cuh"
#include "VariableTypes.h"

namespace deme {

// Generic helper function to run something in a thread and get the result
template <typename Func, typename... Args>
inline auto run_in_thread(Func&& func, Args&&... args) -> decltype(func(args...)) {
    using ReturnType = decltype(func(args...));

    std::promise<ReturnType> prom;
    std::future<ReturnType> fut = prom.get_future();

    // Capture everything in a tuple
    auto bound_args = std::make_tuple(std::forward<Func>(func), std::forward<Args>(args)...);

    std::thread t([&prom, bound_args = std::move(bound_args)]() mutable {
        try {
            auto result = std::apply(std::move(std::get<0>(bound_args)),
                                     std::apply(
                                         [](auto&&, auto&&... args_inner) {
                                             return std::make_tuple(std::forward<decltype(args_inner)>(args_inner)...);
                                         },
                                         std::move(bound_args)));
            prom.set_value(std::move(result));
        } catch (...) {
            prom.set_exception(std::current_exception());
        }
    });

    t.join();
    return fut.get();
}

inline int randomZeroOrOne() {
    std::random_device rd;   // Random number device to seed the generator
    std::mt19937 gen(rd());  // Mersenne Twister generator
    std::uniform_int_distribution<> dist(0, 1);
    return dist(gen);
}

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

// Debug-purpose array printer
template <typename T1>
inline void displayArray(T1* arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        std::cout << +(arr[i]) << " ";
    }
    std::cout << std::endl;
}

// An extremely inefficient device vector view function
template <typename T1>
inline void displayDeviceArray(T1* arr, size_t n) {
    std::vector<T1> tmp(n);
    DEME_GPU_CALL(cudaMemcpy(tmp.data(), arr, n * sizeof(T1), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++) {
        std::cout << +(tmp[i]) << " ";
    }
    std::cout << std::endl;
}

inline void displayDeviceFloat3(float3* arr, size_t n) {
    std::vector<float3> tmp(n);
    DEME_GPU_CALL(cudaMemcpy(tmp.data(), arr, n * sizeof(float3), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++) {
        std::cout << "(" << +(tmp[i].x) << ", " << +(tmp[i].y) << ", " << +(tmp[i].z) << "), ";
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

template <typename T>
inline bool isBetween(const T& x, const T& L, const T& U) {
    if (x < L) {
        return false;
    }
    if (x > U) {
        return false;
    }
    return true;
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
    non_match = "";
    bool res = true;
    for (const auto& word : words) {
        if (!match_whole_word(sentence, word)) {
            non_match += word + ",";
            res = false;
        }
    }
    return res;
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

// Host version of sort and return
template <typename T1>
std::vector<T1> hostSort(std::vector<T1> input) {
    std::sort(input.begin(), input.end());
    return input;  // May be moved or elided
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

template <typename T1, typename T2>
inline std::vector<T1> VecOfVecToReal3Vector(const std::vector<std::vector<T2>>& vec) {
    std::vector<T1> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        T1 tmp;
        tmp.x = vec[i][0];
        tmp.y = vec[i][1];
        tmp.z = vec[i][2];

        res[i] = tmp;
    }
    return res;
}

template <typename T1, typename T2>
inline std::vector<T1> VecOfVecToReal4Vector(const std::vector<std::vector<T2>>& vec) {
    std::vector<T1> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        T1 tmp;
        tmp.x = vec[i][0];
        tmp.y = vec[i][1];
        tmp.z = vec[i][2];
        tmp.w = vec[i][3];

        res[i] = tmp;
    }
    return res;
}

template <typename T1, typename T2>
inline std::vector<std::vector<T1>> Real3VectorToVecOfVec(const std::vector<T2>& vec) {
    std::vector<std::vector<T1>> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        std::vector<T1> tmp = {vec[i].x, vec[i].y, vec[i].z};
        res[i] = tmp;
    }
    return res;
}

template <typename T1, typename T2>
inline std::vector<std::vector<T1>> Real4VectorToVecOfVec(const std::vector<T2>& vec) {
    std::vector<std::vector<T1>> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        std::vector<T1> tmp = {vec[i].x, vec[i].y, vec[i].z, vec[i].w};
        res[i] = tmp;
    }
    return res;
}

template <typename T1, typename T2>
inline std::vector<T1> RealTupleVectorToXComponentVector(const std::vector<T2>& vec) {
    std::vector<T1> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        res[i] = vec[i].x;
    }
    return res;
}
template <typename T1, typename T2>
inline std::vector<T1> RealTupleVectorToYComponentVector(const std::vector<T2>& vec) {
    std::vector<T1> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        res[i] = vec[i].y;
    }
    return res;
}
template <typename T1, typename T2>
inline std::vector<T1> RealTupleVectorToZComponentVector(const std::vector<T2>& vec) {
    std::vector<T1> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        res[i] = vec[i].z;
    }
    return res;
}
template <typename T1, typename T2>
inline std::vector<T1> RealTupleVectorToWComponentVector(const std::vector<T2>& vec) {
    std::vector<T1> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        res[i] = vec[i].w;
    }
    return res;
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
    applyOriQToVector3(pos.x, pos.y, pos.z, rot_Q.w, -rot_Q.x, -rot_Q.y, -rot_Q.z);
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
inline void assertPositive(const T& var, const std::string& func_name, const std::string& var_name) {
    if (var <= (T)0) {
        std::stringstream out;
        out << func_name << "'s " << var_name << " argument needs to be all-positive.\n";
        throw std::runtime_error(out.str());
    }
}
template <typename T>
inline void assertThreeElements(const std::vector<T>& vec, const std::string& func_name, const std::string& var_name) {
    if (vec.size() != 3) {
        std::stringstream out;
        out << func_name << "'s " << var_name
            << " argument needs to be, or be composed of, length-3 lists/vectors. The provided size is " << vec.size()
            << ".\n";
        throw std::runtime_error(out.str());
    }
}
template <typename T>
inline void assertFourElements(const std::vector<T>& vec, const std::string& func_name, const std::string& var_name) {
    if (vec.size() != 4) {
        std::stringstream out;
        out << func_name << "'s " << var_name
            << " argument needs to be, or be composed of, length-4 lists/vectors. The provided size is " << vec.size()
            << ".\n";
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
