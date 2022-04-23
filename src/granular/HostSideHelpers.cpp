//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#ifndef SGPS_DEM_HOST_HELPERS
#define SGPS_DEM_HOST_HELPERS

#pragma once
#include <iostream>
#include <sstream>
#include <list>
#include <cmath>
#include <vector>
#include <algorithm>
#include <helper_math.cuh>

#include <granular/GranularDefines.h>

namespace sgps {

// In an upper-triangular matrix, given i and j and num_of_col, this function returns the index of the corresponding
// flatten-ed non-zero entries. This function does not assume i <= j.
template <typename T1>
inline T1 locateMatPair(const T1& i, const T1& j) {
    if (i > j)
        return locateMatPair(j, i);
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

// We collect h2a in fact, not force
inline void hostCollectForces(clumpBodyInertiaOffset_t* inertiaPropOffsets,
                              bodyID_t* idA,
                              bodyID_t* idB,
                              float3* contactForces,
                              float* clump_aX,
                              float* clump_aY,
                              float* clump_aZ,
                              bodyID_t* ownerClumpBody,
                              float* massClumpBody,
                              double h,
                              size_t n,
                              double l) {
    for (size_t i = 0; i < n; i++) {
        bodyID_t bodyA = idA[i];
        bodyID_t bodyB = idB[i];
        const float3 F = contactForces[i];
        bodyID_t AOwner = ownerClumpBody[bodyA];
        float AMass = massClumpBody[inertiaPropOffsets[AOwner]];
        clump_aX[AOwner] += (double)F.x / AMass / l * h * h;
        clump_aY[AOwner] += (double)F.y / AMass / l * h * h;
        clump_aZ[AOwner] += (double)F.z / AMass / l * h * h;

        bodyID_t BOwner = ownerClumpBody[bodyB];
        float BMass = massClumpBody[inertiaPropOffsets[BOwner]];
        clump_aX[BOwner] += -(double)F.x / BMass / l * h * h;
        clump_aY[BOwner] += -(double)F.y / BMass / l * h * h;
        clump_aZ[BOwner] += -(double)F.z / BMass / l * h * h;
    }
}

// Note we collect h2Alpha, not torque
inline void hostCollectTorques(clumpBodyInertiaOffset_t* inertiaPropOffsets,
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
                               double h,
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
        clumpBodyInertiaOffset_t AMOIOffset = inertiaPropOffsets[AOwner];
        clumpBodyInertiaOffset_t BMOIOffset = inertiaPropOffsets[BOwner];
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

/// A light-weight grid sampler that can be used to generate the initial stage of the granular system
inline std::vector<float3> DEMBoxGridSampler(float3 BoxCenter, float3 HalfDims, float GridSize) {
    std::vector<float3> points;
    for (float z = BoxCenter.z - HalfDims.z; z <= BoxCenter.z + HalfDims.z; z += GridSize) {
        for (float y = BoxCenter.y - HalfDims.y; y <= BoxCenter.y + HalfDims.y; y += GridSize) {
            for (float x = BoxCenter.x - HalfDims.x; x <= BoxCenter.x + HalfDims.x; x += GridSize) {
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

/// Host version of applying a quaternion to a vector
template <typename T1, typename T2>
inline void hostApplyOriQ2Vector3(T1& X, T1& Y, T1& Z, const T2& Q0, const T2& Q1, const T2& Q2, const T2& Q3) {
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

/// Calculate the contact params based on the 2 contact material types given
inline void materialProxyMaterixCalculator(float& E_eff,
                                           float& G_eff,
                                           const float& Y1,
                                           const float& nu1,
                                           const float& Y2,
                                           const float& nu2) {
    double invE = (1 - nu1 * nu1) / Y1 + (1 - nu2 * nu2) / Y2;
    E_eff = 1. / invE;
    double invG = 2 * (2 - nu1) * (1 + nu1) / Y1 + 2 * (2 - nu2) * (1 + nu2) / Y2;
    G_eff = 1. / invG;
}

inline std::string to_string_with_precision(const double a_value, const unsigned int n = 10) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

}  // namespace sgps

#endif
