//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#ifndef SGPS_DEM_HOST_HELPERS
#define SGPS_DEM_HOST_HELPERS

#pragma once
#include <iostream>
#include <list>
#include <cmath>
#include <vector>
#include <algorithm>

namespace sgps {

template <typename T1>
inline void displayArray(T1* arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        std::cout << arr[i] << " ";
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
void hostScanForJumpsNum(T1* arr, size_t n, unsigned int minSegLen, size_t& total_found) {
    size_t i = 0;
    total_found = 0;
    while (i < n - 1) {
        size_t thisIndx = i;
        T1 thisItem = arr[i];
        do {
            i++;
        } while (arr[i] == thisItem && i < n - 1);
        if (i - thisIndx >= minSegLen || (i == n - 1 && i - thisIndx + 1 >= minSegLen && arr[i] == thisItem)) {
            total_found++;
        }
    }
}

// Tell each active bin where to find its touching spheres
template <typename T1, typename T2, typename T3>
void hostScanForJumps(T1* arr, T1* arr_elem, T2* jump_loc, T3* jump_len, size_t n, unsigned int minSegLen) {
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

// Note we assume the ``force'' here is actually acceleration written in terms of multiples of l
template <typename T1>
void hostCollectForces(bodyID_t* idA,
                       bodyID_t* idB,
                       float3* contactForces,
                       float* clump_h2aX,
                       float* clump_h2aY,
                       float* clump_h2aZ,
                       T1* ownerClumpBody,
                       double h,
                       size_t n) {
    for (size_t i = 0; i < n; i++) {
        bodyID_t bodyA = idA[i];
        bodyID_t bodyB = idB[i];
        T1 AOwner = ownerClumpBody[bodyA];
        clump_h2aX[AOwner] += contactForces[i].x * h * h;
        clump_h2aY[AOwner] += contactForces[i].y * h * h;
        clump_h2aZ[AOwner] += contactForces[i].z * h * h;

        T1 BOwner = ownerClumpBody[bodyB];
        clump_h2aX[BOwner] += -contactForces[i].x * h * h;
        clump_h2aY[BOwner] += -contactForces[i].y * h * h;
        clump_h2aZ[BOwner] += -contactForces[i].z * h * h;
    }
}

}  // namespace sgps

#endif
