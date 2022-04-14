//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once
#include <climits>
#include <stdint.h>
#include <algorithm>

#define SGPS_DEM_MIN(a, b) ((a < b) ? a : b)
#define SGPS_DEM_MAX(a, b) ((a > b) ? a : b)

namespace sgps {
#ifndef SGPS_GET_VAR_NAME
    #define SGPS_GET_VAR_NAME(Variable) (#Variable)
#endif

#define SGPS_DEM_MAX_SPHERES_PER_BIN 32  // very tricky; should redo CD kernels to be block--bin based
#define WAIT_GRANULARITY_MS 1
#define TEST_SHARED_SIZE 128
#ifndef SGPS_DEM_TINY_FLOAT
    #define SGPS_DEM_TINY_FLOAT 1e-6f
#endif
#ifndef SGPS_BITS_PER_BYTE
    #define SGPS_BITS_PER_BYTE 8
#endif
#ifndef SGPS_CUDA_WARP_SIZE
    #define SGPS_CUDA_WARP_SIZE 32
#endif

typedef uint16_t subVoxelPos_t;  ///< uint16 or uint32
constexpr uint8_t VOXEL_RES_POWER2 = sizeof(subVoxelPos_t) * SGPS_BITS_PER_BYTE;
typedef uint64_t voxelID_t;
// TODO: oriQ should be int (mapped to [-1,1]); applyOriQ2Vector3 and hostApplyOriQ2Vector3 need to be changed to make
// that happen
typedef float oriQ_t;
typedef unsigned int bodyID_t;
typedef unsigned int binID_t;
typedef unsigned short int materialsOffset_t;
typedef unsigned short int clumpBodyInertiaOffset_t;
typedef unsigned short int clumpComponentOffset_t;
typedef double floatFine_t;
typedef unsigned short int contact_t;  ///< Contact type (sphere--sphere is 0, etc.)
typedef unsigned short int family_t;   ///< Data type for clump presecription type (0 for not prescribed)
typedef char scratch_t;                ///< Data type for DEM scratch-pad array

// How many bin--sphere touch pairs can there be for one sphere, tops? This type should not need to be large.
typedef unsigned short int binsSphereTouches_t;
// This type needs to be large enough to hold the result of a prefix scan of the type binsSphereTouches_t; but normally,
// it should be the same magnitude as bodyID_t.
typedef unsigned int binSphereTouchPairs_t;
// How many spheres a bin can touch, tops? We can assume it will not be too large to save GPU memory. Note this type
// also doubles as the type for the number of contacts in a bin. NOTE!! Seems uint8_t is not supported by CUB???
typedef unsigned short int spheresBinTouches_t;
// Need to be large enough to hold the number of total contact pairs. In general this number should be in the same
// magnitude as bodyID_t.
typedef unsigned int contactPairs_t;

// typedef unsigned int stateVectors_default_t; // what's this for??
// typedef unsigned int distinctSphereRelativePositions_default_t;
// typedef unsigned int distinctSphereRadiiOffset_default_t;

#define SGPS_DEM_NUM_BINS_PER_BLOCK 128
#define SGPS_DEM_NUM_BODIES_PER_BLOCK 512
// It should generally just be the warp size. When a block is launched, at least min(these_numbers) threads will be
// launched so the template loading is always safe.
constexpr clumpComponentOffset_t NUM_ACTIVE_TEMPLATE_LOADING_THREADS =
    SGPS_DEM_MIN(SGPS_DEM_MIN(SGPS_CUDA_WARP_SIZE, SGPS_DEM_NUM_BINS_PER_BLOCK), SGPS_DEM_NUM_BODIES_PER_BLOCK);

// A few pre-computed constants

#ifndef SGPS_TWO_OVER_THREE
    #define SGPS_TWO_OVER_THREE 0.666666666666667
#endif
#ifndef SGPS_TWO_TIMES_SQRT_FIVE_OVER_SIX
    #define SGPS_TWO_TIMES_SQRT_FIVE_OVER_SIX 1.825741858350554
#endif
#ifndef SGPS_PI
    #define SGPS_PI 3.141592653589793
#endif
#ifndef SGPS_PI_SQUARED
    #define SGPS_PI_SQUARED 9.869604401089358
#endif

}  // namespace sgps
