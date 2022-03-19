//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once
#include <climits>
#include <stdint.h>

namespace sgps {
#define BITS_PER_BYTE 8
#define N_MANUFACTURED_ITEMS 4
#define NUM_BINS_PER_BLOCK 128
#define NUM_BODIES_PER_BLOCK 512
#define MAX_SPHERES_PER_BIN 16
#define VOXEL_RES_POWER2 16
#define WAIT_GRANULARITY_MS 1
#define TEST_SHARED_SIZE 128
#ifndef SGPS_DEM_TINY_FLOAT
    #define SGPS_DEM_TINY_FLOAT 1e-6f
#endif

typedef uint16_t subVoxelPos_t;  ///< uint16 or uint32
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
// Contact type (sphere--sphere is 0, etc.)
typedef unsigned char contact_t;
typedef char scratch_t;  // Data type for DEM scratch-pad array

// How many bin--sphere touch pairs can there be, tops? This type needs to be large enough to hold the result of a
// prefix scan, but normally, it should be the same magnitude as bodyID_t.
typedef unsigned int binsSphereTouches_t;
// How many spheres a bin can touch, tops? This one will not double as a container for prefix scans, so we can assume it
// will not be too large. We should keep it as small as possible, since in contact detection...
typedef uint16_t spheresBinTouches_t;
// Need to be large enough to hold the number of total contact pairs. This number should be in the same magnitude as
// bodyID_t.
typedef unsigned int contactPairs_t;

// typedef unsigned int stateVectors_default_t; // what's this for??
// typedef unsigned int distinctSphereRelativePositions_default_t;
// typedef unsigned int distinctSphereRadiiOffset_default_t;

// somehow add array materialsArray and radiiArray??

// A few pre-computed constants
#ifndef SGPS_TWO_OVER_THREE
    #define SGPS_TWO_OVER_THREE 0.666666666666667
#endif
#ifndef SGPS_SQRT_FIVE_OVER_SIX
    #define SGPS_SQRT_FIVE_OVER_SIX 0.912870929175277
#endif
#ifndef SGPS_PI
    #define SGPS_PI 3.141592653589793
#endif
#ifndef SGPS_PI_SQUARED
    #define SGPS_PI_SQUARED 9.869604401089358
#endif

}  // namespace sgps
