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
#define MAX_SPHERES_PER_BIN 16
#define VOXEL_RES_POWER2 16
#define WAIT_GRANULARITY_MS 1
#define TEST_SHARED_SIZE 128

typedef uint16_t subVoxelPos_t;  ///< uint16 or uint32
typedef uint64_t voxelID_t;
typedef int oriQ_t;  // orientation quaternion data type
typedef unsigned int bodyID_t;
typedef unsigned int binID_t;
typedef unsigned short int materialsOffset_t;
typedef unsigned short int clumpBodyInertiaOffset_t;
typedef unsigned short int clumpComponentOffset_t;
typedef double floatFine_t;

// How many bin--sphere touch pairs can there be, tops? This type needs to be large enough to hold the result of a
// prefix scan, but normally, it should be the same magnitude as bodyID_t.
typedef unsigned int binsSphereTouches_t;
// How many spheres a bin can touch, tops? This one will not double as a container for prefix scans, so we can assume it
// will not be too large. We should keep it as small as possible, since in contact detection,
typedef uint16_t spheresBinTouches_t;
// Need to be large enough to hold the number of total contact pairs. This number should be in the same magnitude as
// bodyID_t.
typedef unsigned int contactPairs_t;

// typedef unsigned int stateVectors_default_t; // what's this for??
// typedef unsigned int distinctSphereRelativePositions_default_t;
// typedef unsigned int distinctSphereRadiiOffset_default_t;

// somehow add array materialsArray and radiiArray??

}  // namespace sgps
