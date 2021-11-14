//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once
#include <climits>
#include <stdint.h>

namespace sgps {
#define BITS_PER_BYTE 8
#define N_MANUFACTURED_ITEMS 4
#define N_INPUT_ITEMS 4
#define VOXEL_RES_POWER2 16
#define WAIT_GRANULARITY_MS 1

typedef uint16_t subVoxelPos_t;
typedef uint64_t voxelID_t;
typedef int oriQ_t;  // orientation quaternion data type
typedef unsigned int bodyID_t;
typedef unsigned short int materialsOffset_t;
typedef unsigned short int clumpBodyInertiaOffset_t;
typedef unsigned short int clumpComponentOffset_t;

// typedef unsigned int stateVectors_default_t; // what's this for??
// typedef unsigned int distinctSphereRelativePositions_default_t;
// typedef unsigned int distinctSphereRadiiOffset_default_t;

// somehow add array materialsArray and radiiArray??

}  // namespace sgps
