//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_VAR_TYPES
#define DEME_VAR_TYPES

namespace deme {

// static_assert(sizeof(size_t) >= sizeof(unsigned long long), "This code should be compiled on 64-bit systems!");

#if defined(_WIN64)
typedef long long ssize_t;
#elif defined(_WIN32)
typedef long ssize_t;
#endif

typedef uint16_t subVoxelPos_t;  ///< uint16 or uint32

typedef uint64_t voxelID_t;
typedef float oriQ_t;
typedef unsigned int bodyID_t;
typedef unsigned int binID_t;
typedef uint8_t objID_t;
typedef uint16_t materialsOffset_t;
typedef uint16_t inertiaOffset_t;
typedef uint8_t clumpComponentOffset_t;
typedef uint16_t clumpComponentOffsetExt_t;  ///< Extended component offset type for non-jitified part
typedef double realFine_t;
typedef char scratch_t;  ///< Data type for DEM scratch-pad array (this should NOT be changed, MUST be 1-byte)

// How many bin--sphere touch pairs can there be for one sphere, tops? This type should not need to be large.
typedef unsigned short int binsSphereTouches_t;
// This type needs to be large enough to hold the result of a prefix scan of the type binsSphereTouches_t (and objID_t);
// but normally, it should be the same magnitude as bodyID_t.
typedef unsigned int binSphereTouchPairs_t;
// How many spheres a bin can touch, tops? We can assume it will not be too large to save GPU memory.
typedef unsigned short int spheresBinTouches_t;
// How many contact pairs can there be in one bin? Sometimes, the geometry overlap is significant and there can be a
// lot.
typedef unsigned int binContactPairs_t;
// Need to be large enough to hold the number of total contact pairs. In general this number should be in the same
// magnitude as bodyID_t.
typedef unsigned int contactPairs_t;
// How many other entities can a sphere touch, tops? It does not need to be large unless you have spheres that have
// magnitudes of difference in size, which you should preferrably avoid.
typedef unsigned short int geoSphereTouches_t;

// How many triangle--sphere touch pairs can there be for one sphere, tops? This type should not need to be large.
typedef unsigned int binsTriangleTouches_t;
// This type needs to be large enough to hold the result of a prefix scan of the type binsTriangleTouches_t (and
// objID_t).
typedef unsigned int binsTriangleTouchPairs_t;
// How many triangles a bin can touch, tops? We can assume it will not be too large to save GPU memory.
typedef unsigned short int trianglesBinTouches_t;

typedef uint8_t notStupidBool_t;  ///< Ad-hoc bool (array) type
typedef uint8_t contact_t;        ///< Contact type (sphere--sphere is 1, etc.)
typedef uint8_t family_t;         ///< Data type for clump presecription type
typedef uint8_t ownerType_t;      ///< The type of a owner entity

typedef uint8_t objType_t;
typedef bool objNormal_t;
}  // namespace deme

#endif
