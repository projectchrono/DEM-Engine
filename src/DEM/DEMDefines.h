//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once
#include <limits>
#include <stdint.h>
#include <algorithm>
#include <cmath>

#include <DEM/VariableTypes.h>

#define SGPS_DEM_MIN(a, b) ((a < b) ? a : b)
#define SGPS_DEM_MAX(a, b) ((a > b) ? a : b)

namespace sgps {
// =============================================================================
// NOW DEFINING CONSTANTS USED BY THE DEM MODULE
// =============================================================================

#ifndef SGPS_GET_VAR_NAME
    #define SGPS_GET_VAR_NAME(Variable) (#Variable)
#endif

#define SGPS_DEM_MAX_SPHERES_PER_BIN 32  ///< Can't be too large since one thread processes one bin
#define WAIT_DEMITY_MS 1
#ifndef SGPS_DEM_TINY_FLOAT
    #define SGPS_DEM_TINY_FLOAT 1e-12
#endif
#ifndef SGPS_DEM_HUGE_FLOAT
    #define SGPS_DEM_HUGE_FLOAT 1e15
#endif
#ifndef SGPS_BITS_PER_BYTE
    #define SGPS_BITS_PER_BYTE 8
#endif
#ifndef SGPS_CUDA_WARP_SIZE
    #define SGPS_CUDA_WARP_SIZE 32
#endif

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

constexpr uint8_t DEM_VOXEL_RES_POWER2 = sizeof(subVoxelPos_t) * SGPS_BITS_PER_BYTE;
constexpr int64_t DEM_MAX_SUBVOXEL = (int64_t)1 << DEM_VOXEL_RES_POWER2;

#define SGPS_DEM_NUM_BINS_PER_BLOCK 128
#define SGPS_DEM_NUM_BODIES_PER_BLOCK 512
#define SGPS_DEM_INIT_CNT_MULTIPLIER 4
// It should generally just be the warp size. When a block is launched, at least min(these_numbers) threads will be
// launched so the template loading is always safe.
constexpr clumpComponentOffset_t NUM_ACTIVE_TEMPLATE_LOADING_THREADS =
    SGPS_DEM_MIN(SGPS_DEM_MIN(SGPS_CUDA_WARP_SIZE, SGPS_DEM_NUM_BINS_PER_BLOCK), SGPS_DEM_NUM_BODIES_PER_BLOCK);

const objType_t DEM_ENTITY_TYPE_PLANE = 0;
const objType_t DEM_ENTITY_TYPE_PLATE = 1;
const objNormal_t DEM_ENTITY_NORMAL_INWARD = 0;
const objNormal_t DEM_ENTITY_NORMAL_OUTWARD = 1;

const contact_t DEM_NOT_A_CONTACT = 0;
const contact_t DEM_SPHERE_SPHERE_CONTACT = 1;
const contact_t DEM_SPHERE_PLANE_CONTACT = 2;
const notStupidBool_t DEM_DONT_PREVENT_CONTACT = 0;
const notStupidBool_t DEM_PREVENT_CONTACT = 1;

const unsigned int DEM_DEFAULT_CLUMP_FAMILY_NUM = 0;
constexpr unsigned int DEM_RESERVED_FAMILY_NUM = ((unsigned int)1 << (sizeof(family_t) * SGPS_BITS_PER_BYTE)) - 1;

// Some enums...
// Friction mode
enum class DEM_FRICTION_MODE { FRICTIONLESS, MULTI_STEP };
// Verbosity
enum DEM_VERBOSITY { QUIET = 0, ERROR = 10, WARNING = 20, INFO = 30, DEBUG = 40 };

// =============================================================================
// NOW DEFINING SOME GPU-SIDE DATA STRUCTURES
// =============================================================================

// Structs defined here will be used by GPUs.
// NOTE: All data structs here need to be simple enough to jitify. In general, if you need to include something much
// more complex than DEMDefines for example, then do it in DEMStructs.h.

// A structure for storing simulation parameters. Note these simulation parameters should not change often (changes of
// them usually lead to re-jitification). Those who change often (especially in each time step) should go into
// DEMSolverStateDataKT/DT.
struct DEMSimParams {
    // Number of voxels in the X direction, expressed as a power of 2
    unsigned char nvXp2;
    // Number of voxels in the Y direction, expressed as a power of 2
    unsigned char nvYp2;
    // Number of voxels in the Z direction, expressed as a power of 2
    unsigned char nvZp2;
    // Number of bins in the X direction (actual number)
    binID_t nbX;
    // Number of bins in the Y direction (actual number)
    binID_t nbY;
    // Number of bins in the Z direction (actual number)
    binID_t nbZ;
    // Smallest length unit
    double l;
    // Double-precision single voxel size
    double voxelSize;
    // The edge length of a bin (for contact detection)
    double binSize;
    // Number of clumps, spheres, triangles, mesh-represented objects, analytical components, external objs...
    bodyID_t nSpheresGM;
    triID_t nTriGM;
    objID_t nAnalGM;
    bodyID_t nOwnerBodies;
    bodyID_t nOwnerClumps;
    objID_t nExtObj;
    bodyID_t nTriEntities;

    // Number of the templates (or say the ``types'') of clumps and spheres
    clumpBodyInertiaOffset_t nDistinctClumpBodyTopologies;
    clumpComponentOffset_t nDistinctClumpComponents;  ///< Does not include `big' clump's (external object's) components
    materialsOffset_t nMatTuples;
    // Coordinate of the left-bottom-front point of the simulation ``world''
    float LBFX;
    float LBFY;
    float LBFZ;
    // Grav acceleration
    float Gx;
    float Gy;
    float Gz;
    // Time step size
    double h;
    // Sphere radii/geometry thickness inflation amount (for safer contact detection)
    float beta;
};

// The collection of pointers to DEM template arrays such as radiiSphere. This struct will become useless eventually as
// the string substitution in JIT comes into play.
struct DEMTemplate {
    // unsigned int* inflatedRadiiVoxelRatio;
    float* radiiSphere;
    float* relPosSphereX;
    float* relPosSphereY;
    float* relPosSphereZ;
    float* massClumpBody;
    float* mmiXX;
    float* mmiYY;
    float* mmiZZ;
    // materialsOffset_t* materialTupleOffset;
    float* EProxy;
    float* nuProxy;
    float* CoRProxy;
};

// DEM material proxy, such that when they are used in force kernels, we can use these (which are associated with each
// material individually, mind you) to somehow recover the contact properties such as tangential and normal stiffness k.
struct DEMMaterialProxy {
    float k_proxy = 50000.0;
};

// A struct that holds pointers to data arrays that dT uses
// For more details just look at PhysicsSystem.h
struct DEMDataDT {
    clumpBodyInertiaOffset_t* inertiaPropOffsets;

    family_t* familyID;

    voxelID_t* voxelID;

    subVoxelPos_t* locX;
    subVoxelPos_t* locY;
    subVoxelPos_t* locZ;

    oriQ_t* oriQ0;
    oriQ_t* oriQ1;
    oriQ_t* oriQ2;
    oriQ_t* oriQ3;

    float* vX;
    float* vY;
    float* vZ;

    float* omgBarX;
    float* omgBarY;
    float* omgBarZ;

    float* aX;
    float* aY;
    float* aZ;

    float* alphaX;
    float* alphaY;
    float* alphaZ;

    bodyID_t* idGeometryA;
    bodyID_t* idGeometryB;
    contact_t* contactType;

    // Some dT's own work array pointers
    float3* contactForces;
    float3* contactPointGeometryA;
    float3* contactPointGeometryB;

    // The offset info that indexes into the template arrays
    bodyID_t* ownerClumpBody;
    clumpComponentOffset_t* clumpComponentOffset;
    materialsOffset_t* materialTupleOffset;

    // dT-owned buffer pointers, for itself's usage
    size_t nContactPairs_buffer = 0;
    bodyID_t* idGeometryA_buffer;
    bodyID_t* idGeometryB_buffer;
    contact_t* contactType_buffer;

    // pointer to remote buffer where kinematic thread stores work-order data provided by the dynamic thread
    voxelID_t* pKTOwnedBuffer_voxelID = NULL;
    subVoxelPos_t* pKTOwnedBuffer_locX = NULL;
    subVoxelPos_t* pKTOwnedBuffer_locY = NULL;
    subVoxelPos_t* pKTOwnedBuffer_locZ = NULL;
    oriQ_t* pKTOwnedBuffer_oriQ0 = NULL;
    oriQ_t* pKTOwnedBuffer_oriQ1 = NULL;
    oriQ_t* pKTOwnedBuffer_oriQ2 = NULL;
    oriQ_t* pKTOwnedBuffer_oriQ3 = NULL;
};

// A struct that holds pointers to data arrays that kT uses
// For more details just look at PhysicsSystem.h
struct DEMDataKT {
    family_t* familyID;
    voxelID_t* voxelID;
    subVoxelPos_t* locX;
    subVoxelPos_t* locY;
    subVoxelPos_t* locZ;
    oriQ_t* oriQ0;
    oriQ_t* oriQ1;
    oriQ_t* oriQ2;
    oriQ_t* oriQ3;

    // kT-owned buffer pointers, for itself's usage
    voxelID_t* voxelID_buffer;
    subVoxelPos_t* locX_buffer;
    subVoxelPos_t* locY_buffer;
    subVoxelPos_t* locZ_buffer;
    oriQ_t* oriQ0_buffer;
    oriQ_t* oriQ1_buffer;
    oriQ_t* oriQ2_buffer;
    oriQ_t* oriQ3_buffer;

    // The offset info that indexes into the template arrays
    bodyID_t* ownerClumpBody;
    clumpComponentOffset_t* clumpComponentOffset;

    // kT's own work arrays. Now these array pointers get assigned in contactDetection() which point to shared scratch
    // spaces. No need to do forward declaration anymore. They are left here for reference, should contactDetection()
    // need to be re-visited.
    // binsSphereTouches_t* numBinsSphereTouches;
    // binSphereTouchPairs_t* numBinsSphereTouchesScan;
    // binID_t* binIDsEachSphereTouches;
    // bodyID_t* sphereIDsEachBinTouches;
    // binID_t* activeBinIDs;
    // binSphereTouchPairs_t* sphereIDsLookUpTable;
    // spheresBinTouches_t* numSpheresBinTouches;
    // contactPairs_t* numContactsInEachBin;

    // kT produces contact info, and stores it, temporarily
    bodyID_t* idGeometryA;
    bodyID_t* idGeometryB;
    contact_t* contactType;

    // data pointers that is kT's transfer destination
    size_t* pDTOwnedBuffer_nContactPairs = NULL;
    bodyID_t* pDTOwnedBuffer_idGeometryA = NULL;
    bodyID_t* pDTOwnedBuffer_idGeometryB = NULL;
    contact_t* pDTOwnedBuffer_contactType = NULL;
};

// typedef DEMDataDT* DEMDataDTPtr;
// typedef DEMSimParams* DEMSimParamsPtr;

// =============================================================================
// NOW DEFINING MACRO COMMANDS USED BY THE DEM MODULE
// =============================================================================

#define SGPS_DEM_ERROR(...)                                      \
    {                                                            \
        if (verbosity >= DEM_VERBOSITY::ERROR) {                 \
            char error_message[256];                             \
            sprintf(error_message, __VA_ARGS__);                 \
            printf("\nERROR! ");                                 \
            printf("%s", error_message);                         \
            printf("\n%s", __func__);                            \
        }                                                        \
        throw std::runtime_error("\nEXITING SGPS SIMULATION\n"); \
    }

#define SGPS_DEM_WARNING(...)                      \
    {                                              \
        if (verbosity >= DEM_VERBOSITY::WARNING) { \
            printf("\nWARNING! ");                 \
            printf(__VA_ARGS__);                   \
            printf("\n");                          \
        }                                          \
    }

#define SGPS_DEM_INFO(...)                      \
    {                                           \
        if (verbosity >= DEM_VERBOSITY::INFO) { \
            printf(__VA_ARGS__);                \
            printf("\n");                       \
        }                                       \
    }

#define SGPS_DEM_DEBUG_PRINTF(...)               \
    {                                            \
        if (verbosity >= DEM_VERBOSITY::DEBUG) { \
            printf(__VA_ARGS__);                 \
            printf("\n");                        \
        }                                        \
    }

}  // namespace sgps
