//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef SMUG_DEM_MISC_DEFINES
#define SMUG_DEM_MISC_DEFINES

#include <limits>
#include <stdint.h>
#include <algorithm>
#include <cmath>

#include <DEM/VariableTypes.h>

#define SMUG_DEM_MIN(a, b) ((a < b) ? a : b)
#define SMUG_DEM_MAX(a, b) ((a > b) ? a : b)

namespace smug {
// =============================================================================
// NOW DEFINING CONSTANTS USED BY THE DEM MODULE
// =============================================================================
#define SMUG_GET_VAR_NAME(Variable) (#Variable)
#define SMUG_DEM_KT_CD_NTHREADS_PER_BLOCK 256
#define SMUG_DEM_MAX_SPHERES_PER_BIN 256  ///< Can't be larger than SMUG_DEM_KT_CD_NTHREADS_PER_BLOCK
#define SMUG_DEM_TINY_FLOAT 1e-12
#define SMUG_DEM_HUGE_FLOAT 1e15
#define SMUG_BITS_PER_BYTE 8
#define SMUG_CUDA_WARP_SIZE 32
#define SMUG_DEM_MAX_WILDCARD_NUM 8

// A few pre-computed constants
constexpr double TWO_OVER_THREE = 0.666666666666667;
constexpr double FOUR_OVER_THREE = 1.333333333333333;
constexpr double FIVE_OVER_THREE = 1.666666666666667;
constexpr double TWO_TIMES_SQRT_FIVE_OVER_SIX = 1.825741858350554;
constexpr double PI = 3.1415926535897932385;
constexpr double PI_SQUARED = 9.869604401089358;

constexpr uint8_t DEM_VOXEL_RES_POWER2 = sizeof(subVoxelPos_t) * SMUG_BITS_PER_BYTE;
constexpr uint8_t DEM_VOXEL_COUNT_POWER2 = sizeof(voxelID_t) * SMUG_BITS_PER_BYTE;
constexpr int64_t DEM_MAX_SUBVOXEL = (int64_t)1 << DEM_VOXEL_RES_POWER2;

#define SMUG_DEM_NUM_BODIES_PER_BLOCK 512
#define SMUG_DEM_MAX_THREADS_PER_BLOCK 1024
#define SMUG_DEM_INIT_CNT_MULTIPLIER 5
// If there are more than this number of analytical geometry, we may have difficulty jitify them all
#define SMUG_DEM_THRESHOLD_TOO_MANY_ANAL_GEO 64
// If a clump has more than this number of sphere components, it is automatically considered a non-jitifiable big clump
#define SMUG_DEM_THRESHOLD_BIG_CLUMP 256
// If there are more than this number of sphere components across all clumps (excluding the clumps that are considered
// big clumps), then some of them may have to stay in global memory, rather than being jitified
#define SMUG_DEM_THRESHOLD_TOO_MANY_SPHERE_COMP 512
// It should generally just be the warp size. When a block is launched, at least min(these_numbers) threads will be
// launched so the template loading is always safe.
constexpr clumpComponentOffset_t NUM_ACTIVE_TEMPLATE_LOADING_THREADS =
    SMUG_DEM_MIN(SMUG_DEM_MIN(SMUG_CUDA_WARP_SIZE, SMUG_DEM_KT_CD_NTHREADS_PER_BLOCK), SMUG_DEM_NUM_BODIES_PER_BLOCK);

const objType_t DEM_ANAL_OBJ_TYPE_PLANE = 0;
const objType_t DEM_ANAL_OBJ_TYPE_PLATE = 1;
const objNormal_t DEM_ENTITY_NORMAL_INWARD = 0;
const objNormal_t DEM_ENTITY_NORMAL_OUTWARD = 1;

const contact_t DEM_NOT_A_CONTACT = 0;
const contact_t DEM_SPHERE_SPHERE_CONTACT = 1;
const contact_t DEM_SPHERE_PLANE_CONTACT = 2;

const notStupidBool_t DEM_DONT_PREVENT_CONTACT = 0;
const notStupidBool_t DEM_PREVENT_CONTACT = 1;

// Codes for owner types. We just have a handful of types...
const ownerType_t DEM_OWNER_T_CLUMP = 0;
const ownerType_t DEM_OWNER_T_ANALYTICAL = 1;
const ownerType_t DEM_OWNER_T_MESH = 2;

// This ID marks that this is a new contact, not present when we did contact detection last time
// TODO: half max add half max... so stupid... Better way?? numeric_limit won't work...
constexpr contactPairs_t DEM_NULL_MAPPING_PARTNER =
    ((size_t)1 << (sizeof(contactPairs_t) * SMUG_BITS_PER_BYTE - 1)) +
    ((size_t)1 << (sizeof(contactPairs_t) * SMUG_BITS_PER_BYTE - 1) - 1);
// Reserved bodyID
constexpr bodyID_t DEM_NULL_BODYID = ((size_t)1 << (sizeof(bodyID_t) * SMUG_BITS_PER_BYTE - 1)) +
                                     ((size_t)1 << (sizeof(bodyID_t) * SMUG_BITS_PER_BYTE - 1) - 1);
// Default (user) clump family number
const unsigned int DEM_DEFAULT_CLUMP_FAMILY_NUM = 0;
// Reserved (user) clump family number which is always used for fixities
constexpr unsigned int DEM_RESERVED_FAMILY_NUM = ((unsigned int)1 << (sizeof(family_t) * SMUG_BITS_PER_BYTE)) - 1;
// The number of all possible families is known: it depends on family_t
constexpr size_t DEM_NUM_AVAL_FAMILIES = (size_t)1 << (sizeof(family_t) * SMUG_BITS_PER_BYTE);
// Reserved clump template mark number, used to indicate the largest inertiaOffset number (currently not used since all
// inertia properties are jitified)
constexpr inertiaOffset_t DEM_RESERVED_INERTIA_OFFSET =
    ((size_t)1 << (sizeof(inertiaOffset_t) * SMUG_BITS_PER_BYTE)) - 1;
// Reserved clump component offset number, used to indicate that this sphere's relative pos etc. won't be found in the
// kernel, instead have to be brought from the global memory
constexpr clumpComponentOffset_t DEM_RESERVED_CLUMP_COMPONENT_OFFSET =
    ((size_t)1 << (sizeof(clumpComponentOffset_t) * SMUG_BITS_PER_BYTE)) - 1;
// Used to be compared against, so we know if some of the sphere components need to stay in global memory
constexpr unsigned int DEM_THRESHOLD_CANT_JITIFY_ALL_COMP =
    SMUG_DEM_MIN(SMUG_DEM_MIN(DEM_RESERVED_CLUMP_COMPONENT_OFFSET, SMUG_DEM_THRESHOLD_BIG_CLUMP),
                 SMUG_DEM_THRESHOLD_TOO_MANY_SPHERE_COMP);

// Some enums...
// Verbosity
enum DEM_VERBOSITY { QUIET = 0, ERROR = 10, WARNING = 20, INFO = 30, STEP_STATS = 32, STEP_METRIC = 35, DEBUG = 40 };
// Stepping method
enum class DEM_TIME_INTEGRATOR { FORWARD_EULER, CENTERED_DIFFERENCE, EXTENDED_TAYLOR, CHUNG };
// Owner types
enum class DEM_OWNER_TYPE { CLUMP, ANALYTICAL, MESH };
// Types of entities (can be either owner or geometry entity) that can be inspected by inspection methods
enum class DEM_INSPECT_ENTITY_TYPE { SPHERE, CLUMP, MESH, MESH_FACET };
// Which reduce operation is needed in an inspection
enum class DEM_CUB_REDUCE_FLAVOR { NONE, MAX, MIN, SUM };
// Format of the output files
enum class DEM_OUTPUT_FORMAT { CSV, BINARY, CHPF };
// Force mode type
enum class DEM_FORCE_MODEL { HERTZIAN, HERTZIAN_FRICTIONLESS, CUSTOM };
// The info that should be present in the output files
enum DEM_OUTPUT_CONTENT {
    XYZ = 0,
    QUAT = 1,
    ABSV = 2,
    VEL = 4,
    ANG_VEL = 8,
    ACC = 16,
    ANG_ACC = 32,
    FAMILY = 64,
    MAT = 128,
    // How much this clump expanded in size via ChangeClumpSizes, compared to its `vanilla' template. Can be useful if
    // the user imposed some fine-grain clump size control.
    EXP_FACTOR = 256
};
// Output particles as individual (component) spheres, or as owner clumps (clump CoMs for location, as an example)?
enum class DEM_SPATIAL_DIR { X, Y, Z, NONE };

// =============================================================================
// NOW DEFINING SOME GPU-SIDE DATA STRUCTURES
// =============================================================================

// Structs defined here will be used by GPUs.
// NOTE: All data structs here need to be simple enough to jitify. In general, if you need to include something much
// more complex than DEMDefines for example, then do it in DEMStructs.h.

// A structure for storing simulation parameters.
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
    bodyID_t nTriMeshes;

    // bodyID_t nSpheresJitified;
    // triID_t nTriJitified;

    // Number of the templates (or say the ``types'') of clumps and spheres
    unsigned int nDistinctClumpBodyTopologies;
    unsigned int nDistinctMassProperties;     ///< All mass property items, not just clumps'
    unsigned int nJitifiableClumpComponents;  ///< Does not include `big' clump's (external object's) components
    unsigned int nDistinctClumpComponents;    ///< All clump components, including the ones `big' clumps have
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
    float h;
    // Sphere radii/geometry thickness inflation amount (for safer contact detection)
    float beta;
    // Max velocity, user approximated, we verify during simulation
    float approxMaxVel;
    // Expand safety parameter (multiplier for the CD envelope)
    float expSafetyParam;
    // Stepping method
    DEM_TIME_INTEGRATOR stepping = DEM_TIME_INTEGRATOR::FORWARD_EULER;

    // Number of wildcards (extra property) arrays associated with contacts and owners
    unsigned int nContactWildcards;
    unsigned int nOwnerWildcards;
};

// A struct that holds pointers to data arrays that dT uses
// For more details just look at PhysicsSystem.h
struct DEMDataDT {
    inertiaOffset_t* inertiaPropOffsets;

    family_t* familyID;

    voxelID_t* voxelID;

    subVoxelPos_t* locX;
    subVoxelPos_t* locY;
    subVoxelPos_t* locZ;

    oriQ_t* oriQw;
    oriQ_t* oriQx;
    oriQ_t* oriQy;
    oriQ_t* oriQz;

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
    contactPairs_t* contactMapping;

    // Family mask
    notStupidBool_t* familyMasks;

    // Some dT's own work array pointers
    float3* contactForces;
    float3* contactTorque_convToForce;
    float3* contactPointGeometryA;
    float3* contactPointGeometryB;
    // float3* contactHistory;
    // float* contactDuration;

    // The offset info that indexes into the template arrays
    bodyID_t* ownerClumpBody;
    clumpComponentOffset_t* clumpComponentOffset;
    clumpComponentOffsetExt_t* clumpComponentOffsetExt;
    materialsOffset_t* sphereMaterialOffset;
    bodyID_t* ownerMesh;
    float3* relPosNode1;
    float3* relPosNode2;
    float3* relPosNode3;
    materialsOffset_t* triMaterialOffset;

    // dT-owned buffer pointers, for itself's usage
    size_t nContactPairs_buffer = 0;
    bodyID_t* idGeometryA_buffer;
    bodyID_t* idGeometryB_buffer;
    contact_t* contactType_buffer;
    contactPairs_t* contactMapping_buffer;

    // pointer to remote buffer where kinematic thread stores work-order data provided by the dynamic thread
    voxelID_t* pKTOwnedBuffer_voxelID = NULL;
    subVoxelPos_t* pKTOwnedBuffer_locX = NULL;
    subVoxelPos_t* pKTOwnedBuffer_locY = NULL;
    subVoxelPos_t* pKTOwnedBuffer_locZ = NULL;
    oriQ_t* pKTOwnedBuffer_oriQ0 = NULL;
    oriQ_t* pKTOwnedBuffer_oriQ1 = NULL;
    oriQ_t* pKTOwnedBuffer_oriQ2 = NULL;
    oriQ_t* pKTOwnedBuffer_oriQ3 = NULL;
    family_t* pKTOwnedBuffer_familyID = NULL;

    // The collection of pointers to DEM template arrays such as radiiSphere, still useful when there are template info
    // not directly jitified into the kernels
    float* radiiSphere;
    float* relPosSphereX;
    float* relPosSphereY;
    float* relPosSphereZ;
    float* massOwnerBody;
    float* mmiXX;
    float* mmiYY;
    float* mmiZZ;
    float* volumeOwnerBody;

    // Wildcards. These are some quantities that you can associate with contact pairs and/or owner objects. Very
    // typically, contact history info in Hertzian model in this DEM tool is a wildcard, and electric charges can be
    // registered on granular particles with wildcards.
    float* contactWildcards[SMUG_DEM_MAX_WILDCARD_NUM];
    float* ownerWildcards[SMUG_DEM_MAX_WILDCARD_NUM];
};

// A struct that holds pointers to data arrays that kT uses
// For more details just look at PhysicsSystem.h
struct DEMDataKT {
    family_t* familyID;
    voxelID_t* voxelID;
    subVoxelPos_t* locX;
    subVoxelPos_t* locY;
    subVoxelPos_t* locZ;
    oriQ_t* oriQw;
    oriQ_t* oriQx;
    oriQ_t* oriQy;
    oriQ_t* oriQz;

    // kT-owned buffer pointers, for itself's usage
    voxelID_t* voxelID_buffer;
    subVoxelPos_t* locX_buffer;
    subVoxelPos_t* locY_buffer;
    subVoxelPos_t* locZ_buffer;
    oriQ_t* oriQ0_buffer;
    oriQ_t* oriQ1_buffer;
    oriQ_t* oriQ2_buffer;
    oriQ_t* oriQ3_buffer;
    family_t* familyID_buffer;

    // Family mask
    notStupidBool_t* familyMasks;

    // The offset info that indexes into the template arrays
    bodyID_t* ownerClumpBody;
    clumpComponentOffset_t* clumpComponentOffset;
    clumpComponentOffsetExt_t* clumpComponentOffsetExt;
    bodyID_t* ownerMesh;
    float3* relPosNode1;
    float3* relPosNode2;
    float3* relPosNode3;

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
    bodyID_t* previous_idGeometryA;
    bodyID_t* previous_idGeometryB;
    contact_t* previous_contactType;
    contactPairs_t* contactMapping;

    // data pointers that is kT's transfer destination
    size_t* pDTOwnedBuffer_nContactPairs = NULL;
    bodyID_t* pDTOwnedBuffer_idGeometryA = NULL;
    bodyID_t* pDTOwnedBuffer_idGeometryB = NULL;
    contact_t* pDTOwnedBuffer_contactType = NULL;
    contactPairs_t* pDTOwnedBuffer_contactMapping = NULL;

    // The collection of pointers to DEM template arrays such as radiiSphere, still useful when there are template info
    // not directly jitified into the kernels
    float* radiiSphere;
    float* relPosSphereX;
    float* relPosSphereY;
    float* relPosSphereZ;
};

// typedef DEMDataDT* DEMDataDTPtr;
// typedef DEMSimParams* DEMSimParamsPtr;

}  // namespace smug

#endif
