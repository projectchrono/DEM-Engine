//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <granular/GranularDefines.h>

namespace sgps {
// Structs defined here will be used by GPUs.
// NOTE: All data structs here need to be simple enough to jitify. In general, if you need to include something much
// more complex than GranularDefines for example, then do it in GranularStructs.h.

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
    float l;
    // Double-precision single voxel size
    double voxelSize;
    // The edge length of a bin (for contact detection)
    double binSize;
    // Number of clumps and spheres
    bodyID_t nClumpBodies;
    bodyID_t nSpheresGM;
    // Number of the templates (or say the ``types'') of clumps and spheres
    clumpBodyInertiaOffset_t nDistinctClumpBodyTopologies;
    clumpComponentOffset_t nDistinctClumpComponents;
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
    // Sphere radii inflation ratio (for safer contact detection)
    float beta;
    // Number of bins active
    size_t nActiveBins = 0;
    // Number of active contact pairs. kT figures this number out but it is dT that badly needs this number to do its
    // work.
    size_t nContactPairs = 0;
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
    float* GProxy;
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

    voxelID_t* voxelID;

    subVoxelPos_t* locX;
    subVoxelPos_t* locY;
    subVoxelPos_t* locZ;

    oriQ_t* oriQ0;
    oriQ_t* oriQ1;
    oriQ_t* oriQ2;
    oriQ_t* oriQ3;

    float* hvX;
    float* hvY;
    float* hvZ;

    float* hOmgBarX;
    float* hOmgBarY;
    float* hOmgBarZ;

    float* h2aX;
    float* h2aY;
    float* h2aZ;

    float* h2AlphaX;
    float* h2AlphaY;
    float* h2AlphaZ;

    bodyID_t* idGeometryA;
    bodyID_t* idGeometryB;

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

    // Other kT's own work arrays
    binsSphereTouches_t* numBinsSphereTouches;
    binSphereTouchPairs_t* numBinsSphereTouchesScan;
    binID_t* binIDsEachSphereTouches;
    bodyID_t* sphereIDsEachBinTouches;
    binID_t* activeBinIDs;
    binSphereTouchPairs_t* sphereIDsLookUpTable;
    spheresBinTouches_t* numSpheresBinTouches;
    contactPairs_t* numContactsInEachBin;

    // kT produces contact info, and stores it, temporarily
    bodyID_t* idGeometryA;
    bodyID_t* idGeometryB;

    // data pointers that is kT's transfer destination
    size_t* pDTOwnedBuffer_nContactPairs = NULL;
    bodyID_t* pDTOwnedBuffer_idGeometryA = NULL;
    bodyID_t* pDTOwnedBuffer_idGeometryB = NULL;
};

// typedef DEMDataDT* DEMDataDTPtr;
// typedef DEMSimParams* DEMSimParamsPtr;

}  // namespace sgps