//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <granular/GranularDefines.h>

namespace sgps {
// A structure for storing simulation parameters
struct DEMSimParams {
    // Number of voxels in the X direction, expressed as a power of 2
    unsigned char nvXp2;
    // Number of voxels in the Y direction, expressed as a power of 2
    unsigned char nvYp2;
    // Number of voxels in the Z direction, expressed as a power of 2
    unsigned char nvZp2;
    // Smallest length unit
    float l;
    // Double-precision single voxel size
    double voxelSize;
    // The edge length of a bin (for contact detection), as a multiple of voxelSize
    unsigned int binSize;
    // Number of clumps and spheres
    unsigned int nClumpBodies;
    unsigned int nSpheresGM;
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
};

// A struct that holds pointers to data arrays that dT uses
// For more details just look at PhysicsSystem.h
struct DEMDataDT {
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

    // dT-owned buffer pointers, for itself's usage
    bodyID_t* idGeometryA_buffer;
    bodyID_t* idGeometryB_buffer;

    // pointer to remote buffer where kinematic thread stores work-order data provided by the dynamic thread
    voxelID_t* pKTOwnedBuffer_voxelID = NULL;
    subVoxelPos_t* pKTOwnedBuffer_locX = NULL;
    subVoxelPos_t* pKTOwnedBuffer_locY = NULL;
    subVoxelPos_t* pKTOwnedBuffer_locZ = NULL;
    int* pKTOwnedBuffer_oriQ0 = NULL;
    int* pKTOwnedBuffer_oriQ1 = NULL;
    int* pKTOwnedBuffer_oriQ2 = NULL;
    int* pKTOwnedBuffer_oriQ3 = NULL;
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

    // kT produces contact info, and stores it, temporarily
    bodyID_t* idGeometryA;
    bodyID_t* idGeometryB;

    // data pointers that is kT's transfer destination
    bodyID_t* pDTOwnedBuffer_idGeometryA = NULL;
    bodyID_t* pDTOwnedBuffer_idGeometryB = NULL;
};

// typedef DEMDataDT* DEMDataDTPtr;
// typedef DEMSimParams* DEMSimParamsPtr;

}  // namespace sgps