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
    voxelID_default_t* voxelID;

    subVoxelPos_default_t* locX;
    subVoxelPos_default_t* locY;
    subVoxelPos_default_t* locZ;

    int* oriQ0;
    int* oriQ1;
    int* oriQ2;
    int* oriQ3;

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
};

// A struct that holds pointers to data arrays that kT uses
// For more details just look at PhysicsSystem.h
struct DEMDataKT {
    voxelID_default_t* voxelID;

    subVoxelPos_default_t* locX;
    subVoxelPos_default_t* locY;
    subVoxelPos_default_t* locZ;

    int* oriQ0;
    int* oriQ1;
    int* oriQ2;
    int* oriQ3;
};

// typedef DEMDataDT* DEMDataDTPtr;
// typedef DEMSimParams* DEMSimParamsPtr;

}  // namespace sgps