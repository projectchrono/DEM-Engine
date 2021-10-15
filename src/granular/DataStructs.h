//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

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
    unsigned int* voxelID;

    unsigned int* locX;
    unsigned int* locY;
    unsigned int* locZ;

    int* oriQ0;
    int* oriQ1;
    int* oriQ2;
    int* oriQ3;

    int* hvX;
    int* hvY;
    int* hvZ;

    int* hOmgBarX;
    int* hOmgBarY;
    int* hOmgBarZ;

    int* h2aX;
    int* h2aY;
    int* h2aZ;

    int* h2AlphaX;
    int* h2AlphaY;
    int* h2AlphaZ;
};

// typedef DEMDataDT* DEMDataDTPtr;
// typedef DEMSimParams* DEMSimParamsPtr;

}  // namespace sgps