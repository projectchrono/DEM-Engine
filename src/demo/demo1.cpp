//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <granular/ApiSystem.h>

#include <cstdio>

using namespace sgps;

int main() {
    SGPS_api aa(1.f);
    printf("Constructed!\n");

    aa.LaunchThreads();

    std::cout << aa.GetClumpVoxelID(0) << std::endl;

    return 0;
}
