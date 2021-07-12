//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <granular/ApiSystem.h>

#include <cstdio>

using namespace sgps;

int main() {
    SGPS aa(1.f);

    std::vector<float> a_vec(3, 1);
    std::vector<unsigned char> b_vec(2, 1);
    std::vector<unsigned char> c_vec(3, 1);

    // aa.LoadClumpType(1, 1, 1, 1, a_vec, a_vec, a_vec, a_vec, c_vec);
    aa.LoadClumpSimpleSphere(1, 1, 1);

    aa.LaunchThreads();

    std::cout << aa.GetClumpVoxelID(0) << std::endl;

    return 0;
}
