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
    std::vector<float> b_vec(3, 2);
    std::vector<unsigned int> c_vec(3, 0);

    aa.LoadClumpType(1, 1, 1, 1, a_vec, a_vec, a_vec, a_vec, c_vec);
    aa.LoadClumpType(2, 2, 2, 2, b_vec, a_vec, b_vec, a_vec, c_vec);
    aa.LoadClumpSimpleSphere(3, 3, 0);

    aa.LoadMaterialType(1, 10);

    aa.Initialize();

    aa.LaunchThreads();

    std::cout << aa.GetClumpVoxelID(0) << std::endl;

    return 0;
}
