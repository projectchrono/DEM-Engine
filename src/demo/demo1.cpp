//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <granular/ApiSystem.h>
#include <core/utils/chpf/particle_writer.hpp>

#include <cstdio>

using namespace sgps;

int main() {
    SGPS aa(1.f);

    float3 relative_pos = make_float3(1, 0, 0);

    std::vector<float> radii_a_vec(3, 1);
    std::vector<float> radii_b_vec(3, 2);

    std::vector<float3> pos_a_vec(3, make_float3(6, 7, 5));
    std::vector<float3> pos_b_vec(3, make_float3(6, 3, 4));
    std::vector<float3> pos_c_vec(3, make_float3(1, 3, 4));

    std::vector<unsigned int> mat_vec(3, 0);

    aa.LoadMaterialType(1, 10);

    aa.LoadClumpType(1, make_float3(1), radii_a_vec, pos_a_vec, mat_vec);
    aa.LoadClumpType(2, make_float3(2), radii_b_vec, pos_b_vec, mat_vec);
    aa.LoadClumpType(2, make_float3(3), radii_b_vec, pos_c_vec, mat_vec);
    aa.LoadClumpSimpleSphere(3, 3, 0);

    aa.Initialize();

    aa.LaunchThreads();

    std::cout << aa.GetClumpVoxelID(0) << std::endl;

    return 0;
}
