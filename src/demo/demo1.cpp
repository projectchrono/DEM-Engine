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

    std::vector<float> radii_a_vec(3, .05);
    std::vector<float> radii_b_vec(3, .1);

    std::vector<float3> pos_a_vec(3, make_float3(.1, .12, .08));
    std::vector<float3> pos_b_vec(3, make_float3(.1, .05, .06));
    std::vector<float3> pos_c_vec(3, make_float3(.01, .1, .14));
    std::vector<float3> pos_d_vec;
    pos_d_vec.push_back(make_float3(.1, .1, .07));
    pos_d_vec.push_back(make_float3(.12, .1, .07));
    pos_d_vec.push_back(make_float3(.12, .1, .14));

    std::vector<unsigned int> mat_vec(3, 0);

    aa.LoadMaterialType(1, 10);

    auto type1 = aa.LoadClumpType(1, make_float3(1), radii_a_vec, pos_a_vec, mat_vec);
    auto type2 = aa.LoadClumpType(2, make_float3(2), radii_b_vec, pos_b_vec, mat_vec);
    auto type3 = aa.LoadClumpType(2, make_float3(3), radii_b_vec, pos_d_vec, mat_vec);
    auto type4 = aa.LoadClumpSimpleSphere(3, .15, 0);

    std::vector<unsigned int> input_types;
    std::vector<float3> input_xyz;

    unsigned int total_types = 4;
    unsigned int total_clumps = 6;
    for (unsigned int i = 0; i < total_clumps; i++) {
        input_types.push_back(i / total_types);
        input_xyz.push_back(
            make_float3((float)rand() / (RAND_MAX), (float)rand() / (RAND_MAX), (float)rand() / (RAND_MAX)));
    }
    aa.SetClumps(input_types, input_xyz);

    aa.Initialize();

    aa.LaunchThreads();

    std::cout << aa.GetClumpVoxelID(0) << std::endl;

    return 0;
}
