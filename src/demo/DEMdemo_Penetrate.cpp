//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <core/utils/chpf/particle_writer.hpp>
#include <granular/ApiSystem.h>
#include <granular/HostSideHelpers.cpp>

#include <cstdio>
#include <time.h>

using namespace sgps;

int main() {
    DEMSolver DEM_sim;

    srand(759);

    // total number of random clump templates to generate
    int num_template = 6;

    int min_sphere = 1;
    int max_sphere = 5;

    float min_rad = 0.01;
    float max_rad = 0.02;

    float min_relpos = -0.01;
    float max_relpos = 0.01;

    auto mat_type_1 = DEM_sim.LoadMaterialType(1e8, 0.3, 0.2);

    // First create clump type 0 for representing the ground
    float ground_sp_r = 0.03;
    float ball_sp_r = 0.015;
    auto template_ground = DEM_sim.LoadClumpSimpleSphere(0.5, ground_sp_r, mat_type_1);
    auto template_ball = DEM_sim.LoadClumpSimpleSphere(0.5, ball_sp_r, mat_type_1);

    // generate ground clumps
    std::vector<unsigned int> input_template_num;
    std::vector<unsigned int> family_code;
    std::vector<float3> input_vel;
    auto input_xyz = DEMBoxGridSampler(make_float3(0, 0, 0), make_float3(ground_sp_r * 1, ground_sp_r * 1, 0.001),
                                       ground_sp_r * 1.2);
    // Mark family 1 as fixed
    family_code.insert(family_code.end(), input_xyz.size(), 1);
    input_template_num.insert(input_template_num.end(), input_xyz.size(), template_ground);
    input_vel.insert(input_vel.end(), input_xyz.size(), make_float3(0, 0, 0));

    // Add a ball rolling
    input_template_num.push_back(template_ball);
    input_xyz.push_back(make_float3(0, 0, ground_sp_r + ball_sp_r));
    family_code.push_back(0);
    input_vel.push_back(make_float3(0, 0, 0));

    DEM_sim.SetClumps(input_template_num, input_xyz);
    DEM_sim.SetClumpFamily(family_code);
    DEM_sim.SetClumpVels(input_vel);

    DEM_sim.InstructBoxDomainNumVoxel(21, 21, 22, 7.5e-11);

    DEM_sim.CenterCoordSys();
    DEM_sim.SetTimeStepSize(1e-5);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEM_sim.SetCDUpdateFreq(0);
    DEM_sim.SetExpandFactor(1.0);
    DEM_sim.Initialize();

    DEM_sim.UpdateSimParams();  // Not needed; just testing if this function works...

    for (int i = 0; i < 200; i++) {
        char filename[100];
        sprintf(filename, "./DEMdemo_collide_output_%04d.csv", i);
        DEM_sim.WriteFileAsSpheres(std::string(filename));
        std::cout << "Frame: " << i << std::endl;
        DEM_sim.LaunchThreads(3e-3);
    }

    std::cout << "DEMdemo_Penetrate exiting..." << std::endl;
    // TODO: add end-game report APIs
    return 0;
}
