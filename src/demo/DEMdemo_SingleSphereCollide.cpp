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

    // srand(time(NULL));
    srand(4150);

    auto mat_type_1 = DEM_sim.LoadMaterialType(1e9, 0.3, 0.8);
    auto sph_type_1 = DEM_sim.LoadClumpSimpleSphere(11728., 1., mat_type_1);

    std::vector<float3> input_xyz;
    std::vector<float3> input_vel;
    std::vector<unsigned int> input_clump_type(2, sph_type_1);

    // Inputs are just 2 spheres
    float sphPos = 1.2f;
    input_xyz.push_back(make_float3(-sphPos, 0, 0));
    input_xyz.push_back(make_float3(sphPos, 0, 0));
    DEM_sim.SetClumps(input_clump_type, input_xyz);

    input_vel.push_back(make_float3(1.f, 0, 0));
    input_vel.push_back(make_float3(-1.f, 0, 0));
    DEM_sim.SetClumpVels(input_vel);

    DEM_sim.InstructBoxDomainNumVoxel(22, 21, 21, 3e-11);

    DEM_sim.CenterCoordSys();
    DEM_sim.SetTimeStepSize(1e-4);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, 0));
    DEM_sim.SetCDUpdateFreq(0);

    DEM_sim.Initialize();

    for (int i = 0; i < 100; i++) {
        std::cout << "Frame: " << i << std::endl;

        char filename[100];
        sprintf(filename, "./DEMdemo_collide_output_%04d.csv", i);
        DEM_sim.WriteFileAsSpheres(std::string(filename));

        DEM_sim.LaunchThreads(1e-2);
    }

    std::cout << "DEMdemo_SingleSphereCollide exiting..." << std::endl;
    return 0;
}
