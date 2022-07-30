//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <filesystem>
#include <cstdio>
#include <time.h>
#include <filesystem>

using namespace sgps;
using namespace std::filesystem;

int main() {
    DEMSolver DEM_sim;
    DEM_sim.SetVerbosity(DEBUG);

    // srand(time(NULL));
    srand(4150);

    auto mat_type_1 = DEM_sim.LoadMaterialType(1e9, 0.3, 0.8);
    auto sph_type_1 = DEM_sim.LoadClumpSimpleSphere(11728., 1., mat_type_1);

    std::vector<float3> input_xyz;
    std::vector<float3> input_vel;
    std::vector<std::shared_ptr<DEMClumpTemplate>> input_clump_type(2, sph_type_1);
    // std::vector<unsigned int> input_clump_type(1, sph_type_1);

    // Inputs are just 2 sphere
    float sphPos = 1.2f;
    input_xyz.push_back(make_float3(-sphPos, 0, 0));
    input_xyz.push_back(make_float3(sphPos, 0, 0));
    input_vel.push_back(make_float3(1.f, 0, 0));
    input_vel.push_back(make_float3(-1.f, 0, 0));

    auto particles = DEM_sim.AddClumps(input_clump_type, input_xyz);
    particles->SetVel(input_vel);

    DEM_sim.AddBCPlane(make_float3(0, 0, -1.25), make_float3(0, 0, 1), mat_type_1);

    DEM_sim.InstructBoxDomainNumVoxel(22, 21, 21, 3e-11);

    DEM_sim.SetCoordSysOrigin("center");
    DEM_sim.SetInitTimeStep(2e-5);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // Velocity not to exceed 3.0
    DEM_sim.SetCDUpdateFreq(10);
    DEM_sim.SetMaxVelocity(3.);
    DEM_sim.SetExpandSafetyParam(2.);

    DEM_sim.Initialize();

    path out_dir = current_path();
    out_dir += "/DEMdemo_SingleSphereCollide";
    create_directory(out_dir);
    for (int i = 0; i < 100; i++) {
        std::cout << "Frame: " << i << std::endl;

        char filename[100];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), i);
        DEM_sim.WriteClumpFile(std::string(filename));

        DEM_sim.DoDynamicsThenSync(1e-2);
    }

    std::cout << "DEMdemo_SingleSphereCollide exiting..." << std::endl;
    return 0;
}
