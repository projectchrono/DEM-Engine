//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <core/utils/chpf/particle_writer.hpp>
#include <DEM/ApiSystem.h>
#include <DEM/HostSideHelpers.hpp>

#include <cstdio>
#include <time.h>
#include <filesystem>

using namespace sgps;
using namespace std::filesystem;
const double PI = 3.141592653589793;

void SphereRollUpIncline(DEMSolver& DEM_sim) {
    // Clump initial profile
    std::vector<float3> input_xyz;
    std::vector<std::shared_ptr<DEMClumpTemplate>> input_template_type;
    std::vector<unsigned int> family_code;
    std::vector<float3> input_vel;

    auto mat_type_1 = DEM_sim.LoadMaterialType(1e7, 0.3, 0.5, 0.5, 0.3);
    // A ball
    float sphere_rad = 0.2;
    float mass = 5.0;
    auto sphere_template = DEM_sim.LoadClumpSimpleSphere(mass, sphere_rad, mat_type_1);

    // Incline angle
    float alpha = 35.;
    // Add the incline
    float3 normal_dir = make_float3(-std::sin(2. * PI * (alpha / 360.)), 0., std::cos(2. * PI * (alpha / 360.)));
    float3 tang_dir = make_float3(std::cos(2. * PI * (alpha / 360.)), 0., std::sin(2. * PI * (alpha / 360.)));
    DEM_sim.AddBCPlane(make_float3(0, 0, 0), normal_dir, mat_type_1);

    // Add a ball rolling
    auto sphere = DEM_sim.AddClumpTracked(sphere_template, normal_dir * sphere_rad, tang_dir * 0.5);
    // DEM_sim.SetClumpFamilies(family_code);

    DEM_sim.Initialize();

    float frame_time = 1e-5;
    path out_dir = current_path();
    out_dir += "/DEMdemo_TestPack";
    create_directory(out_dir);
    for (int i = 0; i < 0.15 / frame_time; i++) {
        char filename[100];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), i);
        // DEM_sim.WriteClumpFile(std::string(filename));
        std::cout << "Frame: " << i << std::endl;
        float3 vel = sphere->Vel();
        float3 angVel = sphere->AngVel();
        std::cout << "Time: " << frame_time * i << std::endl;
        std::cout << "Velocity of the sphere: " << vel.x << ", " << vel.y << ", " << vel.z << std::endl;
        std::cout << "Angular velocity of the sphere: " << angVel.x << ", " << angVel.y << ", " << angVel.z
                  << std::endl;

        DEM_sim.DoStepDynamics(frame_time);
    }
}

int main() {
    DEMSolver DEM_sim;
    DEM_sim.SetVerbosity(INFO);
    DEM_sim.SetOutputFormat(DEM_OUTPUT_FORMAT::CSV);

    DEM_sim.InstructBoxDomainNumVoxel(22, 22, 20, 7.5e-11);
    float step_size = 1e-5;
    DEM_sim.CenterCoordSys();
    DEM_sim.SetTimeStepSize(step_size);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    DEM_sim.SetCDUpdateFreq(0);

    // An ellipsoid
    std::vector<float> radii = {0.2, 0.176, 0.128, 0.176, 0.128};
    std::vector<float3> relPos = {make_float3(0, 0, 0), make_float3(0, 0, 0.172), make_float3(0, 0, 0.288),
                                  make_float3(0, 0, -0.172), make_float3(0, 0, -0.288)};
    // Then calculate mass and MOI
    float mass = 5.0;
    auto mat_type_1 = DEM_sim.LoadMaterialType(1e8, 0.3, 0.5, 0.5, 0.3);
    float3 MOI = make_float3(1. / 5. * mass * (0.2 * 0.2 + 0.4 * 0.4), 1. / 5. * mass * (0.2 * 0.2 + 0.4 * 0.4),
                             1. / 5. * mass * (0.2 * 0.2 + 0.2 * 0.2));
    auto ellipsoid_template = DEM_sim.LoadClumpType(mass, MOI, radii, relPos, mat_type_1);

    // Validation tests
    SphereRollUpIncline(DEM_sim);

    std::cout << "DEMdemo_TestPack exiting..." << std::endl;
    // TODO: add end-game report APIs

    return 0;
}
