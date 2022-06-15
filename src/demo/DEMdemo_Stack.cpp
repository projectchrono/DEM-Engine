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

int main() {
    DEMSolver DEM_sim;
    DEM_sim.UseFrictionlessHertzianModel();
    DEM_sim.SetVerbosity(DEBUG);

    srand(759);

    auto mat_type_1 = DEM_sim.LoadMaterialType(1e8, 0.3, 0.2);

    float ground_sp_r = 0.03;
    float ball_sp_r = 0.015;
    float3 Orig = make_float3(0, 0, 0);

    // The ground is a big clump that has many sphere components
    auto ground_comp_xyz =
        DEMBoxGridSampler(Orig, make_float3(ground_sp_r * 50, ground_sp_r * 50, 0.001), ground_sp_r * 1.2);
    unsigned int num_ground_comp = ground_comp_xyz.size();
    std::cout << "The ground has " << num_ground_comp << " spheres" << std::endl;
    auto template_ground = DEM_sim.LoadClumpType(
        1000.0, make_float3(10, 10, 10), std::vector<float>(num_ground_comp, ground_sp_r), ground_comp_xyz, mat_type_1);

    // A ball sits on the ground...
    auto template_ball = DEM_sim.LoadClumpSimpleSphere(0.5, ball_sp_r, mat_type_1);

    // Clump initial profile
    std::vector<float3> input_xyz;
    std::vector<std::shared_ptr<DEMClumpTemplate>> input_template_type;
    std::vector<unsigned int> family_code;
    std::vector<float3> input_vel;

    // Mark family 1 ground
    family_code.push_back(1);
    input_xyz.push_back(Orig);
    input_template_type.push_back(template_ground);
    input_vel.push_back(make_float3(0, 0, 0));

    // Add a ball rolling
    input_template_type.push_back(template_ball);
    input_xyz.push_back(make_float3(0, 0, ground_sp_r + ball_sp_r));
    family_code.push_back(0);
    input_vel.push_back(make_float3(0, 0, 0));

    DEM_sim.AddClumps(input_template_type, input_xyz, input_vel);
    DEM_sim.SetClumpFamilies(family_code);
    // Family 1 (ground) particles have no contact among each other, and fixed
    DEM_sim.DisableContactBetweenFamilies(1, 1);
    DEM_sim.SetFamilyFixed(1);

    DEM_sim.InstructBoxDomainNumVoxel(22, 22, 20, 7.5e-11);

    float step_size = 1e-5;
    DEM_sim.CenterCoordSys();
    DEM_sim.SetTimeStepSize(step_size);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    DEM_sim.SetCDUpdateFreq(0);
    // DEM_sim.SetCDUpdateFreq(10);
    // DEM_sim.SuggestExpandFactor(3.);
    // DEM_sim.SuggestExpandSafetyParam(2.);

    DEM_sim.Initialize();

    path out_dir = current_path();
    out_dir += "/DEMdemo_Stack";
    create_directory(out_dir);
    for (int i = 0; i < 100; i++) {
        char filename[100];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), i);
        DEM_sim.WriteClumpFile(std::string(filename));
        std::cout << "Frame: " << i << std::endl;
        // Seems that after user call, both thread are chilling, not responding; have to resolve it...
        // float KE = DEM_sim.GetTotalKineticEnergy();
        // std::cout << "Total kinetic energy: " << KE << std::endl;
        DEM_sim.DoStepDynamicsSync(3e-3);
    }

    std::cout << "DEMdemo_Stack exiting..." << std::endl;
    // TODO: add end-game report APIs

    return 0;
}
