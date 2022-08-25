//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <cstdio>
#include <chrono>
#include <filesystem>

using namespace sgps;
using namespace std::filesystem;

int main() {
    DEMSolver DEM_sim;
    DEM_sim.UseFrictionalHertzianModel();
    DEM_sim.SetVerbosity(STEP_STATS);
    DEM_sim.SetOutputFormat(DEM_OUTPUT_FORMAT::CSV);

    srand(42);

    // total number of random clump templates to generate
    int num_template = 6;

    int min_sphere = 1;
    int max_sphere = 5;

    float min_rad = 0.01;
    float max_rad = 0.02;

    float min_relpos = -0.01;
    float max_relpos = 0.01;

    auto mat_type_1 = DEM_sim.LoadMaterialType(1e8, 0.3, 0.3);
    auto mat_type_2 = DEM_sim.LoadMaterialType(1e9, 0.3, 0.7);

    // First create clump type 0 for representing the ground
    float ground_sp_r = 0.02;
    auto template_ground = DEM_sim.LoadClumpSimpleSphere(0.5, ground_sp_r, mat_type_1);

    // Make an array to store these generated clump templates
    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_types;

    // Then randomly create some clumps for piling up
    for (int i = 0; i < num_template; i++) {
        // first decide the number of spheres that live in this clump
        int num_sphere = rand() % (max_sphere - min_sphere + 1) + 1;

        // then allocate the clump template definition arrays (all in SI)
        float mass = 0.1 * (float)num_sphere;
        float3 MOI =
            make_float3(2e-5 * (float)num_sphere, 1.5e-5 * (float)num_sphere, 1.8e-5 * (float)num_sphere) * 50.;
        std::vector<float> radii;
        std::vector<float3> relPos;
        std::vector<std::shared_ptr<DEMMaterial>> mat;

        // randomly generate clump template configurations
        // the relPos of a sphere is always seeded from one of the already-generated sphere
        float3 seed_pos = make_float3(0);
        for (int j = 0; j < num_sphere; j++) {
            radii.push_back(((float)rand() / RAND_MAX) * (max_rad - min_rad) + min_rad);
            float3 tmp;
            if (j == 0) {
                tmp.x = 0;
                tmp.y = 0;
                tmp.z = 0;
            } else {
                tmp.x = ((float)rand() / RAND_MAX) * (max_relpos - min_relpos) + min_relpos;
                tmp.y = ((float)rand() / RAND_MAX) * (max_relpos - min_relpos) + min_relpos;
                tmp.z = ((float)rand() / RAND_MAX) * (max_relpos - min_relpos) + min_relpos;
            }
            tmp += seed_pos;
            relPos.push_back(tmp);
            mat.push_back(mat_type_1);

            // seed relPos from one of the previously generated spheres
            int choose_from = rand() % (j + 1);
            seed_pos = relPos.at(choose_from);
        }

        // LoadClumpType returns a shared_ptr that points to this template so you may modify it. Also, material can be
        // vector or a material shared ptr, and in the latter case it will just be applied to all component spheres this
        // clump has.
        auto clump_ptr = DEM_sim.LoadClumpType(mass, MOI, radii, relPos, mat_type_1);
        clump_types.push_back(clump_ptr);
    }

    // Generate ground clumps
    HCPSampler sampler1(ground_sp_r * 1.3);
    std::vector<std::shared_ptr<DEMClumpTemplate>> input_ground_clump_type;
    std::vector<unsigned int> family_code;
    auto input_ground_xyz = sampler1.SampleBox(make_float3(0, 0, -3.8), make_float3(5.0, 5.0, 0.001));
    family_code.insert(family_code.end(), input_ground_xyz.size(), 1);
    DEM_sim.DisableContactBetweenFamilies(1, 1);
    DEM_sim.SetFamilyFixed(1);
    input_ground_clump_type.insert(input_ground_clump_type.end(), input_ground_xyz.size(), template_ground);
    auto ground = DEM_sim.AddClumps(input_ground_clump_type, input_ground_xyz);
    ground->SetFamilies(family_code);

    // Generate initial clumps for piling
    HCPSampler sampler2(0.08);
    std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type;
    float3 sample_center = make_float3(0, 0, -1);
    float sample_halfheight = 2;
    float sample_halfwidth = 0.7;
    auto input_pile_xyz = sampler2.SampleCylinderZ(sample_center, sample_halfwidth, sample_halfheight);
    unsigned int num_clumps = input_pile_xyz.size();
    // Clear family_code for pile particles
    std::vector<unsigned int>().swap(family_code);
    // Casually select from generated clump types
    for (unsigned int i = 0; i < num_clumps; i++) {
        input_pile_template_type.push_back(clump_types.at(i % num_template));
        family_code.push_back(0);
    }
    // Calling AddClumps a second time will just add more clumps to the system
    auto the_pile = DEM_sim.AddClumps(input_pile_template_type, input_pile_xyz);
    the_pile->SetFamilies(family_code);

    DEM_sim.InstructBoxDomainNumVoxel(21, 21, 22, 7.5e-11);
    // DEM_sim.InstructBoxDomainNumVoxel(11, 11, 10, 1e-10);

    // Planes are all defaulted to fixed
    DEM_sim.AddBCPlane(make_float3(0, 4.5, 0), make_float3(0, -1, 0), mat_type_2);
    DEM_sim.AddBCPlane(make_float3(0, -4.5, 0), make_float3(0, 1, 0), mat_type_2);
    DEM_sim.AddBCPlane(make_float3(4.5, 0, 0), make_float3(-1, 0, 0), mat_type_2);
    std::shared_ptr<DEMExternObj> plane_a = DEM_sim.AddExternalObject();
    plane_a->AddPlane(make_float3(-4.5, 0, 0), make_float3(1, 0, 0), mat_type_2);

    DEM_sim.SetCoordSysOrigin("center");
    DEM_sim.SetInitTimeStep(5e-6);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEM_sim.SetCDUpdateFreq(20);
    // DEM_sim.SetExpandFactor(1e-3);
    DEM_sim.SetMaxVelocity(10.);
    DEM_sim.SetExpandSafetyParam(2.);
    DEM_sim.Initialize();

    DEM_sim.UpdateSimParams();  // Not needed; just testing if this function works...

    path out_dir = current_path();
    out_dir += "/DEMdemo_Pile";
    create_directory(out_dir);

    for (int i = 0; i < 200; i++) {
        char filename[100];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), i);
        DEM_sim.WriteSphereFile(std::string(filename));
        std::cout << "Frame: " << i << std::endl;
        // float KE = DEM_sim.GetTotalKineticEnergy();
        // std::cout << "Total kinetic energy: " << KE << std::endl;
        DEM_sim.DoDynamicsThenSync(5e-2);
    }
    DEM_sim.ShowTimingStats();
    DEM_sim.ClearTimingStats();

    std::cout << "DEMdemo_Pile exiting..." << std::endl;
    // TODO: add end-game report APIs
    return 0;
}
