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

using namespace deme;
using namespace std::filesystem;

int main() {
    DEMSolver DEMSim;
    DEMSim.UseFrictionalHertzianModel();
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);

    srand(42);

    // Scale factor
    float scaling = 2;

    // total number of random clump templates to generate
    int num_template = 6;

    int min_sphere = 1;
    int max_sphere = 5;

    float min_rad = 0.01 * scaling;
    float max_rad = 0.02 * scaling;

    float min_relpos = -0.01 * scaling;
    float max_relpos = 0.01 * scaling;

    auto mat_type_walls = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 1}});
    auto mat_type_particles = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.7}, {"mu", 1}});

    /*
    // First create clump type 0 for representing the ground
    float ground_sp_r = 0.02;
    auto template_ground = DEMSim.LoadSphereType(0.5, ground_sp_r, mat_type_walls);
    */

    // Loaded meshes are by-default fixed
    auto funnel = DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/funnel.obj"), mat_type_walls);
    funnel->Scale(0.15);
    float funnel_bottom = 0.f;

    // Make an array to store these generated clump templates
    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_types;

    // Then randomly create some clumps for piling up
    for (int i = 0; i < num_template; i++) {
        // first decide the number of spheres that live in this clump
        int num_sphere = rand() % (max_sphere - min_sphere + 1) + 1;

        // then allocate the clump template definition arrays (all in SI)
        float mass = 0.1 * (float)num_sphere * std::pow(scaling, 3);
        float3 MOI = make_float3(2e-5 * (float)num_sphere, 1.5e-5 * (float)num_sphere, 1.8e-5 * (float)num_sphere) *
                     50. * std::pow(scaling, 5);
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
            mat.push_back(mat_type_walls);

            // seed relPos from one of the previously generated spheres
            int choose_from = rand() % (j + 1);
            seed_pos = relPos.at(choose_from);
        }

        // LoadClumpType returns a shared_ptr that points to this template so you may modify it. Also, material can be
        // vector or a material shared ptr, and in the latter case it will just be applied to all component spheres this
        // clump has.
        auto clump_ptr = DEMSim.LoadClumpType(mass, MOI, radii, relPos, mat_type_walls);
        clump_types.push_back(clump_ptr);
    }

    /*
    HCPSampler sampler1(ground_sp_r * 1.3);
    std::vector<std::shared_ptr<DEMClumpTemplate>> input_ground_clump_type;
    std::vector<unsigned int> family_code;
    auto input_ground_xyz = sampler1.SampleBox(make_float3(0, 0, -3.8), make_float3(5.0, 5.0, 0.001));
    family_code.insert(family_code.end(), input_ground_xyz.size(), 1);
    DEMSim.DisableContactBetweenFamilies(1, 1);
    input_ground_clump_type.insert(input_ground_clump_type.end(), input_ground_xyz.size(), template_ground);
    auto ground = DEMSim.AddClumps(input_ground_clump_type, input_ground_xyz);
    ground->SetFamilies(family_code);
    */

    // Generate initial clumps for piling
    float spacing = 0.08 * scaling;
    float fill_width = 5.f;
    float fill_height = 2.f * fill_width;
    float fill_bottom = funnel_bottom + fill_width + spacing;
    HCPSampler sampler(spacing);
    std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type;
    float3 sample_center = make_float3(0, 0, fill_bottom + fill_height / 2);
    auto input_pile_xyz = sampler.SampleCylinderZ(sample_center, fill_width, fill_height / 2);
    unsigned int num_clumps = input_pile_xyz.size();
    // Casually select from generated clump types
    for (unsigned int i = 0; i < num_clumps; i++) {
        input_pile_template_type.push_back(clump_types.at(i % num_template));
    }
    // Calling AddClumps a second time will just add more clumps to the system
    auto the_pile = DEMSim.AddClumps(input_pile_template_type, input_pile_xyz);

    // DEMSim.InstructBoxDomainNumVoxel(21, 21, 22, 7.5e-11);
    DEMSim.InstructBoxDomainDimension(20, 20, 30);
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_walls);

    DEMSim.SetCoordSysOrigin(make_float3(10, 10, -(funnel_bottom - 10.f)));
    DEMSim.SetInitTimeStep(5e-6);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEMSim.SetCDUpdateFreq(40);
    // DEMSim.SetExpandFactor(1e-3);
    DEMSim.SetMaxVelocity(25.);
    DEMSim.SetExpandSafetyParam(2.);
    DEMSim.SetInitBinSize(min_rad * 6);
    DEMSim.Initialize();

    DEMSim.UpdateSimParams();  // Not needed; just testing if this function works...

    path out_dir = current_path();
    out_dir += "/DemoOutput_Repose";
    create_directory(out_dir);

    for (int i = 0; i < 300; i++) {
        char filename[200], meshfile[200];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), i);
        sprintf(meshfile, "%s/DEMdemo_funnel_%04d.vtk", out_dir.c_str(), i);
        DEMSim.WriteSphereFile(std::string(filename));
        DEMSim.WriteMeshFile(std::string(meshfile));
        std::cout << "Frame: " << i << std::endl;
        DEMSim.DoDynamics(1e-1);
        DEMSim.ShowThreadCollaborationStats();
    }
    DEMSim.ShowTimingStats();
    DEMSim.ClearTimingStats();

    std::cout << "DEMdemo_Repose exiting..." << std::endl;
    return 0;
}
