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
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV | OUTPUT_CONTENT::VEL);
    // You can enforce owner wildcard output by the following call, or directly include OUTPUT_CONTENT::OWNER_WILDCARD
    // in SetOutputContent
    DEMSim.EnableOwnerWildcardOutput();

    // E, nu, CoR, mu, Crr...
    auto mat_type_cube = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.7}, {"mu", 0.5}, {"Crr", 0.0}});
    auto mat_type_granular = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.7}, {"mu", 0.5}, {"Crr", 0.0}});

    float granular_rad = 0.0025;
    auto template_granular = DEMSim.LoadSphereType(granular_rad * granular_rad * granular_rad * 2.6e3 * 4 / 3 * 3.14,
                                                   granular_rad, mat_type_granular);

    float step_size = 1e-6;
    const double world_size = 0.8;
    const float fill_height = world_size / 4.;
    const float chamber_bottom = -world_size / 2.;
    const float fill_bottom = chamber_bottom + granular_rad;

    DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size);
    DEMSim.InstructBoxDomainBoundingBC("all", mat_type_granular);
    DEMSim.SetCoordSysOrigin("center");

    // Now add a cylinderical boundary
    auto walls = DEMSim.AddExternalObject();
    walls->AddCylinder(make_float3(0), make_float3(0, 0, 1), world_size / 2., mat_type_cube, 0);

    auto cube = DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/cube.obj"), mat_type_cube);
    std::cout << "Total num of triangles: " << cube->GetNumTriangles() << std::endl;
    // Make the cube about 10cm by 2cm
    float cube_width = 0.1;
    float cube_height = 0.04;
    double cube_speed = 0.1;  // 0.01;
    cube->Scale(make_float3(cube_width, cube_width, cube_height));
    cube->SetFamily(10);
    DEMSim.SetFamilyFixed(10);
    DEMSim.SetFamilyPrescribedLinVel(11, "0", "0", to_string_with_precision(-cube_speed));
    // Track the cube
    auto cube_tracker = DEMSim.Track(cube);

    // Sampler to use
    HCPSampler sampler(2.002f * granular_rad);
    float3 fill_center = make_float3(0, 0, fill_bottom + fill_height / 2);
    const float fill_radius = world_size / 2. - 2. * granular_rad;
    auto input_xyz = sampler.SampleCylinderZ(fill_center, fill_radius, fill_height / 2);
    auto particles = DEMSim.AddClumps(template_granular, input_xyz);
    particles->SetFamily(1);
    // Initially, no contact between the brick and the granular material
    DEMSim.DisableContactBetweenFamilies(1, 10);

    // Use a owner wildcard to record tangential displacement compared to initial pos
    auto force_model = DEMSim.GetContactForceModel();
    force_model->SetPerOwnerWildcards({"gran_strain"});
    particles->AddOwnerWildcard("gran_strain", 0.0);
    // Or simply DEMSim.SetOwnerWildcards({"gran_strain"}); it does the job too

    unsigned int num_particles = input_xyz.size();
    std::cout << "Total num of particles: " << num_particles << std::endl;
    auto particle_tracker = DEMSim.Track(particles);

    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.SetCDUpdateFreq(30);
    DEMSim.SetInitBinSize(4 * granular_rad);
    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir += "/DemoOutput_Indentation";
    create_directory(out_dir);

    float sim_end = 0.5;     // 3.0;
    unsigned int fps = 100;  // 20;
    float frame_time = 1.0 / fps;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    // Keep tab of some sim quantities
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");

    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int currframe = 0;
    unsigned int curr_step = 0;

    // Settle
    DEMSim.DoDynamicsThenSync(0.5);
    // Record init positions of the particles
    std::vector<float3> particle_xyz(num_particles);
    std::vector<std::vector<bodyID_t>> particle_cnt_map;
    std::vector<std::vector<float3>> particle_init_relative_pos;
    for (unsigned int i = 0; i < num_particles; i++) {
        particle_xyz[i] = particle_tracker->Pos(i);
    }
    // Build contact map (contact partner owner IDs) for all particles
    for (unsigned int i = 0; i < num_particles; i++) {
        // Main particle location
        float3 main_loc = particle_xyz[i];
        // All the main particle's contact partners
        particle_cnt_map.push_back(particle_tracker->GetContactClumps(i));
        std::vector<float3> init_rel_pos;
        // Compute all this guy's partners' relative positions wrt to itself
        for (auto& ID : particle_cnt_map.back()) {
            init_rel_pos.push_back(DEMSim.GetOwnerPosition(ID) - main_loc);
        }
        particle_init_relative_pos.push_back(init_rel_pos);
    }

    // Ready to start indentation
    DEMSim.ChangeFamily(10, 11);
    double cube_zpos = max_z_finder->GetValue() + cube_height / 2;
    cube_tracker->SetPos(make_float3(0, 0, cube_zpos));
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (double t = 0; t < sim_end; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            // Compute relative displacement
            std::vector<float> gran_strain(num_particles);
            for (unsigned int i = 0; i < num_particles; i++) {
                float3 main_loc = particle_tracker->Pos(i);
                // Compute contact partners' new locations
                std::vector<float3> rel_pos;
                for (auto& ID : particle_cnt_map.at(i)) {
                    rel_pos.push_back(DEMSim.GetOwnerPosition(ID) - main_loc);
                }
                // How large is the strain?
                float3 strains = make_float3(0);
                for (int j = 0; j < particle_init_relative_pos.at(i).size(); j++) {
                    strains += particle_init_relative_pos.at(i).at(j) - rel_pos.at(j);
                }
                gran_strain[i] = length(strains) / particle_init_relative_pos.at(i).size();
            }
            // Feed displacement info to wildcard, then leverage the output method to output it to the file
            DEMSim.SetFamilyOwnerWildcardValue(1, "gran_strain", gran_strain);
            char filename[200], meshname[200];
            std::cout << "Outputting frame: " << currframe << std::endl;
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
            sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), currframe++);
            DEMSim.WriteSphereFile(std::string(filename));
            DEMSim.WriteMeshFile(std::string(meshname));
            DEMSim.ShowThreadCollaborationStats();
        }

        DEMSim.DoDynamics(step_size);
        // cube_zpos -= cube_speed * step_size;
        // cube_tracker->SetPos(make_float3(0, 0, cube_zpos));
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << (time_sec.count()) / sim_end / (1e-5 / step_size)
              << " seconds (wall time) to finish 1e5 steps' simulation" << std::endl;

    std::cout << "DEMdemo_Indentation exiting..." << std::endl;
    return 0;
}
