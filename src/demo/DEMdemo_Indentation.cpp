//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// This demo tries to show the strain distribution in the granular material when
// affected by a compressor.
// =============================================================================

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

inline void buildContactMap(std::vector<std::vector<bodyID_t>>& map,
                            std::vector<std::vector<float3>>& relative_pos,
                            const DEMSolver& DEMSim,
                            std::shared_ptr<DEMTracker>& particle_tracker,
                            unsigned int num_particles) {
    // You could easily get the clump contact pairs' A and B IDs, but we go the long way to show some APIs
    // auto cnt_pairs = DEMSim.GetClumpContacts();
    // Now you could also get the full contact pair information. We also use DEME_TINY_FLOAT to exclude contacts that
    // are not producing any force, thus being only a potential contact (preemptively detected).
    std::shared_ptr<ContactInfoContainer> cnt_pairs_container = DEMSim.GetContactDetailedInfo(DEME_TINY_FLOAT);
    std::vector<std::string>& cnt_type = cnt_pairs_container->GetContactType();
    std::vector<bodyID_t>& owner_A = cnt_pairs_container->GetAOwner();
    std::vector<bodyID_t>& owner_B = cnt_pairs_container->GetBOwner();

    map.clear();
    relative_pos.clear();
    map.resize(num_particles);
    relative_pos.resize(num_particles);
    // At system-level, the clump's ID may not start from 0 (although they are consecutive), so we can get the actual
    // starting ID as the offset for later use
    size_t clump_ID_offset = particle_tracker->GetOwnerID();
    // Use Positions to get all particle locations in bulk (rather than using inefficient piecemeal Pos() method)
    std::vector<float3> particle_xyz = particle_tracker->Positions();
    assert(particle_xyz.size() == num_particles);

    for (unsigned int i = 0; i < cnt_type.size(); i++) {
        if (cnt_type[i] != "SS") {
            // We only care about SS contacts, so skip the rest
            continue;
        }
        // Here, what we store is the ID of contact partners but starting from 0 (rather than whatever the system
        // assigns them), so we subtract the offset
        map[owner_A[i] - clump_ID_offset].push_back(owner_B[i] - clump_ID_offset);
        map[owner_B[i] - clump_ID_offset].push_back(owner_A[i] - clump_ID_offset);
    }
    for (unsigned int i = 0; i < num_particles; i++) {
        // Main particle location
        float3 main_loc = particle_xyz[i];
        std::vector<float3> init_rel_pos;
        // Compute all this guy's partners' relative positions wrt to itself
        // The purpose of that we store 0-based partner ID is clear now: We can directly use pre-filled particle_xyz,
        // and it's efficient
        for (const auto& ID : map[i]) {
            init_rel_pos.push_back(particle_xyz[ID] - main_loc);
        }
        relative_pos[i] = init_rel_pos;
    }
}

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
    // We will query the contact info later, so at least we need owners
    DEMSim.SetContactOutputContent({"OWNER"});
    // You can enforce owner wildcard output by the following call, or directly include OUTPUT_CONTENT::OWNER_WILDCARD
    // in SetOutputContent
    DEMSim.EnableOwnerWildcardOutput();

    path out_dir = current_path();
    out_dir /= "DemoOutput_Indentation";
    create_directory(out_dir);

    // E, nu, CoR, mu, Crr...
    auto mat_type_cube = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.8}, {"mu", 0.4}, {"Crr", 0.0}});
    auto mat_type_granular_1 = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.8}, {"mu", 0.3}, {"Crr", 0.0}});
    auto mat_type_granular_2 = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.8}, {"mu", 0.4}, {"Crr", 0.0}});
    // CoR is a pair-wise property, so it should be mentioned here
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_cube, mat_type_granular_1, 0.8);
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_cube, mat_type_granular_2, 0.8);

    float granular_rad = 0.001;  // 0.002;
    auto template_granular = DEMSim.LoadSphereType(granular_rad * granular_rad * granular_rad * 2.6e3 * 4 / 3 * 3.14,
                                                   granular_rad, mat_type_granular_1);

    float step_size = 1e-6;
    const double world_size = 0.6;
    const float fill_height = 0.3;
    const float chamber_bottom = -world_size / 2.;
    const float fill_bottom = chamber_bottom + granular_rad;

    DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size);
    DEMSim.InstructBoxDomainBoundingBC("all", mat_type_granular_2);

    // Now add a cylinderical boundary
    auto walls = DEMSim.AddExternalObject();
    walls->AddCylinder(make_float3(0), make_float3(0, 0, 1), world_size / 2., mat_type_cube, 0);

    auto cube = DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/cube.obj"), mat_type_cube);
    std::cout << "Total num of triangles: " << cube->GetNumTriangles() << std::endl;
    // Make the cube about 10cm by 2cm
    float cube_width = 0.1;
    float cube_height = 0.04;
    double cube_speed = 0.25;  // 0.1 and 0.02, try them too... very similar though
    cube->Scale(make_float3(cube_width, cube_width, cube_height));
    cube->SetFamily(10);
    DEMSim.SetFamilyFixed(10);
    DEMSim.SetFamilyPrescribedLinVel(11, "0", "0", to_string_with_precision(-cube_speed));
    // Track the cube
    auto cube_tracker = DEMSim.Track(cube);

    // Sampler to use
    const float spacing = 2.05f * granular_rad;
    const float fill_radius = world_size / 2. - 2. * granular_rad;

    PDSampler sampler(spacing);
    std::vector<float3> input_xyz;
    float layer_z = 0;
    while (layer_z < fill_height) {
        float3 sample_center = make_float3(0, 0, fill_bottom + layer_z + spacing / 2);
        auto layer_xyz = sampler.SampleCylinderZ(sample_center, fill_radius, 0);
        input_xyz.insert(input_xyz.end(), layer_xyz.begin(), layer_xyz.end());
        layer_z += spacing;
    }

    // HCPSampler sampler(spacing);
    // float3 fill_center = make_float3(0, 0, fill_bottom + fill_height / 2);
    // auto input_xyz = sampler.SampleCylinderZ(fill_center, fill_radius, fill_height / 2);

    // Note: AddClumps can be called multiple times before initialization to add more clumps to the system
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
    DEMSim.SetCDUpdateFreq(20);
    // You usually don't have to worry about initial bin size. But sometimes if you can set the init bin size so that
    // the kT--dT work at a sweet collaboration pattern, it could make the solver run faster.
    DEMSim.SetInitBinNumTarget(5e7);
    DEMSim.Initialize();

    float sim_end = cube_height * 1.5 / cube_speed;  // 3.0;
    unsigned int fps = 200;                          // 20;
    float frame_time = 1.0 / fps;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    // Keep tab of some sim quantities
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");

    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int currframe = 0;
    unsigned int curr_step = 0;

    // Settling phase, no output
    for (double t = 0; t < 0.3; t += frame_time) {
        DEMSim.ShowThreadCollaborationStats();
        DEMSim.DoDynamicsThenSync(frame_time);
    }
    double init_max_z = max_z_finder->GetValue();
    std::cout << "After settling, max particle Z coord is " << init_max_z << std::endl;

    // Record init positions of the particles
    std::vector<std::vector<bodyID_t>> particle_cnt_map;  // bodyID_t is just unsigned int
    std::vector<std::vector<float3>> particle_init_relative_pos;
    // Build contact map (contact partner owner IDs) for all particles
    buildContactMap(particle_cnt_map, particle_init_relative_pos, DEMSim, particle_tracker, num_particles);

    // Ready to start indentation
    std::cout << "Simulation starts..." << std::endl;
    // Let the brick sink with a downward velocity.
    DEMSim.ChangeFamily(10, 11);

    // This is meant to show that you can change the material type of the clumps in mid-simulation.
    // Doing this, we change the mu between particles from 0.3 (lower, for getting something denser
    // after settling) to 0.4 (the value we use for the main simulation).
    DEMSim.SetFamilyClumpMaterial(1, mat_type_granular_2);

    double cube_zpos = max_z_finder->GetValue() + cube_height / 2;
    cube_tracker->SetPos(make_float3(0, 0, cube_zpos));
    std::cout << "Initially the cube is at Z = " << cube_zpos << std::endl;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (double t = 0; t < sim_end; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            // Compute relative displacement
            std::vector<float> gran_strain(num_particles);
            // Pre-fill all particle locations
            std::vector<float3> particle_xyz = particle_tracker->Positions();
            for (unsigned int i = 0; i < num_particles; i++) {
                float3 main_loc = particle_xyz[i];
                // Compute contact partners' new locations, using pre-filled xyz
                std::vector<float3> rel_pos;
                for (auto& ID : particle_cnt_map.at(i)) {
                    rel_pos.push_back(particle_xyz[ID] - main_loc);
                }
                // How large is the strain?
                // float3 strains = make_float3(0);
                float strains = 0.;
                int num_neighbors = particle_init_relative_pos.at(i).size();
                for (int j = 0; j < num_neighbors; j++) {
                    // strains += particle_init_relative_pos.at(i).at(j) - rel_pos.at(j);
                    strains += length(particle_init_relative_pos.at(i).at(j) - rel_pos.at(j));
                }
                gran_strain[i] = (num_neighbors > 0) ? (strains / num_neighbors) : 0.0;
            }
            // Re-build contact map, for the next output step
            buildContactMap(particle_cnt_map, particle_init_relative_pos, DEMSim, particle_tracker, num_particles);
            std::cout << "A new contact map constructed..." << std::endl;

            // Feed displacement info to wildcard, then leverage the output method to output it to the file
            DEMSim.SetFamilyOwnerWildcardValue(1, "gran_strain", gran_strain);
            char filename[100], meshname[100];
            std::cout << "Outputting frame: " << currframe << std::endl;
            sprintf(filename, "DEMdemo_output_%04d.csv", currframe);
            sprintf(meshname, "DEMdemo_mesh_%04d.vtk", currframe++);
            DEMSim.WriteSphereFile(out_dir / filename);
            DEMSim.WriteMeshFile(out_dir / meshname);
            DEMSim.ShowThreadCollaborationStats();
        }

        DEMSim.DoDynamics(step_size);
        // cube_zpos -= cube_speed * step_size;
        // cube_tracker->SetPos(make_float3(0, 0, cube_zpos));
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds (wall time) to finish the simulation" << std::endl;

    DEMSim.ShowTimingStats();

    std::cout << "----------------------------------------" << std::endl;
    DEMSim.ShowMemStats();
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "DEMdemo_Indentation exiting..." << std::endl;
    return 0;
}
