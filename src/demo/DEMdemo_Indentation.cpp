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
    auto cnt_pairs = DEMSim.GetClumpContacts();
    map.clear();
    relative_pos.clear();
    map.resize(num_particles);
    relative_pos.resize(num_particles);
    std::vector<float3> particle_xyz(num_particles);
    // At system-level, the clump's ID may not start from 0; but a batch of clumps loaded together have consecutive IDs.
    size_t clump_ID_offset = particle_tracker->GetOwnerID();
    for (unsigned int i = 0; i < num_particles; i++) {
        particle_xyz[i] = particle_tracker->Pos(i);
    }
    for (unsigned int i = 0; i < cnt_pairs.size(); i++) {
        const auto& pair = cnt_pairs.at(i);
        map[pair.first - clump_ID_offset].push_back(pair.second);
        map[pair.second - clump_ID_offset].push_back(pair.first);
    }
    for (unsigned int i = 0; i < num_particles; i++) {
        // Main particle location
        float3 main_loc = particle_xyz[i];
        std::vector<float3> init_rel_pos;
        // Compute all this guy's partners' relative positions wrt to itself
        for (const auto& ID : map[i]) {
            init_rel_pos.push_back(DEMSim.GetOwnerPosition(ID) - main_loc);
        }
        relative_pos[i] = init_rel_pos;
    }
}

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::VEL | OUTPUT_CONTENT::FAMILY);
    // You can enforce owner wildcard output by the following call, or directly include OUTPUT_CONTENT::OWNER_WILDCARD
    // in SetOutputContent
    DEMSim.EnableOwnerWildcardOutput();

    path out_dir = current_path();
    out_dir += "/DemoOutput_Indentation";
    create_directory(out_dir);

    // E, nu, CoR, mu, Crr...
    auto mat_type_cube = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.4}, {"mu", 0.4}, {"Crr", 0.04}});
    auto mat_type_granular_1 = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.4}, {"mu", 0.3}, {"Crr", 0.02}});
    auto mat_type_granular_2 =
        DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.4}, {"mu", 0.8}, {"Crr", 0.08}, {"Cohesion", 0.50}});
    // CoR is a pair-wise property, so it should be mentioned here
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_cube, mat_type_granular_1, 0.8);
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_cube, mat_type_granular_2, 0.8);

    float granular_rad = 0.02;  // 0.002;
    auto template_granular = DEMSim.LoadSphereType(granular_rad * granular_rad * granular_rad * 2.6e3 * 4 / 3 * 3.14,
                                                   granular_rad, mat_type_granular_1);

    float step_size = 1e-6;
    const double world_size = 4.0;
    const float fill_height = 3.0;
    const float chamber_bottom = -fill_height / 2.;
    const float fill_bottom = -fill_height + granular_rad;

    DEMSim.InstructBoxDomainDimension({-world_size / 2, world_size / 2}, {-world_size / 2, world_size / 2},
                                      {-fill_height, 10.0});
    DEMSim.InstructBoxDomainBoundingBC("all", mat_type_granular_2);

    // Now add a cylinderical boundary
    auto walls = DEMSim.AddExternalObject();
    walls->AddCylinder(make_float3(0, 0, fill_height), make_float3(0, 0, 1), world_size / 2., mat_type_cube, 0);

    double cube_speed = -0.5;

    // float mass = 7;
    // float3 MOI = make_float3(0.1, 0.1, 0.1) * 2000;
    float w_r = 2 * deme::PI;

    // auto pile = DEMSim.AddWavefrontMeshObject("../data/granularFlow/pile.obj", mat_type_cube);
    // float3 move = make_float3(0.00, 0.00, 0.02);  // z
    // float4 rot = make_float4(0, 0, 1, 0);
    // pile->Scale(make_float3(0.01));
    // pile->Move(move, rot);
    // pile->SetMass(mass);
    // pile->SetMOI(MOI);

    // pile->SetFamily(10);
    // DEMSim.SetFamilyFixed(10);

    DEMSim.SetFamilyPrescribedAngVel(11, "0", "0", to_string_with_precision(w_r), false);

    DEMSim.SetFamilyPrescribedLinVel(12, "0", "0", to_string_with_precision(-cube_speed));
    // Track the cube

    // Sampler to use
    auto modelCohesion = DEMSim.ReadContactForceModel("ForceModelSteel.cu");
    modelCohesion->SetMustHaveMatProp({"E", "nu", "CoR", "mu", "Crr"});
    modelCohesion->SetMustPairwiseMatProp({"CoR", "mu", "Crr"});
    modelCohesion->SetPerContactWildcards(
        {"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z", "innerInteraction", "initialLength"});

    const float spacing = 2.02f * granular_rad;
    const float fill_radius = world_size / 2. - 2. * granular_rad;

    // PDSampler sampler(spacing);
    // std::vector<float3> input_xyz;
    // float layer_z = 0;
    // while (layer_z < fill_height) {
    //     float3 sample_center = make_float3(0, 0, fill_bottom + layer_z + spacing / 2);
    //     auto layer_xyz = sampler.SampleCylinderZ(sample_center, fill_radius, 0);
    //     input_xyz.insert(input_xyz.end(), layer_xyz.begin(), layer_xyz.end());
    //     layer_z += spacing;
    // // Drum is a `big clump', we now generate its template

    float3 CylAxis = make_float3(0, 0, 1);
    float CylRad = 0.60;
    float CylHeight = 2.0;
    float3 CylCenter = make_float3(0, 0, 1.1 * CylHeight / 2);

    float CylParticleRad = 0.01;
    float CylMass = 100;
    float sphere_vol = 4. / 3. * deme::PI * CylParticleRad * CylParticleRad * CylParticleRad;
    float mass = 7850 * sphere_vol;
    std::shared_ptr<DEMClumpTemplate> my_template = DEMSim.LoadSphereType(mass, CylParticleRad, mat_type_cube);
    auto monopile = DEMCylSurfSampler(CylCenter, CylAxis, CylRad, CylHeight, CylParticleRad);
    auto particles_pile = DEMSim.AddClumps(my_template, monopile);

    std::cout << monopile.size() << " spheres make up the rotating drum" << std::endl;
    particles_pile->SetFamily(2);
    DEMSim.SetFamilyExtraMargin(2, 1.0 * CylParticleRad);

    float CylHeightAnular = 0.10;
    float3 CylCenter_Up = make_float3(0, 0, 1.1 * CylHeight / 2+CylHeight / 2+ CylHeightAnular/2+6*CylParticleRad/5);
    
   
    auto monopile_Up = DEMCylSurfSampler(CylCenter_Up, CylAxis, CylRad, CylHeightAnular, CylParticleRad);
    auto particles_pile_Up = DEMSim.AddClumps(my_template, monopile_Up);
    
    std::cout << monopile_Up.size() << " spheres make up the rotating drum" << std::endl;
    particles_pile_Up->SetFamily(30);
    DEMSim.SetFamilyExtraMargin(30, 1.0 * CylParticleRad);

    //DEMSim.SetFamilyPrescribedLinVel(2, "none", "none", to_string_with_precision(2 * cube_speed));
    DEMSim.AddFamilyPrescribedAcc(4, "none", to_string_with_precision( -1./ 500),"none");

    // Drum->SetFamily(2);

    // Drum_template->SetFamily(drum_family);
    //  auto monopile_track = DEMSim.Track(Drum_template);

    HCPSampler sampler(spacing);
    float3 fill_center = make_float3(0, 0, fill_bottom + fill_height / 2);
    auto input_xyz = sampler.SampleCylinderZ(fill_center, fill_radius, fill_height / 2);

    // Note: AddClumps can be called multiple times before initialization to add more clumps to the system
    auto particles = DEMSim.AddClumps(template_granular, input_xyz);
    particles->SetFamily(1);
    // DEMSim.SetFamilyExtraMargin(1, 0.5 * granular_rad);
    //  Initially, no contact between the brick and the granular material
    //  DEMSim.DisableContactBetweenFamilies(1, 10);

    // Use a owner wildcard to record tangential displacement compared to initial pos
    // auto force_model = DEMSim.GetContactForceModel();
    modelCohesion->SetPerOwnerWildcards({"gran_strain"});
    particles->AddOwnerWildcard("gran_strain", 0.0);
    particles_pile->AddOwnerWildcard("gran_strain", 0.0);
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

    std::cout << "Initial number of contacts: " << DEMSim.GetNumContacts() << std::endl;

    DEMSim.SetFamilyContactWildcardValueAll(2, "initialLength", 0.0);
    DEMSim.SetFamilyContactWildcardValueAll(3, "initialLength", 0.0);
    DEMSim.SetFamilyContactWildcardValueAll(2, "innerInteraction", 2.0);
    DEMSim.SetFamilyContactWildcardValueAll(3, "innerInteraction", 2.0);
    DEMSim.SetFamilyContactWildcardValueAll(1, "innerInteraction", 0.0);

    DEMSim.SetFamilyClumpMaterial(1, mat_type_granular_1);
    DEMSim.SetFamilyClumpMaterial(2, mat_type_cube);
    DEMSim.SetFamilyClumpMaterial(3, mat_type_cube);

    float sim_end = 15;
    unsigned int fps = 20;  // 20;
    float frame_time = 1.0 / fps;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * frame_time));

    // Keep tab of some sim quantities
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");

    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int currframe = 0;
    unsigned int curr_step = 0;

    bool cond_1 = true;
    bool cond_2 = true;

    // DEMSim.ChangeFamily(10, 11);

    //  Settle
    for (double t = 0; t < 0.1; t += frame_time) {
        char filename[200], meshname[200];
        std::cout << "Outputting frame: " << currframe << std::endl;
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
        sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), currframe++);
        DEMSim.WriteSphereFile(std::string(filename));
        DEMSim.WriteMeshFile(std::string(meshname));
        DEMSim.ShowThreadCollaborationStats();

        DEMSim.DoDynamicsThenSync(frame_time);
    }
    double init_max_z = max_z_finder->GetValue();
    std::cout << "After settling, max particle Z coord is " << init_max_z << std::endl;

    // Record init positions of the particles
    std::vector<std::vector<bodyID_t>> particle_cnt_map;
    std::vector<std::vector<float3>> particle_init_relative_pos;
    // Build contact map (contact partner owner IDs) for all particles
    // buildContactMap(particle_cnt_map, particle_init_relative_pos, DEMSim, particle_tracker, num_particles);

    // Ready to start indentation
    std::cout << "Simulation starts..." << std::endl;
    // Let the brick sink with a downward velocity.

    // This is meant to show that you can change the material type of the clumps in mid-simulation.
    // Doing this, we change the mu between particles from 0.3 (lower, for getting something denser
    // after settling) to 0.4 (the value we use for the main simulation).
    // DEMSim.SetFamilyClumpMaterial(1, mat_type_granular_2);

    std::string nameOutFile = "Indentation.csv";
    std::ofstream csvFile(nameOutFile);

    // double cube_zpos = max_z_finder->GetValue();
    //  cube_tracker->SetPos(make_float3(0, 0, cube_zpos));
    // std::cout << "Initially the cube is at Z = " << cube_zpos << std::endl;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (double t = 0; t < sim_end; t += frame_time, curr_step++) {
        if (curr_step % out_steps == 0) {
            // float3 forces = cube_tracker->ContactAcc();
            // float3 pos = cube_tracker->Pos();
            // float3 rot = cube_tracker->AngVelLocal();
            std::cout << "Time: " << t << std::endl;
            // std::cout << "Pos of pile: " << pos.z << std::endl;
            // std::cout << "Rot of pile: " << rot.z << std::endl;
            // csvFile << t << "; " << pos.z << "; " << forces.x << "; " << forces.y << "; " << forces.z << std::endl;

            // if (pos.z < -0.50 && cond_1) {
            //     DEMSim.DoDynamicsThenSync(0);
            //     DEMSim.ChangeFamily(11, 10);
            //     DEMSim.SetFamilyClumpMaterial(1, mat_type_granular_2);
            //     cond_1 = false;
            // }

            if (t > 0.50 && cond_2) {
                DEMSim.DoDynamicsThenSync(0);
                DEMSim.ChangeFamily(30, 4);

                cond_2 = false;
            }

            // Compute relative displacement
            // std::vector<float> gran_strain(num_particles);
            // for (unsigned int i = 0; i < num_particles; i++) {
            //     float3 main_loc = particle_tracker->Pos(i);
            //     // Compute contact partners' new locations
            //     std::vector<float3> rel_pos;
            //     for (auto& ID : particle_cnt_map.at(i)) {
            //         rel_pos.push_back(DEMSim.GetOwnerPosition(ID) - main_loc);
            //     }
            //     //     // How large is the strain?
            //     //     // float3 strains = make_float3(0);
            //     float strains = 0.;
            //     int num_neighbors = particle_init_relative_pos.at(i).size();
            //     for (int j = 0; j < num_neighbors; j++) {
            //         // strains += particle_init_relative_pos.at(i).at(j) - rel_pos.at(j);
            //         strains += length(particle_init_relative_pos.at(i).at(j) - rel_pos.at(j));
            //     }
            //     gran_strain[i] = (num_neighbors > 0) ? (strains / num_neighbors) : 0.0;
            // }
            // Re-build contact map, for the next output step
            // buildContactMap(particle_cnt_map, particle_init_relative_pos, DEMSim, particle_tracker, num_particles);
            // std::cout << "A new contact map constructed..." << std::endl;

            // Feed displacement info to wildcard, then leverage the output method to output it to the file
            // DEMSim.SetFamilyOwnerWildcardValue(1, "gran_strain", gran_strain);
            char filename[200], meshname[200];
            std::cout << "Outputting frame: " << currframe << std::endl;
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
            sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), currframe++);
            DEMSim.WriteSphereFile(std::string(filename));
            DEMSim.WriteMeshFile(std::string(meshname));
            // DEMSim.ShowThreadCollaborationStats();
        }

        DEMSim.DoDynamics(frame_time);
        std::cout << "Number of contacts: " << DEMSim.GetNumContacts() << std::endl;
        // cube_zpos -= cube_speed * step_size;
        // cube_tracker->SetPos(make_float3(0, 0, cube_zpos));
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << (time_sec.count()) / sim_end / (1e-5 / step_size)
              << " seconds (wall time) to finish 1e5 steps' simulation" << std::endl;

    csvFile.close();
    DEMSim.ShowTimingStats();
    std::cout << "DEMdemo_Indentation exiting..." << std::endl;
    return 0;
}
