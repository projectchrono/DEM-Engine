//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// In GRCPrep demo series, we try to prepare a sample of the GRC simulant, which
// are supposed to be used for extraterrestrial rover mobility simulations. You
// have to finish Part1 first, then run this one. In Part2, we copy-paste particles
// generated and settled in Part1 and form a thicker bed.
// WARNING: This is a huge simulation with millions of particles.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <cstdio>
#include <chrono>
#include <filesystem>
#include <map>
#include <random>
#include <cmath>

using namespace deme;
using namespace std::filesystem;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    // DEMSim.SetOutputContent(OUTPUT_CONTENT::FAMILY);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::XYZ);

    srand(759);

    // Define materials
    auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.5}});
    auto mat_type_wheel = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.5}});

    // Define the simulation world
    double world_y_size = 0.99;
    DEMSim.InstructBoxDomainDimension(world_y_size, world_y_size, world_y_size * 2);
    // Add 5 bounding planes around the simulation world, and leave the top open
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);
    float bottom = -0.5;
    DEMSim.AddBCPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_terrain);

    // Define the terrain particle templates
    // Calculate its mass and MOI
    float terrain_density = 2.6e3;
    float volume1 = 4.2520508;
    float mass1 = terrain_density * volume1;
    float3 MOI1 = make_float3(1.6850426, 1.6375114, 2.1187753) * terrain_density;
    float volume2 = 2.1670011;
    float mass2 = terrain_density * volume2;
    float3 MOI2 = make_float3(0.57402126, 0.60616378, 0.92890173) * terrain_density;
    // Scale the template we just created
    std::vector<double> scales = {0.014, 0.0075833, 0.0044, 0.003, 0.002, 0.0018333, 0.0017};
    // Then load it to system
    std::shared_ptr<DEMClumpTemplate> my_template2 =
        DEMSim.LoadClumpType(mass2, MOI2, GetDEMEDataFile("clumps/triangular_flat_6comp.csv"), mat_type_terrain);
    std::shared_ptr<DEMClumpTemplate> my_template1 =
        DEMSim.LoadClumpType(mass1, MOI1, GetDEMEDataFile("clumps/triangular_flat.csv"), mat_type_terrain);
    std::vector<std::shared_ptr<DEMClumpTemplate>> ground_particle_templates = {my_template2,
                                                                                DEMSim.Duplicate(my_template2),
                                                                                my_template1,
                                                                                DEMSim.Duplicate(my_template1),
                                                                                DEMSim.Duplicate(my_template1),
                                                                                DEMSim.Duplicate(my_template1),
                                                                                DEMSim.Duplicate(my_template1)};
    // Now scale those templates
    for (int i = 0; i < scales.size(); i++) {
        std::shared_ptr<DEMClumpTemplate>& my_template = ground_particle_templates.at(i);
        // Note the mass and MOI are also scaled in the process, automatically. But if you are not happy with this, you
        // can always manually change mass and MOI afterwards.
        my_template->Scale(scales.at(i));
        // Give these templates names, 0000, 0001 etc.
        char t_name[20];
        sprintf(t_name, "%04d", i);
        my_template->AssignName(std::string(t_name));
    }

    // Now we load part1 clump locations from a part1 output file
    auto part1_clump_xyz = DEMSim.ReadClumpXyzFromCsv("./DemoOutput_GRCPrep_Part1/GRC_3e5.csv");
    auto part1_clump_quaternion = DEMSim.ReadClumpQuatFromCsv("./DemoOutput_GRCPrep_Part1/GRC_3e5.csv");
    auto part1_pairs = DEMSim.ReadContactPairsFromCsv("./DemoOutput_GRCPrep_Part1/Contact_pairs_3e5.csv");
    auto part1_wcs = DEMSim.ReadContactWildcardsFromCsv("./DemoOutput_GRCPrep_Part1/Contact_pairs_3e5.csv");

    std::vector<float3> in_xyz;
    std::vector<float4> in_quat;
    std::vector<std::shared_ptr<DEMClumpTemplate>> in_types;
    unsigned int t_num = 0;
    for (int i = 0; i < scales.size(); i++) {
        char t_name[20];
        sprintf(t_name, "%04d", t_num);

        auto this_type_xyz = part1_clump_xyz[std::string(t_name)];
        auto this_type_quat = part1_clump_quaternion[std::string(t_name)];

        size_t n_clump_this_type = this_type_xyz.size();
        // Prepare clump type identification vector for loading into the system (don't forget type 0 in
        // ground_particle_templates is the template for rover wheel)
        std::vector<std::shared_ptr<DEMClumpTemplate>> this_type(n_clump_this_type,
                                                                 ground_particle_templates.at(t_num));

        // Add them to the big long vector
        in_xyz.insert(in_xyz.end(), this_type_xyz.begin(), this_type_xyz.end());
        in_quat.insert(in_quat.end(), this_type_quat.begin(), this_type_quat.end());
        in_types.insert(in_types.end(), this_type.begin(), this_type.end());

        // Our template names are 0000, 0001 etc.
        t_num++;
    }
    // Finally, load them into the system
    DEMClumpBatch base_batch(in_xyz.size());
    base_batch.SetTypes(in_types);
    base_batch.SetPos(in_xyz);
    base_batch.SetOriQ(in_quat);
    base_batch.SetExistingContacts(part1_pairs);
    base_batch.SetExistingContactWildcards(part1_wcs);
    base_batch.SetFamily(0);
    DEMSim.AddClumps(base_batch);

    // I also would like an `inverse batch', which is a batch of clumps that is the base batch flipped around
    DEMClumpBatch inv_batch = base_batch;
    std::vector<float3> inv_xyz = in_xyz;
    std::vector<float4> inv_quat = in_quat;
    float3 flip_center = make_float3(0, 0, bottom);
    float3 flip_axis = make_float3(1, 0, 0);
    std::for_each(inv_xyz.begin(), inv_xyz.end(), [flip_center, flip_axis](float3& xyz) {
        xyz = flip_center + Rodrigues(xyz - flip_center, flip_axis, 3.14159);
    });
    std::for_each(inv_quat.begin(), inv_quat.end(), [flip_axis](float4& Q) { Q = RotateQuat(Q, flip_axis, 3.14159); });
    // inv_batch.SetPos(inv_xyz);
    inv_batch.SetOriQ(inv_quat);

    // Based on the `base_batch', we can create more batches. For example, another batch that is like copy-paste the
    // existing batch, then shift up for a small distance.
    float shift_dist = 0.2;
    // First put the inv batch above the base batch
    std::for_each(inv_xyz.begin(), inv_xyz.end(), [](float3& xyz) { xyz.z += 0.2; });
    inv_batch.SetPos(inv_xyz);
    DEMSim.AddClumps(inv_batch);
    // Add more layers of such graular bed
    for (int i = 0; i < 1; i++) {
        DEMClumpBatch another_batch = base_batch;
        std::for_each(in_xyz.begin(), in_xyz.end(), [shift_dist](float3& xyz) { xyz.z += shift_dist; });
        another_batch.SetPos(in_xyz);
        DEMSim.AddClumps(another_batch);
        DEMClumpBatch another_inv_batch = inv_batch;
        std::for_each(inv_xyz.begin(), inv_xyz.end(), [shift_dist](float3& xyz) { xyz.z += shift_dist; });
        another_inv_batch.SetPos(inv_xyz);
        DEMSim.AddClumps(another_inv_batch);
    }

    // Some inspectors and compressors
    // auto total_volume_finder = DEMSim.CreateInspector("clump_volume", "return (abs(X) <= 0.48) && (abs(Y) <= 0.48) &&
    // (Z <= -0.44);");
    auto total_mass_finder =
        DEMSim.CreateInspector("clump_mass", "return (abs(X) <= 0.48) && (abs(Y) <= 0.48) && (Z <= -0.44);");
    float total_volume = 0.96 * 0.96 * 0.06;
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");

    // Now add a plane to compress the `road'
    auto compressor = DEMSim.AddExternalObject();
    compressor->AddPlane(make_float3(0, 0, 0), make_float3(0, 0, -1), mat_type_terrain);
    compressor->SetFamily(1);
    DEMSim.DisableContactBetweenFamilies(0, 1);
    DEMSim.SetFamilyFixed(1);
    auto compressor_tracker = DEMSim.Track(compressor);

    // Make ready for simulation
    float step_size = 1e-6;
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    // Error out vel is used to force the simulation to abort when something goes wrong.
    DEMSim.SetErrorOutVelocity(15.);
    DEMSim.SetExpandSafetyMultiplier(1.2);
    DEMSim.SetInitBinNumTarget(1e7);
    DEMSim.Initialize();

    unsigned int fps = 20;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    path out_dir = current_path();
    out_dir /= "DemoOutput_GRCPrep_Part2";
    create_directory(out_dir);
    unsigned int currframe = 0;
    unsigned int curr_step = 0;

    float settle_frame_time = 0.05;
    float settle_batch_time = 1.0;

    float matter_mass = total_mass_finder->GetValue();
    std::cout << "Initial bulk density " << matter_mass / total_volume << std::endl;

    for (float t = 0; t < settle_batch_time; t += settle_frame_time) {
        std::cout << "Frame: " << currframe << std::endl;
        char filename[100];
        sprintf(filename, "DEMdemo_output_%04d.csv", currframe++);
        DEMSim.WriteSphereFile(out_dir / filename);
        DEMSim.DoDynamicsThenSync(settle_frame_time);
        DEMSim.ShowThreadCollaborationStats();
    }

    matter_mass = total_mass_finder->GetValue();
    std::cout << "Bulk density after settling " << matter_mass / total_volume << std::endl;

    // Now compress it
    DEMSim.EnableContactBetweenFamilies(0, 1);
    double compress_time = 0.3;
    double now_z = max_z_finder->GetValue();
    compressor_tracker->SetPos(make_float3(0, 0, now_z));
    double compressor_final_dist = (now_z > -0.37) ? now_z - (-0.37) : 0.0;
    double compressor_v = compressor_final_dist / compress_time;

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (double t = 0; t < compress_time; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            std::cout << "Frame: " << currframe << std::endl;
            std::cout << "Highest point is at " << now_z << std::endl;
            matter_mass = total_mass_finder->GetValue();
            std::cout << "Bulk density in compression " << matter_mass / total_volume << std::endl;
            DEMSim.ShowThreadCollaborationStats();
            char filename[100];
            sprintf(filename, "DEMdemo_output_%04d.csv", currframe++);
            DEMSim.WriteSphereFile(out_dir / filename);
        }
        now_z -= compressor_v * step_size;
        compressor_tracker->SetPos(make_float3(0, 0, now_z));
        DEMSim.DoDynamics(step_size);
    }
    // Then gradually remove the compressor
    for (double t = 0; t < compress_time; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            std::cout << "Frame: " << currframe << std::endl;
            std::cout << "Highest point is at " << now_z << std::endl;
            matter_mass = total_mass_finder->GetValue();
            std::cout << "Bulk density in compression " << matter_mass / total_volume << std::endl;
            DEMSim.ShowThreadCollaborationStats();
            char filename[100];
            sprintf(filename, "DEMdemo_output_%04d.csv", currframe++);
            DEMSim.WriteSphereFile(out_dir / filename);
        }
        now_z += compressor_v * step_size;
        compressor_tracker->SetPos(make_float3(0, 0, now_z));
        DEMSim.DoDynamics(step_size);
    }

    DEMSim.DoDynamicsThenSync(0);
    DEMSim.DisableContactBetweenFamilies(0, 1);
    DEMSim.DoDynamicsThenSync(0.3);
    matter_mass = total_mass_finder->GetValue();

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds (wall time) to finish the simulation" << std::endl;

    std::cout << "Bulk density after settling " << matter_mass / total_volume << std::endl;

    // Final write
    char cp_filename[100];
    sprintf(cp_filename, "GRC_3e6.csv");
    DEMSim.WriteClumpFile(out_dir / cp_filename);

    DEMSim.ClearThreadCollaborationStats();

    char cnt_filename[100];
    sprintf(cnt_filename, "Contact_pairs_3e6.csv");
    DEMSim.WriteContactFile(out_dir / cnt_filename);

    std::cout << "----------------------------------------" << std::endl;
    DEMSim.ShowMemStats();
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "DEMdemo_GRCPrep_Part2 exiting..." << std::endl;
    return 0;
}
