//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// In GRCPrep demo series, we try to prepare a sample of the GRC simulant, which
// are supposed to be used for extraterrestrial rover mobility simulations. You
// have to finish Part2 first, then run this one. In Part3, we concatenate several
// patches of particles together to form a 4m * 2m granular bed.
// WARNING: This is a huge simulation with tens of millions of particles.
// WARNING: You probably need recent data-center GPUs to run this one.
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
    auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.4}});
    auto mat_type_wheel = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.4}});

    // Define the simulation world
    double world_y_size = 2.0;
    double world_x_size = 4.0;
    DEMSim.InstructBoxDomainDimension(world_x_size, world_y_size, world_y_size);
    float bottom = -0.5;
    DEMSim.AddBCPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_terrain);
    // Side bounding planes
    DEMSim.AddBCPlane(make_float3(0, world_y_size / 2, 0), make_float3(0, -1, 0), mat_type_terrain);
    DEMSim.AddBCPlane(make_float3(0, -world_y_size / 2, 0), make_float3(0, 1, 0), mat_type_terrain);
    // X-dir bounding planes
    DEMSim.AddBCPlane(make_float3(-world_x_size / 2, 0, 0), make_float3(1, 0, 0), mat_type_terrain);
    DEMSim.AddBCPlane(make_float3(world_x_size / 2, 0, 0), make_float3(-1, 0, 0), mat_type_terrain);

    // Define the terrain particle templates
    // Calculate its mass and MOI
    float terrain_density = 2.6e3;
    float volume1 = 4.2520508;
    float mass1 = terrain_density * volume1;
    float3 MOI1 = make_float3(1.6850426, 1.6375114, 2.1187753) * terrain_density;
    // Scale the template we just created
    std::vector<double> scales = {0.007};
    // Then load it to system
    std::shared_ptr<DEMClumpTemplate> my_template1 =
        DEMSim.LoadClumpType(mass1, MOI1, GetDEMEDataFile("clumps/triangular_flat.csv"), mat_type_terrain);
    std::vector<std::shared_ptr<DEMClumpTemplate>> ground_particle_templates = {my_template1};
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
    // Now we load part2 clump locations from a part1 output file
    auto part2_clump_xyz = DEMSim.ReadClumpXyzFromCsv("./CheapGRCPrep_Part1/GRC_3e5.csv");
    auto part2_clump_quaternion = DEMSim.ReadClumpQuatFromCsv("./CheapGRCPrep_Part1/GRC_3e5.csv");
    auto part2_pairs = DEMSim.ReadContactPairsFromCsv("./CheapGRCPrep_Part1/Contact_pairs_3e5.csv");
    auto part2_wcs = DEMSim.ReadContactWildcardsFromCsv("./CheapGRCPrep_Part1/Contact_pairs_3e5.csv");
    std::vector<float3> in_xyz;
    std::vector<float4> in_quat;
    std::vector<std::shared_ptr<DEMClumpTemplate>> in_types;
    unsigned int t_num = 0;
    for (int i = 0; i < scales.size(); i++) {
        char t_name[20];
        sprintf(t_name, "%04d", t_num);

        auto this_type_xyz = part2_clump_xyz[std::string(t_name)];
        auto this_type_quat = part2_clump_quaternion[std::string(t_name)];

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
    // Remove some elements maybe? I feel this making the surface flatter
    // std::vector<notStupidBool_t> elem_to_remove(in_xyz.size(), 0);
    // for (size_t i = 0; i < in_xyz.size(); i++) {
    //     if (in_xyz.at(i).z > -0.44)
    //         elem_to_remove.at(i) = 1;
    // }
    // in_xyz.erase(
    //     std::remove_if(in_xyz.begin(), in_xyz.end(),
    //                    [&elem_to_remove, &in_xyz](const float3& i) { return elem_to_remove.at(&i - in_xyz.data());
    //                    }),
    //     in_xyz.end());
    // in_quat.erase(
    //     std::remove_if(in_quat.begin(), in_quat.end(),
    //                    [&elem_to_remove, &in_quat](const float4& i) { return elem_to_remove.at(&i - in_quat.data());
    //                    }),
    //     in_quat.end());
    // in_types.erase(
    //     std::remove_if(in_types.begin(), in_types.end(),
    //                    [&elem_to_remove, &in_types](const auto& i) { return elem_to_remove.at(&i - in_types.data());
    //                    }),
    //     in_types.end());

    // Finally, load the info into this batch
    DEMClumpBatch base_batch(in_xyz.size());
    base_batch.SetTypes(in_types);
    base_batch.SetPos(in_xyz);
    base_batch.SetOriQ(in_quat);
    base_batch.SetExistingContacts(part2_pairs);
    base_batch.SetExistingContactWildcards(part2_wcs);

    // Based on the `base_batch', we can create more batches
    std::vector<float> x_shift_dist = {-1.5, -0.5, 0.5, 1.5};
    std::vector<float> y_shift_dist = {-0.5, 0.5};
    // Add some patches of such graular bed
    for (float x_shift : x_shift_dist) {
        for (float y_shift : y_shift_dist) {
            DEMClumpBatch another_batch = base_batch;
            std::vector<float3> my_xyz = in_xyz;
            std::for_each(my_xyz.begin(), my_xyz.end(), [x_shift, y_shift](float3& xyz) {
                xyz.x += x_shift;
                xyz.y += y_shift;
            });
            another_batch.SetPos(my_xyz);
            DEMSim.AddClumps(another_batch);
        }
    }

    // Now add a plane to compress the `road'
    auto compressor = DEMSim.AddExternalObject();
    compressor->AddPlane(make_float3(0, 0, 0), make_float3(0, 0, -1), mat_type_terrain);
    compressor->SetFamily(1);
    DEMSim.DisableContactBetweenFamilies(0, 1);
    DEMSim.SetFamilyFixed(1);
    auto compressor_tracker = DEMSim.Track(compressor);

    // And a z position inspector
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
    // Keep tab of the max velocity in simulation
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");
    // Final mass inspection tool
    // auto total_volume_finder = DEMSim.CreateInspector("clump_volume", "return (abs(X) <= 0.48) && (abs(Y) <= 0.48) &&
    // (Z <= -0.44);");
    auto total_mass_finder =
        DEMSim.CreateInspector("clump_mass", "return (abs(X) <= 0.48) && (abs(Y) <= 0.48) && (Z <= -0.44);");
    float total_volume = 0.96 * 0.96 * 0.06;

    // Make ready for simulation
    double step_size = 2e-6;
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    // Error out vel is used to force the simulation to abort when something goes wrong.
    DEMSim.SetErrorOutVelocity(15.);
    DEMSim.SetExpandSafetyMultiplier(1.1);
    DEMSim.Initialize();

    unsigned int fps = 10;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    path out_dir = current_path();
    out_dir += "/CheapGRCPrep_Part2";
    create_directory(out_dir);
    unsigned int currframe = 0;
    unsigned int curr_step = 0;

    // Settle a bit
    DEMSim.DoDynamicsThenSync(0.3);

    // Now compress it
    DEMSim.EnableContactBetweenFamilies(0, 1);
    double compress_time = 0.2;

    float matter_mass = total_mass_finder->GetValue();
    std::cout << "Initial bulk density " << matter_mass / total_volume << std::endl;

    double now_z = max_z_finder->GetValue();
    compressor_tracker->SetPos(make_float3(0, 0, now_z));
    double compressor_final_dist = (now_z > -0.3) ? now_z - (-0.3) : 0.0;
    double compressor_v = compressor_final_dist / compress_time;
    for (double t = 0; t < compress_time; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            std::cout << "Frame: " << currframe << std::endl;
            std::cout << "Highest point is at " << now_z << std::endl;
            matter_mass = total_mass_finder->GetValue();
            std::cout << "Bulk density in compression " << matter_mass / total_volume << std::endl;
            float max_v = max_v_finder->GetValue();
            std::cout << "Highest velocity is " << max_v << std::endl;
            DEMSim.ShowThreadCollaborationStats();
            char filename[200];
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
            DEMSim.WriteSphereFile(std::string(filename));
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
            float max_v = max_v_finder->GetValue();
            std::cout << "Highest velocity is " << max_v << std::endl;
            DEMSim.ShowThreadCollaborationStats();
            char filename[200];
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
            DEMSim.WriteSphereFile(std::string(filename));
        }
        now_z += compressor_v * step_size;
        compressor_tracker->SetPos(make_float3(0, 0, now_z));
        DEMSim.DoDynamics(step_size);
    }

    DEMSim.DoDynamicsThenSync(0);
    DEMSim.DisableContactBetweenFamilies(0, 1);
    DEMSim.DoDynamicsThenSync(0.2);
    matter_mass = total_mass_finder->GetValue();
    std::cout << "Bulk density after settling " << matter_mass / total_volume << std::endl;

    char cp_filename[200];
    sprintf(cp_filename, "%s/GRC_3e6.csv", out_dir.c_str());
    DEMSim.WriteClumpFile(std::string(cp_filename));

    DEMSim.ClearThreadCollaborationStats();

    std::cout << "DEMdemo_GRCPrep_Part2 exiting..." << std::endl;
    return 0;
}
