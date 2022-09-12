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
#include <map>
#include <random>
#include <cmath>

using namespace smug;
using namespace std::filesystem;

int main() {
    DEMSolver DEM_sim;
    DEM_sim.SetVerbosity(INFO);
    DEM_sim.SetOutputFormat(DEM_OUTPUT_FORMAT::CSV);
    // DEM_sim.SetOutputContent(DEM_OUTPUT_CONTENT::FAMILY);
    DEM_sim.SetOutputContent(DEM_OUTPUT_CONTENT::XYZ);

    srand(759);

    //
    float kg_g_conv = 1;
    // Define materials
    auto mat_type_terrain = DEM_sim.LoadMaterial({{"E", 2e9 * kg_g_conv}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.5}});
    auto mat_type_wheel = DEM_sim.LoadMaterial({{"E", 1e9 * kg_g_conv}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.5}});

    // Define the simulation world
    double world_y_size = 0.99;
    DEM_sim.InstructBoxDomainNumVoxel(21, 21, 22, world_y_size / std::pow(2, 16) / std::pow(2, 21));
    // Add 5 bounding planes around the simulation world, and leave the top open
    DEM_sim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);
    float bottom = -0.5;
    DEM_sim.AddBCPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_terrain);

    // Then the ground particle template
    DEMClumpTemplate shape_template;
    shape_template.ReadComponentFromFile((GET_DATA_PATH() / "clumps/triangular_flat.csv").string());
    // Calculate its mass and MOI
    float mass = 2.6e3 * 5.5886717 * kg_g_conv;  // in kg or g
    float3 MOI = make_float3(1.8327927, 2.1580013, 0.77010059) * 2.6e3 * kg_g_conv;
    float clump_vol = 5.5886717;
    // Scale the template we just created
    std::vector<std::shared_ptr<DEMClumpTemplate>> ground_particle_templates;
    std::vector<double> scales = {0.0014, 0.00063, 0.00033, 0.00022, 0.00015, 0.00009};
    std::for_each(scales.begin(), scales.end(), [](double& r) { r *= 10.; });
    for (double scaling : scales) {
        auto this_template = shape_template;
        this_template.mass = (double)mass * scaling * scaling * scaling;
        this_template.MOI.x = (double)MOI.x * (double)(scaling * scaling * scaling * scaling * scaling);
        this_template.MOI.y = (double)MOI.y * (double)(scaling * scaling * scaling * scaling * scaling);
        this_template.MOI.z = (double)MOI.z * (double)(scaling * scaling * scaling * scaling * scaling);
        std::cout << "Mass: " << this_template.mass << std::endl;
        std::cout << "MOIX: " << this_template.MOI.x << std::endl;
        std::cout << "MOIY: " << this_template.MOI.y << std::endl;
        std::cout << "MOIZ: " << this_template.MOI.z << std::endl;
        std::cout << "=====================" << std::endl;
        std::for_each(this_template.radii.begin(), this_template.radii.end(), [scaling](float& r) { r *= scaling; });
        std::for_each(this_template.relPos.begin(), this_template.relPos.end(), [scaling](float3& r) { r *= scaling; });
        this_template.materials = std::vector<std::shared_ptr<DEMMaterial>>(this_template.nComp, mat_type_terrain);
        // Set the volume of this template
        this_template.SetVolume((double)clump_vol * scaling * scaling * scaling);
        ground_particle_templates.push_back(DEM_sim.LoadClumpType(this_template));
    }

    // Now we load part1 clump locations from a part1 output file
    auto part1_clump_xyz = DEM_sim.ReadClumpXyzFromCsv("GRC_2e5.csv");
    auto part1_clump_quaternion = DEM_sim.ReadClumpQuatFromCsv("GRC_2e5.csv");
    std::vector<float3> in_xyz;
    std::vector<float4> in_quat;
    std::vector<std::shared_ptr<DEMClumpTemplate>> in_types;
    unsigned int t_num;
    for (int i = 0; i < scales.size(); i++) {
        // Our template names are 0001, 0002 etc.
        t_num++;
        char t_name[20];
        sprintf(t_name, "%04d", t_num);

        auto this_type_xyz = part1_clump_xyz[std::string(t_name)];
        auto this_type_quat = part1_clump_quaternion[std::string(t_name)];

        size_t n_clump_this_type = this_type_xyz.size();
        // Prepare clump type identification vector for loading into the system (don't forget type 0 in
        // ground_particle_templates is the template for rover wheel)
        std::vector<std::shared_ptr<DEMClumpTemplate>> this_type(n_clump_this_type,
                                                                 ground_particle_templates.at(t_num - 1));

        // Add them to the big long vector
        in_xyz.insert(in_xyz.end(), this_type_xyz.begin(), this_type_xyz.end());
        in_quat.insert(in_quat.end(), this_type_quat.begin(), this_type_quat.end());
        in_types.insert(in_types.end(), this_type.begin(), this_type.end());
    }
    // Finally, load them into the system
    DEMClumpBatch base_batch(in_xyz.size());
    base_batch.SetTypes(in_types);
    base_batch.SetPos(in_xyz);
    base_batch.SetOriQ(in_quat);
    DEM_sim.AddClumps(base_batch);

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
    float shift_dist = 0.1;
    // First put the inv batch above the base batch
    std::for_each(inv_xyz.begin(), inv_xyz.end(), [shift_dist](float3& xyz) { xyz.z += shift_dist / 2.0; });
    inv_batch.SetPos(inv_xyz);
    DEM_sim.AddClumps(inv_batch);
    // Add 10 layers of such graular bed
    for (int i = 0; i < 4; i++) {
        DEMClumpBatch another_batch = base_batch;
        std::for_each(in_xyz.begin(), in_xyz.end(), [shift_dist](float3& xyz) { xyz.z += shift_dist; });
        another_batch.SetPos(in_xyz);
        DEM_sim.AddClumps(another_batch);
        DEMClumpBatch another_inv_batch = inv_batch;
        std::for_each(inv_xyz.begin(), inv_xyz.end(), [shift_dist](float3& xyz) { xyz.z += shift_dist; });
        another_inv_batch.SetPos(inv_xyz);
        DEM_sim.AddClumps(another_inv_batch);
    }

    // Now add a plane to compress the `road'
    auto compressor = DEM_sim.AddExternalObject();
    compressor->AddPlane(make_float3(0, 0, 0), make_float3(0, 0, -1), mat_type_terrain);
    compressor->SetFamily(DEM_RESERVED_FAMILY_NUM);
    auto compressor_tracker = DEM_sim.Track(compressor);

    // And a z position inspector
    auto max_z_finder = DEM_sim.CreateInspector("clump_max_z");
    auto void_ratio_finder =
        DEM_sim.CreateInspector("clump_volume", "return (abs(X) <= 0.48) && (abs(Y) <= 0.48) && (Z <= -0.45);");

    // Make ready for simulation
    float step_size = 5e-7;
    DEM_sim.SetCoordSysOrigin("center");
    DEM_sim.SetInitTimeStep(step_size);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEM_sim.SetCDUpdateFreq(10);
    // DEM_sim.SetExpandFactor(1e-3);
    DEM_sim.SetMaxVelocity(20.);
    DEM_sim.SetExpandSafetyParam(1.2);
    DEM_sim.SetInitBinSize(scales.at(2));
    DEM_sim.Initialize();

    unsigned int fps = 20;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    path out_dir = current_path();
    out_dir += "/DEMdemo_GRCPrep_Part2";
    create_directory(out_dir);
    unsigned int currframe = 0;
    unsigned int curr_step = 0;

    float settle_frame_time = 0.05;
    float settle_batch_time = 1.2;

    float total_volume = 0.96 * 0.96 * 0.05;
    float matter_volume = void_ratio_finder->GetValue();

    std::cout << "Initial void ratio " << (total_volume - matter_volume) / matter_volume << std::endl;

    for (float t = 0; t < settle_batch_time; t += settle_frame_time) {
        std::cout << "Frame: " << currframe << std::endl;
        char filename[200];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
        DEM_sim.WriteSphereFile(std::string(filename));
        DEM_sim.DoDynamicsThenSync(settle_frame_time);
        DEM_sim.ShowThreadCollaborationStats();
    }

    char cp_filename[200];
    sprintf(cp_filename, "%s/GRC_2e6_mid.csv", out_dir.c_str());
    DEM_sim.WriteClumpFile(std::string(cp_filename));

    matter_volume = void_ratio_finder->GetValue();
    std::cout << "Void ratio after settling " << (total_volume - matter_volume) / matter_volume << std::endl;

    // Now compress it
    float compress_time = 0.5;

    float now_z = max_z_finder->GetValue();
    compressor_tracker->SetPos(make_float3(0, 0, now_z));
    float compressor_final_dist = (now_z > -0.43) ? now_z - (-0.43) : 0.0;
    float compressor_v = compressor_final_dist / compress_time;
    for (float t = 0; t < compress_time; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            std::cout << "Frame: " << currframe << std::endl;
            std::cout << "Highest point is at " << now_z << std::endl;
            matter_volume = void_ratio_finder->GetValue();
            std::cout << "Void ratio in compression " << (total_volume - matter_volume) / matter_volume << std::endl;
            DEM_sim.ShowThreadCollaborationStats();
            char filename[200];
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
            DEM_sim.WriteSphereFile(std::string(filename));
        }
        now_z -= compressor_v * step_size;
        compressor_tracker->SetPos(make_float3(0, 0, now_z));
        DEM_sim.DoDynamics(step_size);
    }
    // Then gradually remove the compressor
    for (float t = 0; t < compress_time; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            std::cout << "Frame: " << currframe << std::endl;
            std::cout << "Highest point is at " << now_z << std::endl;
            matter_volume = void_ratio_finder->GetValue();
            std::cout << "Void ratio in compression " << (total_volume - matter_volume) / matter_volume << std::endl;
            DEM_sim.ShowThreadCollaborationStats();
            char filename[200];
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
            DEM_sim.WriteSphereFile(std::string(filename));
        }
        now_z += compressor_v * step_size;
        compressor_tracker->SetPos(make_float3(0, 0, now_z));
        DEM_sim.DoDynamics(step_size);
    }

    // Final write
    char cp2_filename[200];
    sprintf(cp2_filename, "%s/GRC_2e6.csv", out_dir.c_str());
    DEM_sim.WriteClumpFile(std::string(cp2_filename));

    DEM_sim.ClearThreadCollaborationStats();

    std::cout << "DEMdemo_GRCPrep_Part2 exiting..." << std::endl;
    return 0;
}
