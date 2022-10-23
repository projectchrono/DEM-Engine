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

using namespace deme;
using namespace std::filesystem;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    // DEMSim.SetOutputContent(OUTPUT_CONTENT::FAMILY);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::XYZ);

    srand(759);

    //
    float kg_g_conv = 1;
    // Define materials
    auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 1e9 * kg_g_conv}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.5}});
    auto mat_type_wheel = DEMSim.LoadMaterial({{"E", 1e9 * kg_g_conv}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.5}});

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

    // Then the ground particle template
    DEMClumpTemplate shape_template;
    shape_template.ReadComponentFromFile((GET_SOURCE_DATA_PATH() / "clumps/triangular_flat.csv").string());
    // Calculate its mass and MOI
    float mass = 2.6e3 * 5.5886717 * kg_g_conv;  // in kg or g
    float3 MOI = make_float3(1.8327927, 2.1580013, 0.77010059) * 2.6e3 * kg_g_conv;
    float clump_vol = 5.5886717;
    // Scale the template we just created
    std::vector<std::shared_ptr<DEMClumpTemplate>> ground_particle_templates;
    // std::vector<double> scales = {0.0014, 0.00063, 0.00033, 0.00022, 0.00015, 0.00009};
    std::vector<double> scales = {0.00063, 0.00033, 0.00022, 0.00015, 0.00009};
    std::for_each(scales.begin(), scales.end(), [](double& r) { r *= 20.; });
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
        this_template.SetVolume((double)clump_vol * scaling * scaling * scaling);
        ground_particle_templates.push_back(DEMSim.LoadClumpType(this_template));
    }

    // Now we load part2 clump locations from a part1 output file
    auto part2_clump_xyz = DEMSim.ReadClumpXyzFromCsv("GRC_3e6.csv");
    auto part2_clump_quaternion = DEMSim.ReadClumpQuatFromCsv("GRC_3e6.csv");
    auto part2_pairs = DEMSim.ReadContactPairsFromCsv("Contact_pairs_3e6.csv");
    auto part2_wcs = DEMSim.ReadContactWildcardsFromCsv("Contact_pairs_3e6.csv");
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
    double step_size = 1e-6;
    DEMSim.SetCoordSysOrigin("center");
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEMSim.SetCDUpdateFreq(20);
    // DEMSim.SetExpandFactor(1e-3);
    DEMSim.SetMaxVelocity(15.);
    DEMSim.SetExpandSafetyParam(1.1);
    DEMSim.SetInitBinSize(scales.at(1));
    DEMSim.Initialize();

    unsigned int fps = 20;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    path out_dir = current_path();
    out_dir += "/DemoOutput_GRCPrep_Part3";
    create_directory(out_dir);
    unsigned int currframe = 0;
    unsigned int curr_step = 0;

    // Settle a bit
    DEMSim.DoDynamicsThenSync(0.3);
    DEMSim.SetInitTimeStep(5e-7);
    DEMSim.UpdateSimParams();

    // Now compress it
    DEMSim.EnableContactBetweenFamilies(0, 1);
    double compress_time = 0.3;

    float matter_mass = total_mass_finder->GetValue();
    std::cout << "Initial bulk density " << matter_mass / total_volume << std::endl;

    double now_z = max_z_finder->GetValue();
    compressor_tracker->SetPos(make_float3(0, 0, now_z));
    double compressor_final_dist = (now_z > -0.41) ? now_z - (-0.41) : 0.0;
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
    sprintf(cp_filename, "%s/GRC_20e6.csv", out_dir.c_str());
    DEMSim.WriteClumpFile(std::string(cp_filename));

    DEMSim.ClearThreadCollaborationStats();

    char cnt_filename[200];
    sprintf(cnt_filename, "%s/Contact_pairs.csv", out_dir.c_str());
    DEMSim.WriteContactFile(std::string(cnt_filename));

    std::cout << "DEMdemo_GRCPrep_Part3 exiting..." << std::endl;
    return 0;
}
