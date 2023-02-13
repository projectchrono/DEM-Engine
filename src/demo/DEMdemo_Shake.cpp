//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <random>

using namespace deme;

const double math_PI = 3.14159;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.SetContactOutputContent(OWNER | FORCE | POINT);

    // E, nu, CoR, mu, Crr...
    auto mat_type_cone = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.5}, {"Crr", 0.00}});
    auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.5}, {"Crr", 0.00}});

    float shake_amp = 0.1;
    float shake_speed = 2;  // Num of periods per second
    float step_size = 1e-6;
    double world_size = 2;
    double soil_bin_diameter = 0.584;
    double cone_surf_area = 323e-6;
    double cone_diameter = std::sqrt(cone_surf_area / math_PI) * 2;
    DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size);
    // No need to add simulation `world' boundaries, b/c we'll add a cylinderical container manually
    DEMSim.InstructBoxDomainBoundingBC("none", mat_type_terrain);
    // Now add a cylinderical boundary along with a bottom plane
    double bottom = -0.5;
    auto walls = DEMSim.AddExternalObject();
    walls->AddCylinder(make_float3(0), make_float3(0, 0, 1), soil_bin_diameter / 2., mat_type_terrain, 0);
    walls->AddPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_terrain);
    walls->SetFamily(1);

    // Define the GRC terrain particle templates
    DEMClumpTemplate shape_template;
    shape_template.ReadComponentFromFile(GetDEMEDataFile("clumps/triangular_flat.csv"));
    // Calculate its mass and MOI
    float terrain_density = 2.6e3;
    double clump_vol = 5.5886717;
    float mass = terrain_density * clump_vol;  // in kg or g
    float3 MOI = make_float3(1.8327927, 2.1580013, 0.77010059) * (double)2.6e3;
    // Scale the template we just created
    std::vector<std::shared_ptr<DEMClumpTemplate>> ground_particle_templates;
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
        this_template.SetVolume(clump_vol * scaling * scaling * scaling);
        ground_particle_templates.push_back(DEMSim.LoadClumpType(this_template));
    }

    // Now we load clump locations from a checkpointed file
    {
        std::cout << "Making terrain..." << std::endl;
        auto clump_xyz = DEMSim.ReadClumpXyzFromCsv("./GRC_3e6.csv");
        auto clump_quaternion = DEMSim.ReadClumpQuatFromCsv("./GRC_3e6.csv");
        std::vector<float3> in_xyz;
        std::vector<float4> in_quat;
        std::vector<std::shared_ptr<DEMClumpTemplate>> in_types;
        unsigned int t_num = 0;
        for (int i = 0; i < scales.size(); i++) {
            char t_name[20];
            sprintf(t_name, "%04d", t_num);

            auto this_type_xyz = clump_xyz[std::string(t_name)];
            auto this_type_quat = clump_quaternion[std::string(t_name)];

            size_t n_clump_this_type = this_type_xyz.size();
            std::cout << "Loading clump " << std::string(t_name) << " which has particle num: " << n_clump_this_type
                      << std::endl;
            // Prepare clump type identification vector for loading into the system (don't forget type 0 in
            // ground_particle_templates is the template for rover wheel)
            std::vector<std::shared_ptr<DEMClumpTemplate>> this_type(n_clump_this_type,
                                                                     ground_particle_templates.at(t_num));

            // Add them to the big long vector
            in_xyz.insert(in_xyz.end(), this_type_xyz.begin(), this_type_xyz.end());
            in_quat.insert(in_quat.end(), this_type_quat.begin(), this_type_quat.end());
            in_types.insert(in_types.end(), this_type.begin(), this_type.end());
            std::cout << "Added clump type " << t_num << std::endl;
            // Our template names are 0000, 0001 etc.
            t_num++;
        }
        // Now, we don't need all particles loaded... we just need a cylinderical portion out of it, to fill the soil
        // bin Remove the particles that are outside a cylinderical region
        std::vector<notStupidBool_t> elem_to_remove(in_xyz.size(), 0);
        for (size_t i = 0; i < in_xyz.size(); i++) {
            if (std::pow(in_xyz.at(i).x, 2) + std::pow(in_xyz.at(i).y, 2) >= std::pow(soil_bin_diameter / 2. - 0.02, 2))
                elem_to_remove.at(i) = 1;
        }
        in_xyz.erase(std::remove_if(
                         in_xyz.begin(), in_xyz.end(),
                         [&elem_to_remove, &in_xyz](const float3& i) { return elem_to_remove.at(&i - in_xyz.data()); }),
                     in_xyz.end());
        in_quat.erase(std::remove_if(in_quat.begin(), in_quat.end(),
                                     [&elem_to_remove, &in_quat](const float4& i) {
                                         return elem_to_remove.at(&i - in_quat.data());
                                     }),
                      in_quat.end());
        in_types.erase(std::remove_if(in_types.begin(), in_types.end(),
                                      [&elem_to_remove, &in_types](const auto& i) {
                                          return elem_to_remove.at(&i - in_types.data());
                                      }),
                       in_types.end());
        DEMClumpBatch base_batch(in_xyz.size());
        base_batch.SetTypes(in_types);
        base_batch.SetPos(in_xyz);
        base_batch.SetOriQ(in_quat);

        DEMSim.AddClumps(base_batch);

        // This batch is about 10cm thick... let's add another 2 batches, so we have something like 30cm
        float shift_dist = 0.13;
        for (int i = 0; i < 2; i++) {
            std::for_each(in_xyz.begin(), in_xyz.end(), [shift_dist](float3& xyz) { xyz.z += shift_dist; });
            DEMClumpBatch another_batch = base_batch;
            another_batch.SetPos(in_xyz);
            DEMSim.AddClumps(another_batch);
        }
    }

    // Now add a plane to compress the sample
    auto compressor = DEMSim.AddExternalObject();
    compressor->AddPlane(make_float3(0, 0, 0), make_float3(0, 0, -1), mat_type_terrain);
    compressor->SetFamily(1);
    auto compressor_tracker = DEMSim.Track(compressor);

    // Family 1 shakes, family 2 is fixed
    DEMSim.SetFamilyPrescribedLinVel(1, "0", "0",
                                     to_string_with_precision(shake_amp) + " * sin(" +
                                         to_string_with_precision(shake_speed) + " * 2 * deme::PI * t)");
    DEMSim.SetFamilyFixed(2);

    // Some inspectors
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
    auto min_z_finder = DEMSim.CreateInspector("clump_min_z");
    auto total_mass_finder = DEMSim.CreateInspector("clump_mass");
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");

    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.SetCDUpdateFreq(20);
    DEMSim.SetMaxVelocity(10.);
    DEMSim.SetExpandSafetyMultiplier(1.1);
    DEMSim.SetInitBinSize(2 * scales.at(2));
    DEMSim.Initialize();

    std::filesystem::path out_dir = std::filesystem::current_path();
    out_dir += "/Shake";
    std::filesystem::create_directory(out_dir);

    // Compress until dense enough
    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    unsigned int fps = 5;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));
    double stop_time = 2.0;  // Time before stopping shaking and measure bulk density
    unsigned int stop_steps = (unsigned int)(stop_time * (1.0 / step_size));
    float sim_end = 16.0;
    std::cout << "Output at " << fps << " FPS" << std::endl;
    float bulk_density;

    for (double t = 0; t < sim_end; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            char filename[200];
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
            DEMSim.WriteSphereFile(std::string(filename));
            std::cout << "Max system velocity: " << max_v_finder->GetValue() << std::endl;
            DEMSim.ShowThreadCollaborationStats();
        }

        if (curr_step % stop_steps == 0) {
            // Fix container, then settle a bit before measuring the void ratio
            DEMSim.ChangeFamily(1, 2);
            DEMSim.DoDynamics(0.2);
            // Measure
            float max_z = max_z_finder->GetValue();
            float min_z = min_z_finder->GetValue();
            float matter_mass = total_mass_finder->GetValue();
            float total_volume = math_PI * (soil_bin_diameter * soil_bin_diameter / 4) * (max_z - min_z);
            bulk_density = matter_mass / total_volume;
            std::cout << "Max Z is: " << max_z << std::endl;
            std::cout << "Min Z is: " << min_z << std::endl;
            std::cout << "Bulk density: " << bulk_density << std::endl;
            // Put the cap to its new position and start shaking
            compressor_tracker->SetPos(make_float3(0, 0, max_z));
            DEMSim.ChangeFamily(2, 1);
        }

        DEMSim.DoDynamics(step_size);
    }

    char cp_filename[200];
    sprintf(cp_filename, "%s/GRC_rho%6.f.csv", out_dir.c_str(), bulk_density);
    DEMSim.WriteClumpFile(std::string(cp_filename));

    std::cout << "Shake demo exiting..." << std::endl;
    return 0;
}