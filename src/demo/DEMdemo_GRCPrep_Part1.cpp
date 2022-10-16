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
    double world_y_size = 0.99;
    DEMSim.InstructBoxDomainDimension(world_y_size, world_y_size, world_y_size);
    // Add 5 bounding planes around the simulation world, and leave the top open
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);
    float bottom = -0.5;
    DEMSim.AddBCPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_terrain);

    // Then the ground particle template
    DEMClumpTemplate shape_template;
    shape_template.ReadComponentFromFile((GET_SOURCE_DATA_PATH() / "clumps/triangular_flat.csv").string());
    // Calculate its mass and MOI
    float mass = 2.6e3 * 5.5886717 * kg_g_conv;  // in kg or g
    float3 MOI = make_float3(1.8327927, 2.1580013, 0.77010059) * 2.6e3 * kg_g_conv;
    // Scale the template we just created
    std::vector<std::shared_ptr<DEMClumpTemplate>> ground_particle_templates;
    // std::vector<double> scales = {0.0014, 0.00063, 0.00033, 0.00022, 0.00015, 0.00009};
    std::vector<double> scales = {0.00063, 0.00033, 0.00022, 0.00015, 0.00009};
    std::for_each(scales.begin(), scales.end(), [](double& r) { r *= 20.; });
    unsigned int t_num = 0;
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

        // Give these templates names, 0000, 0001 etc.
        char t_name[20];
        sprintf(t_name, "%04d", t_num);
        this_template.AssignName(std::string(t_name));
        ground_particle_templates.push_back(DEMSim.LoadClumpType(this_template));
        t_num++;
    }

    // std::vector<double> weight_perc = {0.18, 0.20, 0.14, 0.16, 0.17, 0.15};
    std::vector<double> weight_perc = {0.2439, 0.1707, 0.1951, 0.2073, 0.1829};
    std::vector<double> grain_perc;
    for (int i = 0; i < scales.size(); i++) {
        grain_perc.push_back(weight_perc.at(i) / std::pow(scales.at(i), 3));
    }
    {
        double tmp = vector_sum(grain_perc);
        std::for_each(grain_perc.begin(), grain_perc.end(), [tmp](double& p) { p /= tmp; });
        std::cout << "Percentage of grains add up to " << vector_sum(grain_perc) << std::endl;
    }
    std::random_device r;
    std::default_random_engine e1(r());
    // Distribution that defines different weights (17, 10, etc.) for numbers.
    std::discrete_distribution<int> discrete_dist(grain_perc.begin(), grain_perc.end());

    // Sampler to use
    HCPSampler sampler(scales.at(0) * 2.2);

    // Make ready for simulation
    float step_size = 5e-7;
    DEMSim.SetCoordSysOrigin("center");
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEMSim.SetCDUpdateFreq(30);
    // DEMSim.SetExpandFactor(1e-3);
    DEMSim.SetMaxVelocity(15.);
    DEMSim.SetExpandSafetyParam(1.2);
    DEMSim.SetInitBinSize(scales.at(2));
    DEMSim.Initialize();

    float time_end = 10.0;
    unsigned int fps = 20;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    path out_dir = current_path();
    out_dir += "/DemoOutput_GRCPrep_Part1";
    create_directory(out_dir);
    unsigned int currframe = 0;
    unsigned int curr_step = 0;

    float sample_halfheight = 0.4;
    float sample_halfwidth_x = (world_y_size * 0.96) / 2;
    float sample_halfwidth_y = (world_y_size * 0.96) / 2;
    float offset_z = bottom + sample_halfheight + 0.15;
    float settle_frame_time = 0.2;
    float settle_batch_time = 2.0;
    while (DEMSim.GetNumClumps() < 0.25e6) {
        DEMSim.ClearCache();
        float3 sample_center = make_float3(0, 0, offset_z);
        std::vector<std::shared_ptr<DEMClumpTemplate>> heap_template_in_use;
        std::vector<unsigned int> heap_family;
        // Sample and add heap particles
        auto heap_particles_xyz =
            sampler.SampleBox(sample_center, make_float3(sample_halfwidth_x, sample_halfwidth_y, sample_halfheight));
        for (unsigned int i = 0; i < heap_particles_xyz.size(); i++) {
            int ind = std::round(discrete_dist(e1));
            heap_template_in_use.push_back(ground_particle_templates.at(ind));
            heap_family.push_back(ind);
        }
        auto heap_particles = DEMSim.AddClumps(heap_template_in_use, heap_particles_xyz);
        // Give ground particles a small initial velocity so they `collapse' at the start of the simulation
        heap_particles->SetVel(make_float3(0.00, 0, -0.05));
        heap_particles->SetFamilies(heap_family);
        DEMSim.UpdateClumps();
        std::cout << "Current number of clumps: " << DEMSim.GetNumClumps() << std::endl;

        // Allow for some settling
        // Must DoDynamicsThenSync (not DoDynamics), as adding entities to the simulation is only allowed at a sync-ed
        // point of time.
        for (float t = 0; t < settle_batch_time; t += settle_frame_time) {
            std::cout << "Frame: " << currframe << std::endl;
            char filename[200];
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
            DEMSim.WriteSphereFile(std::string(filename));
            DEMSim.DoDynamicsThenSync(settle_frame_time);
        }

        DEMSim.ShowThreadCollaborationStats();
    }

    // Settle for some time more
    DEMSim.DoDynamicsThenSync(1.0);

    char cp_filename[200];
    sprintf(cp_filename, "%s/GRC_3e5.csv", out_dir.c_str());
    DEMSim.WriteClumpFile(std::string(cp_filename));

    DEMSim.ShowThreadCollaborationStats();
    DEMSim.ClearThreadCollaborationStats();

    char cnt_filename[200];
    sprintf(cnt_filename, "%s/Contact_pairs_3e5.csv", out_dir.c_str());
    DEMSim.WriteContactFile(std::string(cnt_filename));

    std::cout << "DEMdemo_GRCPrep_Part1 exiting..." << std::endl;
    return 0;
}
