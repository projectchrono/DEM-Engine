//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

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

using namespace sgps;
using namespace std::filesystem;

int main() {
    DEMSolver DEM_sim;
    DEM_sim.SetVerbosity(INFO);
    DEM_sim.SetOutputFormat(DEM_OUTPUT_FORMAT::CSV);
    // DEM_sim.SetOutputContent(DEM_OUTPUT_CONTENT::FAMILY);

    srand(759);

    //
    float kg_g_conv = 1;
    // Define materials
    auto mat_type_terrain = DEM_sim.LoadMaterialType(2e9 * kg_g_conv, 0.3, 0.3, 0.5, 0.0);
    auto mat_type_wheel = DEM_sim.LoadMaterialType(1e9 * kg_g_conv, 0.3, 0.3, 0.5, 0.0);

    // Define the simulation world
    double world_y_size = 0.99;
    DEM_sim.InstructBoxDomainNumVoxel(21, 21, 22, world_y_size / std::pow(2, 16) / std::pow(2, 21));
    // Add 5 bounding planes around the simulation world, and leave the top open
    DEM_sim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);
    float bottom = -0.5;
    DEM_sim.AddBCPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_terrain);

    // Define the wheel geometry
    float wheel_rad = 0.25;
    float wheel_width = 0.2;
    float wheel_mass = 10.0 * kg_g_conv;  // in g
    // Our shelf wheel geometry is lying flat on ground with z being the axial direction
    float wheel_IZZ = wheel_mass * wheel_rad * wheel_rad / 2;
    float wheel_IXX = (wheel_mass / 12) * (3 * wheel_rad * wheel_rad + wheel_width * wheel_width);
    auto wheel_template = DEM_sim.LoadClumpType(wheel_mass, make_float3(wheel_IXX, wheel_IXX, wheel_IZZ),
                                                "./data/clumps/ViperWheelSimple.csv", mat_type_wheel);
    // The file contains no wheel particles size info, so let's manually set them
    wheel_template->radii = std::vector<float>(wheel_template->nComp, 0.01);

    // Then the ground particle template
    DEMClumpTemplate shape_template;
    shape_template.ReadComponentFromFile("./data/clumps/triangular_flat.csv");
    // Calculate its mass and MOI
    float mass = 2.6e3 * 5.5886717 * kg_g_conv;  // in g
    float3 MOI = make_float3(1.8327927, 2.1580013, 0.77010059) * 2.6e3 * kg_g_conv;
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
        ground_particle_templates.push_back(DEM_sim.LoadClumpType(this_template));
    }

    std::vector<double> weight_perc = {0.18, 0.20, 0.14, 0.16, 0.17, 0.15};
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

    // // Instantiate this wheel
    // auto wheel = DEM_sim.AddClumps(wheel_template, make_float3(-0.5, 0, bottom + 0.3));
    // // Let's `flip' the wheel's initial position so... yeah, it's like how wheel operates normally
    // wheel->SetOriQ(make_float4(0.7071, 0.7071, 0, 0));
    // // Give the wheel a family number so we can potentially add prescription
    // wheel->SetFamily(101);
    // // Note that the added constant ang vel is wrt the wheel's own coord sys, therefore it should be on the z axis:
    // in
    // // line with the orientation at which the wheel is loaded into the simulation system.
    // DEM_sim.SetFamilyPrescribedAngVel(100, "0", "0", "-0.5", false);
    // DEM_sim.SetFamilyFixed(101);
    // DEM_sim.DisableContactBetweenFamilies(101, 0);
    // DEM_sim.DisableContactBetweenFamilies(101, 1);
    // DEM_sim.DisableContactBetweenFamilies(101, 2);
    // DEM_sim.DisableContactBetweenFamilies(101, 3);
    // DEM_sim.DisableContactBetweenFamilies(101, 4);
    // DEM_sim.DisableContactBetweenFamilies(101, 5);
    // DEM_sim.DisableContactBetweenFamilies(101, 101);
    // DEM_sim.DisableContactBetweenFamilies(100, 100);
    DEM_sim.InsertFamily(0);
    DEM_sim.InsertFamily(1);
    DEM_sim.InsertFamily(2);
    DEM_sim.InsertFamily(3);
    DEM_sim.InsertFamily(4);
    DEM_sim.InsertFamily(5);

    // Make ready for simulation
    float step_size = 5e-7;
    DEM_sim.SetCoordSysOrigin("center");
    DEM_sim.SetInitTimeStep(step_size);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEM_sim.SetCDUpdateFreq(10);
    // DEM_sim.SetExpandFactor(1e-3);
    DEM_sim.SetMaxVelocity(15.);
    DEM_sim.SetExpandSafetyParam(1.2);
    DEM_sim.SetInitBinSize(scales.at(2));
    DEM_sim.Initialize();

    float time_end = 10.0;
    unsigned int fps = 20;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    path out_dir = current_path();
    out_dir += "/DEMdemo_GRCPrep";
    create_directory(out_dir);
    unsigned int currframe = 0;
    unsigned int curr_step = 0;

    float sample_halfheight = 0.4;
    float sample_halfwidth_x = (world_y_size * 0.96) / 2;
    float sample_halfwidth_y = (world_y_size * 0.96) / 2;
    float offset_z = bottom + sample_halfheight + 0.15;
    float settle_frame_time = 0.2;
    float settle_batch_time = 2.0;
    while (DEM_sim.GetNumClumps() < 0.2e6) {
        DEM_sim.ClearCache();
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
        auto heap_particles = DEM_sim.AddClumps(heap_template_in_use, heap_particles_xyz);
        // Give ground particles a small initial velocity so they `collapse' at the start of the simulation
        heap_particles->SetVel(make_float3(0.00, 0, -0.05));
        heap_particles->SetFamilies(heap_family);
        DEM_sim.UpdateClumps();
        std::cout << "Current number of clumps: " << DEM_sim.GetNumClumps() << std::endl;

        // Allow for some settling
        // Must DoDynamicsThenSync (not DoDynamics), as adding entities to the simulation is only allowed at a sync-ed
        // point of time.
        for (float t = 0; t < settle_batch_time; t += settle_frame_time) {
            std::cout << "Frame: " << currframe << std::endl;
            char filename[200];
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
            DEM_sim.WriteSphereFile(std::string(filename));
            DEM_sim.DoDynamicsThenSync(settle_frame_time);
        }

        DEM_sim.ShowThreadCollaborationStats();
    }
    char cp_filename[200];
    sprintf(cp_filename, "%s/Final_xyz.csv", out_dir.c_str());
    DEM_sim.WriteClumpFile(std::string(cp_filename));

    // DEM_sim.DoDynamicsThenSync(0.3);
    // DEM_sim.ChangeFamily(101, 100);
    // for (double t = 0; t < (double)time_end; t += step_size, curr_step++) {
    //     if (curr_step % out_steps == 0) {
    //         std::cout << "Frame: " << currframe << std::endl;
    //         DEM_sim.ShowThreadCollaborationStats();
    //         char filename[100];
    //         sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
    //         DEM_sim.WriteSphereFile(std::string(filename));
    //         currframe++;
    //     }

    //     DEM_sim.DoDynamics(step_size);
    // }

    DEM_sim.ShowThreadCollaborationStats();
    DEM_sim.ClearThreadCollaborationStats();

    std::cout << "DEMdemo_GRCPrep exiting..." << std::endl;
    // TODO: add end-game report APIs
    return 0;
}
