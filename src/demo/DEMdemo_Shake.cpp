//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Put particles in a jar, the shake the jar in the hope to change the bulk density.
// =============================================================================

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
    DEMSim.SetContactOutputContent({"OWNER", "FORCE", "POINT"});

    // If you don't need individual force information, then this option makes the solver run a bit faster.
    DEMSim.SetNoForceRecord();

    // E, nu, CoR, mu, Crr...
    auto mat_type_cone = DEMSim.LoadMaterial({{"E", 5e7}, {"nu", 0.3}, {"CoR", 0.5}});
    auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 5e7}, {"nu", 0.3}, {"CoR", 0.5}});
    DEMSim.UseFrictionlessHertzianModel();

    float shake_amp = 0.1;
    float shake_speed = 2;  // Num of periods per second
    float step_size = 1e-5;
    double world_size = 2.;
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

    // Define the terrain particle templates. Two types of clumps.
    float terrain_density = 2.6e3;
    double clump_vol1 = 5.5886717;
    float mass1 = terrain_density * clump_vol1;
    float3 MOI1 = make_float3(2.928, 2.6029, 3.9908) * terrain_density;
    double clump_vol2 = 2.1670011;
    float mass2 = terrain_density * clump_vol2;
    float3 MOI2 = make_float3(0.57402126, 0.60616378, 0.92890173) * terrain_density;
    // Then load them to system
    std::shared_ptr<DEMClumpTemplate> template_1 =
        DEMSim.LoadClumpType(mass1, MOI1, GetDEMEDataFile("clumps/triangular_flat.csv"), mat_type_terrain);
    std::shared_ptr<DEMClumpTemplate> template_2 =
        DEMSim.LoadClumpType(mass2, MOI2, GetDEMEDataFile("clumps/triangular_flat_6comp.csv"), mat_type_terrain);
    // Decide the scalings of the templates we just created (so that they are... like particles, not rocks)
    double scale1 = 0.015;
    double scale2 = 0.004;
    template_1->Scale(scale1);
    template_2->Scale(scale2);

    // Sampler to sample
    GridSampler sampler1(scale1 * 3.);
    float fill_height = 0.5;
    float3 fill_center = make_float3(0, 0, bottom + fill_height / 2);
    float fill_radius = soil_bin_diameter / 2. - scale1 * 2.;
    auto input_xyz = sampler1.SampleCylinderZ(fill_center, fill_radius, fill_height / 2 - scale1 * 2.);
    DEMSim.AddClumps(template_1, input_xyz);
    // Another batch...
    GridSampler sampler2(scale2 * 3.);
    fill_center += make_float3(0, 0, fill_height);
    fill_radius = soil_bin_diameter / 2. - scale2 * 2.;
    input_xyz = sampler2.SampleCylinderZ(fill_center, fill_radius, fill_height / 2 - scale2 * 2.);
    DEMSim.AddClumps(template_2, input_xyz);

    // Now add a `cap' to the container when we shake it
    auto compressor = DEMSim.AddExternalObject();
    compressor->AddPlane(make_float3(0, 0, 0), make_float3(0, 0, -1), mat_type_terrain);
    compressor->SetFamily(1);
    auto compressor_tracker = DEMSim.Track(compressor);

    // Family 1 shakes, family 2 is fixed
    std::string shake_pattern_x = to_string_with_precision(shake_amp) + " * sin(" +
                                  to_string_with_precision(shake_speed) + " * 2 * deme::PI * t)";
    DEMSim.SetFamilyPrescribedLinVel(1, shake_pattern_x, "0", shake_pattern_x);
    DEMSim.SetFamilyFixed(2);

    // Some inspectors
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
    auto min_z_finder = DEMSim.CreateInspector("clump_min_z");
    auto total_mass_finder = DEMSim.CreateInspector("clump_mass");
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");

    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.Initialize();

    std::filesystem::path out_dir = std::filesystem::current_path();
    out_dir /= "DemoOutput_Shake";
    std::filesystem::create_directory(out_dir);

    // Settle phase
    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    unsigned int fps = 10;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));
    compressor_tracker->SetPos(make_float3(0, 0, max_z_finder->GetValue()));
    for (double t = 0; t < 0.5; t += 0.1) {
        char filename[100];
        sprintf(filename, "DEMdemo_output_%04d.csv", currframe++);
        DEMSim.WriteSphereFile(out_dir / filename);
        DEMSim.DoDynamicsThenSync(0.1);
    }

    double stop_time = 2.0;  // Time before stopping shaking and measure bulk density
    unsigned int stop_steps = (unsigned int)(stop_time * (1.0 / step_size));
    float sim_end = 6.0;
    std::cout << "Output at " << fps << " FPS" << std::endl;
    float bulk_density;

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (double t = 0; t < sim_end; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            char filename[100];
            sprintf(filename, "DEMdemo_output_%04d.csv", currframe++);
            DEMSim.WriteSphereFile(out_dir / filename);
            std::cout << "Max system velocity: " << max_v_finder->GetValue() << std::endl;
            DEMSim.ShowThreadCollaborationStats();
        }

        if (curr_step % stop_steps == 0) {
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
        }

        DEMSim.DoDynamics(step_size);
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds (wall time) to finish the simulation" << std::endl;

    // Output the final configuration of the clumps as a file. This is just a demonstration. This particular
    // configuration is not that useful as no other demos actually use it, unlike the GRC-1 soil.
    char cp_filename[100];
    sprintf(cp_filename, "settled_rho%4.f.csv", bulk_density);
    DEMSim.WriteClumpFile(out_dir / cp_filename);

    std::cout << "Shake demo exiting..." << std::endl;
    return 0;
}