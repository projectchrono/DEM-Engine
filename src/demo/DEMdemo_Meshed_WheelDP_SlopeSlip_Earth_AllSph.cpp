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

const double math_PI = 3.1415927;

int main() {
    std::filesystem::path out_dir = std::filesystem::current_path();
    // out_dir += "/DEMdemo_Meshed_WheelDP_SlopeSlip_Earth_AllSph_HalfSize";
    out_dir += "/DEMdemo_Meshed_WheelDP_SlopeSlip_Earth_AllSph_0.003size";
    // out_dir += "/DEMdemo_Meshed_WheelDP_SlopeSlip_Earth_AllSph_0.0002Size";
    // out_dir += "/DEMdemo_Meshed_WheelDP_SlopeSlip_Earth_KenScaled_AllSph";
    std::filesystem::create_directory(out_dir);

    // `World'
    float G_mag = 9.81;
    float step_size = 2e-6;
    double world_size_y = 0.52;
    double world_size_x = 2.04;  // 4.08;
    double world_size_z = 4.0;

    // Define the wheel geometry
    float wheel_rad = 0.25;
    float wheel_width = 0.2;
    float wheel_mass = 11.;
    float total_pressure = 22. * 9.81;
    float added_pressure = (total_pressure - wheel_mass * G_mag);
    float wheel_IYY = wheel_mass * wheel_rad * wheel_rad / 2;
    float wheel_IXX = (wheel_mass / 12) * (3 * wheel_rad * wheel_rad + wheel_width * wheel_width);

    float Slopes_deg[] = {5, 10, 15, 20, 25};
    // float Slopes_deg[] = {25};
    unsigned int run_mode = 0;
    unsigned int currframe = 0;

    for (float Slope_deg : Slopes_deg) {
        DEMSolver DEMSim;
        DEMSim.SetVerbosity(INFO);
        DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
        DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
        DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
        DEMSim.SetContactOutputContent(OWNER | FORCE | POINT);

        // E, nu, CoR, mu, Crr...
        float mu = 0.2;
        auto mat_type_wall = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.8}, {"mu", 0.9}, {"Crr", 0.00}});
        auto mat_type_wheel = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.8}, {"mu", 0.1}, {"Crr", 0.00}});
        auto mat_type_terrain =
            DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.8}, {"mu", 0.9}, {"Crr", 0.00}});

        DEMSim.InstructBoxDomainDimension(world_size_x, world_size_y, world_size_z);
        DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);

        float bottom = -0.5;
        auto bot_wall = DEMSim.AddBCPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_terrain);
        auto bot_wall_tracker = DEMSim.Track(bot_wall);

        auto wheel =
            DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/rover_wheels/Moon_rover_wheel.obj"), mat_type_wheel);
        wheel->SetMass(wheel_mass);
        wheel->SetMOI(make_float3(wheel_IXX, wheel_IYY, wheel_IXX));
        // Give the wheel a family number so we can potentially add prescription
        wheel->SetFamily(10);
        // Track it
        auto wheel_tracker = DEMSim.Track(wheel);

        float granular_rad = 0.003;
        auto sphere_type = DEMSim.LoadSphereType(2.6e3 * 4 / 3 * math_PI * granular_rad * granular_rad * granular_rad,
                                                 granular_rad, mat_type_terrain);

        float sample_height = 0.15;
        float3 sample_center = make_float3(0, 0, -0.5 + 0.03 + sample_height / 2.);
        float3 sample_half_size =
            make_float3((world_size_x - 0.06) / 2., (world_size_y - 0.06) / 2., sample_height / 2.);
        HCPSampler sampler(2.01f * granular_rad);
        auto input_xyz = sampler.SampleBox(sample_center, sample_half_size);
        DEMSim.AddClumps(sphere_type, input_xyz);

        // Now add a plane to compress the sample
        // auto compressor = DEMSim.AddExternalObject();
        // compressor->AddPlane(make_float3(0, 0, 0), make_float3(0, 0, -1), mat_type_terrain);
        // compressor->SetFamily(2);
        // auto compressor_tracker = DEMSim.Track(compressor);

        // Families' prescribed motions (Earth)
        // float w_r = 0.8 * 2.45;
        float w_r = 0.8;
        float v_ref = w_r * wheel_rad;
        double G_ang = Slope_deg * math_PI / 180.;

        double sim_end = 8.;
        // Note: this wheel is not `dictated' by our prescrption of motion because it can still fall onto the ground
        // (move freely linearly)
        DEMSim.SetFamilyPrescribedAngVel(1, "0", to_string_with_precision(w_r), "0", false);
        DEMSim.AddFamilyPrescribedAcc(1, to_string_with_precision(-added_pressure * std::sin(G_ang) / wheel_mass),
                                      "none", to_string_with_precision(-added_pressure * std::cos(G_ang) / wheel_mass));
        DEMSim.SetFamilyFixed(10);

        // Some inspectors
        auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
        auto min_z_finder = DEMSim.CreateInspector("clump_min_z");
        auto total_mass_finder = DEMSim.CreateInspector("clump_mass");
        auto partial_mass_finder = DEMSim.CreateInspector("clump_mass", "return (Z <= -0.41);");
        auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");

        float3 this_G = make_float3(-G_mag * std::sin(G_ang), 0, -G_mag * std::cos(G_ang));
        DEMSim.SetGravitationalAcceleration(this_G);

        DEMSim.SetInitTimeStep(step_size);
        DEMSim.SetCDUpdateFreq(30);
        DEMSim.SetExpandSafetyAdder(0.5);
        DEMSim.SetMaxVelocity(40);
        DEMSim.SetInitBinSizeAsMultipleOfSmallestSphere(3);
        DEMSim.Initialize();

        // Compress until dense enough
        unsigned int curr_step = 0;
        unsigned int fps = 10;
        unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));
        double frame_time = 1.0 / fps;
        unsigned int report_ps = 1000;
        unsigned int report_steps = (unsigned int)(1.0 / (report_ps * step_size));
        std::cout << "Output at " << fps << " FPS" << std::endl;

        // Put the wheel in place, then let the wheel sink in initially
        float init_x = -0.0;  // - 1.0;
        if (Slope_deg < 14) {
            init_x = -0.6;  // - 1.0;
        }
        // Put the wheel in place, then let the wheel sink in initially
        float max_z = max_z_finder->GetValue();
        wheel_tracker->SetPos(make_float3(init_x, 0, max_z + 0.03 + wheel_rad));

        // Settling
        for (double t = 0; t < 0.4; t += 0.05) {
            DEMSim.DoDynamicsThenSync(0.05);
        }

        float bulk_den_high = partial_mass_finder->GetValue() / ((-0.41 + 0.5) * world_size_x * world_size_y);
        float bulk_den_low = total_mass_finder->GetValue() / ((max_z + 0.5) * world_size_x * world_size_y);
        std::cout << "Bulk density high: " << bulk_den_high << std::endl;
        std::cout << "Bulk density low: " << bulk_den_low << std::endl;

        DEMSim.ChangeFamily(10, 1);

        bool start_measure = false;
        for (double t = 0; t < sim_end; t += step_size, curr_step++) {
            if (curr_step % out_steps == 0) {
                char filename[200], meshname[200];
                std::cout << "Outputting frame: " << currframe << std::endl;
                sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
                sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), currframe);
                DEMSim.WriteSphereFile(std::string(filename));
                DEMSim.WriteMeshFile(std::string(meshname));
                DEMSim.ShowThreadCollaborationStats();
                currframe++;
            }

            if (t >= 2. && !start_measure) {
                start_measure = true;
            }

            if (curr_step % report_steps == 0 && start_measure) {
                float3 V = wheel_tracker->Vel();
                // float Vup = V.x * std::cos(G_ang) + V.z * std::sin(G_ang);
                float slip = 1.0 - V.x / (w_r * wheel_rad);
                std::cout << "Current slope: " << Slope_deg << std::endl;
                std::cout << "Time: " << t << std::endl;
                // std::cout << "Distance: " << dist_moved << std::endl;
                std::cout << "X: " << wheel_tracker->Pos().x << std::endl;
                std::cout << "V: " << V.x << std::endl;
                std::cout << "Slip: " << slip << std::endl;
                std::cout << "Max system velocity: " << max_v_finder->GetValue() << std::endl;
            }

            DEMSim.DoDynamics(step_size);
        }

        run_mode++;
        DEMSim.ShowTimingStats();
        DEMSim.ShowAnomalies();
    }

    std::cout << "DEMdemo_WheelDP_SlopeSlip demo exiting..." << std::endl;
    return 0;
}