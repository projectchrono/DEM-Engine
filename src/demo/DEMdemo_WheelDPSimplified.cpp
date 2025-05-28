//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A simplified wheel drawbar-pull test, featuring Curiosity wheel geometry and
// terrain particles represented by irregular DEM elements. Unlike WheelDP, this
// demo does not have a prerequisite and can run on its own, and the terrain is
// simpler since there is only one type of particle (3-sphere clump). The velocity
// and angular velocity of the wheel are prescribed and the terrain force on the
// wheel is measured (we only test one slip case in this demo, to make it faster).
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

const double math_PI = 3.1415927;

int main() {
    std::filesystem::path out_dir = std::filesystem::current_path();
    out_dir /= "DemoOutput_WheelDPSimplified";
    std::filesystem::create_directory(out_dir);

    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.SetContactOutputContent({"OWNER", "FORCE", "POINT"});

    // If you don't need individual force information, then this option makes the solver run a bit faster.
    DEMSim.SetNoForceRecord();

    // E, nu, CoR, mu, Crr...
    auto mat_type_wheel = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.01}});
    auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.4}, {"mu", 0.5}, {"Crr", 0.01}});
    // If you don't have this line, then mu between drum material and granular material will be the average of the
    // two.
    DEMSim.SetMaterialPropertyPair("mu", mat_type_wheel, mat_type_terrain, 0.8);
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_wheel, mat_type_terrain, 0.6);

    // `World'
    float G_mag = 9.81;
    float step_size = 5e-6;
    double world_size_y = 1.;
    double world_size_x = 2.;
    double world_size_z = 2.;
    DEMSim.InstructBoxDomainDimension(world_size_x, world_size_y, world_size_z);
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);
    float bottom = -0.5;
    auto bot_wall = DEMSim.AddBCPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_terrain);
    auto bot_wall_tracker = DEMSim.Track(bot_wall);

    // Define the wheel geometry
    float wheel_rad = 0.25;
    float wheel_width = 0.2;
    float wheel_weight = 100.;
    float wheel_mass = wheel_weight / G_mag;
    float total_pressure = 200.0;
    float added_pressure = total_pressure - wheel_weight;
    float wheel_IYY = wheel_mass * wheel_rad * wheel_rad / 2;
    float wheel_IXX = (wheel_mass / 12) * (3 * wheel_rad * wheel_rad + wheel_width * wheel_width);
    auto wheel =
        DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/rover_wheels/viper_wheel_right.obj"), mat_type_wheel);
    wheel->SetMass(wheel_mass);
    wheel->SetMOI(make_float3(wheel_IXX, wheel_IYY, wheel_IXX));
    // Give the wheel a family number so we can potentially add prescription
    wheel->SetFamily(1);
    // Track it
    auto wheel_tracker = DEMSim.Track(wheel);

    // Define the terrain particle templates
    // Calculate its mass and MOI
    float terrain_density = 2.6e3;
    float volume1 = 4.2520508;
    float mass1 = terrain_density * volume1;
    float3 MOI1 = make_float3(1.6850426, 1.6375114, 2.1187753) * terrain_density;
    // Scale the template we just created
    double scale = 0.02;
    // Then load it to system
    std::shared_ptr<DEMClumpTemplate> my_template =
        DEMSim.LoadClumpType(mass1, MOI1, GetDEMEDataFile("clumps/triangular_flat.csv"), mat_type_terrain);
    // Now scale the template
    // Note the mass and MOI are also scaled in the process, automatically. But if you are not happy with this, you
    // can always manually change mass and MOI afterwards.
    my_template->Scale(scale);
    // Give these templates names, 0000, 0001 etc.
    char t_name[20];
    sprintf(t_name, "%04d", 0);
    my_template->AssignName(std::string(t_name));

    // Sampler to use
    HCPSampler sampler(scale * 2.7);
    float sample_halfheight = 0.25;
    float sample_halfwidth_x = (world_size_x * 0.95) / 2;
    float sample_halfwidth_y = (world_size_y * 0.95) / 2;
    float offset_z = bottom + sample_halfheight + 0.03;
    // Sample initial particles
    float3 sample_center = make_float3(0, 0, offset_z);
    auto terrain_particles_xyz =
        sampler.SampleBox(sample_center, make_float3(sample_halfwidth_x, sample_halfwidth_y, sample_halfheight));
    std::vector<std::shared_ptr<DEMClumpTemplate>> terrain_template_in_use(terrain_particles_xyz.size(), my_template);
    std::vector<unsigned int> heap_family(terrain_particles_xyz.size(), 0);
    auto terrain_particles = DEMSim.AddClumps(terrain_template_in_use, terrain_particles_xyz);
    // Give ground particles a small initial velocity so they `collapse' at the start of the simulation
    terrain_particles->SetVel(make_float3(0.00, 0, -0.05));
    terrain_particles->SetFamilies(heap_family);
    std::cout << "Current number of clumps: " << terrain_particles_xyz.size() << std::endl;

    // Families' prescribed motions
    float w_r = math_PI / 4;
    float v_ref = w_r * wheel_rad;

    double sim_end = 6.;
    // Note: this wheel is not `dictated' by our prescrption of motion because it can still fall onto the ground
    // (move freely linearly)
    DEMSim.SetFamilyPrescribedAngVel(1, "0", to_string_with_precision(w_r), "0", false);
    // An extra force (acceleration) is addedd to simulate the load that the wheel carries
    DEMSim.AddFamilyPrescribedAcc(1, "none", "none", to_string_with_precision(-added_pressure / wheel_mass));
    // `Real sim' family number
    DEMSim.SetFamilyPrescribedAngVel(2, "0", to_string_with_precision(w_r), "0", false);
    // Note: this wheel is not `dictated' by our prescrption of motion (hence the false argument), because it
    // can sink into the ground (move on Z dir); but its X and Y motions are explicitly controlled.
    // This one says when the experiment is going, the slip ratio is 0.5 (by our prescribing linear and angular vel)
    DEMSim.SetFamilyPrescribedLinVel(2, to_string_with_precision(v_ref * 0.5), "0", "none", false);
    // An extra force (acceleration) is addedd to simulate the load that the wheel carries
    DEMSim.AddFamilyPrescribedAcc(2, "none", "none", to_string_with_precision(-added_pressure / wheel_mass));

    // Some inspectors
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
    auto min_z_finder = DEMSim.CreateInspector("clump_min_z");
    auto total_mass_finder = DEMSim.CreateInspector("clump_mass");
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");

    // Make ready for simulation
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -G_mag));
    // Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    // wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    // happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(20.);
    // Error out vel is used to force the simulation to abort when something goes wrong.
    DEMSim.SetErrorOutVelocity(35.);
    DEMSim.SetExpandSafetyMultiplier(1.);
    //// TODO: Implement the better CDUpdateFreq adapt algorithm that overperforms the current one...
    DEMSim.SetCDUpdateFreq(40);
    DEMSim.DisableAdaptiveUpdateFreq();
    DEMSim.Initialize();

    unsigned int fps = 10;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));
    unsigned int curr_step = 0;
    unsigned int currframe = 0;
    double frame_time = 1.0 / fps;
    unsigned int report_ps = 100;
    unsigned int report_steps = (unsigned int)(1.0 / (report_ps * step_size));
    std::cout << "Output at " << fps << " FPS" << std::endl;

    // Put the wheel in place, then let the wheel sink in initially
    float max_z = max_z_finder->GetValue();
    wheel_tracker->SetPos(make_float3(-0.45, 0, max_z + 0.03 + wheel_rad));
    for (double t = 0; t < 1.; t += frame_time) {
        char filename[100], meshname[100];
        std::cout << "Outputting frame: " << currframe << std::endl;
        sprintf(filename, "DEMdemo_output_%04d.csv", currframe);
        sprintf(meshname, "DEMdemo_mesh_%04d.vtk", currframe++);
        DEMSim.WriteSphereFile(out_dir / filename);
        DEMSim.WriteMeshFile(out_dir / meshname);

        DEMSim.DoDynamics(frame_time);
    }

    // Switch wheel from free fall into DP test
    DEMSim.DoDynamicsThenSync(0);
    // You don't have to do this! I am just testing if sync-ing it twice breaks the system.
    DEMSim.DoDynamicsThenSync(0);
    DEMSim.ChangeFamily(1, 2);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    for (double t = 0; t < sim_end; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            char filename[100], meshname[100];
            std::cout << "Outputting frame: " << currframe << std::endl;
            sprintf(filename, "DEMdemo_output_%04d.csv", currframe);
            sprintf(meshname, "DEMdemo_mesh_%04d.vtk", currframe++);
            DEMSim.WriteSphereFile(out_dir / filename);
            DEMSim.WriteMeshFile(out_dir / meshname);
            DEMSim.ShowThreadCollaborationStats();
        }

        if (curr_step % report_steps == 0) {
            float3 forces = wheel_tracker->ContactAcc();
            forces *= wheel_mass;
            std::cout << "Time: " << t << std::endl;
            std::cout << "Force on wheel: " << forces.x << ", " << forces.y << ", " << forces.z << std::endl;
            std::cout << "Drawbar pull coeff: " << forces.x / total_pressure << std::endl;
            std::cout << "Max system velocity: " << max_v_finder->GetValue() << std::endl;
        }

        DEMSim.DoDynamics(step_size);
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds (wall time) to finish the simulation" << std::endl;

    DEMSim.ShowTimingStats();
    std::cout << "----------------------------------------" << std::endl;
    DEMSim.ShowMemStats();
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "WheelDPSimpilified demo exiting..." << std::endl;
    return 0;
}