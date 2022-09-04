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

using namespace sgps;
using namespace std::filesystem;

int main() {
    DEMSolver DEM_sim;
    DEM_sim.SetVerbosity(INFO);
    DEM_sim.SetOutputFormat(DEM_OUTPUT_FORMAT::CSV);
    DEM_sim.SetOutputContent(DEM_OUTPUT_CONTENT::ABSV);

    // Define materials
    auto mat_type_terrain = DEM_sim.LoadMaterial(2e9, 0.3, 0.6, 0.5, 0.01);
    auto mat_type_wheel = DEM_sim.LoadMaterial(1e9, 0.3, 0.5, 0.5, 0.01);

    // Define the simulation world
    double world_size = 1.5;
    DEM_sim.InstructBoxDomainNumVoxel(22, 21, 21, world_size / std::pow(2, 16) / std::pow(2, 21));
    // Add 5 bounding planes around the simulation world, and leave the top open
    DEM_sim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);

    // Define the wheel geometry
    float wheel_rad = 0.25;
    float wheel_width = 0.25;
    float wheel_mass = 10.0;
    // Our shelf wheel geometry is lying flat on ground with z being the axial direction
    float wheel_IYY = wheel_mass * wheel_rad * wheel_rad / 2;
    float wheel_IXX = (wheel_mass / 12) * (3 * wheel_rad * wheel_rad + wheel_width * wheel_width);
    auto wheel_template = DEM_sim.LoadClumpType(wheel_mass, make_float3(wheel_IXX, wheel_IYY, wheel_IXX),
                                                "./data/clumps/ViperWheelSimple.csv", mat_type_wheel);
    // The file contains no wheel particles size info, so let's manually set them
    wheel_template->radii = std::vector<float>(wheel_template->nComp, 0.01);
    // This wheel template is `lying down', but our reported MOI info is assuming it's in a position to roll along X
    // direction. Let's make it clear its principal axes is not what we used to report its component sphere relative
    // positions.
    wheel_template->InformCentroidPrincipal(make_float3(0), make_float4(0.7071, 0.7071, 0, 0));

    // Then the ground particle template
    DEMClumpTemplate ellipsoid_template;
    ellipsoid_template.ReadComponentFromFile("./data/clumps/ellipsoid_2_1_1.csv");
    // Calculate its mass and MOI
    float mass = 2.6e3 * 4. / 3. * PI * 2 * 1 * 1;
    float3 MOI = make_float3(1. / 5. * mass * (1 * 1 + 2 * 2), 1. / 5. * mass * (1 * 1 + 2 * 2),
                             1. / 5. * mass * (1 * 1 + 1 * 1));
    // Scale the template we just created
    double scaling = 0.005;
    ellipsoid_template.mass = mass * scaling * scaling * scaling;
    ellipsoid_template.MOI = MOI * scaling * scaling * scaling * scaling * scaling;
    std::for_each(ellipsoid_template.radii.begin(), ellipsoid_template.radii.end(),
                  [scaling](float& r) { r *= scaling; });
    std::for_each(ellipsoid_template.relPos.begin(), ellipsoid_template.relPos.end(),
                  [scaling](float3& r) { r *= scaling; });
    ellipsoid_template.materials =
        std::vector<std::shared_ptr<DEMMaterial>>(ellipsoid_template.nComp, mat_type_terrain);
    auto ground_particle_template = DEM_sim.LoadClumpType(ellipsoid_template);

    // Instantiate this wheel
    auto wheel = DEM_sim.AddClumps(wheel_template, make_float3(-1.2, 0, 0.4));
    // Give the wheel a family number so we can potentially add prescription
    wheel->SetFamily(10);
    // Note that the added constant ang vel is wrt the wheel's own principal coord system
    DEM_sim.SetFamilyPrescribedAngVel(10, "0", "2.0", "0", false);

    // Sample and add ground particles
    float3 sample_center = make_float3(0, 0, -0.3);
    float sample_halfheight = 0.4;
    float sample_halfwidth_x = 1.45;
    float sample_halfwidth_y = 0.725;
    auto ground_particles_xyz =
        DEMBoxGridSampler(sample_center, make_float3(sample_halfwidth_x, sample_halfwidth_y, sample_halfheight),
                          scaling * std::cbrt(2.0) * 2.1, scaling * std::cbrt(2.0) * 2.1, scaling * 2 * 2.1);
    auto ground_particles = DEM_sim.AddClumps(ground_particle_template, ground_particles_xyz);
    // Give ground particles a small initial velocity so they `collapse' at the start of the simulation
    ground_particles->SetVel(make_float3(0.002, 0, 0));

    // Create an absv inspector
    auto max_v_finder = DEM_sim.CreateInspector("clump_max_absv");

    // Make ready for simulation
    float step_size = 5e-6;
    DEM_sim.SetCoordSysOrigin("center");
    DEM_sim.SetInitTimeStep(step_size);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEM_sim.SetCDUpdateFreq(20);
    // DEM_sim.SetExpandFactor(1e-3);
    DEM_sim.SetMaxVelocity(5.);
    DEM_sim.SetExpandSafetyParam(1.2);
    DEM_sim.SetInitBinSize(scaling / 1.5);
    DEM_sim.Initialize();

    float time_end = 10.0;
    unsigned int fps = 20;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    path out_dir = current_path();
    out_dir += "/DEMdemo_RoverWheel";
    create_directory(out_dir);
    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (double t = 0; t < (double)time_end; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            std::cout << "Frame: " << currframe << std::endl;
            DEM_sim.ShowThreadCollaborationStats();
            char filename[100];
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
            DEM_sim.WriteSphereFile(std::string(filename));
            float max_v = max_v_finder->GetValue();
            std::cout << "Max velocity of any point in simulation is " << max_v << std::endl;
        }

        DEM_sim.DoDynamics(step_size);
        // We can query info out of this drum, since it is tracked
        // float3 drum_pos = Drum_tracker->Pos();
        // float3 drum_angVel = Drum_tracker->AngVelLocal();
        // std::cout << "Position of the drum: " << drum_pos.x << ", " << drum_pos.y << ", " << drum_pos.z
        //           << std::endl;
        // std::cout << "Angular velocity of the drum: " << drum_angVel.x << ", " << drum_angVel.y << ", "
        //           << drum_angVel.z << std::endl;
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << (time_sec.count()) / time_end << " seconds (wall time) to finish 1 seconds' simulation" << std::endl;
    DEM_sim.ShowThreadCollaborationStats();
    DEM_sim.ClearThreadCollaborationStats();

    std::cout << "DEMdemo_RoverWheel exiting..." << std::endl;
    return 0;
}
