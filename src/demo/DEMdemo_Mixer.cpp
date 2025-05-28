//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// This demo features a mesh-represented bladed mixer interacting with clump-represented
// DEM particles.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <cstdio>
#include <chrono>
#include <filesystem>

using namespace deme;
using namespace std::filesystem;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(STEP_METRIC);
    // For general use cases, you want to set the verbosity to INFO: It's also a bit faster than STEP_METRIC.
    // DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);

    // If you don't need individual force information, then this option makes the solver run a bit faster.
    DEMSim.SetNoForceRecord();

    // E, nu, CoR, mu, Crr...
    auto mat_type_mixer = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.0}});
    auto mat_type_granular = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.2}, {"Crr", 0.0}});
    // If you don't have this line, then mu between mixer material and granular material will be 0.35 (average of the
    // two).
    DEMSim.SetMaterialPropertyPair("mu", mat_type_mixer, mat_type_granular, 0.5);

    float step_size = 5e-6;
    const double world_size = 1;
    const float chamber_height = world_size / 3.;
    const float fill_height = chamber_height;
    const float chamber_bottom = -world_size / 2.;
    const float fill_bottom = chamber_bottom + chamber_height;

    DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size);
    DEMSim.InstructBoxDomainBoundingBC("all", mat_type_granular);

    // Now add a cylinderical boundary
    auto walls = DEMSim.AddExternalObject();
    walls->AddCylinder(make_float3(0), make_float3(0, 0, 1), world_size / 2., mat_type_mixer, 0);

    auto mixer = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/internal_mixer.obj").string(), mat_type_mixer);
    std::cout << "Total num of triangles: " << mixer->GetNumTriangles() << std::endl;
    mixer->Scale(make_float3(world_size / 2, world_size / 2, chamber_height));
    mixer->SetFamily(10);
    // Define the prescribed motion of mixer
    DEMSim.SetFamilyPrescribedAngVel(10, "0", "0", "3.14159");

    float granular_rad = 0.005;
    // auto template_granular = DEMSim.LoadSphereType(granular_rad * granular_rad * granular_rad * 2.8e3 * 4 / 3 * 3.14,
    //                                                granular_rad, mat_type_granular);
    // Calculate its mass and MOI
    float mass = 2.6e3 * 5.5886717;
    float3 MOI = make_float3(2.928, 2.6029, 3.9908) * 2.6e3;
    std::shared_ptr<DEMClumpTemplate> template_granular =
        DEMSim.LoadClumpType(mass, MOI, GetDEMEDataFile("clumps/3_clump.csv"), mat_type_granular);
    template_granular->Scale(granular_rad);

    // Track the mixer
    auto mixer_tracker = DEMSim.Track(mixer);

    // Sampler to use
    HCPSampler sampler(3.f * granular_rad);
    float3 fill_center = make_float3(0, 0, fill_bottom + fill_height / 2);
    const float fill_radius = world_size / 2. - 2. * granular_rad;
    auto input_xyz = sampler.SampleCylinderZ(fill_center, fill_radius, fill_height / 2);
    DEMSim.AddClumps(template_granular, input_xyz);
    std::cout << "Total num of particles: " << input_xyz.size() << std::endl;

    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.SetCDUpdateFreq(40);
    // Mixer has a big angular velocity-contributed linear speed at its blades, this is something the solver do not
    // account for, for now. And that means it needs to be added as an estimated value.
    DEMSim.SetExpandSafetyAdder(2.0);
    // You usually don't have to worry about initial bin size. In very rare cases, init bin size is so bad that auto bin
    // size adaption is effectless, and you should notice in that case kT runs extremely slow. Then in that case setting
    // init bin size may save the simulation.
    // DEMSim.SetInitBinSize(25 * granular_rad);
    DEMSim.SetCDNumStepsMaxDriftMultipleOfAvg(1.2);
    DEMSim.SetCDNumStepsMaxDriftAheadOfAvg(6);
    DEMSim.SetSortContactPairs(true);
    // DEMSim.DisableAdaptiveBinSize();
    DEMSim.SetErrorOutVelocity(20.);
    // Force the solver to error out if something went crazy. A good practice to add them, but not necessary.
    DEMSim.SetErrorOutAvgContacts(50);

    // The two following methods set how proactive the solver is in avoiding having its bins (for contact detection) too
    // large or too small, and numbers close to 1 means more proactive. Usually, the user do not have to manually set it
    // and the default values work fine.
    DEMSim.SetAdaptiveBinSizeUpperProactivity(0.5);
    DEMSim.SetAdaptiveBinSizeLowerProactivity(0.15);

    // Initialize the simulation system
    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir /= "DemoOutput_Mixer";
    create_directory(out_dir);

    float sim_end = 10.0;
    unsigned int fps = 20;
    float frame_time = 1.0 / fps;

    // Keep tab of the max velocity in simulation
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");

    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int currframe = 0;

    mixer_tracker->SetPos(make_float3(0, 0, chamber_bottom + chamber_height / 2.0));
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (float t = 0; t < sim_end; t += frame_time) {
        std::cout << "Frame: " << currframe << std::endl;
        char filename[100], meshfilename[100], cnt_filename[100];
        sprintf(filename, "DEMdemo_output_%04d.csv", currframe);
        sprintf(meshfilename, "DEMdemo_mesh_%04d.vtk", currframe);
        sprintf(cnt_filename, "Contact_pairs_%04d.csv", currframe++);
        DEMSim.WriteSphereFile(out_dir / filename);
        DEMSim.WriteMeshFile(out_dir / meshfilename);
        // DEMSim.WriteContactFile(out_dir / cnt_filename);

        float max_v = max_v_finder->GetValue();
        std::cout << "Max velocity of any point in simulation is " << max_v << std::endl;
        std::cout << "Solver's current update frequency (auto-adapted): " << DEMSim.GetUpdateFreq() << std::endl;
        std::cout << "Average contacts each sphere has: " << DEMSim.GetAvgSphContacts() << std::endl;

        float3 mixer_moi = mixer_tracker->MOI();
        float3 mixer_acc = mixer_tracker->ContactAngAccLocal();
        float3 mixer_torque = mixer_acc * mixer_moi;
        std::cout << "Contact torque on the mixer is " << mixer_torque.x << ", " << mixer_torque.y << ", "
                  << mixer_torque.z << std::endl;

        DEMSim.DoDynamics(frame_time);
        DEMSim.ShowThreadCollaborationStats();
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds (wall time) to finish the simulation" << std::endl;

    DEMSim.ShowTimingStats();

    std::cout << "----------------------------------------" << std::endl;
    DEMSim.ShowMemStats();
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "DEMdemo_Mixer exiting..." << std::endl;
    return 0;
}
