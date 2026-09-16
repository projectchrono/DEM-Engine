//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// This demo is the mixer example with the granular clumps replaced by
// cube-shaped mesh particles. Mesh universal contact is enabled so the cubes
// can contact the mixer, the cylindrical chamber, and each other as meshes.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/Samplers.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>

using namespace deme;
using namespace std::filesystem;

int main() {
    std::cout << "==== DEME demo/test: DEMdemo_MixerCubes ====" << std::endl;
    std::cout << "========================================" << std::endl;
    DEMSolver DEMSim;
    DEMSim.SetVerbosity("METRIC");
    // For general use cases, you want to set the verbosity to INFO: It's also a bit faster than "METRIC".
    // DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.SetMeshOutputContent(MESH_OUTPUT_CONTENT::ABSV | MESH_OUTPUT_CONTENT::OWNER);
    DEMSim.SetMeshUniversalContact(true);
    DEMSim.SetMeshParticlesLowPoly(true);
    DEMSim.SetSimplePatchCombination(true);

    // If you don't need individual force information, then this option makes the solver run a bit faster.
    DEMSim.SetNoForceRecord();

    // E, nu, CoR, mu, Crr...
    auto mat_type_mixer = DEMSim.LoadMaterial({{"E", 1e7}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.0}});
    auto mat_type_granular = DEMSim.LoadMaterial({{"E", 1e7}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.2}, {"Crr", 0.0}});
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

    // Now add a cylindrical boundary
    auto walls = DEMSim.AddExternalObject();
    walls->AddCylinder(make_float3(0), make_float3(0, 0, 1), world_size / 2., mat_type_mixer, 0);

    auto mixer = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/internal_mixer.obj").string(), mat_type_mixer);
    std::cout << "Total num of mixer triangles: " << mixer->GetNumTriangles() << std::endl;
    mixer->Scale(make_float3(world_size / 2, world_size / 2, chamber_height));
    mixer->SetFamily(10);
    // Define the prescribed motion of mixer
    const float mixer_ang_vel = 3.14159f;
    DEMSim.SetFamilyPrescribedAngVel(10, "0", "0", to_string_with_precision(mixer_ang_vel));

    const float granular_rad = 0.005f;
    const float cube_size = 2.f * granular_rad;
    const float cube_density = 2.6e3f;
    const float cube_mass = cube_density * cube_size * cube_size * cube_size;
    const float cube_moi = cube_mass * cube_size * cube_size / 6.f;
    const float cube_half_diag = 0.5f * cube_size * std::sqrt(3.f);
    const float fill_spacing = 3.f * granular_rad;

    // Load a 12-triangle unit cube once, then instantiate it many times. Marking it convex and never-winner follows the
    // mesh-particle demos and reduces unnecessary contact-island work for these tiny rigid particles.
    auto cube_template =
        DEMSim.LoadMeshType((GET_DATA_PATH() / "mesh/cube.obj").string(), mat_type_granular, true, false);
    cube_template->Scale(cube_size);
    cube_template->SetConvex(true);
    cube_template->SetNeverWinner(true);

    // Track the mixer
    auto mixer_tracker = DEMSim.Track(mixer);

    // Sampler to use. The original mixer uses this same center, height, and nominal spacing; the tighter radial
    // clearance accounts for the cube's half diagonal so cubes do not start outside the cylindrical chamber.
    HCPSampler sampler(fill_spacing);
    float3 fill_center = make_float3(0, 0, fill_bottom + fill_height / 2);
    const float fill_radius = world_size / 2. - 2.f * cube_half_diag;
    auto input_xyz = sampler.SampleCylinderZ(fill_center, fill_radius, fill_height / 2);

    // A full one-for-one mesh replacement of the original fill would instantiate more than a million moving triangles.
    // Keep the same filled volume but cap the count so this remains a practical universal-mesh-contact demo.
    const unsigned int target_cubes = 45000;
    std::mt19937 rng(42);
    std::shuffle(input_xyz.begin(), input_xyz.end(), rng);
    if (input_xyz.size() > target_cubes) {
        input_xyz.resize(target_cubes);
    }

    for (const auto& pos : input_xyz) {
        auto cube = DEMSim.AddMeshFromTemplate(cube_template, pos);
        cube->SetFamily(1);
        cube->SetMass(cube_mass);
        cube->SetMOI(make_float3(cube_moi, cube_moi, cube_moi));
        cube->SetInitQuat(make_float4(0.f, 0.f, 0.f, 1.0f));
    }
    std::cout << "Total num of cube particles: " << input_xyz.size() << std::endl;
    std::cout << "Cube edge length: " << cube_size << " m, mass: " << cube_mass << " kg" << std::endl;

    DEMSim.SetTimeStepSize(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.SetErrorOutVelocity(40.);
    // Mesh universal contact uses triangle primitives, so let the safety margin follow the measured mesh velocity and
    // seed it with the rotating mixer tip speed.
    DEMSim.SetExpandSafetyType("auto");
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
    out_dir /= "DemoOutput_MixerCubes";
    create_directory(out_dir);

    float sim_end = 10.0;
    unsigned int fps = 20;
    float frame_time = 1.0 / fps;

    // Keep tab of the max velocity in simulation. Mesh particles need the generic inspector, not the clump-only one.
    auto max_v_finder = DEMSim.CreateInspector("max_absv");

    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int currframe = 0;

    mixer_tracker->SetPos(make_float3(0, 0, chamber_bottom + chamber_height / 2.0));
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (float t = 0; t < sim_end; t += frame_time) {
        std::cout << "Frame: " << currframe << std::endl;
        char analytical_filename[100], meshfilename[100], cnt_filename[100];
        sprintf(analytical_filename, "DEMdemo_analytical_%04d.vtk", currframe);
        sprintf(meshfilename, "DEMdemo_mesh_%04d.vtk", currframe);
        sprintf(cnt_filename, "Contact_pairs_%04d.csv", currframe++);
        DEMSim.WriteAnalyticalFile(out_dir / analytical_filename);
        DEMSim.WriteMeshFile(out_dir / meshfilename);
        // DEMSim.WriteContactFile(out_dir / cnt_filename);

        float max_v = max_v_finder->GetValue();
        std::cout << "Max velocity of any point in simulation is " << max_v << std::endl;
        std::cout << "Solver's current update frequency (auto-adapted): " << DEMSim.GetUpdateFreq() << std::endl;
        std::cout << "Average contacts each primitive has: " << DEMSim.GetAvgPrimitiveContacts() << std::endl;

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

    std::cout << "DEMdemo_MixerCubes exiting..." << std::endl;
    return 0;
}
