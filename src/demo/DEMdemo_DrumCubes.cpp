//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Rotating drum centrifuge demo with only cube mesh particles.
// Matches the output style of DEMdemo_Centrifuge but uses 10 mm cubes
// (12-triangle mesh) inside an analytically defined cylinder and lids.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/Samplers.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>

using namespace deme;
using namespace std::filesystem;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::FAMILY);
    DEMSim.SetNoForceRecord();
    DEMSim.SetMeshUniversalContact(true);

    auto mat_type_cube = DEMSim.LoadMaterial({{"E", 1e6}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.01}});
    auto mat_type_drum = DEMSim.LoadMaterial({{"E", 2e6}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.01}});
    DEMSim.SetMaterialPropertyPair("mu", mat_type_cube, mat_type_drum, 0.5);

    const float cube_size = 0.01f;
    const float cube_density = 2600.0f;
    const float cube_mass = cube_density * cube_size * cube_size * cube_size;
    const float cube_moi = cube_mass * cube_size * cube_size / 6.0f;
    const float half_diag = 0.5f * cube_size * std::sqrt(3.0f);

    // Load cube mesh template (12 triangles) and scale to 10 mm
    auto cube_template = DEMSim.LoadMeshType((GET_DATA_PATH() / "mesh/cube.obj").string(), mat_type_cube, true, false);
    cube_template->Scale(cube_size);

    // Drum definition
    float3 CylCenter = make_float3(0, 0, 0);
    float3 CylAxis = make_float3(0, 0, 1);
    float CylRad = 0.08f;
    float CylHeight = 0.2f;
    float CylMass = 1.0f;
    float safe_delta = 0.003f;
    float IZZ = CylMass * CylRad * CylRad / 2;
    float IYY = (CylMass / 12) * (3 * CylRad * CylRad + CylHeight * CylHeight);
    auto Drum = DEMSim.AddExternalObject();
    // Drum->AddCylinder(CylCenter, CylAxis, CylRad, mat_type_drum, 0);
    Drum->AddPlane(make_float3(CylRad, 0, 0), make_float3(-1, 0, 0), mat_type_drum);
    Drum->AddPlane(make_float3(-CylRad, 0, 0), make_float3(1, 0, 0), mat_type_drum);
    Drum->AddPlane(make_float3(0, CylRad, 0), make_float3(0, -1, 0), mat_type_drum);
    Drum->AddPlane(make_float3(0, -CylRad, 0), make_float3(0, 1, 0), mat_type_drum);
    Drum->SetMass(CylMass);
    Drum->SetMOI(make_float3(IYY, IYY, IZZ));
    auto Drum_tracker = DEMSim.Track(Drum);
    unsigned int drum_family = 100;
    Drum->SetFamily(drum_family);
    const float rpm = 200.0f;
    const float drum_ang_vel = rpm * 2.0f * PI / 60.0f;
    DEMSim.SetFamilyPrescribedAngVel(drum_family, "0", "0", to_string_with_precision(drum_ang_vel));
    auto top_bot_planes = DEMSim.AddExternalObject();
    top_bot_planes->AddPlane(make_float3(0, 0, CylHeight / 2. - safe_delta), make_float3(0, 0, -1), mat_type_drum);
    top_bot_planes->AddPlane(make_float3(0, 0, -CylHeight / 2. + safe_delta), make_float3(0, 0, 1), mat_type_drum);
    top_bot_planes->SetFamily(drum_family);
    auto planes_tracker = DEMSim.Track(top_bot_planes);

    // Place 1000 cubes on a grid inside the drum
    const unsigned int target_cubes = 1000;
    float sample_radius = CylRad - half_diag - safe_delta;
    float sample_halfheight = CylHeight / 2.0f - half_diag - safe_delta;
    float fill_spacing = cube_size * 1.25f;  // leave gap so meshes don't start in contact
    std::mt19937 rng(42);
    unsigned int created = 0;
    for (float z = -sample_halfheight; z <= sample_halfheight && created < target_cubes; z += fill_spacing) {
        for (float y = -sample_radius; y <= sample_radius && created < target_cubes; y += fill_spacing) {
            for (float x = -sample_radius; x <= sample_radius && created < target_cubes; x += fill_spacing) {
                if (x * x + y * y > sample_radius * sample_radius) {
                    continue;
                }
                auto cube = DEMSim.AddMeshFromTemplate(cube_template, make_float3(x, y, z));
                cube->SetFamily(1);
                cube->SetMass(cube_mass);
                cube->SetMOI(make_float3(cube_moi, cube_moi, cube_moi));
                cube->SetInitQuat(make_float4(0.f, 0.f, 0.f, 1.0f));
                created++;
            }
        }
    }
    std::cout << "Placed " << created << " cubes inside the drum." << std::endl;

    auto max_v_finder = DEMSim.CreateInspector("max_absv");
    float max_v;

    DEMSim.InstructBoxDomainDimension(0.4, 0.4, 0.4);
    float step_size = 1e-4f;
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.SetExpandSafetyType("auto");
    DEMSim.SetExpandSafetyAdder(drum_ang_vel * CylRad);
    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir /= "DemoOutput_DrumCubes";
    create_directory(out_dir);

    float time_end = 3.0f;
    unsigned int fps = 20;
    float frame_time = 1.0f / fps;

    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (double t = 0; t < (double)time_end; t += frame_time, curr_step++) {
        std::cout << "Frame: " << currframe << std::endl;
        DEMSim.ShowThreadCollaborationStats();
        char filename[100];
        sprintf(filename, "DEMdemo_output_%04d.vtk", currframe);
        DEMSim.WriteMeshFile(out_dir / filename);
        currframe++;
        max_v = max_v_finder->GetValue();
        std::cout << "Max velocity of any point in simulation is " << max_v << std::endl;

        float3 drum_moi = Drum_tracker->MOI();
        float3 drum_acc = Drum_tracker->ContactAngAccLocal();
        float3 drum_torque = drum_acc * drum_moi;
        std::cout << "Contact torque on the side walls is " << drum_torque.x << ", " << drum_torque.y << ", "
                  << drum_torque.z << std::endl;

        float3 force_on_BC = planes_tracker->ContactAcc() * planes_tracker->Mass();
        std::cout << "Contact force on bottom plane is " << std::abs(force_on_BC.z) << std::endl;

        DEMSim.DoDynamics(frame_time);
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << (time_sec.count()) / time_end * 10.0 << " seconds (wall time) to finish 10 seconds' simulation"
              << std::endl;
    DEMSim.ShowThreadCollaborationStats();
    DEMSim.ClearThreadCollaborationStats();

    DEMSim.ShowTimingStats();
    std::cout << "----------------------------------------" << std::endl;
    DEMSim.ShowMemStats();
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "DEMdemo_DrumCubes exiting..." << std::endl;
    return 0;
}