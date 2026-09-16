//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//  SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Drop one 1 mm sphere inside a reverse-wound 50 mm mesh box. This is the C++
// regression counterpart of the pyDEME 01_mesh_box_1particle_1s.py case.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <system_error>

using namespace deme;
using namespace std::filesystem;

namespace {

int fail(const char* reason) {
    std::cerr << "DEMTest_MeshBox1Particle FAILED: " << reason << std::endl;
    return 1;
}

bool isFinite(const float3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

}  // namespace

int main() {
    constexpr float mm_to_m = 1.0e-3f;
    constexpr float box_size = 0.050f;
    constexpr float sphere_radius = 0.001f;
    constexpr float sphere_density = 2600.0f;
    constexpr float step_size = 1.0e-6f;
    constexpr float sample_time = 0.02f;
    constexpr float total_time = 1.0f;
    constexpr float gravity = 9.81f;
    const float3 initial_position = make_float3(0.010f, 0.020f, 0.010f);  // make_float3(0.0125f, 0.020f, 0.0125f);

    DEMSolver solver;
    solver.SetOutputFormat(OUTPUT_FORMAT::CSV);
    solver.SetOutputContent({"ABSV", "XYZ"});
    solver.SetMeshOutputFormat(MESH_FORMAT::VTK);
    solver.SetNoForceRecord();

    // World size
    const std::pair<float, float> mesh_domain = {-0.005, 0.055};
    solver.InstructBoxDomainDimension(mesh_domain, mesh_domain, mesh_domain);

    auto particle_material =
        solver.LoadMaterial({{"E", 1.0e8f}, {"nu", 0.30f}, {"CoR", 0.80f}, {"mu", 0.20f}, {"Crr", 0.0f}});
    auto wall_material =
        solver.LoadMaterial({{"E", 1.0e8f}, {"nu", 0.30f}, {"CoR", 0.60f}, {"mu", 0.50f}, {"Crr", 0.0f}});
    solver.SetMaterialPropertyPair("mu", particle_material, wall_material, 0.35f);
    solver.SetMaterialPropertyPair("CoR", particle_material, wall_material, 0.50f);

    // The reverse winding makes the box's active triangle faces point toward
    // the enclosed particle, matching the reference pyDEME mesh.
    auto mesh_wall =
        solver.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/box_50mm_reverse_winding.obj").string(), wall_material);
    mesh_wall->Scale(mm_to_m);
    mesh_wall->SetMass(1.0f);
    mesh_wall->SetMOI(make_float3(1.0f, 1.0f, 1.0f));
    mesh_wall->SetFamily(10);
    solver.SetFamilyFixed(10);

    const float sphere_mass =
        (4.0f / 3.0f) * static_cast<float>(PI) * sphere_radius * sphere_radius * sphere_radius * sphere_density;
    auto sphere_type = solver.LoadSphereType(sphere_mass, sphere_radius, particle_material);
    auto sphere_batch = solver.AddClumps(sphere_type, initial_position);
    auto sphere_tracker = solver.Track(sphere_batch);

    solver.SetTimeStepSize(step_size);
    solver.SetGravitationalAcceleration(make_float3(0.0f, -gravity, 0.0f));
    solver.Initialize();

    // Match the pyDEME driver by reporting solver state and writing sphere and
    // mesh snapshots at every sample boundary, including the initial state.
    const path output_dir = current_path() / "modular_test_output" / "DEMTest_MeshBox1Particle";
    std::error_code dir_error;
    create_directories(output_dir, dir_error);
    if (dir_error || !is_directory(output_dir)) {
        std::cerr << "Failed to create output directory: " << output_dir << " (" << dir_error.message() << ")"
                  << std::endl;
        return 1;
    }
    auto max_velocity = solver.CreateInspector("clump_max_absv");

    bool observed_downward_motion = false;
    bool observed_rebound = false;
    float max_speed = 0.0f;
    const int sample_count = static_cast<int>(total_time / sample_time);
    for (int sample = 0; sample < sample_count; sample++) {
        const float current_time = static_cast<float>(sample) * sample_time;
        std::cout << "Sample " << sample << ": t=" << current_time << " s, max_v=" << max_velocity->GetValue()
                  << " m/s, avg_contacts=" << solver.GetAvgSphContacts() << ", update_freq=" << solver.GetUpdateFreq()
                  << std::endl;

        char sphere_filename[64];
        char mesh_filename[64];
        std::snprintf(sphere_filename, sizeof(sphere_filename), "sphere_%04d.csv", sample);
        std::snprintf(mesh_filename, sizeof(mesh_filename), "mesh_%04d.vtk", sample);
        solver.WriteSphereFile(output_dir / sphere_filename);
        solver.WriteMeshFile(output_dir / mesh_filename);

        solver.DoDynamicsThenSync(sample_time);

        const float3 position = sphere_tracker->Pos();
        const float3 velocity = sphere_tracker->Vel();
        if (!isFinite(position) || !isFinite(velocity)) {
            return fail("sphere state became non-finite");
        }
        const float speed = std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
        max_speed = std::max(max_speed, speed);
        observed_downward_motion = observed_downward_motion || velocity.y < -0.1f;
        observed_rebound = observed_rebound || (observed_downward_motion && velocity.y > 0.05f);

        // Permit a small contact-scale tolerance while requiring the sphere's
        // center to remain inside the closed mesh throughout the run.
        constexpr float containment_tolerance = 0.25f * sphere_radius;
        if (position.x < sphere_radius - containment_tolerance ||
            position.x > box_size - sphere_radius + containment_tolerance ||
            position.y < sphere_radius - containment_tolerance ||
            position.y > box_size - sphere_radius + containment_tolerance ||
            position.z < sphere_radius - containment_tolerance ||
            position.z > box_size - sphere_radius + containment_tolerance) {
            return fail("sphere escaped the mesh box");
        }
    }

    if (!observed_downward_motion) {
        return fail("gravity-driven downward motion was not observed");
    }
    if (!observed_rebound) {
        return fail("a rebound from the bottom mesh wall was not observed");
    }

    const float expected_first_impact_speed = std::sqrt(2.0f * gravity * (initial_position.y - sphere_radius));
    std::cout << "DEMTest_MeshBox1Particle passed. simulated_time=" << total_time
              << " s, max_sampled_speed=" << max_speed
              << " m/s, expected_first_impact_speed=" << expected_first_impact_speed << " m/s" << std::endl;
    return 0;
}
