//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
// SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Hopper discharge with analytical sphere particles and low-poly mesh cylinders.
// Cylinders have an octagonal cross section, flat ends, and 28 triangles each.
// Spheres and cylinders settle together above a fixed mesh plug. Disabling the
// plug's contacts and output opens the outlet for discharge into the lower bin.
// Run with --smoke-test for a small bed and a short, self-checking discharge.
// =============================================================================

#include "DEM/API.h"
#include "DEM/utils/Samplers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace deme;

int main(int argc, char* argv[]) {
    const bool smoke_test = argc == 2 && std::string(argv[1]) == "--smoke-test";
    if (argc > 1 && !smoke_test) {
        std::cout << "Usage: DEMdemo_HopperSphereMeshedCylinder [--smoke-test]" << std::endl;
        return std::string(argv[1]) == "--help" ? 0 : 1;
    }
    std::cout << "==== DEME demo/test: DEMdemo_HopperSphereMeshedCylinder ====" << std::endl;

    DEMSolver DEMSim;
    DEMSim.UseFrictionalHertzianModel();
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat("CSV");
    DEMSim.SetOutputContent(OUTPUT_CONTENT::XYZ | OUTPUT_CONTENT::VEL | OUTPUT_CONTENT::FAMILY);
    DEMSim.SetMeshOutputFormat("VTK");
    // Attach each triangle's owner family so cylinders, hopper walls, and the gate can be colored separately.
    DEMSim.SetMeshOutputContent({"FAMILY"});
    DEMSim.SetNoForceRecord();
    // Cylinders collide with spheres, other cylinders, the mesh gate, and analytical bin walls.
    DEMSim.SetMeshUniversalContact(true);
    DEMSim.SetMeshParticlesLowPoly(true);
    DEMSim.SetErrorOutAvgContacts(80);

    const size_t total_cylinders = smoke_test ? 24 : 5250;
    const size_t total_spheres = smoke_test ? 16 : 3500;
    const double radius_cylinder = 0.002;
    const double cylinder_length = 0.008;
    const double density_cylinder = 1128;
    const double radius_sphere = 0.003;
    const double density_sphere = 1592;
    const float step_size = 5e-6f;
    const double hopper_width = 0.04;
    const double gate_width = 0.1295;
    constexpr unsigned int cylinder_family = 100;
    constexpr unsigned int sphere_family = 99;
    constexpr unsigned int fixed_family = 10;
    constexpr unsigned int closed_gate_family = 3;
    constexpr unsigned int disabled_gate_family = 4;

    auto mat_flume = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.60}});
    auto mat_walls = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.60}});
    auto mat_spheres = DEMSim.LoadMaterial({{"E", 1e7}, {"nu", 0.35}, {"CoR", 0.20}, {"mu", 0.40}, {"Crr", 0.04}});
    auto mat_cylinders = DEMSim.LoadMaterial({{"E", 1e7}, {"nu", 0.35}, {"CoR", 0.20}, {"mu", 0.30}, {"Crr", 0.03}});
    // Apply the wall/flume contact coefficients to both particle materials.
    for (const auto& wall : {mat_walls, mat_flume}) {
        for (const auto& particle : {mat_spheres, mat_cylinders}) {
            DEMSim.SetMaterialPropertyPair("CoR", wall, particle, 0.20);
            DEMSim.SetMaterialPropertyPair("Crr", wall, particle, 0.05);
            DEMSim.SetMaterialPropertyPair("mu", wall, particle, 0.30);
        }
    }

    DEMSim.InstructBoxDomainDimension({-0.10, 0.10}, {-0.02, 0.02}, {-0.50, 1.0});
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_walls);
    DEMSim.SetTimeStepSize(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.SetMaxVelocity(25.);
    DEMSim.SetInitBinSize(2 * radius_cylinder);
    DEMSim.SetFamilyFixed(fixed_family);
    DEMSim.SetFamilyFixed(closed_gate_family);
    // Keep the disabled plug stationary, non-contacting, and absent from output for every family in the solver.
    DEMSim.SetFamilyFixed(disabled_gate_family);
    DEMSim.DisableFamilyOutput(disabled_gate_family);
    for (unsigned int family = 0; family < NUM_AVAL_FAMILIES; family++) {
        DEMSim.DisableContactBetweenFamilies(disabled_gate_family, family);
    }

    // The three rectangular mesh pieces form the hopper floor and its removable central plug.
    const auto funnel_file = GetDEMEDataFile("mesh/funnel_left.obj");
    const float4 funnel_rotation = make_float4(std::sqrt(0.5f), 0, 0, std::sqrt(0.5f));
    auto left = DEMSim.AddWavefrontMeshObject(funnel_file, mat_flume);
    left->Move(make_float3(-hopper_width / 2, 0, -0.01), funnel_rotation);
    left->SetFamily(fixed_family);
    auto right = DEMSim.AddWavefrontMeshObject(funnel_file, mat_flume);
    right->Move(make_float3(gate_width + hopper_width / 2, 0, -0.01), funnel_rotation);
    right->SetFamily(fixed_family);
    auto gate = DEMSim.AddWavefrontMeshObject(funnel_file, mat_flume);
    gate->Move(make_float3(gate_width / 2, 0, -0.011), funnel_rotation);
    gate->SetFamily(closed_gate_family);
    auto gate_tracker = DEMSim.Track(gate);
    // The floor pieces overlap the closed gate; their mutual contacts do not contribute to the bed dynamics.
    DEMSim.DisableContactBetweenFamilies(fixed_family, closed_gate_family);
    DEMSim.DisableContactBetweenFamilies(fixed_family, fixed_family);

    // The template is centered on its principal axes, with its long axis along X.
    // Explicit circular-cylinder mass properties describe the nominal particle dimensions and density.
    auto cylinder_type = DEMSim.LoadMeshType(GetDEMEDataFile("mesh/cyl_x_r1_h2_8sides.obj"), mat_cylinders, false);
    cylinder_type->Scale(make_float3(cylinder_length / 2, radius_cylinder, radius_cylinder));
    const float cylinder_mass = PI * radius_cylinder * radius_cylinder * cylinder_length * density_cylinder;
    const float axial_moi = 0.5 * cylinder_mass * radius_cylinder * radius_cylinder;
    const float transverse_moi =
        cylinder_mass * (3 * radius_cylinder * radius_cylinder + cylinder_length * cylinder_length) / 12;
    cylinder_type->SetMass(cylinder_mass);
    cylinder_type->SetMOI(make_float3(axial_moi, transverse_moi, transverse_moi));
    cylinder_type->SetConvex(true);
    cylinder_type->SetFamily(cylinder_family);

    const float sphere_mass = 4. / 3. * PI * radius_sphere * radius_sphere * radius_sphere * density_sphere;
    auto sphere_type = DEMSim.LoadSphereType(sphere_mass, radius_sphere, mat_spheres);

    // Build the complete bed before Initialize so mesh topology and mass tables are available during JIT compilation.
    // An anisotropic grid leaves gaps between horizontal cylinders. Sorting by height fills the lower layers first.
    const float half_width = smoke_test ? 0.02f : 0.10f;
    const float half_depth = smoke_test ? 0.013f : 0.02f;
    GridSampler cylinder_sampler(make_float3(1.1 * cylinder_length, 2.2 * radius_cylinder, 2.2 * radius_cylinder));
    auto cylinder_positions = cylinder_sampler.SampleBox(
        make_float3(0, 0, 0.16),
        make_float3(half_width - 0.55 * cylinder_length, half_depth - 1.1 * radius_cylinder, 0.14));
    std::stable_sort(cylinder_positions.begin(), cylinder_positions.end(),
                     [](const float3& a, const float3& b) { return a.z < b.z; });
    if (cylinder_positions.size() < total_cylinders) {
        DEME_ERROR("Cylinder loading region is too small for %zu particles.", total_cylinders);
    }
    cylinder_positions.resize(total_cylinders);
    std::shared_ptr<DEMTracker> first_cylinder_tracker;
    for (const auto& pos : cylinder_positions) {
        auto cylinder = DEMSim.AddMeshFromTemplate(cylinder_type, pos);
        if (!first_cylinder_tracker) {
            first_cylinder_tracker = DEMSim.Track(cylinder);
        }
    }

    // Place analytical spheres above the cylinders with a gap; both particle types move from the first step.
    const float sphere_bottom = cylinder_positions.back().z + radius_cylinder + radius_sphere + 0.01;
    const float sphere_top = 0.60f;
    GridSampler sphere_sampler(2.2 * radius_sphere);
    auto sphere_positions =
        sphere_sampler.SampleBox(make_float3(0, 0, (sphere_bottom + sphere_top) / 2),
                                 make_float3(half_width - 1.1 * radius_sphere, half_depth - 1.1 * radius_sphere,
                                             (sphere_top - sphere_bottom) / 2));
    std::stable_sort(sphere_positions.begin(), sphere_positions.end(),
                     [](const float3& a, const float3& b) { return a.z < b.z; });
    if (sphere_positions.size() < total_spheres) {
        DEME_ERROR("Sphere loading region is too small for %zu particles.", total_spheres);
    }
    sphere_positions.resize(total_spheres);
    auto spheres = DEMSim.AddClumps(sphere_type, sphere_positions);
    spheres->SetFamily(sphere_family);
    spheres->SetVel(make_float3(0, 0, -0.8));
    auto max_speed = DEMSim.CreateInspector("max_absv");
    DEMSim.Initialize();

    // Mesh instances were added consecutively; a single bulk velocity update initializes their downward motion.
    const auto first_cylinder = first_cylinder_tracker->GetOwnerID();
    DEMSim.SetOwnerVelocity(first_cylinder, std::vector<float3>(total_cylinders, make_float3(0, 0, -0.8)));
    const float gate_initial_z = gate_tracker->Pos().z;
    std::cout << "Particles: " << total_cylinders << " meshed cylinders (" << cylinder_type->GetNumTriangles()
              << " triangles each), " << total_spheres << " analytical spheres" << std::endl;

    auto out_dir = std::filesystem::current_path() / "DemoOutput_HopperSphereMeshedCylinder";
    if (smoke_test) {
        out_dir += "_smoke";
    }
    std::filesystem::create_directories(out_dir);
    unsigned int frame = 0;
    double elapsed = 0;
    // Write matching CSV and VTK frames through every phase; the disabled plug is filtered from mesh output.
    auto write_frame = [&]() {
        char sphere_file[100], mesh_file[100];
        std::snprintf(sphere_file, sizeof(sphere_file), "DEMdemo_spheres_%04u.csv", frame);
        std::snprintf(mesh_file, sizeof(mesh_file), "DEMdemo_mesh_%04u.vtk", frame);
        DEMSim.WriteSphereFile(out_dir / sphere_file);
        DEMSim.WriteMeshFile(out_dir / mesh_file);
        std::cout << "Frame " << frame++ << ": t = " << elapsed << " s, max speed = " << max_speed->GetValue()
                  << std::endl;
        // Report host and device memory at output intervals throughout settling and discharge.
        DEMSim.ShowMemStats();
    };
    // Advance each phase in integer step counts, with output at regular simulation-time intervals.
    const unsigned int output_steps = static_cast<unsigned int>(std::llround(0.01 / step_size));
    auto run_phase = [&](const char* name, unsigned int steps) {
        std::cout << name << std::endl;
        while (steps > 0) {
            const auto chunk = std::min(steps, output_steps);
            const double duration = chunk * static_cast<double>(step_size);
            DEMSim.DoDynamicsThenSync(duration);
            elapsed += duration;
            steps -= chunk;
            write_frame();
        }
    };
    auto steps_for = [&](double duration) { return static_cast<unsigned int>(std::llround(duration / step_size)); };
    const auto start = std::chrono::high_resolution_clock::now();
    write_frame();
    run_phase("Settling spheres and cylinders", steps_for(smoke_test ? 0.16 : 0.70));

    const auto discharge_steps = steps_for(smoke_test ? 0.14 : 7.50);
    // Settling ends synchronized. Switch the plug to its disabled family before any discharge step or output.
    DEMSim.ChangeFamily(closed_gate_family, disabled_gate_family);
    write_frame();
    run_phase("Discharging with the plug disabled", discharge_steps);
    DEMSim.WaitForPendingOutput();

    if (smoke_test) {
        // Check the representation, free cylinder motion, finite particle states, and stationary disabled plug.
        bool valid = DEMSim.GetNumClumps() == total_spheres && cylinder_type->GetNumTriangles() == 28;
        const auto positions = DEMSim.GetOwnerPosition(0, static_cast<bodyID_t>(DEMSim.GetNumOwners()));
        const auto velocities = DEMSim.GetOwnerVelocity(0, static_cast<bodyID_t>(DEMSim.GetNumOwners()));
        for (size_t i = 0; i < positions.size(); i++) {
            valid = valid && std::isfinite(positions[i].x) && std::isfinite(positions[i].y) &&
                    std::isfinite(positions[i].z) && std::isfinite(velocities[i].x) && std::isfinite(velocities[i].y) &&
                    std::isfinite(velocities[i].z);
        }
        valid = valid && gate_tracker->GetFamily() == disabled_gate_family &&
                std::abs(gate_tracker->Pos().z - gate_initial_z) < 1e-4 &&
                first_cylinder_tracker->Pos().z < cylinder_positions.front().z - radius_cylinder;
        std::cout << (valid ? "PASS" : "FAIL") << ": hopper mesh-cylinder smoke test" << std::endl;
        if (!valid) {
            return 1;
        }
    }
    const std::chrono::duration<double> wall_time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Simulated " << elapsed << " s in " << wall_time.count() << " s wall time" << std::endl;
    DEMSim.ShowTimingStats();
    DEMSim.ShowMemStats();
    std::cout << "DEMdemo_HopperSphereMeshedCylinder exiting..." << std::endl;
    return 0;
}
