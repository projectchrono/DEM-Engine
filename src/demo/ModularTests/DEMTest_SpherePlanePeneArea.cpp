//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Sphere-plane penetration and area test: a sphere mesh (sphere_highres.obj)
// is placed at several fixed positions so that it penetrates an analytical
// plane by a controlled depth.  A custom force model (no actual force computed)
// writes the patch-based penetration depth and contact area into owner
// wildcards after each simulation step.  The solver values are then compared
// against the analytical expressions:
//
//   penetration_analytical = d
//   area_analytical        = pi * (2*R*d - d^2)
//
// where R is the sphere radius and d is the intended penetration depth.
// For a high-resolution sphere mesh the two should agree very closely.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/HostSideHelpers.hpp>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <filesystem>
#include <system_error>

using namespace std::filesystem;
using namespace deme;

int main() {
    // =========================================================================
    // Parameters
    // =========================================================================
    const float R = 0.1f;  // sphere radius after scaling (m)

    // Penetration depths to test (m)
    const std::vector<float> depths = {0.001f, 0.003f, 0.007f, 0.015f, 0.03f};

    // Material (values are irrelevant since we generate no force, but must be set)
    const float E = 1e8f;
    const float nu = 0.3f;
    const float CoR = 0.3f;
    const float mu = 0.3f;

    // Simulation time step – we only need the solver to run one step per depth
    const float step_size = 1e-5f;

    std::cout << "=====================================================" << std::endl;
    std::cout << "DEMTest_SpherePlanePeneArea" << std::endl;
    std::cout << "  Sphere radius R = " << R << " m" << std::endl;
    std::cout << "  Analytical plane at z=0 with normal (0,0,1)" << std::endl;
    std::cout << "=====================================================" << std::endl;

    // =========================================================================
    // Solver setup
    // =========================================================================
    DEMSolver DEMSim;
    DEMSim.SetVerbosity("ERROR");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    // Box large enough to contain the sphere at all test positions
    DEMSim.InstructBoxDomainDimension(20, 20, 20);
    DEMSim.SetGravitationalAcceleration(make_float3(0.f, 0.f, 0.f));
    // Mesh-related contacts must be enabled
    DEMSim.SetMeshUniversalContact(true);

    auto mat = DEMSim.LoadMaterial({{"E", E}, {"nu", nu}, {"CoR", CoR}, {"mu", mu}, {"Crr", 0.f}});

    // =========================================================================
    // Custom force model: write patch penetration and area to owner wildcards.
    // No actual contact force is generated.
    // =========================================================================
    auto force_model = DEMSim.DefineContactForceModel(R"(
        // Record patch-based penetration depth and contact area in owner wildcards.
        // AOwner is the sphere mesh owner; BOwner is the analytical plane owner.
        if (overlapDepth > 0) {
            penetration[AOwner] = (float)overlapDepth;
            area[AOwner]        = (float)overlapArea;
        } else {
            penetration[AOwner] = 0.f;
            area[AOwner]        = 0.f;
        }
        // Explicitly zero the force so the solver does not emit a warning.
        force = make_float3(0.f, 0.f, 0.f);
    )");
    force_model->SetPerOwnerWildcards({"penetration", "area"});

    // =========================================================================
    // Analytical plane: z = 0, normal = (0,0,1), family 100 (fixed)
    // =========================================================================
    auto plane = DEMSim.AddBCPlane(make_float3(0.f, 0.f, 0.f), make_float3(0.f, 0.f, 1.f), mat);
    plane->SetFamily(100);
    DEMSim.SetFamilyFixed(100);

    // =========================================================================
    // Sphere mesh: sphere_highres.obj (unit sphere), scaled to radius R.
    // Family 1 (fixed); initial position is placed at the first test depth.
    // =========================================================================
    const float sphere_mass = 2600.f * (4.f / 3.f) * PI * R * R * R;
    const float sphere_moi = 2.f / 5.f * sphere_mass * R * R;

    // Initial position: sphere centre above plane, no penetration yet
    const float d0 = depths[0];
    auto sphere = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/sphere_highres.obj").string(), mat);
    sphere->Scale(R);
    sphere->SetMass(sphere_mass);
    sphere->SetMOI(make_float3(sphere_moi, sphere_moi, sphere_moi));
    sphere->SetInitPos(make_float3(0.f, 0.f, R - d0));
    sphere->SetFamily(1);
    DEMSim.SetFamilyFixed(1);

    auto sphere_tracker = DEMSim.Track(sphere);

    // =========================================================================
    // Initialise (JIT-compiles the GPU kernels once)
    // =========================================================================
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetCDUpdateFreq(1);
    DEMSim.Initialize();

    // Output directory
    path out_dir = current_path() / "modular_test_output" / "DEMTest_SpherePlanePeneArea";
    std::error_code dir_ec;
    create_directories(out_dir, dir_ec);

    // =========================================================================
    // Loop over all penetration depths
    // =========================================================================
    std::cout << std::endl;
    std::cout << std::setw(12) << "depth_d(m)" << std::setw(18) << "pen_solver(m)" << std::setw(18) << "pen_analyt(m)"
              << std::setw(14) << "pen_err(%)" << std::setw(18) << "area_solver(m2)" << std::setw(18)
              << "area_analyt(m2)" << std::setw(14) << "area_err(%)" << std::endl;
    std::cout << std::string(112, '-') << std::endl;

    for (float d : depths) {
        // Move sphere to the desired penetration depth
        sphere_tracker->SetPos(make_float3(0.f, 0.f, R - d));

        // Run one dynamics step so the force model executes
        DEMSim.DoDynamicsThenSync(step_size);

        // Read back owner wildcards for the sphere (family 1)
        std::vector<float> pen_vec = DEMSim.GetFamilyOwnerWildcardValue(1, "penetration");
        std::vector<float> area_vec = DEMSim.GetFamilyOwnerWildcardValue(1, "area");

        const float pen_solver = pen_vec.empty() ? 0.f : pen_vec[0];
        const float area_solver = area_vec.empty() ? 0.f : area_vec[0];

        // Analytical values
        const float pen_analyt = d;
        const float area_analyt = PI * (2.f * R * d - d * d);

        const float pen_err = (pen_analyt > 1e-12f) ? 100.f * std::abs(pen_solver - pen_analyt) / pen_analyt : 0.f;
        const float area_err = (area_analyt > 1e-12f) ? 100.f * std::abs(area_solver - area_analyt) / area_analyt : 0.f;

        std::cout << std::setw(12) << std::fixed << std::setprecision(5) << d << std::setw(18) << std::setprecision(6)
                  << pen_solver << std::setw(18) << std::setprecision(6) << pen_analyt << std::setw(14)
                  << std::setprecision(2) << pen_err << std::setw(18) << std::setprecision(6) << area_solver
                  << std::setw(18) << std::setprecision(6) << area_analyt << std::setw(14) << std::setprecision(2)
                  << area_err << std::endl;
    }

    std::cout << std::string(112, '-') << std::endl;
    std::cout << std::endl;
    std::cout << "Note: solver penetration uses the maximum projected vertex depth" << std::endl;
    std::cout << "      across all triangles in the contact patch; analytical area" << std::endl;
    std::cout << "      = pi*(2*R*d - d^2).  Differences reflect mesh discretisation." << std::endl;

    DEMSim.ShowTimingStats();
    std::cout << "=====================================================" << std::endl;
    std::cout << "DEMTest_SpherePlanePeneArea exiting..." << std::endl;
    return 0;
}
