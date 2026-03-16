//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Sphere-sphere penetration and area test: two sphere meshes (sphere_highres.obj)
// are fixed in place along the Z-axis at several separation distances so that
// their surfaces overlap by a controlled depth.  A custom force model (no
// actual force computed) writes the patch-based penetration depth and contact
// area into owner wildcards after each simulation step.  The solver values are
// compared against the analytical expressions for two equal spheres of radius R:
//
//   penetration_analytical = d
//   area_analytical        = pi * (R*d - d^2/4)
//
// where d is the intended penetration depth (= 2R - centre_distance).
// For a high-resolution mesh the values should be similar to the analytical
// result, though the mesh discretisation introduces some approximation error.
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
    const float R = 0.1f;  // radius of each sphere after scaling (m)

    // Penetration depths to test (m)
    const std::vector<float> depths = {0.001f, 0.003f, 0.007f, 0.015f, 0.03f};

    // Material (values are irrelevant since we generate no force, but must be set)
    const float E = 1e8f;
    const float nu = 0.3f;
    const float CoR = 0.3f;
    const float mu = 0.3f;

    // Simulation time step – we only need one step per depth measurement
    const float step_size = 1e-5f;

    std::cout << "=====================================================" << std::endl;
    std::cout << "DEMTest_SphereSphereArea" << std::endl;
    std::cout << "  Sphere radius R = " << R << " m (each)" << std::endl;
    std::cout << "  Sphere A fixed at (0,0,R); Sphere B moves along Z" << std::endl;
    std::cout << "=====================================================" << std::endl;

    // =========================================================================
    // Solver setup
    // =========================================================================
    DEMSolver DEMSim;
    DEMSim.SetVerbosity("ERROR");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.InstructBoxDomainDimension(20, 20, 20);
    DEMSim.SetGravitationalAcceleration(make_float3(0.f, 0.f, 0.f));
    // Mesh-mesh contacts must be enabled for two mesh spheres
    DEMSim.SetMeshUniversalContact(true);

    auto mat = DEMSim.LoadMaterial({{"E", E}, {"nu", nu}, {"CoR", CoR}, {"mu", mu}, {"Crr", 0.f}});

    // =========================================================================
    // Custom force model: write patch penetration and area to owner wildcards.
    // No actual contact force is generated.
    // AOwner and BOwner are the two sphere mesh owners respectively.
    // =========================================================================
    auto force_model = DEMSim.DefineContactForceModel(R"(
        // Record patch-based penetration depth and contact area in owner wildcards.
        // Both sphere owners receive the same (patch-level) depth and area.
        if (overlapDepth > 0) {
            penetration[AOwner] = (float)overlapDepth;
            area[AOwner]        = (float)overlapArea;
            penetration[BOwner] = (float)overlapDepth;
            area[BOwner]        = (float)overlapArea;
        } else {
            penetration[AOwner] = 0.f;
            area[AOwner]        = 0.f;
            penetration[BOwner] = 0.f;
            area[BOwner]        = 0.f;
        }
        // Explicitly zero the force so the solver does not emit a warning.
        force = make_float3(0.f, 0.f, 0.f);
    )");
    force_model->SetPerOwnerWildcards({"penetration", "area"});

    // =========================================================================
    // Sphere A: fixed at (0, 0, R), family 1
    // Sphere B: initially set for the first test depth, family 2
    // =========================================================================
    const float sphere_mass = 2600.f * (4.f / 3.f) * PI * R * R * R;
    const float sphere_moi = 2.f / 5.f * sphere_mass * R * R;

    const float d0 = depths[0];

    // Sphere A centre: (0, 0, R) — just resting on the z=0 level
    auto sphereA = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/sphere_highres.obj").string(), mat);
    sphereA->Scale(R);
    sphereA->SetMass(sphere_mass);
    sphereA->SetMOI(make_float3(sphere_moi, sphere_moi, sphere_moi));
    sphereA->SetInitPos(make_float3(0.f, 0.f, R));
    sphereA->SetInitQuat(make_float4(0.7071, 0.7071, 0, 0));
    sphereA->SetFamily(1);
    DEMSim.SetFamilyFixed(1);

    auto trackerA = DEMSim.Track(sphereA);

    // Sphere B centre: (0, 0, 3R - d0), so |AB| = 2R - d0 → penetration = d0
    auto sphereB = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/sphere_highres.obj").string(), mat);
    sphereB->Scale(R);
    sphereB->SetMass(sphere_mass);
    sphereB->SetMOI(make_float3(sphere_moi, sphere_moi, sphere_moi));
    sphereB->SetInitPos(make_float3(0.f, 0.f, 3.f * R - d0));
    sphereB->SetInitQuat(make_float4(0.7071, 0.7071, 0, 0));
    sphereB->SetFamily(2);
    DEMSim.SetFamilyFixed(2);

    auto trackerB = DEMSim.Track(sphereB);

    // =========================================================================
    // Initialise (JIT-compiles the GPU kernels once)
    // =========================================================================
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetCDUpdateFreq(1);
    DEMSim.SetExpandSafetyAdder(1000.);
    DEMSim.SetErrorOutAvgContacts(20000);
    DEMSim.Initialize();

    // Output directory
    path out_dir = current_path() / "modular_test_output" / "DEMTest_SphereSphereArea";
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
        // Sphere A stays at (0,0,R); move sphere B so centre distance = 2R - d
        trackerA->SetPos(make_float3(0.f, 0.f, R));
        trackerB->SetPos(make_float3(0.f, 0.f, 3.f * R - d));

        DEMSim.SetTriTriPenetration(R);
        // Run one dynamics step so the force model executes
        DEMSim.DoDynamicsThenSync(step_size);

        // Read back owner wildcards: sphere A is family 1, sphere B is family 2
        std::vector<float> penA_vec = DEMSim.GetFamilyOwnerWildcardValue(1, "penetration");
        std::vector<float> areaA_vec = DEMSim.GetFamilyOwnerWildcardValue(1, "area");

        // Use sphere A's values (both spheres receive the same patch values)
        const float pen_solver = penA_vec.empty() ? 0.f : penA_vec[0];
        const float area_solver = areaA_vec.empty() ? 0.f : areaA_vec[0];

        // Analytical values for two equal spheres of radius R with penetration d
        const float pen_analyt = d;
        const float area_analyt = PI * (R * d - d * d / 4.f);

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
    std::cout << "Note: for mesh-mesh sphere contacts the solver's patch-based" << std::endl;
    std::cout << "      penetration and area are approximate (not exact)." << std::endl;
    std::cout << "      Analytical area = pi*(R*d - d^2/4) for equal spheres." << std::endl;

    DEMSim.ShowTimingStats();
    std::cout << "=====================================================" << std::endl;
    std::cout << "DEMTest_SphereSphereArea exiting..." << std::endl;
    return 0;
}
