// Copyright (c) 2021, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause

// Verify that lightweight simulation-parameter setters update initialized workers without a full UpdateSimParams().

#include <core/ApiVersion.h>
#include <DEM/API.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using namespace deme;

namespace {

int fail(const std::string& message) {
    std::cerr << "DEMTest_RuntimeSimParams failed: " << message << std::endl;
    return 1;
}

}  // namespace

int main() {
    constexpr double initial_step = 1.0e-4;
    constexpr double runtime_step = 2.0e-4;
    constexpr double runtime_duration = 2.0e-3;
    constexpr float runtime_gravity = -5.0f;

    DEMSolver solver;
    solver.SetVerbosity("ERROR");
    solver.InstructBoxDomainDimension(2.0, 2.0, 2.0);
    solver.SetInitBinNumTarget(8);
    solver.SetTimeStepSize(initial_step);
    solver.SetGravitationalAcceleration(make_float3(0.f, 0.f, 0.f));

    auto material = solver.LoadMaterial({{"E", 1.0e7f}, {"nu", 0.3f}, {"CoR", 0.2f}, {"mu", 0.4f}, {"Crr", 0.f}});

    // The solver-level DEME 2 policy affects only meshes loaded while it is enabled.
    DEMMesh deme2_mesh;
    if (!deme2_mesh.LoadWavefrontMesh((GET_DATA_PATH() / "mesh/cube.obj").string())) {
        return fail("could not load the DEME 2 policy test mesh");
    }
    deme2_mesh.SetMaterial(material);
    const size_t deme2_triangle_count = deme2_mesh.GetNumTriangles();
    solver.SetDEME2MeshBehavior();
    auto deme2_template = solver.LoadMeshType(deme2_mesh);
    if (deme2_template->GetNumPatches() != deme2_triangle_count) {
        return fail("enabled SetDEME2MeshBehavior did not create one patch per triangle");
    }

    DEMMesh normal_mesh;
    if (!normal_mesh.LoadWavefrontMesh((GET_DATA_PATH() / "mesh/cube.obj").string())) {
        return fail("could not load the normal-policy test mesh");
    }
    normal_mesh.SetMaterial(material);
    solver.SetDEME2MeshBehavior(false);
    auto normal_template = solver.LoadMeshType(normal_mesh);
    if (normal_template->GetNumPatches() != 1) {
        return fail("disabled SetDEME2MeshBehavior modified a newly loaded mesh");
    }

    auto sphere_type = solver.LoadSphereType(1.0f, 0.05f, material);
    auto batch = solver.AddClumps(sphere_type, make_float3(0.f, 0.f, 0.f));
    auto tracker = solver.Track(batch);
    solver.Initialize();

    solver.DoDynamicsThenSync(runtime_duration);
    if (std::abs(tracker->GetVel()[2]) > 1.0e-6f) {
        return fail("zero setup gravity did not preserve zero vertical velocity");
    }

    solver.SetTimeStepSize(runtime_step);
    solver.SetGravitationalAcceleration(make_float3(0.f, 0.f, runtime_gravity));
    solver.SetMaxTriTriPenetration(0.25);
    solver.SetTriTriContactRejectionRatio(-1.f);
    solver.SetMaxSphereInBin(std::numeric_limits<unsigned int>::max());
    solver.SetMaxTriangleInBin(std::numeric_limits<unsigned int>::max());
    solver.SetErrorOutVelocity(1.0e6f);
    solver.SetErrorOutAngularVelocity(1.0e6f);
    solver.SetErrorOutAvgContacts(1.0e6f);
    solver.SetUseAngularVelocityMargin(false);

    if (std::abs(solver.GetTimeStepSize() - runtime_step) > 1.0e-12) {
        return fail("SetTimeStepSize did not update the public timestep state");
    }

    const double gravity_start_time = solver.GetSimTime();
    solver.DoDynamicsThenSync(runtime_duration);
    const double gravity_end_time = solver.GetSimTime();
    const double actual_duration = gravity_end_time - gravity_start_time;
    const std::vector<float> velocity = tracker->GetVel();
    const std::vector<float> position = tracker->GetPos();
    const float actual_velocity = velocity[2];
    const float expected_velocity = runtime_gravity * static_cast<float>(actual_duration);
    if (std::abs(actual_velocity - expected_velocity) > 5.0e-4f) {
        std::cerr << "Runtime gravity diagnostics:\n"
                  << "  requested duration: " << runtime_duration << "\n"
                  << "  actual time range: [" << gravity_start_time << ", " << gravity_end_time << "]\n"
                  << "  actual duration: " << actual_duration << "\n"
                  << "  configured timestep: " << solver.GetTimeStepSize() << "\n"
                  << "  configured gravity z: " << runtime_gravity << "\n"
                  << "  expected velocity z: " << expected_velocity << "\n"
                  << "  actual velocity: [" << velocity[0] << ", " << velocity[1] << ", " << velocity[2] << "]\n"
                  << "  actual position: [" << position[0] << ", " << position[1] << ", " << position[2] << "]\n"
                  << "  velocity z error: " << std::abs(actual_velocity - expected_velocity) << std::endl;
        return fail("runtime gravity did not take effect without UpdateSimParams");
    }

    // Exercise both compatibility names and confirm they share the unified implementation.
    solver.UpdateStepSize(initial_step);
    solver.SetInitTimeStep(runtime_step);
    if (std::abs(solver.GetTimeStepSize() - runtime_step) > 1.0e-12) {
        return fail("legacy timestep methods do not delegate to SetTimeStepSize");
    }

    std::cout << "DEMTest_RuntimeSimParams passed." << std::endl;
    return 0;
}
