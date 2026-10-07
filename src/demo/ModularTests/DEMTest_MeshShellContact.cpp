// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "DEM/API.h"

using namespace deme;

namespace {
// The spheres clear the bare surfaces by 0.03; shell thickness yields penetrations of 0.02 and 0.03.
// A tiny fixed search margin ensures shell thickness itself must reach contact detection.
void checkShellForce(const std::shared_ptr<DEMTracker>& tracker, float expected = 0.02f) {
    std::vector<float3> points, forces;
    tracker->GetContactForces(points, forces);
    float total = 0.f;
    for (const auto& force : forces)
        total += length(force);
    if (std::abs(total - expected) > 2.e-4f)
        throw std::runtime_error("Expected shell contact force " + std::to_string(expected) + ", got " +
                                 std::to_string(total));
}
}  // namespace

// Cover initial shell metadata and its preservation when clump insertion grows the owner arrays.
int main() try {
    DEMSolver solver(1);
    solver.SetVerbosity("ERROR");
    solver.InstructBoxDomainDimension(12, 12, 12);
    solver.SetGravitationalAcceleration(make_float3(0));
    solver.SetTimeStepSize(1.e-5);
    solver.SetExpandFactor(1.e-4f);
    solver.SetCDUpdateFreq(1);
    // Separate the bare surface and sphere AABBs into disjoint bins; only shell expansion bridges the gap.
    solver.SetInitBinSize(0.025);
    solver.DefineContactForceModel("force = make_float3(0.f, 0.f, static_cast<float>(overlapDepth));");
    auto material = solver.LoadMaterial({{"E", 1.e7}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.4}, {"Crr", 0.0}});
    auto sphere = solver.LoadSphereType(1, 0.05f, material);
    auto spheres = solver.AddClumps(sphere, std::vector<float3>{make_float3(-1.75f, 0.125f, 0.08f)});
    spheres->SetFamily(1);
    solver.SetFamilyFixed(1);
    solver.SetFamilyFixed(100);
    auto addShell = [&](float x, float thickness) {
        auto mesh = solver.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/plane_20by20.obj").string(), material);
        mesh->Scale(0.05f);
        mesh->SetInitPos(make_float3(x, 0, 0));
        mesh->SetShellThickness(thickness);
        mesh->SetFamily(100);
        return solver.Track(mesh);
    };
    auto first = addShell(-2.f, 0.1f);
    auto second = addShell(2.f, 0.12f);
    solver.Initialize();
    solver.DoDynamicsThenSync(1.e-5);
    checkShellForce(first);
    checkShellForce(second, 0.f);

    solver.ClearCache();
    auto added = solver.AddClumps(sphere, std::vector<float3>{make_float3(2.25f, 0.125f, 0.08f)});
    added->SetFamily(1);
    solver.Update();
    solver.DoDynamicsThenSync(1.e-5);
    checkShellForce(first);
    checkShellForce(second, 0.03f);
    std::cout << "PASS: shell thickness reaches contact detection at initialization and after insertion.\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
