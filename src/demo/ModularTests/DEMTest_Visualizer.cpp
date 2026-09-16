// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <chrono>
#include <thread>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "DEM/API.h"
#include "../VisualizerDemoLoop.h"
#include "DEM/utils/DEMVisualizer.h"

using namespace deme;
namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
}  // namespace

// Requires a display and CUDA. Exercise actual shaders, depth picking, filters, UI, resize, and resource recreation.
int main(int argc, char** argv) try {
    DEMSolver solver(1);
    solver.SetVerbosity("ERROR");
    solver.InstructBoxDomainDimension(20, 20, 20);
    solver.SetGravitationalAcceleration(make_float3(0));
    auto material = solver.LoadMaterial({{"E", 1.e7}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.4}, {"Crr", 0.0}});
    auto type = solver.LoadSphereType(1, 0.5f, material);
    auto sphere = solver.AddClumps(type, std::vector<float3>{make_float3(0)});
    auto track = solver.Track(sphere);
    auto mesh = solver.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/plane_20by20.obj").string(), material);
    mesh->Scale(0.15f);
    mesh->SetInitPos(make_float3(0, 1, 0));
    mesh->SetInitQuat(make_float4(std::sqrt(0.5f), 0, 0, std::sqrt(0.5f)));
    mesh->SetFamily(1);
    solver.SetFamilyFixed(1);
    auto mesh_track = solver.Track(mesh);
    solver.Initialize();
    DEMVisualizer viewer(solver);
    viewer.SetWindowSize(1000, 700);
    viewer.SetTargetFPS(120);
    viewer.SetCameraPosition(make_float3(0, -5, 0));
    viewer.SetCameraTarget(make_float3(0));
    viewer.Initialize();
    viewer.Render();
    viewer.Render();
    require(viewer.ShouldStep(), "initial stepping disabled");
    viewer.SetPaused(true);
    require(!viewer.ShouldStep(), "pause did not stop step requests");
    viewer.RequestStep();
    require(viewer.ShouldStep() && !viewer.ShouldStep(), "single step was not consumed exactly once");
    viewer.PickAt(330, 350);
    require(viewer.GetSelectedOwner() == track->GetOwnerID() && viewer.IsSelectedSphere(),
            "front sphere picking failed");
    // Pose and family changes must reach the existing GPU buffers without rebuilding scene geometry.
    track->SetPos(make_float3(2, 0, 0));
    viewer.Render();
    viewer.PickAt(330, 350);
    require(viewer.GetSelectedOwner() == mesh_track->GetOwnerID(), "owner pose upload did not move the sphere");
    track->SetPos(make_float3(0));
    track->SetFamily(7);
    viewer.SetFamilyVisible(7, false);
    viewer.Render();
    viewer.PickAt(330, 350);
    require(viewer.GetSelectedOwner() == mesh_track->GetOwnerID(), "runtime family filter did not update");
    track->SetFamily(0);
    viewer.SetFamilyColor(0, {80, 160, 230, 0});
    viewer.Render();
    viewer.PickAt(330, 350);
    require(viewer.GetSelectedOwner() == mesh_track->GetOwnerID(), "zero-alpha geometry remained pickable");
    viewer.SetFamilyColor(0, {80, 160, 230, 255});
    viewer.SetFamilyVisible(0, false);
    viewer.Render();
    viewer.PickAt(330, 350);
    require(viewer.GetSelectedOwner() == mesh_track->GetOwnerID() && !viewer.IsSelectedSphere(),
            "hidden sphere occluded mesh picking");
    // Same-count deformation must invalidate mesh buffers; move the backdrop out of the center ray and restore it.
    auto nodes = mesh->m_vertices;
    auto shifted = nodes;
    for (auto& p : shifted)
        p.x += 5;
    solver.SetTriNodeRelPos(mesh_track->GetOwnerID(), 0, shifted);
    viewer.Render();
    viewer.PickAt(330, 350);
    require(viewer.GetSelectedOwner() == NULL_BODYID, "deformed geometry stayed in the old picking location");
    solver.SetTriNodeRelPos(mesh_track->GetOwnerID(), 0, nodes);
    viewer.Render();
    viewer.PickAt(330, 350);
    require(viewer.GetSelectedOwner() == mesh_track->GetOwnerID(), "restored mesh was not uploaded");
    viewer.SetRenderTriangles(false);
    viewer.Render();
    viewer.PickAt(330, 350);
    require(viewer.GetSelectedOwner() == NULL_BODYID, "background picking did not clear selection");
    viewer.SetRenderTriangles(true);
    viewer.SetFamilyVisible(0, true);
    viewer.SetColorMode(DEMVisualizerColorMode::SPEED);
    viewer.Render();
    viewer.SetColorMode(DEMVisualizerColorMode::HEIGHT);
    viewer.Render();
    viewer.SetColorMode(DEMVisualizerColorMode::FAMILY);
    viewer.SetWindowSize(1200, 800);
    viewer.Render();
    viewer.Render();
    viewer.PickAt(430, 400);
    require(viewer.GetSelectedOwner() == track->GetOwnerID(), "picking after resize failed");
    viewer.FrameSelected();
    viewer.Render();
    viewer.FrameAll();
    viewer.Render();
    if (argc > 1) {
        viewer.RequestScreenshot(argv[1]);
        viewer.Render();
        require(std::filesystem::is_regular_file(argv[1]), "screenshot was not saved at the requested path");
    }
    // Output need not occur at all while the timestep loop services wall-clock display deadlines.
    viewer.SetPaused(false);
    // A slow FPS limiter must not force a redraw on every solver step.
    viewer.SetTargetFPS(30);
    VisualizerDemoLoop loop;
    const double initial_time = solver.GetSimTime();
    const float dt = static_cast<float>(solver.GetTimeStepSize());
    int redraws = 0;
    for (int i = 0; i < 20; ++i) {
        redraws += loop.Update(viewer);
        solver.DoDynamics(dt);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    solver.DoDynamicsThenSync(0.0);
    require(redraws < 10, "display FPS limiter starved the physics work interval");
    require(redraws >= 4, "wall-clock redraws waited for a simulation output frame");
    require(std::abs(solver.GetSimTime() - initial_time - 20 * static_cast<double>(dt)) < dt * 0.1,
            "display refresh changed the number of dynamics steps");
    viewer.Close();
    viewer.Close();
    viewer.Initialize();
    viewer.Render();
    viewer.Close();
    std::cout << "PASS: native rendering, depth picking, filtering, controls, resize and recreation.\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
