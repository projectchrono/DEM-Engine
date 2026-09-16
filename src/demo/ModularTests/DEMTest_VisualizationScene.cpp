// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "DEM/API.h"

using namespace deme;
namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
bool near(float a, float b) {
    return std::abs(a - b) < 1.e-4f;
}

// Independently apply the quaternion-vector formula and compare cached local geometry against legacy snapshots.
float3 transform(float3 p, float3 position, float4 q) {
    float3 t = make_float3(2 * (q.y * p.z - q.z * p.y), 2 * (q.z * p.x - q.x * p.z), 2 * (q.x * p.y - q.y * p.x));
    return make_float3(position.x + p.x + q.w * t.x + q.y * t.z - q.z * t.y,
                       position.y + p.y + q.w * t.y + q.z * t.x - q.x * t.z,
                       position.z + p.z + q.w * t.z + q.x * t.y - q.y * t.x);
}
bool near(float3 a, float3 b) {
    return near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z);
}
void compare(DEMSolver& solver, const DEMVisualizationScene& scene, DEMVisualizationFrame& frame) {
    solver.GetVisualizationFrame(frame, true);
    auto snapshot = solver.GetVisualizationSnapshot();
    require(frame.revision == scene.revision, "unexpected geometry invalidation");
    require(scene.spheres.size() == snapshot.spheres.size(), "sphere count mismatch");
    for (size_t i = 0; i < scene.spheres.size(); ++i) {
        const auto& s = scene.spheres[i];
        require(near(transform(s.offset, frame.positions[s.owner], frame.orientations[s.owner]),
                     snapshot.spheres[i].position),
                "cached sphere transform mismatch");
        require(near(s.radius, snapshot.spheres[i].radius), "sphere radius mismatch");
        require(frame.families[s.owner] == snapshot.spheres[i].family, "family mismatch");
    }
    require(scene.triangles.size() == snapshot.triangles.size(), "triangle count mismatch");
    for (size_t i = 0; i < scene.triangles.size(); ++i) {
        const auto& t = scene.triangles[i];
        auto p = frame.positions[t.owner];
        auto q = frame.orientations[t.owner];
        require(near(transform(t.a, p, q), snapshot.triangles[i].a) &&
                    near(transform(t.b, p, q), snapshot.triangles[i].b) &&
                    near(transform(t.c, p, q), snapshot.triangles[i].c),
                "cached triangle transform mismatch");
    }
}
}  // namespace

// Exercise rigid motion, combined members, runtime families, same-count deformation, and insertion without a window.
int main() try {
    DEMSolver solver(1);
    solver.SetVerbosity("ERROR");
    solver.InstructBoxDomainDimension(20, 20, 20);
    solver.SetGravitationalAcceleration(make_float3(0));
    solver.SetTimeStepSize(1.e-5);
    auto material = solver.LoadMaterial({{"E", 1.e7}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.4}, {"Crr", 0.0}});
    auto clump = solver.LoadClumpType(1, make_float3(1), {0.2f, 0.3f},
                                      {make_float3(-0.4f, 0, 0), make_float3(0.4f, 0, 0)}, material);
    auto batch = solver.AddClumps(clump, std::vector<float3>{make_float3(1, 2, 3)});
    auto track = solver.Track(batch);
    auto sphere = solver.LoadSphereType(1, 0.2f, material);
    auto combined_type =
        solver.LoadCombinedClumpType({sphere, sphere}, {make_float3(-0.5f, 0, 0), make_float3(0.5f, 0, 0)}, {}, 0);
    auto combined = solver.AddCombinedFromTemplate(combined_type, make_float3(-3, 0, 2), make_float4(0, 0, 0, 1));
    auto combined_track = solver.Track(combined);
    auto mesh = solver.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/plane_20by20.obj").string(), material);
    mesh->Scale(0.1f);
    mesh->SetFamily(1);
    solver.SetFamilyFixed(1);
    auto mesh_track = solver.Track(mesh);
    solver.Initialize();
    auto scene = solver.GetVisualizationScene();
    DEMVisualizationFrame frame;
    compare(solver, scene, frame);
    auto* storage = frame.positions.data();
    track->SetOriQ(make_float4(0, 0, std::sqrt(0.5f), std::sqrt(0.5f)));
    track->SetPos(make_float3(2, 1, 3));
    track->SetFamily(7);
    track->SetVel(make_float3(0.1f, 0.2f, 0.3f));
    combined_track->SetOriQ(make_float4(0, 0, std::sqrt(0.5f), std::sqrt(0.5f)));
    compare(solver, scene, frame);
    require(frame.positions.data() == storage, "unchanged frame reallocated owner storage");
    auto owner = track->GetOwnerID();
    require(near(frame.velocities[owner], make_float3(0.1f, 0.2f, 0.3f)), "velocity field mismatch");
    solver.GetVisualizationFrame(frame, false);
    require(frame.velocities.empty(), "unrequested velocities retained");
    solver.DoDynamicsThenSync(1.e-4);
    compare(solver, scene, frame);
    auto nodes = mesh->m_vertices;
    for (auto& p : nodes)
        p.z += 0.3f;
    solver.SetTriNodeRelPos(mesh_track->GetOwnerID(), 0, nodes);
    solver.GetVisualizationFrame(frame);
    require(frame.revision != scene.revision, "deformation did not invalidate same-count geometry");
    scene = solver.GetVisualizationScene();
    compare(solver, scene, frame);
    size_t count = scene.spheres.size();
    auto revision = scene.revision;
    solver.ClearCache();
    solver.AddClumps(clump, std::vector<float3>{make_float3(5, 3, 2)});
    solver.Update();
    scene = solver.GetVisualizationScene();
    require(scene.revision != revision && scene.spheres.size() == count + 2, "insertion did not refresh geometry");
    compare(solver, scene, frame);
    std::cout << "PASS: cached scene/frame transforms, families, combined owners, deformation, insertion.\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
