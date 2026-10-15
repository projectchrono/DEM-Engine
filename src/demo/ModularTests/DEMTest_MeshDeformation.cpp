// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "DEM/API.h"
#include "algorithms/DEMMeshDeformation.h"
#include "core/utils/CudaDebugSync.hpp"

using namespace deme;
namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
bool approxEqual(float3 a, float3 b) {
    return std::abs(a.x - b.x) < 2.e-5f && std::abs(a.y - b.y) < 2.e-5f && std::abs(a.z - b.z) < 2.e-5f;
}
void compare(const std::vector<float3>& actual, const std::vector<float3>& expected, const char* message) {
    require(actual.size() == expected.size(), message);
    for (size_t i = 0; i < actual.size(); ++i)
        require(approxEqual(actual[i], expected[i]), message);
}

// Keep input allocations alive across calls, as an external GPU solid-mechanics solver would.
struct DeviceInput {
    int device;
    float3* data = nullptr;
    DeviceInput(size_t count, int device) : device(device) {
        ScopedCudaDevice scope(device);
        DEME_GPU_CALL(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float3)));
    }
    ~DeviceInput() {
        ScopedCudaDevice scope(device);
        cudaFree(data);
    }
    void upload(const std::vector<float3>& values) {
        ScopedCudaDevice scope(device);
        DEME_GPU_CALL(cudaMemcpy(data, values.data(), values.size() * sizeof(float3), cudaMemcpyHostToDevice));
    }
};

// Reading through the tracker must download both GPU vertices and the actual relPosPatch values used by forces.
void checkAutomatic(const std::shared_ptr<DEMTracker>& tracker, const std::vector<float3>& expected) {
    auto mesh = tracker->GetMesh();
    compare(mesh->GetCoordsVertices(), expected, "device/host vertex mismatch");
    compare(mesh->GetPatchLocations(), mesh->ComputePatchLocations(0.7f, 0.03f, 1.f),
            "automatic GPU patch centers are stale");
}
// Compare the CUDA reduction to CPU geometry for a patch spanning multiple block strides, opposing normals,
// and a zero-area patch. Include an unused vertex so both paths must use whole-mesh bounds, not patch bounds.
void checkCenterReduction(int device) {
    ScopedCudaDevice scope(device);
    DEMMesh mesh;
    mesh.m_vertices = {make_float3(0, 0, 2), make_float3(1, 0, 2), make_float3(0, 1, 2), make_float3(5, 0, 0)};
    mesh.m_face_v_indices.assign(257, make_int3(0, 1, 2));
    mesh.m_patch_ids.assign(257, 0);
    mesh.m_face_v_indices.insert(mesh.m_face_v_indices.end(),
                                 {make_int3(0, 1, 2), make_int3(0, 2, 1), make_int3(0, 0, 0)});
    mesh.m_patch_ids.insert(mesh.m_patch_ids.end(), {1, 1, 2});
    mesh.nTri = mesh.m_face_v_indices.size();
    mesh.nPatches = 3;
    DualArray<float3> vertices(mesh.m_vertices), a(mesh.nTri), b(mesh.nTri), c(mesh.nTri), centers(size_t(3));
    DualArray<size_t> offsets(std::vector<size_t>{0, 257, 259, 260}), triangles(mesh.nTri);
    DualArray<double> diagonal(size_t(1));
    for (size_t i = 0; i < mesh.nTri; ++i) {
        const auto face = mesh.m_face_v_indices[i];
        a[i] = mesh.m_vertices[face.x];
        b[i] = mesh.m_vertices[face.y];
        c[i] = mesh.m_vertices[face.z];
        triangles[i] = i;
    }
    vertices.toDevice();
    a.toDevice();
    b.toDevice();
    c.toDevice();
    offsets.toDevice();
    triangles.toDevice();
    for (MeshPatchCenterOptions options :
         {MeshPatchCenterOptions{}, {0.f, 0.01f, 0.8f}, {2.f, 0.2f, 0.f}, {0.7f, 0.03f, 1.f}}) {
        RefreshMeshPatchCenters(offsets.data(), triangles.data(), size_t(3), a.data(), b.data(), c.data(),
                                centers.data(), vertices.data(), vertices.size(), diagonal.data(), options, nullptr);
        DEME_GPU_CALL(cudaDeviceSynchronize());
        centers.toHost();
        compare(std::vector<float3>(centers.host(), centers.host() + 3),
                mesh.ComputePatchLocations(options.alpha, options.beta, options.gamma),
                "CPU/GPU center reduction disagreement");
    }
}

// A cube below an upward-facing open plane produces deep back-face triangle pairings. Fresh centers reject these;
// centers left at the cube's previous local position admit them. The open plane cannot mask that mistake with its
// own center guard. Compare deformation against loading the target geometry directly, without changing the rule.
float contactForceAfterDeformation(bool deform, bool refresh) {
    const bool trace = std::getenv("DEME_MESH_TRACE") != nullptr;
    if (trace)
        std::cerr << "CONTACT CASE deform=" << deform << " refresh=" << refresh << ": construct\n";
    DEMSolver solver(1);
    solver.SetVerbosity("ERROR");
    solver.InstructBoxDomainDimension(8, 8, 8);
    solver.SetGravitationalAcceleration(make_float3(0));
    solver.SetMeshUniversalContact(true);
    solver.SetCDUpdateFreq(1);
    solver.SetInitBinNumTarget(8);
    solver.SetErrorOutAvgContacts(10000);
    solver.SetTimeStepSize(1.e-5);
    solver.DefineContactForceModel("force = make_float3(0.f, 0.f, static_cast<float>(overlapDepth));");
    auto material = solver.LoadMaterial({{"E", 1.e7}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.4}, {"Crr", 0.0}});
    auto cube = solver.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cube.obj").string(), material);
    cube->Scale(0.2f);
    std::vector<patchID_t> patches(cube->GetNumTriangles());
    for (size_t i = 0; i < patches.size(); ++i)
        patches[i] = i % 2;
    cube->SetPatchIDs(patches);
    cube->SetMaterial(material);
    cube->SetFamily(100);
    solver.SetFamilyFixed(100);
    const auto target = cube->GetCoordsVertices();
    if (deform) {
        for (auto& v : cube->m_vertices)
            v.x += 2.f;
    }
    auto tracker = solver.Track(cube);
    auto plane = solver.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/plane_20by20.obj").string(), material);
    plane->Scale(0.05f);
    plane->SetInitPos(make_float3(0, 0, 0.096f));
    plane->SetFamily(101);
    solver.SetFamilyFixed(101);
    solver.Initialize();
    require(cube->IsWatertight() && !plane->IsWatertight(), "contact regression mesh topology changed");
    if (deform) {
        const int device = solver.GetGPUDeviceIDs().front();
        DeviceInput input(target.size(), device);
        input.upload(target);
        tracker->UpdateMeshFromDevice(input.data, device, true, refresh);
    }
    if (trace)
        std::cerr << "CONTACT CASE: initialized and deformed; starting dynamics\n";
    solver.SetTriTriPenetration(0.4f);
    solver.DoDynamicsThenSync(1.e-4);
    if (trace)
        std::cerr << "CONTACT CASE: dynamics complete; reading forces\n";
    std::vector<float3> points, forces;
    tracker->GetContactForces(points, forces);
    float total = 0;
    for (const auto& force : forces)
        total += length(force);
    if (trace)
        std::cerr << "CONTACT CASE: force=" << total << "; destroying solver\n";
    return total;
}
}  // namespace

// A cube resting shallowly on a plane puts each bottom-triangle contact at its old surface centroid.
// The legacy center guard rejects those valid contacts. Offset centers must keep them while tangential motion
// across the plane leaves the summed response unchanged. Per-triangle patches exercise the limiting patch size.
float shallowContactForce(float shift, bool legacy) {
    DEMSolver solver(1);
    solver.SetVerbosity("ERROR");
    if (legacy)
        solver.SetMeshPatchCenterOffsetAlpha(0);
    solver.InstructBoxDomainDimension(8, 8, 8);
    solver.SetGravitationalAcceleration(make_float3(0));
    solver.SetMeshUniversalContact(true);
    solver.SetCDUpdateFreq(1);
    solver.SetInitBinNumTarget(8);
    solver.SetErrorOutAvgContacts(10000);
    solver.SetTimeStepSize(1.e-5);
    solver.DefineContactForceModel("force = make_float3(0.f, 0.f, static_cast<float>(overlapDepth));");
    auto material = solver.LoadMaterial({{"E", 1.e7}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.4}});
    auto cube = solver.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cube.obj").string(), material);
    cube->Scale(0.2f);
    cube->SetEachTriangleAsPatch();
    cube->SetMaterial(material);
    cube->SetInitPos(make_float3(shift, 0, 0));
    cube->SetFamily(100);
    solver.SetFamilyFixed(100);
    auto tracker = solver.Track(cube);
    auto plane = solver.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/plane_20by20.obj").string(), material);
    // One large triangle contains both cube faces' projections completely. A triangulated plane seam would
    // split the contact polygons and move their centroids away from the cube's patch centers, hiding the bug.
    plane->m_vertices = {make_float3(-1, -1, 0), make_float3(1, -1, 0), make_float3(0, 1, 0)};
    plane->m_face_v_indices = {make_int3(0, 1, 2)};
    plane->nTri = 1;
    plane->SetPatchIDs({0});
    plane->SetInitPos(make_float3(0, 0, -0.096f));
    plane->SetFamily(101);
    solver.SetFamilyFixed(101);
    solver.Initialize();
    solver.SetTriTriPenetration(0.4f);
    solver.DoDynamicsThenSync(1.e-4);
    std::vector<float3> points, forces;
    tracker->GetContactForces(points, forces);
    float total = 0;
    for (const auto& force : forces)
        total += std::abs(force.z);
    return total;
}

// Cover shared-node increments, deferred centers, manual policy, lazy readers, and host/device interleaving.
int main() try {
    // Opt-in diagnostics preserve ordinary asynchronous execution for reproducing timing-sensitive failures.
    deme::SetCudaDebugSyncEnabled(std::getenv("DEME_MESH_DEBUG_SYNC") != nullptr);
    DEMSolver solver(1);
    solver.SetVerbosity("ERROR");
    // Exercise non-default settings through initialization, every deformation flavor, and later clump insertion.
    solver.SetMeshPatchCenterOffsetAlpha(0.7f);
    solver.SetMeshPatchCenterOffsetBeta(0.03f);
    solver.SetMeshPatchCenterOffsetGamma(1.f);
    using Setter = void (DEMSolver::*)(float);
    const Setter setters[] = {&DEMSolver::SetMeshPatchCenterOffsetAlpha, &DEMSolver::SetMeshPatchCenterOffsetBeta,
                              &DEMSolver::SetMeshPatchCenterOffsetGamma};
    std::cout << "Negative parameter and coordinate checks intentionally emit ERROR diagnostics.\n";
    for (auto setter : setters) {
        for (float invalid : {-1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            bool rejected = false;
            try {
                (solver.*setter)(invalid);
            } catch (const std::exception&) {
                rejected = true;
            }
            require(rejected, "invalid solver offset parameter accepted");
        }
    }
    solver.InstructBoxDomainDimension(40, 40, 40);
    solver.SetGravitationalAcceleration(make_float3(0));
    solver.SetTimeStepSize(1.e-5);
    auto material = solver.LoadMaterial({{"E", 1.e7}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.4}, {"Crr", 0.0}});
    auto sphere = solver.LoadSphereType(1, 0.1f, material);
    solver.AddClumps(sphere, std::vector<float3>{make_float3(0, 0, 10)});
    auto mesh = solver.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/plane_20by20.obj").string(), material);
    mesh->Scale(0.1f);
    // Multi-patch automatic centers must use initialization's supplied-normal winding correction too.
    mesh->UseNormals(true);
    for (auto& face : mesh->m_face_v_indices)
        std::swap(face.y, face.z);
    mesh->SetEachTriangleAsPatch();
    mesh->SetMaterial(material);
    mesh->SetFamily(1);
    solver.SetFamilyFixed(1);
    auto tracker = solver.Track(mesh);
    auto single = solver.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/plane_20by20.obj").string(), material);
    // Deliberately reverse cached winding; initialization must restore the supplied +Z normals, and deformation
    // must keep that corrected triangle order rather than scattering the raw face order back onto the GPU.
    single->UseNormals(true);
    for (auto& face : single->m_face_v_indices)
        std::swap(face.y, face.z);
    single->SetInitPos(make_float3(0, 0, -5));
    single->SetFamily(1);
    single->SetPatchLocations({make_float3(1, 2, 3)});
    auto single_tracker = solver.Track(single);
    solver.Initialize();
    for (auto setter : setters) {
        bool rejected = false;
        try {
            (solver.*setter)(1.f);
        } catch (const std::exception&) {
            rejected = true;
        }
        require(rejected, "runtime offset parameter mutation accepted");
    }
    checkAutomatic(tracker, mesh->GetCoordsVertices());
    const int device = solver.GetGPUDeviceIDs().front();
    checkCenterReduction(device);
    const size_t count = mesh->GetNumNodes();
    require(mesh->GetNumPatches() > 1, "test requires multiple triangle patches");
    DeviceInput input(std::max(count, (size_t)mesh->GetNumPatches()), device);
    auto expected = mesh->GetCoordsVertices();
    for (size_t i = 0; i < count; ++i) {
        expected[i].x += 0.25f * i;
        expected[i].z += 0.1f * i;
    }
    input.upload(expected);
    tracker->UpdateMeshFromDevice(input.data, device);
    checkAutomatic(tracker, expected);

    const auto old_centers = mesh->GetPatchLocations();
    std::vector<float3> increment(count, make_float3(0.125f, -0.25f, 0.375f));
    input.upload(increment);
    for (int step = 0; step < 8; ++step) {
        tracker->UpdateMeshByIncrementFromDevice(input.data, device, true, false);
        for (auto& v : expected)
            v += increment[0];
    }
    compare(tracker->GetMesh()->GetPatchLocations(), old_centers, "disabled refresh changed centers");
    tracker->UpdateMeshByIncrementFromDevice(input.data, device, false, true);
    for (auto& v : expected)
        v += increment[0];
    checkAutomatic(tracker, expected);

    // Do not read the cache between device and host updates: increments must use GPU-authoritative vertices.
    tracker->UpdateMeshByIncrementFromDevice(input.data, device);
    tracker->UpdateMeshByIncrement(increment);
    for (auto& v : expected)
        v += increment[0] * 2.f;
    checkAutomatic(tracker, expected);
    expected[0].z += 0.75f;
    tracker->UpdateMesh(expected);
    checkAutomatic(tracker, expected);

    // Exercise each CPU entry point directly. This is the first mesh, so its triangle start is zero.
    // Skipping refresh must still update vertices; resuming must include every skipped deformation.
    for (int api = 0; api < 4; ++api) {
        auto update = [&](bool refresh) {
            for (auto& v : expected)
                v += increment[0];
            switch (api) {
                case 0:
                    tracker->UpdateMesh(expected, refresh);
                    break;
                case 1:
                    tracker->UpdateMeshByIncrement(increment, refresh);
                    break;
                case 2:
                    solver.SetTriNodeRelPos(tracker->GetOwnerID(), 0, expected, refresh);
                    break;
                case 3:
                    solver.UpdateTriNodeRelPos(tracker->GetOwnerID(), 0, increment, refresh);
                    break;
            }
        };
        const auto saved_centers = mesh->GetPatchLocations();
        update(false);
        update(false);
        compare(mesh->GetCoordsVertices(), expected, "CPU skipped refresh failed to update vertices");
        compare(mesh->GetPatchLocations(), saved_centers, "CPU disabled refresh changed centers");
        update(true);
        checkAutomatic(tracker, expected);
        const std::vector<float3> explicit_centers(mesh->GetNumPatches(), make_float3(4, 5, 6));
        tracker->UpdateMeshPatchLocations(explicit_centers);
        update(false);
        update(true);
        compare(mesh->GetCoordsVertices(), expected, "CPU manual-center update failed to update vertices");
        compare(mesh->GetPatchLocations(), explicit_centers, "CPU toggle overwrote explicit centers");
        tracker->UseAutomaticMeshPatchLocations();
        checkAutomatic(tracker, expected);
    }
    // Omitted arguments on the public solver methods must retain the default refresh behavior too.
    for (auto& v : expected)
        v += increment[0];
    solver.SetTriNodeRelPos(tracker->GetOwnerID(), 0, expected);
    checkAutomatic(tracker, expected);
    solver.UpdateTriNodeRelPos(tracker->GetOwnerID(), 0, increment);
    for (auto& v : expected)
        v += increment[0];
    checkAutomatic(tracker, expected);

    std::vector<float3> manual(mesh->GetNumPatches(), make_float3(4, 5, 6));
    tracker->UpdateMeshPatchLocations(manual);
    input.upload(increment);
    tracker->UpdateMeshByIncrementFromDevice(input.data, device, true, true);
    for (auto& v : expected)
        v += increment[0];
    compare(tracker->GetMesh()->GetPatchLocations(), manual, "deformation overwrote user centers");
    manual[0] = make_float3(7, 8, 9);
    input.upload(manual);
    tracker->UpdateMeshPatchLocationsFromDevice(input.data, device);
    compare(tracker->GetMesh()->GetPatchLocations(), manual, "device center setter did not update GPU centers");
    tracker->UseAutomaticMeshPatchLocations();
    require(!mesh->ArePatchLocationsExplicitlySet(), "automatic policy did not resume");
    checkAutomatic(tracker, expected);

    auto single_nodes = single->GetCoordsVertices();
    for (auto& v : single_nodes)
        v.z += 1;
    single_tracker->UpdateMesh(single_nodes);
    compare(single->GetPatchLocations(), {make_float3(1, 2, 3)}, "setup explicit center was overwritten");
    single_tracker->UseAutomaticMeshPatchLocations();
    compare(single->GetPatchLocations(), {make_float3(0)}, "single-patch origin convention changed");

    // Visualization and async output are observation boundaries even without an intervening GetMesh call.
    input.upload(increment);
    tracker->UpdateMeshByIncrementFromDevice(input.data, device);
    for (auto& v : expected)
        v += increment[0];
    const auto scene = solver.GetVisualizationScene();
    for (const auto& triangle : scene.triangles) {
        if (triangle.owner == single_tracker->GetOwnerID()) {
            require(cross(triangle.b - triangle.a, triangle.c - triangle.a).z > 0,
                    "deformation lost initialization's winding correction");
        }
    }
    require(approxEqual(scene.triangles.front().a, expected[mesh->GetIndicesVertexes()[0].x]),
            "stale visualization geometry");
    tracker->UpdateMeshByIncrementFromDevice(input.data, device);
    for (auto& v : expected)
        v += increment[0];
    auto world = tracker->GetMeshNodesGlobal();
    const auto position = tracker->Pos();
    auto expected_world = expected;
    for (auto& v : expected_world)
        v += position;
    compare(world, expected_world, "global node getter returned stale coordinates");
    // Dirty the cache once more so output itself must perform the synchronization.
    tracker->UpdateMeshByIncrementFromDevice(input.data, device);
    for (auto& v : expected)
        v += increment[0];
    const auto output = std::filesystem::temp_directory_path() / "deme_mesh_deformation_test.vtk";
    solver.SetMeshOutputFormat("VTK");
    solver.WriteMeshFile(output.string());
    solver.WaitForPendingOutput();
    compare(mesh->GetCoordsVertices(), expected, "output failed to synchronize mesh vertices");
    require(std::filesystem::file_size(output) > 0, "mesh output is empty");
    std::filesystem::remove(output);

    // Input validation must reject non-finite values before mutating geometry.
    auto invalid = expected;
    invalid[0].x = std::numeric_limits<float>::quiet_NaN();
    input.upload(invalid);
    bool rejected = false;
    try {
        tracker->UpdateMeshFromDevice(input.data, device);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "non-finite device coordinates were accepted");
    checkAutomatic(tracker, expected);

    // Runtime insertion must not restore original host triangle/center arrays over device deformation.
    input.upload(increment);
    tracker->UpdateMeshByIncrementFromDevice(input.data, device);
    for (auto& v : expected)
        v += increment[0];
    solver.ClearCache();
    solver.AddClumps(sphere, std::vector<float3>{make_float3(3, 0, 10)});
    solver.Update();
    checkAutomatic(tracker, expected);
    solver.DoDynamicsThenSync(1.e-5);
    checkAutomatic(tracker, expected);

    int devices = 0;
    DEME_GPU_CALL(cudaGetDeviceCount(&devices));
    if (devices > 1) {
        DeviceInput remote(count, (device + 1) % devices);
        remote.upload(increment);
        tracker->UpdateMeshByIncrementFromDevice(remote.data, remote.device);
        for (auto& v : expected)
            v += increment[0];
        checkAutomatic(tracker, expected);
    } else {
        std::cout << "SKIP: remote-device update (only one CUDA device).\n";
    }
    const float reference_force = contactForceAfterDeformation(false, true);
    const float refreshed_force = contactForceAfterDeformation(true, true);
    const float stale_force = contactForceAfterDeformation(true, false);
    std::cout << "Contact guard forces: reference=" << reference_force << ", refreshed=" << refreshed_force
              << ", stale=" << stale_force << '\n';
    require(std::abs(reference_force) < 1.e-5f, "reference failed to reject deep back-face contacts");
    require(std::abs(reference_force - refreshed_force) < 1.e-5f, "deformation changed filtered patch forces");
    require(std::abs(stale_force - reference_force) > 1.e-3f,
            "contact regression did not exercise stale-center rejection");
    const float shallow = shallowContactForce(0.f, false);
    const float shifted = shallowContactForce(0.2f, false);
    const float legacy = shallowContactForce(0.f, true);
    std::cout << "Shallow forces: offset=" << shallow << ", shifted=" << shifted << ", legacy=" << legacy << '\n';
    require(shallow > 1.e-3f, "offset centers rejected valid shallow contacts");
    require(std::abs(shallow - shifted) < 1.e-5f, "sliding across the plane changed shallow force");
    require(legacy < shallow * 0.5f, "shallow regression did not distinguish old center policy");
    std::cout << "PASS: mesh device deformation, center policies, mixed updates, output, and insertion.\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
