// Copyright (c) 2021, SBEL GPU Development Team
// Copyright (c) 2021, University of Wisconsin - Madison
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "DEM/API.h"

using namespace deme;

namespace {

bool near(float actual, float expected, float tolerance = 1.0e-5f) {
    return std::abs(actual - expected) <= tolerance;
}

}  // namespace

// Verify that visualization snapshots expose current transformed geometry and honor category filtering.
int main() {
    DEMSolver solver(1);
    solver.SetVerbosity("ERROR");
    solver.InstructBoxDomainDimension(8.0f, 8.0f, 8.0f);

    auto material = solver.LoadMaterial({{"E", 1.0e7}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.4}, {"Crr", 0.0}});
    auto sphere = solver.LoadSphereType(1.0f, 0.25f, material);
    solver.AddClumps(sphere, std::vector<float3>{make_float3(1.0f, 2.0f, 3.0f)});

    auto mesh = solver.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/plane_20by20.obj").string(), material);
    mesh->Scale(0.1f);
    mesh->SetFamily(1);
    mesh->AddGeometryWildcard("mesh_test_value", 7.0f);
    solver.SetMeshOutputContent({"QUAT", "ABSV", "VEL", "ANG_VEL", "ABS_ACC", "ACC", "ANG_ACC", "FAMILY", "MAT",
                                 "GEO_WILDCARD", "OWNER", "MESH_ID", "TRI_ID", "PATCH_ID"});
    solver.SetFamilyFixed(1);
    solver.AddBCPlane(make_float3(0, 0, 0), make_float3(0, 0, 1), material);
    solver.Initialize();

    const auto sphere_snapshot = solver.GetVisualizationSnapshot(true, false);
    if (sphere_snapshot.spheres.size() != 1 || !sphere_snapshot.triangles.empty()) {
        std::cerr << "FAIL: sphere-only snapshot returned the wrong geometry categories." << std::endl;
        return 1;
    }
    const auto& rendered_sphere = sphere_snapshot.spheres.front();
    if (!near(rendered_sphere.position.x, 1.0f) || !near(rendered_sphere.position.y, 2.0f) ||
        !near(rendered_sphere.position.z, 3.0f) || !near(rendered_sphere.radius, 0.25f)) {
        std::cerr << "FAIL: visualization sphere does not match its initialized world transform." << std::endl;
        return 1;
    }

    // Exercise the public asynchronous writer and check the core legacy-VTK sections that ParaView consumes.
    const auto vtk_path = std::filesystem::temp_directory_path() /
                          ("deme_sphere_output_" + std::to_string(reinterpret_cast<std::uintptr_t>(&solver)) + ".vtk");
    solver.SetOutputFormat(OUTPUT_FORMAT::VTK);
    solver.WriteSphereFile(vtk_path);
    solver.WaitForPendingOutput();
    std::ifstream vtk_file(vtk_path);
    std::ostringstream vtk_contents;
    vtk_contents << vtk_file.rdbuf();
    std::filesystem::remove(vtk_path);
    const std::string vtk_text = vtk_contents.str();
    if (vtk_text.find("DATASET POLYDATA") == std::string::npos ||
        vtk_text.find("POINTS 1 float") == std::string::npos ||
        vtk_text.find("SCALARS r float 1") == std::string::npos) {
        std::cerr << "FAIL: VTK sphere output is missing required ParaView point-data sections." << std::endl;
        return 1;
    }

    const auto analytical_vtk_path =
        std::filesystem::temp_directory_path() /
        ("deme_analytical_output_" + std::to_string(reinterpret_cast<std::uintptr_t>(&solver)) + ".vtk");
    solver.WriteAnalyticalFile(analytical_vtk_path);
    solver.WaitForPendingOutput();
    std::ifstream analytical_vtk_file(analytical_vtk_path);
    std::ostringstream analytical_vtk_contents;
    analytical_vtk_contents << analytical_vtk_file.rdbuf();
    std::filesystem::remove(analytical_vtk_path);
    const std::string analytical_vtk_text = analytical_vtk_contents.str();
    if (analytical_vtk_text.find("DATASET POLYDATA") == std::string::npos ||
        analytical_vtk_text.find("POLYGONS 2 8") == std::string::npos ||
        analytical_vtk_text.find("SCALARS component_type int 1") == std::string::npos) {
        std::cerr << "FAIL: analytical VTK output is missing the clipped plane surface or metadata." << std::endl;
        return 1;
    }

    const auto mesh_vtk_path =
        std::filesystem::temp_directory_path() /
        ("deme_mesh_output_" + std::to_string(reinterpret_cast<std::uintptr_t>(&solver)) + ".vtk");
    solver.WriteMeshFile(mesh_vtk_path);
    solver.WaitForPendingOutput();
    std::ifstream mesh_vtk_file(mesh_vtk_path);
    std::ostringstream mesh_vtk_contents;
    mesh_vtk_contents << mesh_vtk_file.rdbuf();
    std::filesystem::remove(mesh_vtk_path);
    const std::string mesh_vtk_text = mesh_vtk_contents.str();
    const std::vector<std::string> required_mesh_fields = {"CELL_DATA 2",
                                                           "SCALARS quaternion float 4",
                                                           "SCALARS absv float 1",
                                                           "VECTORS velocity float",
                                                           "VECTORS angular_velocity float",
                                                           "SCALARS abs_acc float 1",
                                                           "VECTORS acceleration float",
                                                           "VECTORS angular_acceleration float",
                                                           "SCALARS family int 1",
                                                           "SCALARS material int 1",
                                                           "SCALARS owner int 1",
                                                           "SCALARS mesh_id int 1",
                                                           "SCALARS tri_id int 1",
                                                           "SCALARS patch_id int 1",
                                                           "SCALARS mesh_test_value float 1"};
    for (const auto& required_field : required_mesh_fields) {
        if (mesh_vtk_text.find(required_field) == std::string::npos) {
            std::cerr << "FAIL: mesh VTK output is missing requested CELL_DATA field: " << required_field << std::endl;
            return 1;
        }
    }

    const auto triangle_snapshot = solver.GetVisualizationSnapshot(false, true);
    if (!triangle_snapshot.spheres.empty() || triangle_snapshot.triangles.size() != mesh->GetNumTriangles()) {
        std::cerr << "FAIL: triangle-only snapshot returned the wrong geometry categories." << std::endl;
        return 1;
    }

    std::cout << "PASS: visualization snapshots expose and filter current solver geometry." << std::endl;
    return 0;
}
