//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A demo that tests mesh patch splitting functionality.
// This demo loads a mesh and splits it into convex patches based on angle
// thresholds, demonstrating the mesh patch splitting utility.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/Samplers.hpp>

#include <filesystem>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <limits>

using namespace deme;
using namespace std::filesystem;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "DEM Mesh Patch Splitting Demo" << std::endl;
    std::cout << "========================================" << std::endl;

    // Test with a simple cube mesh
    std::cout << "\n--- Test 1: Cube Mesh with Default Patch Info ---" << std::endl;
    auto cube_mesh = std::make_shared<DEMMesh>();
    bool loaded = cube_mesh->LoadWavefrontMesh((GET_DATA_PATH() / "mesh/cube.obj").string());

    if (loaded) {
        std::cout << "Loaded cube mesh successfully" << std::endl;
        std::cout << "Number of triangles: " << cube_mesh->GetNumTriangles() << std::endl;
        std::cout << "Number of vertices: " << cube_mesh->GetNumNodes() << std::endl;

        // Test default patch info (should be all in patch 0)
        std::cout << "\nDefault patch info (assuming convex mesh):" << std::endl;
        std::cout << "Number of patches: " << cube_mesh->GetNumPatches() << std::endl;
        std::cout << "Patches explicitly set: " << (cube_mesh->ArePatchesExplicitlySet() ? "yes" : "no") << std::endl;
        const auto& default_patch_ids = cube_mesh->GetPatchIDs();
        std::cout << "All triangles in patch 0: "
                  << (std::all_of(default_patch_ids.begin(), default_patch_ids.end(), [](int id) { return id == 0; })
                          ? "yes"
                          : "no")
                  << std::endl;

        // Test different angle thresholds
        std::cout << "\n--- Test 2: Automatic Patch Splitting ---" << std::endl;
        float thresholds[] = {10.0f, 45.0f, 90.0f, 300.0f};

        for (float threshold : thresholds) {
            size_t num_patches = cube_mesh->SplitIntoConvexPatches(threshold);
            std::cout << "\nAngle threshold: " << std::fixed << std::setprecision(1) << threshold << " degrees"
                      << std::endl;
            std::cout << "Number of patches: " << num_patches << std::endl;
            std::cout << "Patches explicitly set: " << (cube_mesh->ArePatchesExplicitlySet() ? "yes" : "no")
                      << std::endl;

            // Show patch distribution
            const auto& patch_ids = cube_mesh->GetPatchIDs();

            // Count triangles per patch
            std::map<int, int> patch_counts;
            for (int patch_id : patch_ids) {
                patch_counts[patch_id]++;
            }

            std::cout << "Patch distribution:" << std::endl;
            for (const auto& entry : patch_counts) {
                std::cout << "  Patch " << entry.first << ": " << entry.second << " triangles" << std::endl;
            }
        }

        // Optimized patch settings for convex-focused splitting (prefer single patch)
        std::cout << "\n--- Test 2b: Optimized Convex Patch Splitting (Cube) ---" << std::endl;
        DEMMesh::PatchSplitOptions opt;
        opt.soft_angle_deg = -1.0f;
        opt.patch_normal_max_deg = -1.0f;
        opt.block_concave_edges = true;
        opt.concave_allow_deg = 0.0f;
        opt.patch_min = 1;
        opt.patch_max = std::numeric_limits<unsigned int>::max();
        opt.seed_largest_first = true;
        opt.auto_tune.enabled = false;

        DEMMesh::PatchQualityReport rep_cube;
        size_t num_patches_opt = cube_mesh->SplitIntoConvexPatches(120.0f, opt, &rep_cube);
        std::cout << "Optimized patches: " << num_patches_opt << " (quality "
                  << static_cast<int>(rep_cube.overall) << ")" << std::endl;

        // Test manual patch ID setting
        std::cout << "\n--- Test 3: Manual Patch ID Setting ---" << std::endl;
        size_t num_tris = cube_mesh->GetNumTriangles();
        std::vector<patchID_t> manual_patches(num_tris);
        // Split triangles into 3 patches based on index
        for (size_t i = 0; i < num_tris; ++i) {
            manual_patches[i] = i % 3;  // Assign patches 0, 1, 2 cyclically
        }

        cube_mesh->SetPatchIDs(manual_patches);
        std::cout << "Manually set patch IDs (cycling 0, 1, 2)" << std::endl;
        std::cout << "Number of patches: " << cube_mesh->GetNumPatches() << std::endl;
        std::cout << "Patches explicitly set: " << (cube_mesh->ArePatchesExplicitlySet() ? "yes" : "no") << std::endl;

        // Count triangles per patch
        const auto& manual_patch_ids = cube_mesh->GetPatchIDs();
        std::map<int, int> manual_patch_counts;
        for (int patch_id : manual_patch_ids) {
            manual_patch_counts[patch_id]++;
        }
        std::cout << "Manual patch distribution:" << std::endl;
        for (const auto& entry : manual_patch_counts) {
            std::cout << "  Patch " << entry.first << ": " << entry.second << " triangles" << std::endl;
        }
    } else {
        std::cout << "Failed to load cube mesh" << std::endl;
    }

    // Test with sphere mesh if available
    std::cout << "\n--- Test 4: Sphere Mesh ---" << std::endl;
    auto sphere_mesh = std::make_shared<DEMMesh>();
    loaded = sphere_mesh->LoadWavefrontMesh((GET_DATA_PATH() / "mesh/sphere.obj").string());

    if (loaded) {
        std::cout << "Loaded sphere mesh successfully" << std::endl;
        std::cout << "Number of triangles: " << sphere_mesh->GetNumTriangles() << std::endl;
        std::cout << "Number of vertices: " << sphere_mesh->GetNumNodes() << std::endl;

        // Optimized patch split (prefer single patch)
        DEMMesh::PatchSplitOptions opt;
        opt.soft_angle_deg = -1.0f;
        opt.patch_normal_max_deg = -1.0f;
        opt.block_concave_edges = true;
        opt.concave_allow_deg = 0.0f;
        opt.patch_min = 1;
        opt.patch_max = std::numeric_limits<unsigned int>::max();
        opt.seed_largest_first = true;
        opt.auto_tune.enabled = false;

        DEMMesh::PatchQualityReport rep_sphere;
        size_t num_patches = sphere_mesh->SplitIntoConvexPatches(120.0f, opt, &rep_sphere);
        std::cout << "Split into " << num_patches << " patches (optimized, quality "
                  << static_cast<int>(rep_sphere.overall) << ")" << std::endl;

        if (sphere_mesh->ArePatchesExplicitlySet()) {
            const auto& patch_ids = sphere_mesh->GetPatchIDs();

            // Count triangles per patch
            std::map<int, int> patch_counts;
            for (int patch_id : patch_ids) {
                patch_counts[patch_id]++;
            }

            std::cout << "Number of patches with different sizes:" << std::endl;
            std::map<int, int> size_distribution;
            for (const auto& entry : patch_counts) {
                size_distribution[entry.second]++;
            }
            for (const auto& entry : size_distribution) {
                std::cout << "  " << entry.second << " patches with " << entry.first << " triangles each" << std::endl;
            }
        }
    } else {
        std::cout << "Sphere mesh not available, skipping" << std::endl;
    }

    // Test edge case: empty mesh
    std::cout << "\n--- Test 5: Empty Mesh ---" << std::endl;
    auto empty_mesh = std::make_shared<DEMMesh>();
    std::cout << "Empty mesh default patches: " << empty_mesh->GetNumPatches() << " (expected: 1)" << std::endl;
    std::cout << "Patches explicitly set: " << (empty_mesh->ArePatchesExplicitlySet() ? "yes" : "no")
              << " (expected: no)" << std::endl;

    // Test concave mesh (drum)
    std::cout << "\n--- Test 6: Concave Drum Mesh (STL) ---" << std::endl;
    auto drum_mesh = std::make_shared<DEMMesh>();
    loaded = drum_mesh->LoadSTLMesh((GET_DATA_PATH() / "mesh/drum.stl").string());
    if (loaded) {
        std::cout << "Loaded drum mesh successfully" << std::endl;
        std::cout << "Number of triangles: " << drum_mesh->GetNumTriangles() << std::endl;
        std::cout << "Number of vertices: " << drum_mesh->GetNumNodes() << std::endl;

        DEMMesh::PatchSplitOptions opt;
        opt.soft_angle_deg = -1.0f;
        opt.patch_normal_max_deg = -1.0f;
        opt.block_concave_edges = true;
        opt.concave_allow_deg = 0.0f;
        opt.patch_min = 1;
        opt.patch_max = std::numeric_limits<unsigned int>::max();
        opt.seed_largest_first = true;
        opt.auto_tune.enabled = false;

        DEMMesh::PatchQualityReport rep_drum;
        size_t num_patches = drum_mesh->SplitIntoConvexPatches(120.0f, opt, &rep_drum);
        std::cout << "Split into " << num_patches << " patches (concave, quality "
                  << static_cast<int>(rep_drum.overall) << ")" << std::endl;
    } else {
        std::cout << "Drum mesh not available, skipping" << std::endl;
    }

    // Test PLY export with per-patch colors (debug view)
    std::cout << "\n--- Test 7: PLY Export with Patch Colors (per mesh) ---" << std::endl;
    {
        path out_dir = current_path();
        out_dir /= "DemoOutput_MeshPatch";
        create_directory(out_dir);

        auto export_mesh = [&](const std::string& label, const path& mesh_path, bool is_stl) {
            DEMSolver DEMSim;
            DEMSim.SetVerbosity("INFO");
            DEMSim.SetMeshOutputFormat("PLY");
            DEMSim.EnableMeshPatchColorOutput(true);
            DEMSim.InstructBoxDomainDimension(10, 10, 10);
            DEMSim.SetMeshUniversalContact(true);

            auto mat_type = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}});

            std::shared_ptr<DEMMesh> mesh_template;
            if (is_stl) {
                mesh_template = DEMSim.LoadMeshType(mesh_path.string(), mat_type, true, false);
            } else {
                mesh_template = DEMSim.LoadMeshType(mesh_path.string(), mat_type, true, false);
            }

            if (!mesh_template) {
                std::cout << "Failed to load mesh template for " << label << std::endl;
                return;
            }

            DEMMesh::PatchSplitOptions opt;
            opt.soft_angle_deg = -1.0f;
            opt.patch_normal_max_deg = -1.0f;
            opt.block_concave_edges = true;
            opt.concave_allow_deg = 0.0f;
            opt.patch_min = 1;
            opt.patch_max = std::numeric_limits<unsigned int>::max();
            opt.seed_largest_first = true;
            opt.auto_tune.enabled = false;

            mesh_template->SplitIntoConvexPatches(120.0f, opt);
            mesh_template->SetMaterial(mat_type);

            auto mesh_instance = DEMSim.AddMeshFromTemplate(mesh_template, make_float3(0, 0, 0));
            mesh_instance->SetFamily(0);
            mesh_instance->SetMass(1000.);
            mesh_instance->SetMOI(make_float3(200., 200., 200.));

            DEMSim.Initialize();

            path ply_file = out_dir / ("mesh_patch_colors_" + label + ".ply");
            DEMSim.WriteMeshFile(ply_file);
            DEMSim.WaitForPendingOutput();
            std::cout << "Wrote patch-colored PLY to: " << ply_file << std::endl;
        };

        export_mesh("cube", GET_DATA_PATH() / "mesh/cube.obj", false);
        export_mesh("sphere", GET_DATA_PATH() / "mesh/sphere.obj", false);
        export_mesh("drum", GET_DATA_PATH() / "mesh/drum.stl", true);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
