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

        // Test with 30 degree threshold
        size_t num_patches = sphere_mesh->SplitIntoConvexPatches(30.0f);
        std::cout << "Split into " << num_patches << " patches (threshold: 30 degrees)" << std::endl;

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

    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
