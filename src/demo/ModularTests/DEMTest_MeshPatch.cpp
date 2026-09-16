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

#include <algorithm>
#include <filesystem>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <map>

using namespace deme;
using namespace std::filesystem;

int main() {
    auto fail = [](const std::string& message) {
        std::cerr << "FAIL: " << message << std::endl;
        return 1;
    };

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
            if (threshold > 180.0f && num_patches != 1) {
                return fail(
                    "high-threshold vanilla SplitIntoConvexPatches should merge the connected cube mesh into one "
                    "patch");
            }
        }

        // Test the extended patch-split API added
        std::cout << "\n--- Test 3: Extended Patch Split Options and Quality Report ---" << std::endl;
        auto legacy_merge_mesh = std::make_shared<DEMMesh>();
        auto extended_mesh = std::make_shared<DEMMesh>();
        if (!legacy_merge_mesh->LoadWavefrontMesh((GET_DATA_PATH() / "mesh/cube.obj").string()) ||
            !extended_mesh->LoadWavefrontMesh((GET_DATA_PATH() / "mesh/cube.obj").string())) {
            return fail("failed to reload cube mesh for extended patch split test");
        }

        const unsigned int legacy_patches = legacy_merge_mesh->SplitIntoConvexPatches(180.0f);
        DEMMesh::PatchSplitOptions split_opt;
        split_opt.patch_normal_max_deg = 30.0f;
        split_opt.patch_min = 2;
        split_opt.patch_max = 4;
        split_opt.block_concave_edges = true;
        DEMMesh::PatchQualityOptions quality_opt;
        DEMMesh::PatchQualityReport report;
        const unsigned int extended_patches =
            extended_mesh->SplitIntoConvexPatches(180.0f, split_opt, &report, quality_opt);

        std::cout << "Legacy 180-degree split patches: " << legacy_patches << std::endl;
        std::cout << "Extended split patches: " << extended_patches << std::endl;
        std::cout << "Report status: " << static_cast<int>(report.constraint_status)
                  << ", overall quality: " << static_cast<int>(report.overall) << std::endl;

        if (legacy_patches != 1) {
            return fail("legacy 180-degree split should merge the connected cube mesh into one patch");
        }
        if (extended_patches <= legacy_patches) {
            return fail("patch_normal_max_deg option did not split the cube more strictly than the legacy path");
        }
        if (extended_patches != extended_mesh->GetNumPatches() || report.achieved_patches != extended_patches) {
            return fail("extended patch count disagrees with the mesh or quality report");
        }
        if (!extended_mesh->ArePatchesExplicitlySet()) {
            return fail("extended split did not mark patch IDs as explicitly set");
        }
        if (report.requested_min != split_opt.patch_min || report.requested_max != split_opt.patch_max) {
            return fail("quality report did not preserve requested patch-count bounds");
        }
        if (report.per_patch.size() != extended_patches) {
            return fail("quality report per-patch array size does not match achieved patch count");
        }

        unsigned int reported_tris = 0;
        std::vector<unsigned int> counted_tris(extended_patches, 0);
        for (const auto& patch_report : report.per_patch) {
            reported_tris += patch_report.n_tris;
        }
        for (patchID_t patch_id : extended_mesh->GetPatchIDs()) {
            if (patch_id < 0 || patch_id >= static_cast<patchID_t>(extended_patches)) {
                return fail("extended split produced an out-of-range patch ID");
            }
            counted_tris[patch_id]++;
        }
        if (reported_tris != extended_mesh->GetNumTriangles()) {
            return fail("quality report does not account for every triangle");
        }
        for (unsigned int p = 0; p < extended_patches; ++p) {
            if (report.per_patch[p].n_tris != counted_tris[p]) {
                return fail("quality report per-patch triangle count disagrees with patch IDs");
            }
        }

        DEMMesh::PatchConstraintStatus expected_status = DEMMesh::PatchConstraintStatus::SATISFIED;
        if (extended_patches > split_opt.patch_max) {
            expected_status = DEMMesh::PatchConstraintStatus::TOO_MANY_UNMERGEABLE;
        } else if (extended_patches < split_opt.patch_min) {
            expected_status = DEMMesh::PatchConstraintStatus::TOO_FEW_UNSPLITTABLE;
        }
        if (report.constraint_status != expected_status) {
            return fail("quality report constraint status does not match achieved patch count");
        }

        // Test manual patch ID setting
        std::cout << "\n--- Test 4: Manual Patch ID Setting ---" << std::endl;
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

        // Both convenience names must assign one consecutive patch ID to every triangle.
        cube_mesh->SetEachTriangleAsPatch();
        if (cube_mesh->GetNumPatches() != num_tris || cube_mesh->GetPatchIDs().size() != num_tris) {
            return fail("SetEachTriangleAsPatch did not create one patch per triangle");
        }
        for (size_t triangle = 0; triangle < num_tris; triangle++) {
            if (cube_mesh->GetPatchIDs()[triangle] != static_cast<patchID_t>(triangle)) {
                return fail("SetEachTriangleAsPatch did not assign consecutive patch IDs");
            }
        }

    } else {
        std::cout << "Failed to load cube mesh" << std::endl;
    }

    // Test with sphere mesh if available
    std::cout << "\n--- Test 5: Sphere Mesh ---" << std::endl;
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
    std::cout << "\n--- Test 6: Empty Mesh ---" << std::endl;
    auto empty_mesh = std::make_shared<DEMMesh>();
    std::cout << "Empty mesh default patches: " << empty_mesh->GetNumPatches() << " (expected: 1)" << std::endl;
    std::cout << "Patches explicitly set: " << (empty_mesh->ArePatchesExplicitlySet() ? "yes" : "no")
              << " (expected: no)" << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
