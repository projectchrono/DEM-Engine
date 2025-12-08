//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A test that verifies mesh patch location functionality.
// This test checks:
// 1. Default patch locations for single-patch meshes (should be 0,0,0)
// 2. Computed patch locations for multi-patch meshes
// 3. Explicitly set patch locations
// =============================================================================

#include <DEM/BdrsAndObjs.h>
#include <DEM/utils/HostSideHelpers.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>

using namespace deme;

// Helper to check if two float3 values are approximately equal
bool approxEqual(const float3& a, const float3& b, float tolerance = 1e-5f) {
    return std::abs(a.x - b.x) < tolerance && 
           std::abs(a.y - b.y) < tolerance && 
           std::abs(a.z - b.z) < tolerance;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "DEM Mesh Patch Location Test" << std::endl;
    std::cout << "========================================" << std::endl;

    int test_failures = 0;

    // Test 1: Single patch mesh - default location should be (0,0,0)
    std::cout << "\n--- Test 1: Single Patch Mesh Default Location ---" << std::endl;
    auto mesh1 = std::make_shared<DEMMesh>();
    
    // Create a simple triangle mesh
    mesh1->m_vertices = {
        make_float3(0, 0, 0),
        make_float3(1, 0, 0),
        make_float3(0, 1, 0)
    };
    mesh1->m_face_v_indices = {make_int3(0, 1, 2)};
    mesh1->nTri = 1;
    mesh1->nPatches = 1;
    mesh1->m_patch_ids = {0};
    
    auto locations1 = mesh1->ComputePatchLocations();
    std::cout << "Number of patches: " << mesh1->GetNumPatches() << std::endl;
    std::cout << "Computed location for patch 0: (" 
              << locations1[0].x << ", " 
              << locations1[0].y << ", " 
              << locations1[0].z << ")" << std::endl;
    
    if (approxEqual(locations1[0], make_float3(0, 0, 0))) {
        std::cout << "✓ PASS: Single patch has location (0,0,0)" << std::endl;
    } else {
        std::cout << "✗ FAIL: Single patch should have location (0,0,0)" << std::endl;
        test_failures++;
    }

    // Test 2: Multiple patches - should compute centroids
    std::cout << "\n--- Test 2: Multi-Patch Mesh Computed Locations ---" << std::endl;
    auto mesh2 = std::make_shared<DEMMesh>();
    
    // Create a mesh with two triangles in different patches
    mesh2->m_vertices = {
        make_float3(0, 0, 0),    // Triangle 1 vertices (patch 0)
        make_float3(1, 0, 0),
        make_float3(0, 1, 0),
        make_float3(2, 0, 0),    // Triangle 2 vertices (patch 1)
        make_float3(3, 0, 0),
        make_float3(2, 1, 0)
    };
    mesh2->m_face_v_indices = {
        make_int3(0, 1, 2),  // Triangle 1
        make_int3(3, 4, 5)   // Triangle 2
    };
    mesh2->nTri = 2;
    mesh2->nPatches = 2;
    mesh2->m_patch_ids = {0, 1};
    
    auto locations2 = mesh2->ComputePatchLocations();
    std::cout << "Number of patches: " << mesh2->GetNumPatches() << std::endl;
    
    // Expected centroids:
    // Patch 0 triangle centroid: ((0+1+0)/3, (0+0+1)/3, 0) = (0.333, 0.333, 0)
    // Patch 1 triangle centroid: ((2+3+2)/3, (0+0+1)/3, 0) = (2.333, 0.333, 0)
    float3 expected_loc0 = make_float3(1.0f/3.0f, 1.0f/3.0f, 0.0f);
    float3 expected_loc1 = make_float3(7.0f/3.0f, 1.0f/3.0f, 0.0f);
    
    std::cout << "Computed location for patch 0: (" 
              << locations2[0].x << ", " 
              << locations2[0].y << ", " 
              << locations2[0].z << ")" << std::endl;
    std::cout << "Expected location for patch 0: (" 
              << expected_loc0.x << ", " 
              << expected_loc0.y << ", " 
              << expected_loc0.z << ")" << std::endl;
    
    if (approxEqual(locations2[0], expected_loc0)) {
        std::cout << "✓ PASS: Patch 0 location is correct" << std::endl;
    } else {
        std::cout << "✗ FAIL: Patch 0 location is incorrect" << std::endl;
        test_failures++;
    }
    
    std::cout << "Computed location for patch 1: (" 
              << locations2[1].x << ", " 
              << locations2[1].y << ", " 
              << locations2[1].z << ")" << std::endl;
    std::cout << "Expected location for patch 1: (" 
              << expected_loc1.x << ", " 
              << expected_loc1.y << ", " 
              << expected_loc1.z << ")" << std::endl;
    
    if (approxEqual(locations2[1], expected_loc1)) {
        std::cout << "✓ PASS: Patch 1 location is correct" << std::endl;
    } else {
        std::cout << "✗ FAIL: Patch 1 location is incorrect" << std::endl;
        test_failures++;
    }

    // Test 3: Explicitly set patch locations
    std::cout << "\n--- Test 3: Explicitly Set Patch Locations ---" << std::endl;
    auto mesh3 = std::make_shared<DEMMesh>();
    // Initialize mesh with proper patch setup
    mesh3->nTri = 0;  // No actual triangles needed for this test
    mesh3->nPatches = 2;
    mesh3->m_patch_ids.resize(0);  // Ensure consistency
    
    std::vector<float3> manual_locations = {
        make_float3(1.0f, 2.0f, 3.0f),
        make_float3(4.0f, 5.0f, 6.0f)
    };
    
    mesh3->SetPatchLocations(manual_locations);
    
    std::cout << "Patch locations explicitly set: " 
              << (mesh3->ArePatchLocationsExplicitlySet() ? "yes" : "no") << std::endl;
    
    const auto& retrieved_locations = mesh3->GetPatchLocations();
    std::cout << "Retrieved location for patch 0: (" 
              << retrieved_locations[0].x << ", " 
              << retrieved_locations[0].y << ", " 
              << retrieved_locations[0].z << ")" << std::endl;
    std::cout << "Retrieved location for patch 1: (" 
              << retrieved_locations[1].x << ", " 
              << retrieved_locations[1].y << ", " 
              << retrieved_locations[1].z << ")" << std::endl;
    
    if (approxEqual(retrieved_locations[0], manual_locations[0]) && 
        approxEqual(retrieved_locations[1], manual_locations[1])) {
        std::cout << "✓ PASS: Manually set patch locations are correct" << std::endl;
    } else {
        std::cout << "✗ FAIL: Manually set patch locations are incorrect" << std::endl;
        test_failures++;
    }

    // Summary
    std::cout << "\n========================================" << std::endl;
    if (test_failures == 0) {
        std::cout << "All tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } else {
        std::cout << "FAILED " << test_failures << " test(s)" << std::endl;
        std::cout << "========================================" << std::endl;
        return 1;
    }
}
