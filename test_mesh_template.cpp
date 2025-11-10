//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A test/demo that demonstrates the new mesh template functionality.
// This test shows:
// 1. Loading a mesh as a template (not immediately in simulation)
// 2. Instantiating multiple mesh instances from the template at different locations
// 3. Duplicating an existing mesh object
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/Samplers.hpp>

#include <filesystem>
#include <cstdio>
#include <iostream>

using namespace deme;
using namespace std::filesystem;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "DEM Mesh Template Test" << std::endl;
    std::cout << "========================================" << std::endl;

    DEMSolver DEMSim(1);
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);

    // Define simulation world size
    float world_size = 10.0;
    DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size);
    DEMSim.InstructBoxDomainBoundingBC("none", NULL);

    // Set gravitational acceleration
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));

    // Load a material
    auto mat_type = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}});

    std::cout << "\n--- Test 1: Load Mesh Template ---" << std::endl;
    // Load a mesh as a template (not yet in simulation)
    auto mesh_template = DEMSim.LoadMeshType(
        (GET_DATA_PATH() / "mesh/cube.obj").string(),
        mat_type,
        true,  // load_normals
        false  // load_uv
    );
    
    if (mesh_template) {
        std::cout << "Successfully loaded mesh template" << std::endl;
        std::cout << "Number of triangles: " << mesh_template->GetNumTriangles() << std::endl;
        std::cout << "Number of vertices: " << mesh_template->GetNumNodes() << std::endl;
    } else {
        std::cout << "Failed to load mesh template" << std::endl;
        return 1;
    }

    std::cout << "\n--- Test 2: Instantiate Meshes from Template ---" << std::endl;
    // Create multiple instances of the mesh at different locations
    auto mesh1 = DEMSim.AddMeshFromTemplate(mesh_template, make_float3(0, 0, 0));
    std::cout << "Created mesh instance 1 at (0, 0, 0)" << std::endl;
    
    auto mesh2 = DEMSim.AddMeshFromTemplate(mesh_template, make_float3(2, 0, 0));
    std::cout << "Created mesh instance 2 at (2, 0, 0)" << std::endl;
    
    auto mesh3 = DEMSim.AddMeshFromTemplate(mesh_template, make_float3(0, 2, 0));
    std::cout << "Created mesh instance 3 at (0, 2, 0)" << std::endl;

    std::cout << "\n--- Test 3: Duplicate a Mesh Object ---" << std::endl;
    // Duplicate an existing mesh
    auto mesh_dup = DEMSim.Duplicate(mesh1);
    mesh_dup->SetInitPos(make_float3(-2, 0, 0));
    std::cout << "Duplicated mesh1 and moved to (-2, 0, 0)" << std::endl;

    std::cout << "\n--- Test 4: Vector-based Position Interface ---" << std::endl;
    // Test the vector-based interface
    std::vector<float> pos = {0, -2, 0};
    auto mesh4 = DEMSim.AddMeshFromTemplate(mesh_template, pos);
    std::cout << "Created mesh instance 4 at (0, -2, 0) using vector interface" << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "All tests completed successfully!" << std::endl;
    std::cout << "Total mesh instances created: 5" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
