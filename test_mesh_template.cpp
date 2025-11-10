//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A demo that demonstrates the new mesh template functionality.
// This demo shows:
// 1. Loading a mesh as a template (not immediately in simulation)
// 2. Instantiating multiple mesh instances from the template at different locations
// 3. Duplicating an existing mesh object
// 4. Running a simple simulation with template-based meshes
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
    std::cout << "DEM Mesh Template Demo" << std::endl;
    std::cout << "========================================" << std::endl;

    DEMSolver DEMSim;
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetContactOutputContent({"OWNER", "FORCE", "POINT"});
    DEMSim.InstructBoxDomainDimension(10, 10, 10);
    // Enable mesh-mesh contacts
    DEMSim.SetMeshUniversalContact(true);

    // Load a material
    auto mat_type = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}});

    // Add a bottom plane
    DEMSim.AddBCPlane(make_float3(0, 0, -4), make_float3(0, 0, 1), mat_type);

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
    auto mesh1 = DEMSim.AddMeshFromTemplate(mesh_template, make_float3(-1.5, 0, 0));
    mesh1->SetFamily(0);
    mesh1->SetMass(1000.);
    mesh1->SetMOI(make_float3(200., 200., 200.));
    std::cout << "Created mesh instance 1 at (-1.5, 0, 0)" << std::endl;
    auto tracker1 = DEMSim.Track(mesh1);
    
    auto mesh2 = DEMSim.AddMeshFromTemplate(mesh_template, make_float3(1.5, 0, 2));
    mesh2->SetFamily(0);
    mesh2->SetMass(1000.);
    mesh2->SetMOI(make_float3(200., 200., 200.));
    std::cout << "Created mesh instance 2 at (1.5, 0, 2)" << std::endl;
    auto tracker2 = DEMSim.Track(mesh2);
    
    auto mesh3 = DEMSim.AddMeshFromTemplate(mesh_template, make_float3(0, 2, 1));
    mesh3->SetFamily(0);
    mesh3->SetMass(1000.);
    mesh3->SetMOI(make_float3(200., 200., 200.));
    std::cout << "Created mesh instance 3 at (0, 2, 1)" << std::endl;

    std::cout << "\n--- Test 3: Duplicate a Mesh Object ---" << std::endl;
    // Duplicate an existing mesh and modify its properties
    auto mesh_dup = DEMSim.Duplicate(mesh1);
    mesh_dup->SetInitPos(make_float3(0, -2, 1));
    mesh_dup->SetFamily(0);
    mesh_dup->SetMass(1000.);
    mesh_dup->SetMOI(make_float3(200., 200., 200.));
    std::cout << "Duplicated mesh1 and moved to (0, -2, 1)" << std::endl;

    std::cout << "\n--- Test 4: Vector-based Position Interface ---" << std::endl;
    // Test the vector-based interface
    std::vector<float> pos = {3, 0, 1};
    auto mesh4 = DEMSim.AddMeshFromTemplate(mesh_template, pos);
    mesh4->SetFamily(0);
    mesh4->SetMass(1000.);
    mesh4->SetMOI(make_float3(200., 200., 200.));
    std::cout << "Created mesh instance 4 at (3, 0, 1) using vector interface" << std::endl;

    std::cout << "\n--- Initializing Simulation ---" << std::endl;
    float step_time = 1e-5;
    DEMSim.SetInitTimeStep(step_time);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.SetExpandSafetyType("auto");
    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir /= "DemoOutput_MeshTemplate";
    create_directory(out_dir);

    std::cout << "\n--- Running Short Simulation ---" << std::endl;
    float frame_time = 1e-2;
    int frame = 0;
    int frame_step = (int)(frame_time / step_time);
    double final_time = 0.1;  // Short simulation
    
    for (int i = 0; i < (int)(final_time / step_time); i++) {
        if (i % frame_step == 0) {
            frame++;
            std::cout << "Frame: " << frame << " Time: " << i * step_time << "s" << std::endl;

            char meshfilename[100];
            sprintf(meshfilename, "mesh_template_demo_%04d.vtk", frame);
            DEMSim.WriteMeshFile(out_dir / meshfilename);

            float3 pos1 = tracker1->Pos();
            float3 pos2 = tracker2->Pos();
            std::cout << "  Mesh 1 position: (" << pos1.x << ", " << pos1.y << ", " << pos1.z << ")" << std::endl;
            std::cout << "  Mesh 2 position: (" << pos2.x << ", " << pos2.y << ", " << pos2.z << ")" << std::endl;
        }

        DEMSim.DoDynamics(step_time);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo completed successfully!" << std::endl;
    std::cout << "Total mesh instances created: 5" << std::endl;
    std::cout << "Output written to: " << out_dir << std::endl;
    DEMSim.ShowTimingStats();
    std::cout << "========================================" << std::endl;

    return 0;
}
