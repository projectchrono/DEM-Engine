//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A demo that tests persistent contact mapping with meshed boxes sliding on
// meshed planes. Boxes periodically break contact by being lifted off the
// plane, then returned to sliding contact. This is useful for finding potential
// problems in persistent contact mapping generation mechanisms.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/Samplers.hpp>

#include <filesystem>
#include <cstdio>
#include <iostream>
#include <cmath>

using namespace deme;
using namespace std::filesystem;

// Helper function to set Z coordinate for a collection of boxes
void SetBoxesHeight(const std::vector<std::shared_ptr<DEMTracker>>& trackers, float height) {
    for (size_t j = 0; j < trackers.size(); j++) {
        float3 current_pos = trackers[j]->Pos();
        trackers[j]->SetPos(make_float3(current_pos.x, current_pos.y, height));
        trackers[j]->SetVel(make_float3(0, 0, 0));  // Reset velocity to avoid sudden impacts
    }
}

// Helper function to set Z coordinate for a collection of boxes
void SetBoxesInitVel(const std::vector<std::shared_ptr<DEMTracker>>& trackers, float vel) {
    for (size_t j = 0; j < trackers.size(); j++) {
        trackers[j]->SetVel(make_float3(0, vel, 0));  
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "DEM Persistent Contact Mapping Demo" << std::endl;
    std::cout << "========================================" << std::endl;

    DEMSolver DEMSim;
    DEMSim.SetVerbosity("METRIC");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetContactOutputContent({"OWNER", "FORCE", "POINT", "NORMAL"});
    DEMSim.InstructBoxDomainDimension(100, 100, 20);
    // Enable mesh-mesh contacts for persistent contact testing
    DEMSim.SetMeshUniversalContact(true);
    // DEMSim.UseFrictionlessHertzianModel();

    // Load materials
    auto mat_type_box = DEMSim.LoadMaterial({{"E", 1e6}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.4}});
    auto mat_type_plane = DEMSim.LoadMaterial({{"E", 1e6}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.4}});

    std::cout << "\n--- Loading Mesh Templates ---" << std::endl;
    // Load mesh templates
    auto cube_template = DEMSim.LoadMeshType((GET_DATA_PATH() / "mesh/cube.obj").string(), mat_type_box,
                                             true,    // load_normals
                                             false);  // load_uv
    auto plane_template = DEMSim.LoadMeshType((GET_DATA_PATH() / "mesh/plane_20by20.obj").string(), mat_type_plane,
                                              true,    // load_normals
                                              false);  // load_uv

    if (!cube_template) {
        std::cout << "Failed to load cube mesh template from mesh/cube.obj" << std::endl;
        return 1;
    }
    if (!plane_template) {
        std::cout << "Failed to load plane mesh template from mesh/plane_20by20.obj" << std::endl;
        return 1;
    }

    std::cout << "Cube mesh triangles: " << cube_template->GetNumTriangles() << std::endl;
    std::cout << "Plane mesh triangles: " << plane_template->GetNumTriangles() << std::endl;

    std::cout << "\n--- Creating Plane ---" << std::endl;
    auto plane1 = DEMSim.AddMeshFromTemplate(plane_template, make_float3(0, 0, 0));
    plane1->Scale(2.0);
    plane1->SetFamily(10);  // Plane family 1
    plane1->SetMass(10000.);
    plane1->SetMOI(make_float3(10000., 10000., 10000.));
    DEMSim.SetFamilyFixed(10);  // Planes are fixed in place
    // auto plane1 = DEMSim.AddBCPlane(make_float3(-5, 0, 0), make_float3(0,0,1), mat_type_plane);

    std::cout << "\n--- Creating Boxes on Plane 1 ---" << std::endl;
    // Create boxes on plane 1 (family 1)
    std::vector<std::shared_ptr<DEMTracker>> plane1_box_trackers;
    for (int i = 0; i < 3; i++) {
        float x = -5.0f + i * 1.2f;
        float y = -2.0f + i * 2.0f;
        float z = 0.51f;  // Start above plane

        auto box = DEMSim.AddMeshFromTemplate(cube_template, make_float3(x, y, z));
        box->SetFamily(1);  // Box group 1
        box->SetMass(1000.);
        box->SetMOI(make_float3(200., 200., 200.));

        auto tracker = DEMSim.Track(box);

        std::cout << "  Box " << i << " at (" << x << ", " << y << ", " << z << ")" << std::endl;
    }

    std::cout << "\n--- Creating Boxes on Plane 2 ---" << std::endl;
    // Create boxes on plane 2 (family 2)
    std::vector<std::shared_ptr<DEMTracker>> plane2_box_trackers;
    for (int i = 0; i < 3; i++) {
        float x = 5.0f + i * 1.2f;
        float y = -2.0f + i * 2.0f;
        float z = 0.51f;  // Start above plane

        auto box = DEMSim.AddMeshFromTemplate(cube_template, make_float3(x, y, z));
        box->SetFamily(2);  // Box group 2
        box->SetMass(1000.);
        box->SetMOI(make_float3(200., 200., 200.));

        auto tracker = DEMSim.Track(box);
        plane2_box_trackers.push_back(tracker);

        std::cout << "  Box " << i << " at (" << x << ", " << y << ", " << z << ")" << std::endl;
    }

    // std::cout << "\n--- Setting Up Prescribed Motion ---" << std::endl;
    // // Set prescribed velocities for sliding motion
    // // Family 1 boxes slide in +Y direction with 1 m/s
    // DEMSim.SetFamilyPrescribedLinVel(1, "0", "0.1", "none", false);
    // std::cout << "Family 1 boxes: sliding in +Y direction at 1 m/s" << std::endl;

    // // Family 2 boxes slide in -Y direction with 1 m/s
    // DEMSim.SetFamilyPrescribedLinVel(2, "0", "-0.1", "none", false);
    // std::cout << "Family 2 boxes: sliding in -Y direction at 1 m/s" << std::endl;

    std::cout << "\n--- Initializing Simulation ---" << std::endl;
    float step_time = 1e-4;
    DEMSim.SetInitTimeStep(step_time);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.SetExpandSafetyType("auto");
    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir /= "DemoOutput_PersistentContact";
    create_directory(out_dir);

    std::cout << "\n--- Running Simulation ---" << std::endl;
    float frame_time = 5e-2;  // Output every 0.05 seconds
    int frame = 0;
    int frame_step = (int)(frame_time / step_time);
    double final_time = 5.0;  // 5 seconds simulation

    // Parameters for periodic lifting
    double lift_period = 1.0;     // Lift boxes every 1 second
    double lift_duration = 0.3;   // Keep lifted for 0.3 seconds
    double lift_height = 2.;     // Lift 3 units above contact plane
    double contact_height = 0.51;  // Normal contact height

    double last_lift_time = 0.0;
    bool boxes_lifted = false;

    SetBoxesInitVel(plane1_box_trackers, 0.2f);
    SetBoxesInitVel(plane2_box_trackers, -0.2f);

    for (int i = 0; i < (int)(final_time / step_time); i++) {
        double sim_time = i * step_time;

        // // Check if we should lift boxes
        // if (!boxes_lifted && sim_time - last_lift_time >= lift_period) {
        //     std::cout << "\n*** Lifting boxes at time " << sim_time << "s ***" << std::endl;
        //     // Lift all boxes
        //     SetBoxesHeight(plane1_box_trackers, lift_height);
        //     SetBoxesHeight(plane2_box_trackers, lift_height);
        //     boxes_lifted = true;
        //     last_lift_time = sim_time;
        // }

        // // Check if we should return boxes to contact
        // if (boxes_lifted && sim_time - last_lift_time >= lift_duration) {
        //     std::cout << "*** Returning boxes to contact at time " << sim_time << "s ***" << std::endl;
        //     // Return all boxes to contact
        //     SetBoxesHeight(plane1_box_trackers, contact_height);
        //     SetBoxesHeight(plane2_box_trackers, contact_height);
        //     boxes_lifted = false;
        // }

        if (i % frame_step == 0) {
            frame++;
            std::cout << "\nFrame: " << frame << " Time: " << sim_time << "s" << std::endl;

            char meshfilename[100];
            sprintf(meshfilename, "persistent_contact_%04d.vtk", frame);
            DEMSim.WriteMeshFile(out_dir / meshfilename);

            // Print some box positions
            if (plane1_box_trackers.size() > 0) {
                float3 pos = plane1_box_trackers[0]->Pos();
                std::cout << "  Box 0 (Plane1): (" << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
            }
            if (plane2_box_trackers.size() > 0) {
                float3 pos = plane2_box_trackers[0]->Pos();
                std::cout << "  Box 0 (Plane2): (" << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
            }
            std::cout << "  Boxes lifted: " << (boxes_lifted ? "YES" : "NO") << std::endl;
        }

        DEMSim.DoDynamics(step_time);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo completed successfully!" << std::endl;
    std::cout << "Total boxes created: " << (plane1_box_trackers.size() + plane2_box_trackers.size()) << std::endl;
    std::cout << "Output written to: " << out_dir << std::endl;
    DEMSim.ShowTimingStats();
    std::cout << "========================================" << std::endl;

    return 0;
}
