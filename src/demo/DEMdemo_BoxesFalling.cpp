//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A demo featuring multiple boxes falling onto an analytical plane, testing
// triangle-triangle contacts between boxes and box-plane contacts.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/Samplers.hpp>

#include <filesystem>
#include <cstdio>
#include <time.h>
#include <random>

using namespace deme;
using namespace std::filesystem;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetContactOutputContent({"OWNER", "FORCE", "POINT", "NORMAL", "TORQUE", "CNT_WILDCARD"});
    DEMSim.InstructBoxDomainDimension(20, 20, 15);
    
    // Enable mesh-mesh contacts so boxes can collide with each other
    DEMSim.SetMeshUniversalContact(true);

    // Define material properties
    auto mat_box = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.4}, {"Crr", 0.01}});
    auto mat_plane = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.3}, {"Crr", 0.01}});

    // Add a bottom plane at z = 0
    DEMSim.AddBCPlane(make_float3(0, 0, 0), make_float3(0, 0, 1), mat_plane);

    // Create multiple boxes in a grid pattern above the plane
    const int num_boxes_x = 4;
    const int num_boxes_y = 4;
    const float box_spacing = 2.0;
    const float initial_height = 8.0;
    const float box_size = 0.5;  // Assume cube.obj is roughly 1x1x1, scale to 0.5
    
    std::vector<std::shared_ptr<DEMTrackedObj>> trackers;
    
    // Random number generator for small position variations
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> pos_dist(-0.1, 0.1);
    std::uniform_real_distribution<> rot_dist(-0.2, 0.2);
    
    for (int i = 0; i < num_boxes_x; i++) {
        for (int j = 0; j < num_boxes_y; j++) {
            float x = (i - num_boxes_x / 2.0 + 0.5) * box_spacing + pos_dist(gen);
            float y = (j - num_boxes_y / 2.0 + 0.5) * box_spacing + pos_dist(gen);
            float z = initial_height + (i + j) * 0.5 + pos_dist(gen) * 2;
            
            auto box = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cube.obj").string(), mat_box);
            box->SetFamily(0);
            box->SetInitPos(make_float3(x, y, z));
            box->Scale(box_size);  // Scale the box
            
            // Set mass and MOI for the box (scaled cubic)
            float mass = 1000.0 * box_size * box_size * box_size;
            float moi = mass * box_size * box_size / 6.0;  // MOI for cube
            box->SetMass(mass);
            box->SetMOI(make_float3(moi, moi, moi));
            
            // Add small initial rotation for more interesting dynamics
            box->SetInitOriQ(make_float4(rot_dist(gen), rot_dist(gen), rot_dist(gen), 1.0));
            
            auto tracker = DEMSim.Track(box);
            trackers.push_back(tracker);
        }
    }

    float step_time = 1e-5;
    DEMSim.SetInitTimeStep(step_time);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.SetExpandSafetyType("auto");
    DEMSim.Initialize();

    // Setup output directory
    path out_dir = current_path();
    out_dir /= "DemoOutput_BoxesFalling";
    create_directory(out_dir);

    float frame_time = 1e-2;
    int frame = 0;
    int frame_step = (int)(frame_time / step_time);
    double final_time = 3.0;  // 3 seconds of simulation
    
    std::cout << "Starting simulation with " << num_boxes_x * num_boxes_y << " boxes" << std::endl;
    std::cout << "Frame time: " << frame_time << " s, Time step: " << step_time << " s" << std::endl;
    std::cout << "Total frames: " << (int)(final_time / frame_time) << std::endl;
    
    for (int i = 0; i < (int)(final_time / step_time); i++) {
        if (i % frame_step == 0) {
            frame++;
            std::cout << "Frame: " << frame << " (t = " << i * step_time << " s)" << std::endl;

            char meshfilename[100];
            sprintf(meshfilename, "DEMdemo_boxes_%04d.vtk", frame);
            DEMSim.WriteMeshFile(out_dir / meshfilename);

            // Print some statistics
            int boxes_settled = 0;
            float avg_height = 0.0;
            for (size_t k = 0; k < trackers.size(); k++) {
                float3 pos = trackers[k]->Pos();
                avg_height += pos.z;
                if (pos.z < 2.0) {  // Consider boxes below 2m as settled
                    boxes_settled++;
                }
            }
            avg_height /= trackers.size();
            
            std::cout << "  Average height: " << avg_height << " m" << std::endl;
            std::cout << "  Boxes settled: " << boxes_settled << " / " << trackers.size() << std::endl;
            
            DEMSim.ShowMemStats();
            std::cout << "----------------------------------------" << std::endl;
        }

        DEMSim.DoDynamics(step_time);
    }

    DEMSim.ShowThreadCollaborationStats();
    DEMSim.ShowTimingStats();
    std::cout << "DEMdemo_BoxesFalling exiting..." << std::endl;
    return 0;
}
