//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A demo featuring multiple particles (boxes, spheres, cones, cylinders) 
// falling onto an analytical plane, testing triangle-triangle contacts 
// between diverse mesh types and mesh-plane contacts.
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

const double math_PI = 3.1415927;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetContactOutputContent({"OWNER", "FORCE", "POINT", "NORMAL", "TORQUE", "CNT_WILDCARD"});
    DEMSim.InstructBoxDomainDimension(20, 20, 15);

    // Enable mesh-mesh contacts so boxes can collide with each other
    DEMSim.SetMeshUniversalContact(true);

    // Define material properties
    auto mat_box = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.4}, {"Crr", 0.1}});
    auto mat_plane = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.3}, {"Crr", 0.1}});

    // Add a bottom plane at z = 0
    DEMSim.AddBCPlane(make_float3(0, 0, 0), make_float3(0, 0, 1), mat_plane);

    // Create multiple particles in a grid pattern above the plane
    const int num_particles_x = 6;
    const int num_particles_y = 6;
    const float particle_spacing = 2.0;
    const float initial_height = 8.0;
    const float base_size = 0.5;  // Base scale for all meshes
    const float cylinder_scale_factor = 0.5;  // Cylinders scaled down since they're taller

    std::vector<std::shared_ptr<DEMTracker>> trackers;

    // Random number generator for variations
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> pos_dist(-0.1, 0.1);
    std::uniform_real_distribution<> rot_dist(-0.2, 0.2);
    std::uniform_real_distribution<> scale_dist(0.8, 1.2);  // Scale variation
    std::uniform_int_distribution<> mesh_type_dist(0, 3);  // 4 mesh types

    for (int i = 0; i < num_particles_x; i++) {
        for (int j = 0; j < num_particles_y; j++) {
            float x = (i - num_particles_x / 2.0 + 0.5) * particle_spacing + pos_dist(gen);
            float y = (j - num_particles_y / 2.0 + 0.5) * particle_spacing + pos_dist(gen);
            float z = initial_height + (i + j) * 0.5 + pos_dist(gen) * 2;

            // Select mesh type randomly: 0=cube, 1=sphere, 2=cone, 3=cylinder
            int mesh_type = mesh_type_dist(gen);
            std::shared_ptr<DEMClumpTemplate> particle;
            
            if (mesh_type == 0) {
                // Cube with non-uniform scaling
                particle = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cube.obj").string(), mat_box);
                float scale_x = base_size * scale_dist(gen);
                float scale_y = base_size * scale_dist(gen);
                float scale_z = base_size * scale_dist(gen);
                particle->Scale(make_float3(scale_x, scale_y, scale_z));  // Non-uniform scaling
                
                // Set mass and MOI for the box (approximate as uniform density)
                float mass = 1000.0 * scale_x * scale_y * scale_z;
                float moi_x = mass * (scale_y * scale_y + scale_z * scale_z) / 12.0;
                float moi_y = mass * (scale_x * scale_x + scale_z * scale_z) / 12.0;
                float moi_z = mass * (scale_x * scale_x + scale_y * scale_y) / 12.0;
                particle->SetMass(mass);
                particle->SetMOI(make_float3(moi_x, moi_y, moi_z));
            } else if (mesh_type == 1) {
                // Sphere (unit sphere in mesh)
                particle = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/sphere.obj").string(), mat_box);
                float scale = base_size * scale_dist(gen);
                particle->Scale(scale);
                
                // Set mass and MOI for sphere
                float mass = 1000.0 * (4.0/3.0) * math_PI * scale * scale * scale;
                float moi = 0.4 * mass * scale * scale;  // MOI for sphere
                particle->SetMass(mass);
                particle->SetMOI(make_float3(moi, moi, moi));
            } else if (mesh_type == 2) {
                // Cone (height ~1, radius ~1 in mesh)
                particle = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cone.obj").string(), mat_box);
                float scale = base_size * scale_dist(gen);
                particle->Scale(scale);
                
                // Set mass and MOI for cone (approximate)
                float mass = 1000.0 * (1.0/3.0) * math_PI * scale * scale * scale;
                float moi_base = 0.3 * mass * scale * scale;  // Approximate MOI
                float moi_height = 0.15 * mass * scale * scale;
                particle->SetMass(mass);
                particle->SetMOI(make_float3(moi_base, moi_base, moi_height));
            } else {
                // Cylinder (radius ~1, height ~2 in mesh)
                particle = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cyl_r1_h2.obj").string(), mat_box);
                float scale = base_size * cylinder_scale_factor * scale_dist(gen);
                particle->Scale(scale);
                
                // Set mass and MOI for cylinder
                float radius = scale;
                float height = 2.0 * scale;
                float mass = 1000.0 * math_PI * radius * radius * height;
                float moi_radial = mass * (3.0 * radius * radius + height * height) / 12.0;
                float moi_axial = 0.5 * mass * radius * radius;
                particle->SetMass(mass);
                particle->SetMOI(make_float3(moi_radial, moi_radial, moi_axial));
            }
            
            particle->SetFamily(0);
            particle->SetInitPos(make_float3(x, y, z));

            // Add small initial rotation for more interesting dynamics
            particle->SetInitQuat(make_float4(rot_dist(gen), rot_dist(gen), rot_dist(gen), 1.0));

            auto tracker = DEMSim.Track(particle);
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

    std::cout << "Starting simulation with " << num_particles_x * num_particles_y << " particles (diverse meshes)" << std::endl;
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
            int particles_settled = 0;
            float avg_height = 0.0;
            for (size_t k = 0; k < trackers.size(); k++) {
                float3 pos = trackers[k]->Pos();
                avg_height += pos.z;
                if (pos.z < 2.0) {  // Consider particles below 2m as settled
                    particles_settled++;
                }
            }
            avg_height /= trackers.size();

            std::cout << "  Average height: " << avg_height << " m" << std::endl;
            std::cout << "  Particles settled: " << particles_settled << " / " << trackers.size() << std::endl;

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
