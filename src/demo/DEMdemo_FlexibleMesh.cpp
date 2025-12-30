//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// This demo shows how to control the deformation of a flexibile mesh and use
// it in a DEM simulation.
// We show how mesh node coordinates can be extracted and modified. This can
// potentially be used with an external solid mechanics solver to do co-simulations
// involving flexible bodies.
// =============================================================================

#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>

using namespace deme;
const double math_PI = 3.1415927;
const std::string force_csv_header = "point_x,point_y,point_z,force_x,force_y,force_z";

// Used to write pairwise force concerning the mesh to a file.
void writeFloat3VectorsToCSV(const std::string& header,
                             const std::vector<std::vector<float3>>& vectors,
                             const std::string& filename,
                             size_t num_items);

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat("CSV");
    DEMSim.SetOutputContent({"ABSV"});
    DEMSim.SetMeshOutputFormat("VTK");
    DEMSim.SetContactOutputContent({"OWNER", "FORCE", "POINT", "TORQUE"});

    // E, nu, CoR, mu, Crr...
    auto mat_type_mesh = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.7}, {"Crr", 0.00}});
    auto mat_type_particle = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.4}, {"Crr", 0.00}});
    // If you don't have this line, then values will take average between 2 materials, when they are in contact
    DEMSim.SetMaterialPropertyPair("mu", mat_type_mesh, mat_type_particle, 0.5);

    double world_size = 2;
    /*
    // You have the choice to not specify the world size. In that case, the world is 20^3. And you might want to add
    world boundaries yourself, like we did in this demo.

    DEMSim.InstructBoxDomainDimension({-world_size / 2, world_size / 2},
                                      {-world_size / 2, world_size / 2}, {0, world_size});
    */

    // No need to add simulation `world' boundaries, b/c we'll add a cylinderical container manually
    DEMSim.InstructBoxDomainBoundingBC("none", mat_type_mesh);
    // Now manually add boundaries (you can choose to add them automatically using InstructBoxDomainBoundingBC, too)
    auto walls = DEMSim.AddExternalObject();
    walls->AddPlane(make_float3(0, 0, 0), make_float3(0, 0, 1), mat_type_mesh);
    walls->AddPlane(make_float3(0, 0, world_size), make_float3(0, 0, -1), mat_type_mesh);
    walls->AddPlane(make_float3(world_size / 2, 0, 0), make_float3(-1, 0, 0), mat_type_mesh);
    walls->AddPlane(make_float3(-world_size / 2, 0, 0), make_float3(1, 0, 0), mat_type_mesh);
    walls->AddPlane(make_float3(0, world_size / 2, 0), make_float3(0, -1, 0), mat_type_mesh);
    walls->AddPlane(make_float3(0, -world_size / 2, 0), make_float3(0, 1, 0), mat_type_mesh);

    // Define the terrain particle templates
    // Calculate its mass and MOI
    float terrain_density = 2.6e3;
    double clump_vol = 4. / 3. * math_PI;
    float mass = terrain_density * clump_vol;
    float3 MOI = make_float3(2. / 5.) * mass;
    // Then load it to system
    std::shared_ptr<DEMClumpTemplate> my_template =
        DEMSim.LoadClumpType(mass, MOI, GetDEMEDataFile("clumps/spiky_sphere.csv"), mat_type_particle);
    my_template->SetVolume(clump_vol);
    // Decide the scalings of the templates we just created (so that they are... like particles, not rocks)
    double scale = 0.05;
    my_template->Scale(scale);

    // Sample 2 chunks of materials in this part
    {
        HCPSampler sampler(scale * 2.2);
        float fill_height = 1.75;
        float3 fill_center1 = make_float3(-world_size / 4, 0, fill_height / 2 + 2 * scale);
        float3 fill_center2 = make_float3(world_size / 4, 0, fill_height / 2 + 2 * scale);
        float3 fill_halfsize = make_float3(world_size / 4 - 2 * scale, world_size / 4 - 2 * scale, fill_height / 2);
        auto input_xyz1 = sampler.SampleBox(fill_center1, fill_halfsize);
        auto input_xyz2 = sampler.SampleBox(fill_center2, fill_halfsize);
        input_xyz1.insert(input_xyz1.end(), input_xyz2.begin(), input_xyz2.end());
        auto particles = DEMSim.AddClumps(my_template, input_xyz1);
        std::cout << "Total num of particles: " << particles->GetNumClumps() << std::endl;
        std::cout << "Total num of spheres: " << particles->GetNumSpheres() << std::endl;
    }

    // Load in the mesh which is a 2x2 (yz) plate. Its thickness is 0.05 in the x direction.
    auto flex_mesh = DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/thin_plate.obj"), mat_type_mesh);
    unsigned int num_tri = flex_mesh->GetNumTriangles();
    std::cout << "Total num of triangles: " << num_tri << std::endl;

    // The define the properties
    float body_mass = 1.5e3 * 1. * 1. * 0.05;
    flex_mesh->SetMass(body_mass);
    float Iz = 1. / 12. * (0.05 * 0.05 + 1 * 1) * body_mass;
    float Ix = 1. / 12. * (1 * 1 + 1 * 1) * body_mass;
    flex_mesh->SetMOI(make_float3(Ix, Iz, Iz));
    // This mesh is created with CoM being 0,0,0, so no effect by InformCentroidPrincipal. But it is good that you know
    // this method exists.
    flex_mesh->InformCentroidPrincipal(make_float3(0, 0, 0), make_float4(0, 0, 0, 1));
    // Attach it to the ceiling
    flex_mesh->SetInitPos(make_float3(0, 0, 1.2));
    flex_mesh->SetFamily(1);
    // If you call SetFamilyPrescribedPosition and SetFamilyPrescribedQuaternion without specifying what position it
    // actually take, then its position is kept `as is' during simulation, without being affected by physics. It's
    // similar to fixing it but allows you manually impose velocities (which may have implications on your force model),
    // even though the velocity won't change its location. If you prescribe position by do not prescribe velocities, it
    // may make the object accumulate `phantom' velocity and de-stabilize the simulation. Fixing both position and
    // velocity is equivalent to fixing the family.
    DEMSim.SetFamilyPrescribedPosition(1);
    DEMSim.SetFamilyPrescribedQuaternion(1);
    DEMSim.SetFamilyPrescribedLinVel(1);
    DEMSim.SetFamilyPrescribedAngVel(1);
    // DEMSim.SetFamilyFixed(1);

    // Track the mesh
    auto flex_mesh_tracker = DEMSim.Track(flex_mesh);

    // Some inspectors
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");

    float step_size = 5e-6;
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    // Mesh has user-enforced deformation that the solver won't expect, so it can be better to allow larger safety
    // adder.
    DEMSim.SetExpandSafetyAdder(1.0);
    DEMSim.SetErrorOutAvgContacts(50);
    DEMSim.Initialize();

    // After system initialization, you can still get an handle of the mesh components using trackers (GetMesh method).
    // But note that if you do changes using this handle, the changes are done to the mesh and it immediately affects
    // the simulation. So sometimes you want to copy the information you are after and keep for your record.
    auto mesh_handle = flex_mesh_tracker->GetMesh();
    // This is keeping a copy of the RELATIVE (to the CoM) locations of the mesh nodes. In our case, the Z coordinates
    // of these nodes range from -0.5 to 0.5.
    std::vector<float3> node_resting_location(mesh_handle->GetCoordsVertices());

    std::filesystem::path out_dir = std::filesystem::current_path();
    out_dir /= "DemoOutput_FlexibleMesh";
    std::filesystem::create_directory(out_dir);

    float sim_end = 9.0;
    unsigned int fps = 20;
    float frame_time = 1.0 / fps;
    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));
    unsigned int frame_count = 0;
    unsigned int step_count = 0;

    // Used to store forces and points of contact
    std::vector<float3> forces, points;
    size_t num_force_pairs = 0;

    // Settle
    for (float t = 0; t < 0.5; t += frame_time) {
        char filename[100], meshname[100], force_filename[100];
        std::cout << "Outputting frame: " << frame_count << std::endl;
        sprintf(filename, "DEMdemo_output_%04d.csv", frame_count);
        sprintf(force_filename, "DEMdemo_forces_%04d.csv", frame_count);
        sprintf(meshname, "DEMdemo_mesh_%04d.vtk", frame_count++);
        DEMSim.WriteSphereFile(out_dir / filename);
        DEMSim.WriteMeshFile(out_dir / meshname);
        std::filesystem::path force_filepath = out_dir / force_filename;
        writeFloat3VectorsToCSV(force_csv_header, {points, forces}, force_filepath.string(), num_force_pairs);
        DEMSim.ShowThreadCollaborationStats();
        DEMSim.DoDynamics(frame_time);
    }

    // It's possible that you don't have to update the mesh every time step so you can set this number larger than 1.
    // However, you have to then ensure the simulation does not de-stabilize because the mesh--particles contacts are
    // running in a delayed fashion and large penetrations can occur. If the mesh is super soft, then it's probably OK.
    int ts_per_mesh_update = 5;
    // Some constants that are used to define the artificial mesh motion. You'll see in the main simulation loop.
    float max_wave_magnitude = 0.3;
    float wave_period = 3.0;

    // Main simulation loop starts...
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (float t = 0; t < sim_end; t += step_size, step_count++) {
        if (step_count % out_steps == 0) {
            char filename[100], meshname[100], force_filename[100];
            std::cout << "Outputting frame: " << frame_count << std::endl;
            sprintf(filename, "DEMdemo_output_%04d.csv", frame_count);
            sprintf(force_filename, "DEMdemo_forces_%04d.csv", frame_count);
            sprintf(meshname, "DEMdemo_mesh_%04d.vtk", frame_count++);
            DEMSim.WriteSphereFile(out_dir / filename);
            DEMSim.WriteMeshFile(out_dir / meshname);
            std::filesystem::path force_filepath = out_dir / force_filename;
            // We write force pairs that are related to the mesh to a file
            num_force_pairs = flex_mesh_tracker->GetContactForces(points, forces);
            writeFloat3VectorsToCSV(force_csv_header, {points, forces}, force_filepath.string(), num_force_pairs);
            DEMSim.ShowThreadCollaborationStats();
        }

        // We probably don't have to update the mesh every time step
        if (step_count % ts_per_mesh_update == 0) {
            // For real use cases, you probably will use an external solver to solve the defomration of the mesh,
            // then feed it to DEME. Here, we create an artificial defomration pattern for the mesh based on mesh node
            // location and time. This is just for show.

            // First, get where the mesh nodes are currently.
            std::vector<float3> node_current_location = flex_mesh_tracker->GetMeshNodesGlobal();
            // If you need the current RELATIVE (to the CoM) locations of the mesh nodes instead of global coordinates,
            // you can get it like the following:
            // std::vector<float3> node_current_location(mesh_handle->GetCoordsVertices());

            // Now calculate how much each node should `wave' and update the node location array. Remember z = 1 is
            // where the highest (relative) mesh node is. Again, this is artificial and only for showcasing this
            // utility.
            for (unsigned int i = 0; i < node_current_location.size(); i++) {
                // Use resting locations to calculate the magnitude of waving for nodes...
                float my_wave_distance = std::pow((1. - node_resting_location[i].z) / 2., 2) * max_wave_magnitude *
                                         std::sin(t / wave_period * 2 * math_PI);
                // Then update the current location array...
                node_current_location[i].x = node_resting_location[i].x + my_wave_distance;
            }

            // Now instruct the mesh to deform. Two things to pay attention to:

            // 1. We should respect the actual CoM location of the mesh. We get the global coords of mesh nodes using
            // GetMeshNodesGlobal, but UpdateMesh works with mesh's local or say relative coordinates, and that is why
            // we do applyFrameTransformGlobalToLocal first. And depending on your setup, the CoM and coord frame of
            // your mesh might be moving, and if it moves and rotates then you probably need to move and rotate the
            // points you got to offset the influence of CoM and local frame first. That said, if you use
            // mesh_handle->GetCoordsVertices() as I mentioned above to get the relative node positions of the mesh,
            // then no need to applyFrameTransformGlobalToLocal the CoM and rotate the frame.

            // 2. UpdateMesh will update the relative locations of mesh nodes to your specified locations. But if you
            // just have the information on the amount of mesh deformation, then you can use UpdateMeshByIncrement
            // instead, to incremenet mesh nodes' relative locations.

            float3 mesh_CoM_pos = flex_mesh_tracker->Pos();
            float4 mesh_frame_oriQ = flex_mesh_tracker->OriQ();
            for (auto& node : node_current_location) {
                applyFrameTransformGlobalToLocal(node, mesh_CoM_pos, mesh_frame_oriQ);
            }
            flex_mesh_tracker->UpdateMesh(node_current_location);

            // Forces need to be extracted, if you want to use an external solver to solve the mesh's deformation. You
            // can do it like shown below. In this example, we did not use it other than writing it to a file; however
            // you may want to feed the array directly to your soild mechanics solver.
            num_force_pairs = flex_mesh_tracker->GetContactForces(points, forces);
        }

        // Means advance simulation by one time step
        DEMSim.DoStepDynamics();
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds (wall time) to finish the simulation" << std::endl;

    DEMSim.ShowTimingStats();

    std::cout << "----------------------------------------" << std::endl;
    DEMSim.ShowMemStats();
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "FlexibleMesh demo exiting..." << std::endl;
    return 0;
}

void writeFloat3VectorsToCSV(const std::string& header,
                             const std::vector<std::vector<float3>>& vectors,
                             const std::string& filename,
                             size_t num_items) {
    std::ofstream file(filename);

    // Check if the file was successfully opened
    if (!file.is_open()) {
        std::cout << "Failed to open the CSV file used to store mesh forces!" << std::endl;
        return;
    }

    file << header << "\n";

    // Write vectors as columns
    for (size_t i = 0; i < num_items; ++i) {
        for (size_t j = 0; j < vectors.size(); ++j) {
            if (i < vectors[j].size()) {
                file << vectors[j][i].x << "," << vectors[j][i].y << "," << vectors[j][i].z;
            }
            if (j != vectors.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    // Close the file
    file.close();
}
