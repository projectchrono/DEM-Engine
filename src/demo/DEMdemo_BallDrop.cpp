//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A meshed ball hitting a granular bed under gravity. A collection of different
// ball densities and drop heights are tested against loosely packed material.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <cstdio>
#include <chrono>
#include <filesystem>
#include <random>

using namespace deme;
using namespace std::filesystem;

double randomBetween0and1() {
    static std::mt19937 gen(std::random_device{}());                       // Random number generator
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);  // Uniform distribution between 0 and 1

    return distribution(gen);
}

int main() {
    float ball_densities[] = {2.2e3, 3.8e3, 7.8e3, 15e3};
    float Hs[] = {0.05, 0.1, 0.2};
    double R = 0.0254 / 2.;

    int run_num = 0;
    for (float ball_density : ball_densities) {
        for (float H : Hs) {
            double terrain_rad = 0.0025 / 2.;

            DEMSolver DEMSim;
            // Output less info at initialization
            DEMSim.SetVerbosity("ERROR");
            DEMSim.SetOutputFormat("CSV");
            DEMSim.SetOutputContent({"ABSV"});
            DEMSim.SetMeshOutputFormat("VTK");

            path out_dir = current_path();
            out_dir /= "DemoOutput_BallDrop";
            create_directory(out_dir);

            // E, nu, CoR, mu, Crr...
            auto mat_type_ball =
                DEMSim.LoadMaterial({{"E", 7e7}, {"nu", 0.24}, {"CoR", 0.9}, {"mu", 0.3}, {"Crr", 0.0}});
            auto mat_type_terrain =
                DEMSim.LoadMaterial({{"E", 7e7}, {"nu", 0.24}, {"CoR", 0.9}, {"mu", 0.3}, {"Crr", 0.0}});
            auto mat_type_terrain_sim =
                DEMSim.LoadMaterial({{"E", 7e7}, {"nu", 0.24}, {"CoR", 0.9}, {"mu", 0.3}, {"Crr", 0.0}});

            float step_size = 2e-6;
            double world_size = 0.2;
            DEMSim.InstructBoxDomainDimension({-world_size / 2., world_size / 2.}, {-world_size / 2., world_size / 2.},
                                              {0, 10 * world_size});
            DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);

            auto projectile =
                DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/sphere.obj").string(), mat_type_ball);
            projectile->Scale(R);
            std::cout << "Total num of triangles: " << projectile->GetNumTriangles() << std::endl;

            projectile->SetInitPos(make_float3(0, 0, 8 * world_size));
            float ball_mass = ball_density * 4. / 3. * PI * R * R * R;
            projectile->SetMass(ball_mass);
            projectile->SetMOI(
                make_float3(ball_mass * 2 / 5 * R * R, ball_mass * 2 / 5 * R * R, ball_mass * 2 / 5 * R * R));
            projectile->SetFamily(2);
            DEMSim.SetFamilyFixed(2);
            DEMSim.DisableContactBetweenFamilies(0, 2);
            // Track the projectile
            auto proj_tracker = DEMSim.Track(projectile);

            // 11 types of spheres, diameter from 0.25cm to 0.35cm
            std::vector<std::shared_ptr<DEMClumpTemplate>> templates_terrain;
            for (int i = 0; i < 11; i++) {
                templates_terrain.push_back(DEMSim.LoadSphereType(
                    terrain_rad * terrain_rad * terrain_rad * 2.5e3 * 4 / 3 * PI, terrain_rad, mat_type_terrain));
                terrain_rad += 0.0001 / 2.;
            }

            unsigned int num_particle = 0;
            float sample_z = 1.5 * terrain_rad;
            float fullheight = world_size * 2.;
            float sample_halfwidth = world_size / 2 - 2 * terrain_rad;
            float init_v = 0.01;

            // If first run, settle the material bed, then save to file; if not first run, just load the saved material
            // bed file.
            if (run_num > 0) {
                path cp_filename = out_dir / "bed.csv";

                auto clump_xyz = DEMSim.ReadClumpXyzFromCsv(cp_filename.string());
                auto clump_quaternion = DEMSim.ReadClumpQuatFromCsv(cp_filename.string());
                for (int i = 0; i < templates_terrain.size(); i++) {
                    char t_name[20];
                    sprintf(t_name, "%04d", i);

                    auto this_xyz = clump_xyz[std::string(t_name)];
                    auto this_quaternion = clump_quaternion[std::string(t_name)];
                    auto batch = DEMSim.AddClumps(templates_terrain[i], this_xyz);
                    batch->SetOriQ(this_quaternion);
                    num_particle += this_quaternion.size();
                }
            } else {
                std::random_device rd;   // Random number device to seed the generator
                std::mt19937 gen(rd());  // Mersenne Twister generator
                std::uniform_int_distribution<> dist(
                    0, templates_terrain.size() - 1);  // Uniform distribution of integers between 0 and n

                PDSampler sampler(2.01 * terrain_rad);
                while (sample_z < fullheight) {
                    float3 sample_center = make_float3(0, 0, sample_z);
                    auto input_xyz =
                        sampler.SampleBox(sample_center, make_float3(sample_halfwidth, sample_halfwidth, 0.000001));
                    std::vector<std::shared_ptr<DEMClumpTemplate>> template_to_use(input_xyz.size());
                    for (unsigned int i = 0; i < input_xyz.size(); i++) {
                        template_to_use[i] = templates_terrain[dist(gen)];
                    }
                    DEMSim.AddClumps(template_to_use, input_xyz);
                    num_particle += input_xyz.size();
                    sample_z += 2.01 * terrain_rad;
                }
            }

            std::cout << "Total num of particles: " << num_particle << std::endl;

            // Now add a plane to compress the sample
            auto compressor = DEMSim.AddExternalObject();
            compressor->AddPlane(make_float3(0, 0, 0), make_float3(0, 0, -1), mat_type_terrain);
            compressor->SetFamily(10);
            DEMSim.SetFamilyFixed(10);
            DEMSim.DisableContactBetweenFamilies(0, 10);
            auto compressor_tracker = DEMSim.Track(compressor);

            auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
            auto total_mass_finder = DEMSim.CreateInspector("clump_mass");

            DEMSim.SetInitTimeStep(step_size);
            DEMSim.SetMaxVelocity(30.);
            DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));

            DEMSim.Initialize();

            float sim_time = 3.0;
            float settle_time = 1.0;
            unsigned int fps = 10;
            float frame_time = 1.0 / fps;
            unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

            std::cout << "Output at " << fps << " FPS" << std::endl;
            unsigned int currframe = 0;
            double terrain_max_z;

            if (run_num == 0) {
                // We can let it settle first
                for (float t = 0; t < settle_time; t += frame_time) {
                    std::cout << "Frame: " << currframe << std::endl;
                    char filename[100], meshfilename[100];
                    sprintf(filename, "DEMdemo_output_%04d.csv", currframe);
                    sprintf(meshfilename, "DEMdemo_mesh_%04d.vtk", currframe);
                    DEMSim.WriteSphereFile(out_dir / filename);
                    DEMSim.WriteMeshFile(out_dir / meshfilename);
                    currframe++;

                    DEMSim.DoDynamicsThenSync(frame_time);
                    DEMSim.ShowThreadCollaborationStats();
                }

                path cp_filename = out_dir / "bed.csv";
                DEMSim.WriteClumpFile(cp_filename);
            }

            // This is to show that you can change the material for all the particles in a family... although here,
            // mat_type_terrain_sim and mat_type_terrain are the same material so there is no effect; you can define
            // them differently though.
            DEMSim.SetFamilyClumpMaterial(0, mat_type_terrain_sim);
            DEMSim.DoDynamicsThenSync(0.2);
            terrain_max_z = max_z_finder->GetValue();
            float matter_mass = total_mass_finder->GetValue();
            float total_volume = (world_size * world_size) * (terrain_max_z - 0.);
            float bulk_density = matter_mass / total_volume;
            std::cout << "Original terrain height: " << terrain_max_z << std::endl;
            std::cout << "Bulk density: " << bulk_density << std::endl;

            // Then drop the ball
            DEMSim.ChangeFamily(2, 0);
            proj_tracker->SetPos(make_float3(0, 0, terrain_max_z + R + H));
            std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
            for (float t = 0; t < sim_time; t += frame_time) {
                // Just output files for the first test. You can output all of them if you want.
                if (run_num == 0) {
                    std::cout << "Frame: " << currframe << std::endl;
                    char filename[100], meshfilename[100], cnt_filename[100];
                    sprintf(filename, "DEMdemo_output_%04d.csv", currframe);
                    sprintf(meshfilename, "DEMdemo_mesh_%04d.vtk", currframe);
                    // sprintf(cnt_filename, "Contact_pairs_%04d.csv", currframe);
                    DEMSim.WriteSphereFile(out_dir / filename);
                    DEMSim.WriteMeshFile(out_dir / meshfilename);
                    // DEMSim.WriteContactFile(out_dir / cnt_filename);
                    currframe++;
                }

                DEMSim.DoDynamics(frame_time);
                DEMSim.ShowThreadCollaborationStats();

                if (std::abs(proj_tracker->Vel().z) < 1e-4) {
                    break;
                }
            }
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_sec =
                std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            std::cout << time_sec.count() << " seconds (wall time) to finish the simulation" << std::endl;

            DEMSim.ShowTimingStats();

            float3 final_pos = proj_tracker->Pos();
            std::cout << "Ball density: " << ball_density << std::endl;
            std::cout << "Ball rad: " << R << std::endl;
            std::cout << "Drop height: " << H << std::endl;
            std::cout << "Penetration: " << terrain_max_z - (final_pos.z - R) << std::endl;

            std::cout << "==============================================================" << std::endl;

            run_num++;
        }
    }
    std::cout << "DEMdemo_BallDrop exiting..." << std::endl;
    return 0;
}
