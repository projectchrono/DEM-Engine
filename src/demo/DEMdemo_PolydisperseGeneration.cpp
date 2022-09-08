//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <cstdio>
#include <chrono>
#include <random>
#include <filesystem>

using namespace smug;
using namespace std::filesystem;

int main() {
    DEMSolver DEM_sim;
    DEM_sim.SetVerbosity(INFO);
    DEM_sim.SetOutputFormat(DEM_OUTPUT_FORMAT::CSV);
    DEM_sim.SetOutputContent(DEM_OUTPUT_CONTENT::ABSV);

    // auto projectile = DEM_sim.AddWavefrontMeshObject("./data/mesh/sphere.obj");
    // std::cout << "Total num of triangles: " << projectile->GetNumTriangles() << std::endl;

    // Shrink both mass and stiffness to make the settling not challenging
    float shrink_factor = 3e3;
    auto mat_type_ball = DEM_sim.LoadMaterial({{"E", 1e9 / shrink_factor}, {"nu", 0.3}, {"CoR", 0.3}});
    auto mat_type_terrain = DEM_sim.LoadMaterial({{"E", 1e9 / shrink_factor}, {"nu", 0.3}, {"CoR", 0.2}});

    float ball_rad = 0.2;
    auto template_ball =
        DEM_sim.LoadClumpSimpleSphere(ball_rad * ball_rad * ball_rad * 7e3 * 4 / 3 * 3.14, ball_rad, mat_type_ball);

    float terrain_rad = 0.02;
    auto template_terrain = DEM_sim.LoadClumpSimpleSphere(
        terrain_rad * terrain_rad * terrain_rad * 2.6e3 * 4 / 3 * 3.14, terrain_rad, mat_type_terrain);

    float step_size = 5e-6;
    double world_size = 1.5;

    float3 sample_center = make_float3(0, 0, -world_size / 4);
    float sample_halfheight = world_size / 4 * 0.95;
    float sample_halfwidth = world_size / 2 * 0.95;
    auto input_xyz = DEMBoxHCPSampler(sample_center, make_float3(sample_halfwidth, sample_halfwidth, sample_halfheight),
                                      2.01 * terrain_rad);
    auto terrain_particles = DEM_sim.AddClumps(template_terrain, input_xyz);
    auto terrain_tracker = DEM_sim.Track(terrain_particles);
    unsigned int nTerrainParticles = input_xyz.size();

    // Create a inspector to find out the highest point of this granular pile
    auto max_z_finder = DEM_sim.CreateInspector("clump_max_z");

    DEM_sim.InstructBoxDomainDimension(world_size, world_size, world_size * 2);
    DEM_sim.InstructBoxDomainBoundingBC("all", mat_type_terrain);
    DEM_sim.AddBCPlane(make_float3(0, 0, -world_size / 2), make_float3(0, 0, 1), mat_type_terrain);
    DEM_sim.SetCoordSysOrigin("center");
    DEM_sim.SetInitTimeStep(step_size);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8 / shrink_factor));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEM_sim.SetCDUpdateFreq(10);
    // DEM_sim.SetExpandFactor(1e-3);
    DEM_sim.SetMaxVelocity(2.);
    DEM_sim.SetExpandSafetyParam(1.1);
    DEM_sim.SetInitBinSize(4 * terrain_rad);
    DEM_sim.Initialize();

    path out_dir = current_path();
    out_dir += "/DEMdemo_PolydisperseGeneration";
    create_directory(out_dir);

    unsigned int currframe = 0;

    float high_z = max_z_finder->GetValue();
    std::cout << "Max Z is at " << high_z << std::endl;

    for (float t = 0; t < 0.1; t += 0.05) {
        {
            std::cout << "Frame: " << currframe << std::endl;
            DEM_sim.ShowThreadCollaborationStats();
            char filename[200];
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
            DEM_sim.WriteSphereFile(std::string(filename));
        }
        DEM_sim.DoDynamicsThenSync(0.05);
    }
    high_z = max_z_finder->GetValue();
    std::cout << "Max Z is at " << high_z << std::endl;

    ///////////////////////////////////////////////////////////////
    // As a proof of concept, we enlarge some of the particles on
    // the fly
    ///////////////////////////////////////////////////////////////
    std::random_device r_device;
    std::default_random_engine r_engine(r_device());
    std::discrete_distribution<unsigned int> discrete_dist({1, 2});
    std::vector<unsigned int> enlarge_ids;
    // Pick around 1/3 particles to enlarge
    for (unsigned int i = 0; i < nTerrainParticles; i++) {
        unsigned int num = std::round(discrete_dist(r_engine));
        if (num == 0) {
            enlarge_ids.push_back(i);
        }
    }

    step_size = 1e-6;
    DEM_sim.SetInitTimeStep(step_size);
    DEM_sim.UpdateSimParams();

    float sim_end = 10.0;
    unsigned int steps_togo = (unsigned int)(0.1 / step_size);
    unsigned int out_steps = steps_togo * 5;
    unsigned int curr_step = 0;

    float cur_terrain_rad = terrain_rad;
    float totalKE;
    // while (cur_terrain_rad < terrain_rad * 1.5) {
    //     double prev_rad = cur_terrain_rad;
    //     cur_terrain_rad += terrain_rad / 10;
    //     std::cout << "Current particle size: " << cur_terrain_rad << std::endl;
    //     std::vector<float> enlarge_ratio(enlarge_ids.size(), (double)cur_terrain_rad / prev_rad);
    //     terrain_tracker->ChangeClumpSizes(enlarge_ids, enlarge_ratio);

    //     do {
    //         // Must DoDynamicsThenSync (not DoDynamics), as adding entities to the simulation is only allowed at a
    //         // sync-ed point of time.
    //         DEM_sim.DoDynamicsThenSync(step_size * steps_togo);
    //         if (curr_step % out_steps == 0) {
    //             std::cout << "Frame: " << currframe << std::endl;
    //             char filename[200];
    //             sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
    //             DEM_sim.WriteSphereFile(std::string(filename));
    //         }
    //         totalKE = DEM_sim.GetTotalKineticEnergy();
    //         std::cout << "Total kinematic energy now: " << totalKE << std::endl;
    //         curr_step += steps_togo;
    //     } while (totalKE > 0.5);
    // }
    // DEM_sim.ShowThreadCollaborationStats();

    // Finally, output a `checkpoint'
    char cp_filename[200];
    sprintf(cp_filename, "%s/Final_xyz.csv", out_dir.c_str());
    DEM_sim.WriteClumpFile(std::string(cp_filename));

    // for (double t = 0; t < (double)sim_end; t += frame_time, currframe++) {
    //     std::cout << "Frame: " << currframe << std::endl;
    //     DEM_sim.ShowThreadCollaborationStats();
    //     char filename[200];
    //     sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
    //     DEM_sim.WriteSphereFile(std::string(filename));

    //     DEM_sim.DoDynamics(frame_time);

    //     //
    // }

    std::cout << "DEMdemo_PolydisperseGeneration exiting..." << std::endl;
    return 0;
}
