//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <cstdio>
#include <chrono>
#include <filesystem>

using namespace sgps;
using namespace std::filesystem;

int main() {
    DEMSolver DEM_sim;
    DEM_sim.SetVerbosity(INFO);
    DEM_sim.SetOutputFormat(DEM_OUTPUT_FORMAT::CSV);
    DEM_sim.SetOutputContent(DEM_OUTPUT_CONTENT::ABSV);

    // auto projectile = DEM_sim.AddWavefrontMeshObject("./data/mesh/sphere.obj");
    // std::cout << "Total num of triangles: " << projectile->GetNumTriangles() << std::endl;

    auto mat_type_ball = DEM_sim.LoadMaterialType(1e10, 0.3, 0.3);
    auto mat_type_terrain = DEM_sim.LoadMaterialType(5e9, 0.3, 0.2);

    float ball_rad = 0.2;
    auto template_ball =
        DEM_sim.LoadClumpSimpleSphere(ball_rad * ball_rad * ball_rad * 7e3 * 4 / 3 * 3.14, ball_rad, mat_type_ball);

    float terrain_rad = 0.01;
    auto template_terrain = DEM_sim.LoadClumpSimpleSphere(
        terrain_rad * terrain_rad * terrain_rad * 2.6e3 * 4 / 3 * 3.14, terrain_rad, mat_type_terrain);

    float step_size = 1e-5;
    double world_size = 1.5;

    float3 sample_center = make_float3(0, 0, -world_size / 4);
    float sample_halfheight = world_size / 4 * 0.95;
    float sample_halfwidth = world_size / 2 * 0.95;
    auto input_xyz = DEMBoxHCPSampler(sample_center, make_float3(sample_halfwidth, sample_halfwidth, sample_halfheight),
                                      2.01 * terrain_rad);
    auto terrain_particles = DEM_sim.AddClumps(template_terrain, input_xyz);
    auto terrain_tracker = DEM_sim.Track(terrain_particles);
    unsigned int nTerrainParticles = input_xyz.size();

    DEM_sim.InstructBoxDomainNumVoxel(21, 21, 22, world_size / std::pow(2, 16) / std::pow(2, 21));
    DEM_sim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);
    DEM_sim.AddBCPlane(make_float3(0, 0, -world_size / 2), make_float3(0, 0, 1), mat_type_terrain);
    DEM_sim.SetCoordSysOrigin("center");
    DEM_sim.SetInitTimeStep(step_size);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEM_sim.SetCDUpdateFreq(10);
    // DEM_sim.SetExpandFactor(1e-3);
    DEM_sim.SetMaxVelocity(2.);
    DEM_sim.SetExpandSafetyParam(1.1);
    DEM_sim.SetInitBinSize(6 * terrain_rad);
    DEM_sim.Initialize();

    path out_dir = current_path();
    out_dir += "/DEMdemo_BallDropPolydisperse";
    create_directory(out_dir);

    float sim_end = 20.0;
    unsigned int fps = 100;
    float frame_time = 1.0 / fps;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int currframe = 0;
    unsigned int curr_step = 0;

    for (float t = 0; t < 0.1; t += 0.05) {
        {
            std::cout << "Frame: " << currframe << std::endl;
            DEM_sim.ShowThreadCollaborationStats();
            char filename[200];
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
            DEM_sim.WriteClumpFile(std::string(filename));
        }
        DEM_sim.DoDynamics(0.05);
    }

    ///////////////////////////////////////////////////////////////
    // As a proof of concept, we enlarge some of the particles on
    // the fly
    ///////////////////////////////////////////////////////////////
    std::vector<unsigned int> enlarge_ids;
    std::vector<float> enlarge_ratio;
    // Pick around 1/3 particles to enlarge
    for (unsigned int i = 0; i < nTerrainParticles; i++) {
        if (i % 3 == 0) {
            enlarge_ids.push_back(i);
            enlarge_ratio.push_back(1.00002);
        }
    }

    float cur_terrain_rad = terrain_rad;
    while (cur_terrain_rad < terrain_rad * 2) {
        if (curr_step % out_steps == 0) {
            std::cout << "Frame: " << currframe << std::endl;
            DEM_sim.ShowThreadCollaborationStats();
            char filename[200];
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
            DEM_sim.WriteClumpFile(std::string(filename));
        }

        // Must DoDynamicsThenSync (not DoDynamics), as adding entities to the simulation is only allowed at a sync-ed
        // point of time.
        std::cout << "Current particle size: " << cur_terrain_rad << std::endl;
        terrain_tracker->ChangeClumpSizes(enlarge_ids, enlarge_ratio);
        DEM_sim.DoDynamicsThenSync(step_size);
        cur_terrain_rad *= enlarge_ratio.at(0);

        curr_step++;
    }
    DEM_sim.DoDynamicsThenSync(0.1);

    // for (double t = 0; t < (double)sim_end; t += frame_time, currframe++) {
    //     std::cout << "Frame: " << currframe << std::endl;
    //     DEM_sim.ShowThreadCollaborationStats();
    //     char filename[200];
    //     sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
    //     DEM_sim.WriteClumpFile(std::string(filename));

    //     DEM_sim.DoDynamics(frame_time);

    //     //
    // }

    std::cout << "DEMdemo_BallDropPolydisperse exiting..." << std::endl;
    // TODO: add end-game report APIs
    return 0;
}
