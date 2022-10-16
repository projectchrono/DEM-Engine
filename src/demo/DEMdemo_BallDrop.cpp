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
#include <filesystem>

using namespace deme;
using namespace std::filesystem;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.EnsureKernelErrMsgLineNum();

    // E, nu, CoR, mu, Crr...
    auto mat_type_ball = DEMSim.LoadMaterial({{"E", 1e10}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.3}, {"Crr", 0.01}});
    auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 5e9}, {"nu", 0.3}, {"CoR", 0.2}, {"mu", 0.3}, {"Crr", 0.01}});

    float step_size = 1e-5;
    double world_size = 10;
    DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size);
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);
    DEMSim.SetCoordSysOrigin("-13");

    auto projectile =
        DEMSim.AddWavefrontMeshObject((GET_SOURCE_DATA_PATH() / "mesh/sphere.obj").string(), mat_type_ball);
    std::cout << "Total num of triangles: " << projectile->GetNumTriangles() << std::endl;

    projectile->SetInitPos(make_float3(world_size / 2, world_size / 2, world_size / 3 * 2));
    float ball_mass = 2.6e3 * 4 / 3 * 3.1416;
    projectile->SetMass(ball_mass);
    projectile->SetMOI(make_float3(ball_mass * 2 / 5, ball_mass * 2 / 5, ball_mass * 2 / 5));
    projectile->SetFamily(1);

    float terrain_rad = 0.05;
    auto template_terrain = DEMSim.LoadSphereType(terrain_rad * terrain_rad * terrain_rad * 2.6e3 * 4 / 3 * 3.14,
                                                  terrain_rad, mat_type_terrain);

    // Track the projectile
    auto proj_tracker = DEMSim.Track(projectile);

    float sample_halfheight = world_size / 8;
    float3 sample_center = make_float3(world_size / 2, world_size / 2, sample_halfheight + 0.05);
    float sample_halfwidth = world_size / 2 * 0.95;
    auto input_xyz = DEMBoxHCPSampler(sample_center, make_float3(sample_halfwidth, sample_halfwidth, sample_halfheight),
                                      2.01 * terrain_rad);
    DEMSim.AddClumps(template_terrain, input_xyz);
    std::cout << "Total num of particles: " << input_xyz.size() << std::endl;

    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEMSim.SetCDUpdateFreq(20);
    // DEMSim.SetExpandFactor(1e-3);
    DEMSim.SetMaxVelocity(15.);
    DEMSim.SetExpandSafetyParam(1.1);
    DEMSim.SetInitBinSize(4 * terrain_rad);
    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir += "/DemoOutput_BallDrop";
    create_directory(out_dir);

    float sim_end = 10.0;
    unsigned int fps = 20;
    float frame_time = 1.0 / fps;

    // Testing...
    proj_tracker->UpdateMesh(projectile);

    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int currframe = 0;

    for (float t = 0; t < sim_end; t += frame_time) {
        std::cout << "Frame: " << currframe << std::endl;
        char filename[200], meshfilename[200], cnt_filename[200];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
        sprintf(meshfilename, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), currframe);
        // sprintf(cnt_filename, "%s/Contact_pairs_%04d.csv", out_dir.c_str(), currframe);
        DEMSim.WriteSphereFile(std::string(filename));
        DEMSim.WriteMeshFile(std::string(meshfilename));
        // DEMSim.WriteContactFile(std::string(cnt_filename));
        currframe++;

        DEMSim.DoDynamicsThenSync(frame_time);
        DEMSim.ShowThreadCollaborationStats();
    }

    std::cout << "DEMdemo_BallDrop exiting..." << std::endl;
    return 0;
}
