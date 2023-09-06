//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A meshed ball hitting a granular bed under gravity.
// =============================================================================

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
    DEMSim.SetVerbosity(STEP_METRIC);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.EnsureKernelErrMsgLineNum();

    // E, nu, CoR, mu, Crr...
    auto mat_type_ball = DEMSim.LoadMaterial({{"E", 1e10}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.3}, {"Crr", 0.01}});
    auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 5e9}, {"nu", 0.3}, {"CoR", 0.8}, {"mu", 0.3}, {"Crr", 0.01}});
    // If you don't have this line, then CoR between mixer material and granular material will be 0.7 (average of the
    // two).
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_ball, mat_type_terrain, 0.6);
    // Should do the same for mu and Crr, but since they are the same across 2 materials, it won't have an effect...

    float step_size = 1e-5;
    double world_size = 10;
    DEMSim.InstructBoxDomainDimension({0, world_size}, {0, world_size}, {0, world_size});
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);

    auto projectile = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/sphere.obj").string(), mat_type_ball);
    std::cout << "Total num of triangles: " << projectile->GetNumTriangles() << std::endl;

    projectile->SetInitPos(make_float3(world_size / 2, world_size / 2, world_size / 3 * 2));
    float ball_mass = 7.8e3 * 4 / 3 * 3.1416;
    projectile->SetMass(ball_mass);
    projectile->SetMOI(make_float3(ball_mass * 2 / 5, ball_mass * 2 / 5, ball_mass * 2 / 5));
    projectile->SetFamily(2);
    DEMSim.SetFamilyFixed(2);

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
    // Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    // wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    // happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(15.);
    // In general you don't have to worry about SetExpandSafetyAdder, unless if an entity has the property that a point
    // on it can move much faster than its CoM. In this demo, you are dealing with a meshed ball and you in fact don't
    // have this problem. In the Centrifuge demo though, this can be a problem since the centrifuge's CoM is not moving,
    // but its pointwise velocity can be high, so it needs to be accounted for using this method.
    DEMSim.SetExpandSafetyAdder(5.);
    // You usually don't have to worry about initial bin size. In very rare cases, init bin size is so bad that auto bin
    // size adaption is effectless, and you should notice in that case kT runs extremely slow. Then in that case setting
    // init bin size may save the simulation.
    // DEMSim.SetInitBinSize(4 * terrain_rad);

    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir += "/DemoOutput_BallDrop";
    create_directory(out_dir);

    float sim_time = 6.0;
    float settle_time = 2.0;
    unsigned int fps = 20;
    float frame_time = 1.0 / fps;

    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int currframe = 0;

    // We can let it settle first
    for (float t = 0; t < settle_time; t += frame_time) {
        std::cout << "Frame: " << currframe << std::endl;
        char filename[200], meshfilename[200];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
        sprintf(meshfilename, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), currframe);
        DEMSim.WriteSphereFile(std::string(filename));
        DEMSim.WriteMeshFile(std::string(meshfilename));
        currframe++;

        DEMSim.DoDynamicsThenSync(frame_time);
        DEMSim.ShowThreadCollaborationStats();
    }

    // Then drop the ball. I also wanted to test if changing step size method works fine here...
    step_size *= 0.5;
    DEMSim.UpdateStepSize(step_size);
    DEMSim.ChangeFamily(2, 1);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (float t = 0; t < sim_time; t += frame_time) {
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
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds (wall time) to finish the simulation" << std::endl;

    DEMSim.ShowTimingStats();
    DEMSim.ShowAnomalies();
    std::cout << "DEMdemo_BallDrop exiting..." << std::endl;
    return 0;
}
