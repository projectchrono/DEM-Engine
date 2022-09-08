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

using namespace sgps;
using namespace std::filesystem;

int main() {
    DEMSolver DEM_sim;
    DEM_sim.SetVerbosity(DEBUG);
    DEM_sim.SetOutputFormat(DEM_OUTPUT_FORMAT::CSV);
    DEM_sim.SetOutputContent(DEM_OUTPUT_CONTENT::ABSV);

    // E, nu, CoR, mu, Crr...
    auto mat_type_ball = DEM_sim.LoadMaterial({{"E", 1e10}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.0}, {"Crr", 0.0}});

    auto projectile = DEM_sim.AddWavefrontMeshObject("./data/mesh/sphere.obj", mat_type_ball);
    std::cout << "Total num of triangles: " << projectile->GetNumTriangles() << std::endl;

    float step_size = 1e-5;
    double x_size = 15.;
    double y_size = 2;
    double z_size = 2.2;
    // DEM_sim.InstructBoxDomainNumVoxel(21, 21, 22, world_size / std::pow(2, 16) / std::pow(2, 21));
    DEM_sim.InstructBoxDomainDimension(x_size, y_size, z_size, DEM_SPATIAL_DIR::X);
    DEM_sim.SetCoordSysOrigin("center");
    DEM_sim.SetInitTimeStep(step_size);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEM_sim.SetCDUpdateFreq(10);
    // DEM_sim.SetExpandFactor(1e-3);
    DEM_sim.SetMaxVelocity(2.);
    DEM_sim.SetExpandSafetyParam(1.1);
    DEM_sim.Initialize();

    std::cout << "DEMdemo_TestWorldSizeGen exiting..." << std::endl;
    return 0;
}
