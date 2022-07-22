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

    auto projectile = DEM_sim.AddWavefrontMeshObject("./data/mesh/sphere.obj");
    std::cout << "Total num of triangles: " << projectile->GetNumTriangles() << std::endl;

    std::cout << "DEMdemo_BallDrop exiting..." << std::endl;
    // TODO: add end-game report APIs
    return 0;
}
